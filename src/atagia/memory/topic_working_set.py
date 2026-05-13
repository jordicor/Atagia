"""Offline Topic Working Set updates driven by structured LLM plans."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.llm_output_limits import TOPIC_WORKING_SET_MAX_OUTPUT_TOKENS
from atagia.core.repositories import MessageRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.intimacy_boundary_policy import strongest_intimacy_boundary
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    known_intimacy_context_metadata,
)
from atagia.services.model_resolution import resolve_component_model


class TopicUpdateActionType(StrEnum):
    """Supported offline topic working-set mutations."""

    CREATE = "create"
    UPDATE = "update"
    PARK = "park"
    REOPEN = "reopen"
    CLOSE = "close"
    NOOP = "noop"


_TOPIC_STRING_SENTINEL = ""
_TOPIC_FLOAT_SENTINEL = -1.0
_TOPIC_INT_SENTINEL = -1
_TOPIC_BOUNDARY_SENTINEL = ""
_INTIMACY_BOUNDARY_VALUES = {boundary.value for boundary in IntimacyBoundary}


class TopicUpdateAction(BaseModel):
    """One structured mutation proposed by the topic updater model."""

    model_config = ConfigDict(extra="ignore")

    action: TopicUpdateActionType
    topic_id: str = _TOPIC_STRING_SENTINEL
    parent_topic_id: str = _TOPIC_STRING_SENTINEL
    title: str = _TOPIC_STRING_SENTINEL
    summary: str = _TOPIC_STRING_SENTINEL
    active_goal: str = _TOPIC_STRING_SENTINEL
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    confidence: float = _TOPIC_FLOAT_SENTINEL
    privacy_level: int = _TOPIC_INT_SENTINEL
    intimacy_boundary: str = _TOPIC_BOUNDARY_SENTINEL
    intimacy_boundary_confidence: float = _TOPIC_FLOAT_SENTINEL

    @model_validator(mode="before")
    @classmethod
    def normalize_wire_sentinels(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in (
            "topic_id",
            "parent_topic_id",
            "title",
            "summary",
            "active_goal",
            "intimacy_boundary",
        ):
            if normalized.get(field_name) is None:
                normalized[field_name] = _TOPIC_STRING_SENTINEL
        for field_name in ("confidence", "intimacy_boundary_confidence"):
            if normalized.get(field_name) is None:
                normalized[field_name] = _TOPIC_FLOAT_SENTINEL
        if normalized.get("privacy_level") is None:
            normalized["privacy_level"] = _TOPIC_INT_SENTINEL
        return normalized

    @model_validator(mode="after")
    def validate_action_shape(self) -> "TopicUpdateAction":
        if self.action is TopicUpdateActionType.CREATE and not self.title.strip():
            raise ValueError("create actions require a title")
        if self.action in {
            TopicUpdateActionType.UPDATE,
            TopicUpdateActionType.PARK,
            TopicUpdateActionType.REOPEN,
            TopicUpdateActionType.CLOSE,
        } and not self.topic_id.strip():
            raise ValueError(f"{self.action.value} actions require topic_id")
        if self.confidence != _TOPIC_FLOAT_SENTINEL and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be -1.0 or between 0.0 and 1.0")
        if self.privacy_level != _TOPIC_INT_SENTINEL and not 0 <= self.privacy_level <= 3:
            raise ValueError("privacy_level must be -1 or between 0 and 3")
        boundary = self.intimacy_boundary.strip()
        if boundary and boundary not in _INTIMACY_BOUNDARY_VALUES:
            raise ValueError("intimacy_boundary is not recognized")
        if (
            self.intimacy_boundary_confidence != _TOPIC_FLOAT_SENTINEL
            and not 0.0 <= self.intimacy_boundary_confidence <= 1.0
        ):
            raise ValueError(
                "intimacy_boundary_confidence must be -1.0 or between 0.0 and 1.0"
            )
        return self


def _wire_string_to_none(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _wire_float_to_none(value: float) -> float | None:
    return None if value == _TOPIC_FLOAT_SENTINEL else value


def _wire_int_to_none(value: int) -> int | None:
    return None if value == _TOPIC_INT_SENTINEL else value


def _wire_boundary_to_none(value: str) -> IntimacyBoundary | None:
    stripped = value.strip()
    if not stripped:
        return None
    return IntimacyBoundary(stripped)


class TopicWorkingSetPlan(BaseModel):
    """Structured output returned by the topic updater model."""

    model_config = ConfigDict(extra="ignore")

    actions: list[TopicUpdateAction] = Field(default_factory=list)
    nothing_to_update: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_root_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"actions": value, "nothing_to_update": not value}
        return value

    @model_validator(mode="after")
    def validate_nothing_to_update_consistency(self) -> "TopicWorkingSetPlan":
        meaningful_actions = [
            action for action in self.actions if action.action is not TopicUpdateActionType.NOOP
        ]
        if self.nothing_to_update and meaningful_actions:
            raise ValueError("nothing_to_update=true but actions are non-empty")
        return self


class TopicWorkingSetUpdater:
    """Applies model-planned topic working-set updates outside the response path."""

    def __init__(
        self,
        *,
        llm_client: LLMClient[Any],
        clock: Clock,
        topic_repository: TopicRepository,
        message_repository: MessageRepository,
        settings: Settings | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._clock = clock
        self._topic_repository = topic_repository
        self._message_repository = message_repository
        resolved_settings = settings or Settings.from_env()
        self._model = resolve_component_model(resolved_settings, "topic_working_set")

    async def update_from_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Plan and persist topic updates for an offline message batch."""
        if not messages:
            return []
        snapshot = await self._topic_repository.get_topic_snapshot(
            user_id=user_id,
            conversation_id=conversation_id,
            active_limit=6,
            parked_limit=12,
        )
        plan = await self._plan_updates(
            user_id=user_id,
            conversation_id=conversation_id,
            snapshot=snapshot,
            messages=messages,
        )
        if plan.nothing_to_update:
            await self._record_processed_batch(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=messages,
            )
            return []
        return await self._apply_plan(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=messages,
            plan=plan,
        )

    async def _plan_updates(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> TopicWorkingSetPlan:
        prompt = self._build_prompt(
            conversation_id=conversation_id,
            snapshot=snapshot,
            messages=messages,
        )
        request = LLMCompletionRequest(
            model=self._model,
            messages=[
                LLMMessage(
                    role="system",
                    content="Maintain conversation topic working sets as structured JSON.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_output_tokens=TOPIC_WORKING_SET_MAX_OUTPUT_TOKENS,
            response_schema=TopicWorkingSetPlan.model_json_schema(),
            metadata={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "purpose": "topic_working_set_update",
                **self._intimacy_metadata_from_snapshot(snapshot),
            },
        )
        return await self._llm_client.complete_structured(request, TopicWorkingSetPlan)

    def _build_prompt(
        self,
        *,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> str:
        boundary_values = ", ".join(boundary.value for boundary in IntimacyBoundary)
        message_payload = [
            {
                "id": str(message["id"]),
                "seq": message.get("seq"),
                "role": str(message["role"]),
                "text": self._message_text_for_topic_prompt(message),
                "raw_text_included": self._message_raw_text_allowed(message),
                "content_kind": message.get("content_kind") or "text",
                "policy_reason": message.get("policy_reason") or "normal",
                "created_at": message.get("created_at"),
                "artifact_refs": self._message_artifact_refs(message),
            }
            for message in messages
        ]
        return "\n".join(
            [
                "Update the read-only Topic Working Set for this conversation.",
                "Return only JSON matching the provided schema.",
                "Do not include markdown fences, preambles, tags, or explanations.",
                "Anything outside the first JSON object will be ignored.",
                "Do not create dataset-specific or benchmark-specific topics.",
                "Prefer updating existing topics when the new messages continue the same thread.",
                "Use source_message_ids only from the provided message IDs.",
                (
                    "Use wire sentinels when a field is not applicable or should "
                    "not be changed: omit any field other than action, or use "
                    '"" for absent string fields; -1.0 for absent confidence '
                    "fields; -1 for absent privacy_level."
                ),
                (
                    'intimacy_boundary="" means not provided/no boundary change; '
                    'intimacy_boundary="ordinary" means explicitly ordinary.'
                ),
                f"Valid intimacy_boundary values are exactly: {boundary_values}.",
                (
                    "For each created or updated topic, set intimacy_boundary "
                    "semantically. Use ordinary unless the topic itself is private "
                    "romantic/intimate context, an intimacy boundary, or "
                    "ambiguous intimate context."
                ),
                (
                    "For non-ordinary intimacy_boundary values, set privacy_level "
                    "to at least 2 and avoid exposing sensitive wording in "
                    "topic titles where a neutral local label is enough."
                ),
                (
                    "Use artifact_ids only from provided artifact_refs when a topic "
                    "depends on an attachment."
                ),
                f"conversation_id={conversation_id}",
                "<existing_topic_snapshot>",
                json_utils.dumps(snapshot, indent=2, sort_keys=True),
                "</existing_topic_snapshot>",
                "<messages>",
                json_utils.dumps(message_payload, indent=2, sort_keys=True),
                "</messages>",
            ]
        )

    @staticmethod
    def _message_text_for_topic_prompt(message: dict[str, Any]) -> str:
        if TopicWorkingSetUpdater._message_raw_text_allowed(message):
            return str(message["text"])
        placeholder = str(message.get("context_placeholder") or "").strip()
        if placeholder:
            return placeholder
        return (
            "[Message omitted from Topic Working Set raw processing | "
            f"id={message.get('id')} "
            f"seq={message.get('seq')} "
            f"role={message.get('role')} "
            f"content_kind={message.get('content_kind') or 'text'} "
            f"policy_reason={message.get('policy_reason') or 'skip_by_default'}]"
        )

    @staticmethod
    def _message_raw_text_allowed(message: dict[str, Any]) -> bool:
        include_raw = message.get("include_raw", True)
        if isinstance(include_raw, bool):
            raw_allowed = include_raw
        elif isinstance(include_raw, (int, float)):
            raw_allowed = bool(include_raw)
        elif isinstance(include_raw, str):
            raw_allowed = include_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            raw_allowed = bool(include_raw)
        return raw_allowed and not bool(message.get("skip_by_default"))

    @staticmethod
    def _intimacy_metadata_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        topics = [
            *(snapshot.get("active_topics") or []),
            *(snapshot.get("parked_topics") or []),
        ]
        rows = [
            topic
            for topic in topics
            if isinstance(topic, dict)
            and str(topic.get("intimacy_boundary") or "ordinary") != "ordinary"
        ]
        if not rows:
            return {}
        boundary = strongest_intimacy_boundary(rows)
        confidence = max(
            (
                float(topic.get("intimacy_boundary_confidence", 0.0) or 0.0)
                for topic in rows
            ),
            default=0.0,
        )
        return known_intimacy_context_metadata(
            reason="topic_working_set_intimacy_boundary",
            boundary=boundary.value,
            confidence=confidence,
        )

    @staticmethod
    def _message_artifact_refs(message: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = message.get("metadata_json")
        if not isinstance(metadata, dict):
            return []
        refs_by_id: dict[str, dict[str, Any]] = {}
        attachments = metadata.get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                artifact_id = attachment.get("artifact_id")
                if not artifact_id:
                    continue
                artifact_ref = {
                    "artifact_id": str(artifact_id),
                    "artifact_type": attachment.get("artifact_type"),
                    "source_kind": attachment.get("source_kind"),
                    "mime_type": attachment.get("mime_type"),
                    "filename": attachment.get("filename"),
                    "title": attachment.get("title"),
                    "privacy_level": attachment.get("privacy_level"),
                    "preserve_verbatim": attachment.get("preserve_verbatim"),
                    "requires_explicit_request": attachment.get("requires_explicit_request"),
                    "relevance_state": attachment.get("relevance_state"),
                }
                refs_by_id[str(artifact_id)] = {
                    key: value for key, value in artifact_ref.items() if value is not None
                }
        attachment_ids = metadata.get("attachment_artifact_ids")
        if isinstance(attachment_ids, list):
            for artifact_id in attachment_ids:
                if artifact_id and str(artifact_id) not in refs_by_id:
                    refs_by_id[str(artifact_id)] = {"artifact_id": str(artifact_id)}
        return list(refs_by_id.values())

    async def _apply_plan(
        self,
        *,
        user_id: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
        plan: TopicWorkingSetPlan,
    ) -> list[dict[str, Any]]:
        changed_topics: list[dict[str, Any]] = []
        provided_artifact_ids = self._provided_artifact_ids(messages)
        source_start_seq, source_end_seq = self._message_seq_bounds(messages)
        for action in plan.actions:
            if action.action is TopicUpdateActionType.NOOP:
                continue
            valid_artifact_ids = [
                artifact_id
                for artifact_id in action.artifact_ids
                if artifact_id in provided_artifact_ids
            ]
            topic = await self._apply_action(
                user_id=user_id,
                conversation_id=conversation_id,
                action=action,
                artifact_ids=valid_artifact_ids,
                source_start_seq=source_start_seq,
                source_end_seq=source_end_seq,
            )
            if topic is None:
                continue
            for source_message_id in await self._valid_source_message_ids(
                user_id=user_id,
                conversation_id=conversation_id,
                source_message_ids=action.source_message_ids,
            ):
                await self._topic_repository.link_source(
                    user_id=user_id,
                    topic_id=str(topic["id"]),
                    source_kind="message",
                    source_id=source_message_id,
                    relation_kind="evidence",
                    commit=False,
                )
            for artifact_id in valid_artifact_ids:
                await self._topic_repository.link_source(
                    user_id=user_id,
                    topic_id=str(topic["id"]),
                    source_kind="artifact",
                    source_id=artifact_id,
                    relation_kind="evidence",
                    commit=False,
                )
            changed_topics.append(topic)
        if not changed_topics:
            await self._record_processed_batch(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=messages,
                commit=False,
            )
        await self._topic_repository.commit()
        return changed_topics

    async def _record_processed_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
        commit: bool = True,
    ) -> None:
        _source_start_seq, source_end_seq = self._message_seq_bounds(messages)
        if source_end_seq is None:
            return
        source_message_id = str(messages[-1]["id"]) if messages and messages[-1].get("id") else None
        await self._topic_repository.create_event(
            user_id=user_id,
            conversation_id=conversation_id,
            topic_id=None,
            event_type="updated",
            source_message_id=source_message_id,
            payload={
                "source": "offline_topic_working_set_updater",
                "processed_through_seq": source_end_seq,
                "changed_topic_count": 0,
            },
            commit=commit,
        )

    async def _apply_action(
        self,
        *,
        user_id: str,
        conversation_id: str,
        action: TopicUpdateAction,
        artifact_ids: list[str],
        source_start_seq: int | None,
        source_end_seq: int | None,
    ) -> dict[str, Any] | None:
        parent_topic_id_value = _wire_string_to_none(action.parent_topic_id)
        topic_id_value = _wire_string_to_none(action.topic_id)
        title_value = _wire_string_to_none(action.title)
        summary_value = _wire_string_to_none(action.summary)
        active_goal_value = _wire_string_to_none(action.active_goal)
        confidence_value = _wire_float_to_none(action.confidence)
        privacy_level_value = _wire_int_to_none(action.privacy_level)
        boundary_value = _wire_boundary_to_none(action.intimacy_boundary)
        boundary_confidence_value = _wire_float_to_none(action.intimacy_boundary_confidence)

        parent_topic_id = await self._valid_parent_topic_id(
            user_id=user_id,
            conversation_id=conversation_id,
            parent_topic_id=parent_topic_id_value,
        )
        if action.action is TopicUpdateActionType.CREATE:
            if title_value is None:
                raise ValueError("create actions require a title")
            create_boundary = boundary_value or IntimacyBoundary.ORDINARY
            create_privacy_level = privacy_level_value if privacy_level_value is not None else 0
            if create_boundary is not IntimacyBoundary.ORDINARY:
                create_privacy_level = max(create_privacy_level, 2)
            return await self._topic_repository.create_topic(
                user_id=user_id,
                conversation_id=conversation_id,
                parent_topic_id=parent_topic_id,
                title=title_value,
                summary=summary_value or "",
                active_goal=active_goal_value,
                open_questions=action.open_questions,
                decisions=action.decisions,
                artifact_ids=artifact_ids,
                source_message_start_seq=source_start_seq,
                source_message_end_seq=source_end_seq,
                last_touched_seq=source_end_seq,
                confidence=confidence_value if confidence_value is not None else 0.5,
                privacy_level=create_privacy_level,
                intimacy_boundary=create_boundary,
                intimacy_boundary_confidence=(
                    boundary_confidence_value if boundary_confidence_value is not None else 0.0
                ),
                last_touched_at=self._clock.now().isoformat(),
                commit=False,
            )

        status = _status_for_action(action.action)
        existing = (
            await self._topic_repository.get_topic(topic_id_value, user_id)
            if topic_id_value is not None
            else None
        )
        if existing is None or str(existing["conversation_id"]) != conversation_id:
            return None
        update_privacy_level = privacy_level_value
        if boundary_value is not None and boundary_value is not IntimacyBoundary.ORDINARY:
            update_privacy_level = max(
                int(existing.get("privacy_level") or 0),
                int(privacy_level_value or 0),
                2,
            )
        return await self._topic_repository.update_topic(
            topic_id=str(topic_id_value),
            user_id=user_id,
            status=status,
            title=title_value,
            summary=summary_value,
            active_goal=active_goal_value,
            open_questions=action.open_questions if action.open_questions else None,
            decisions=action.decisions if action.decisions else None,
            artifact_ids=artifact_ids if artifact_ids else None,
            source_message_start_seq=(
                existing.get("source_message_start_seq")
                if existing is not None and existing.get("source_message_start_seq") is not None
                else source_start_seq
            ),
            source_message_end_seq=source_end_seq,
            last_touched_seq=source_end_seq,
            confidence=confidence_value,
            privacy_level=update_privacy_level,
            intimacy_boundary=boundary_value,
            intimacy_boundary_confidence=boundary_confidence_value,
            last_touched_at=self._clock.now().isoformat(),
            event_type=_event_type_for_action(action.action),
            event_payload={
                "source": "offline_topic_working_set_updater",
                "processed_through_seq": source_end_seq,
            },
            commit=False,
        )

    async def _valid_parent_topic_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
        parent_topic_id: str | None,
    ) -> str | None:
        if parent_topic_id is None:
            return None
        parent = await self._topic_repository.get_topic(parent_topic_id, user_id)
        if parent is None or str(parent["conversation_id"]) != conversation_id:
            return None
        return parent_topic_id

    async def _valid_source_message_ids(
        self,
        *,
        user_id: str,
        conversation_id: str,
        source_message_ids: list[str],
    ) -> list[str]:
        valid_ids: list[str] = []
        for source_message_id in source_message_ids:
            source_message = await self._message_repository.get_message(source_message_id, user_id)
            if source_message is None or source_message["conversation_id"] != conversation_id:
                continue
            valid_ids.append(source_message_id)
        return valid_ids

    def _provided_artifact_ids(self, messages: list[dict[str, Any]]) -> set[str]:
        artifact_ids: set[str] = set()
        for message in messages:
            for artifact_ref in self._message_artifact_refs(message):
                artifact_id = artifact_ref.get("artifact_id")
                if artifact_id:
                    artifact_ids.add(str(artifact_id))
        return artifact_ids

    @staticmethod
    def _message_seq_bounds(messages: list[dict[str, Any]]) -> tuple[int | None, int | None]:
        seqs: list[int] = []
        for message in messages:
            seq = message.get("seq")
            if isinstance(seq, int):
                seqs.append(seq)
            elif isinstance(seq, str) and seq.isdigit():
                seqs.append(int(seq))
        if not seqs:
            return None, None
        return min(seqs), max(seqs)


def _status_for_action(action: TopicUpdateActionType) -> str | None:
    if action is TopicUpdateActionType.PARK:
        return "parked"
    if action is TopicUpdateActionType.REOPEN:
        return "active"
    if action is TopicUpdateActionType.CLOSE:
        return "closed"
    return None


def _event_type_for_action(action: TopicUpdateActionType) -> str:
    if action is TopicUpdateActionType.PARK:
        return "parked"
    if action is TopicUpdateActionType.REOPEN:
        return "reopened"
    if action is TopicUpdateActionType.CLOSE:
        return "closed"
    return "updated"
