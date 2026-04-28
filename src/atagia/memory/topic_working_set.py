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
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
from atagia.services.model_resolution import resolve_component_model


class TopicUpdateActionType(StrEnum):
    """Supported offline topic working-set mutations."""

    CREATE = "create"
    UPDATE = "update"
    PARK = "park"
    REOPEN = "reopen"
    CLOSE = "close"
    NOOP = "noop"


class TopicUpdateAction(BaseModel):
    """One structured mutation proposed by the topic updater model."""

    model_config = ConfigDict(extra="ignore")

    action: TopicUpdateActionType
    topic_id: str | None = None
    parent_topic_id: str | None = None
    title: str | None = None
    summary: str | None = None
    active_goal: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    privacy_level: int | None = Field(default=None, ge=0, le=3)

    @model_validator(mode="after")
    def validate_action_shape(self) -> "TopicUpdateAction":
        if self.action is TopicUpdateActionType.CREATE and not self.title:
            raise ValueError("create actions require a title")
        if self.action in {
            TopicUpdateActionType.UPDATE,
            TopicUpdateActionType.PARK,
            TopicUpdateActionType.REOPEN,
            TopicUpdateActionType.CLOSE,
        } and not self.topic_id:
            raise ValueError(f"{self.action.value} actions require topic_id")
        return self


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
        message_payload = [
            {
                "id": str(message["id"]),
                "seq": message.get("seq"),
                "role": str(message["role"]),
                "text": str(message["text"]),
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
        await self._topic_repository.commit()
        return changed_topics

    async def _apply_action(
        self,
        *,
        user_id: str,
        conversation_id: str,
        action: TopicUpdateAction,
        artifact_ids: list[str],
    ) -> dict[str, Any] | None:
        if action.action is TopicUpdateActionType.CREATE:
            return await self._topic_repository.create_topic(
                user_id=user_id,
                conversation_id=conversation_id,
                parent_topic_id=action.parent_topic_id,
                title=str(action.title),
                summary=action.summary or "",
                active_goal=action.active_goal,
                open_questions=action.open_questions,
                decisions=action.decisions,
                artifact_ids=artifact_ids,
                confidence=action.confidence if action.confidence is not None else 0.5,
                privacy_level=action.privacy_level if action.privacy_level is not None else 0,
                last_touched_at=self._clock.now().isoformat(),
                commit=False,
            )

        status = _status_for_action(action.action)
        return await self._topic_repository.update_topic(
            topic_id=str(action.topic_id),
            user_id=user_id,
            status=status,
            title=action.title,
            summary=action.summary,
            active_goal=action.active_goal,
            open_questions=action.open_questions if action.open_questions else None,
            decisions=action.decisions if action.decisions else None,
            artifact_ids=artifact_ids if artifact_ids else None,
            confidence=action.confidence,
            privacy_level=action.privacy_level,
            last_touched_at=self._clock.now().isoformat(),
            event_type=_event_type_for_action(action.action),
            event_payload={"source": "offline_topic_working_set_updater"},
            commit=False,
        )

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
