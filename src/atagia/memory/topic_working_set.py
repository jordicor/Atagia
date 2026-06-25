"""Offline Topic Working Set updates driven by small LLM cards."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import MessageRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.intimacy_boundary_policy import (
    normalize_intimacy_boundary,
    strongest_intimacy_boundary,
)
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    known_intimacy_context_metadata,
)
from atagia.services.model_resolution import (
    examples_enabled_for_component,
    resolve_component_model,
)

logger = logging.getLogger(__name__)


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

TopicWorkingSetCardName = Literal["route", "content", "boundary"]

TOPIC_WORKING_SET_CARD_CONCURRENCY = 2

_CARD_PURPOSES: dict[TopicWorkingSetCardName, str] = {
    "route": "topic_working_set_route_card",
    "content": "topic_working_set_content_card",
    "boundary": "topic_working_set_boundary_card",
}
_CARD_MAX_OUTPUT_TOKENS: dict[TopicWorkingSetCardName, int] = {
    "route": 192,
    "content": 512,
    "boundary": 192,
}
_MAX_TOPIC_CARD_ACTIONS = 6
_CONTENT_ACTIONS = {TopicUpdateActionType.CREATE, TopicUpdateActionType.UPDATE}


@dataclass(frozen=True, slots=True)
class _TopicRoute:
    action: TopicUpdateActionType
    target_id: str
    source_message_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TopicContent:
    title: str | None = None
    summary: str | None = None
    active_goal: str | None = None
    open_questions: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TopicBoundary:
    boundary: IntimacyBoundary
    privacy_level: int | None = None
    confidence: float = 0.7


@dataclass(frozen=True, slots=True)
class _TopicCardPlan:
    routes: tuple[_TopicRoute, ...]
    contents: dict[str, _TopicContent] = field(default_factory=dict)
    boundaries: dict[str, _TopicBoundary] = field(default_factory=dict)
    artifacts: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
        self._include_examples = examples_enabled_for_component(
            resolved_settings, "topic_working_set"
        )
        self._card_models = {
            card_name: self._model for card_name in _CARD_PURPOSES
        }

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
        existing_routes = await self._run_existing_route_card(
            user_id=user_id,
            conversation_id=conversation_id,
            snapshot=snapshot,
            messages=messages,
        )
        uncovered_messages = self._messages_not_covered_by_routes(
            messages,
            existing_routes,
        )
        new_routes = (
            await self._run_new_topic_track_card(
                user_id=user_id,
                conversation_id=conversation_id,
                snapshot=snapshot,
                messages=uncovered_messages,
            )
            if uncovered_messages
            else ()
        )
        routes = _dedupe_routes([*existing_routes, *new_routes])
        if not routes:
            return TopicWorkingSetPlan(actions=[], nothing_to_update=True)

        content_routes = tuple(
            route for route in routes if route.action in _CONTENT_ACTIONS
        )
        contents = await self._run_content_cards(
            user_id=user_id,
            conversation_id=conversation_id,
            snapshot=snapshot,
            messages=messages,
            routes=content_routes,
        )
        boundaries = await self._run_boundary_cards(
            user_id=user_id,
            conversation_id=conversation_id,
            snapshot=snapshot,
            messages=messages,
            routes=content_routes,
            contents=contents,
        )
        artifacts = self._artifact_ids_from_route_messages(messages, tuple(routes))
        return _topic_card_plan_to_structured_plan(
            _TopicCardPlan(
                routes=tuple(routes),
                contents=contents,
                boundaries=boundaries,
                artifacts=artifacts,
            )
        )

    async def _run_existing_route_card(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> tuple[_TopicRoute, ...]:
        request = self._card_request(
            card_name="route",
            user_id=user_id,
            conversation_id=conversation_id,
            prompt=self._build_existing_route_prompt(
                conversation_id=conversation_id,
                snapshot=snapshot,
                messages=messages,
            ),
            snapshot=snapshot,
        )
        response = await self._llm_client.complete(request)
        return tuple(
            route
            for route in _parse_route_card_output(
                response.output_text,
                valid_topic_ids=_topic_ids_from_snapshot(snapshot),
                valid_message_ids=_message_ids_from_messages(messages),
                conversation_id=conversation_id,
            )
            if route.action is not TopicUpdateActionType.CREATE
        )

    async def _run_new_topic_track_card(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> tuple[_TopicRoute, ...]:
        request = self._card_request(
            card_name="route",
            user_id=user_id,
            conversation_id=conversation_id,
            prompt=self._build_new_topic_track_prompt(
                conversation_id=conversation_id,
                snapshot=snapshot,
                messages=messages,
            ),
            snapshot=snapshot,
        )
        response = await self._llm_client.complete(request)
        return _parse_new_topic_track_output(
            response.output_text,
            valid_message_ids=_message_ids_from_messages(messages),
            conversation_id=conversation_id,
        )

    async def _run_content_cards(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
        routes: tuple[_TopicRoute, ...],
    ) -> dict[str, _TopicContent]:
        if not routes:
            return {}
        topics_by_id = _topics_by_id_from_snapshot(snapshot)
        semaphore = asyncio.Semaphore(TOPIC_WORKING_SET_CARD_CONCURRENCY)

        async def run_card(route: _TopicRoute) -> tuple[str, _TopicContent]:
            async with semaphore:
                request = self._card_request(
                    card_name="content",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    prompt=self._build_content_prompt(
                        conversation_id=conversation_id,
                        snapshot=snapshot,
                        messages=messages,
                        route=route,
                        existing_topic=topics_by_id.get(route.target_id),
                    ),
                    snapshot=snapshot,
                    target_id=route.target_id,
                )
                response = await self._llm_client.complete(request)
                return route.target_id, _parse_content_card_output(response.output_text)

        results = await asyncio.gather(*(run_card(route) for route in routes))
        return {target_id: content for target_id, content in results}

    async def _run_boundary_cards(
        self,
        *,
        user_id: str,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
        routes: tuple[_TopicRoute, ...],
        contents: dict[str, _TopicContent],
    ) -> dict[str, _TopicBoundary]:
        if not routes:
            return {}
        semaphore = asyncio.Semaphore(TOPIC_WORKING_SET_CARD_CONCURRENCY)

        async def run_card(route: _TopicRoute) -> tuple[str, _TopicBoundary | None]:
            async with semaphore:
                request = self._card_request(
                    card_name="boundary",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    prompt=self._build_target_boundary_prompt(
                        conversation_id=conversation_id,
                        messages=messages,
                        route=route,
                        content=contents.get(route.target_id, _TopicContent()),
                    ),
                    snapshot=snapshot,
                    target_id=route.target_id,
                )
                response = await self._llm_client.complete(request)
                boundaries = _parse_boundary_card_output(
                    response.output_text,
                    valid_target_ids={route.target_id},
                )
                return route.target_id, boundaries.get(route.target_id)

        results = await asyncio.gather(*(run_card(route) for route in routes))
        return {
            target_id: boundary
            for target_id, boundary in results
            if boundary is not None
        }

    def _card_request(
        self,
        *,
        card_name: TopicWorkingSetCardName,
        user_id: str,
        conversation_id: str,
        prompt: str,
        snapshot: dict[str, Any],
        target_id: str | None = None,
    ) -> LLMCompletionRequest:
        purpose = _CARD_PURPOSES[card_name]
        metadata: dict[str, Any] = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "purpose": purpose,
            "topic_working_set_card": card_name,
            **self._intimacy_metadata_from_snapshot(snapshot),
        }
        if target_id is not None:
            metadata["topic_working_set_target_id"] = target_id
        return LLMCompletionRequest(
            model=self._card_models[card_name],
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Keep track of the topics in this conversation, using the data below. "
                        "Write only the requested plain-text lines. No JSON. No explanation."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_output_tokens=_CARD_MAX_OUTPUT_TOKENS[card_name],
            metadata=metadata,
        )

    def _build_existing_route_prompt(
        self,
        *,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> str:
        instruction_head = "\n".join(
            [
                "Decide whether this message batch changes one of the topics we are already tracking.",
                "Only consider existing topics from the snapshot.",
                "Never create a new topic in this card.",
                "Write one line per touched existing topic, or exactly: none",
                "Allowed actions:",
                "update = the same topic continues and should remain active",
                "park = the user pauses, postpones, defers, or puts this topic aside",
                "reopen = the user resumes a parked topic",
                "close = the user says this topic is done, finished, resolved, or no longer active",
                "Format: action topic_id message_id [message_id ...]",
                "Use only topic ids from the snapshot.",
                "Use only message ids from the provided messages.",
                "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            ]
        )
        examples_block = "\n".join(
            [
                "Snapshot topic tpc_42 (invoice cleanup); user says 'also add vendor IDs to the invoice audit notes' in msg_8.",
                "update tpc_42 msg_8",
                "Snapshot topic tpc_7 (model comparison); user says 'let's pause that comparison until the run finishes' in msg_3.",
                "park tpc_7 msg_3",
                "Snapshot has a parked topic tpc_11 (moving plan); user says 'back to the moving plan' in msg_5.",
                "reopen tpc_11 msg_5",
                "Snapshot topic tpc_19 (bug triage); user says 'that bug is fixed, we can close it' in msg_2.",
                "close tpc_19 msg_2",
                "No snapshot topic matches, or the batch is only a greeting.",
                "none",
            ]
        )
        body = compose_card_prompt(
            instruction_head,
            examples_block,
            include_examples=self._include_examples,
        )
        return "\n".join(
            [
                body,
                f"conversation_id={conversation_id}",
                "<existing_topic_snapshot>",
                json_utils.dumps(snapshot, indent=2, sort_keys=True),
                "</existing_topic_snapshot>",
                "<messages>",
                json_utils.dumps(self._message_payload(messages), indent=2, sort_keys=True),
                "</messages>",
            ]
        )

    def _build_new_topic_track_prompt(
        self,
        *,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> str:
        instruction_head = "\n".join(
            [
                "This card only sees messages not already assigned to an existing topic.",
                "Decide whether these remaining messages introduce one new local topic.",
                "Default answer: track.",
                "Write ignore only when there is no local subject to carry forward.",
                "Ignore pure greetings, thanks, empty chatter, or non-subject fragments.",
                "Track any subject the assistant may need as local conversation context.",
                "Track tasks, decisions, problems, plans, personal situations, attachments, and ongoing discussions.",
                "Track private or sensitive subjects too; privacy is handled by a later card.",
                "Write exactly one line.",
                "Format when tracking: track message_id [message_id ...]",
                "Otherwise write exactly: ignore",
                "Use only message ids from the provided messages.",
                "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            ]
        )
        examples_block = "\n".join(
            [
                "Remaining message msg_1: 'I want to plan a quiet birthday dinner next month.'",
                "track msg_1",
                "Remaining messages msg_2 and msg_3 describe a new bug and how to reproduce it.",
                "track msg_2 msg_3",
                "Remaining message: 'Thanks, that helps.'",
                "ignore",
            ]
        )
        body = compose_card_prompt(
            instruction_head,
            examples_block,
            include_examples=self._include_examples,
        )
        return "\n".join(
            [
                body,
                f"conversation_id={conversation_id}",
                "<existing_topic_snapshot>",
                json_utils.dumps(snapshot, indent=2, sort_keys=True),
                "</existing_topic_snapshot>",
                "<uncovered_messages>",
                json_utils.dumps(self._message_payload(messages), indent=2, sort_keys=True),
                "</uncovered_messages>",
            ]
        )

    def _build_content_prompt(
        self,
        *,
        conversation_id: str,
        snapshot: dict[str, Any],
        messages: list[dict[str, Any]],
        route: _TopicRoute,
        existing_topic: dict[str, Any] | None,
    ) -> str:
        source_messages = self._messages_for_route(messages, route)
        instruction_head = "\n".join(
            [
                "Write content fields for one topic.",
                "Write only fields that should be created or changed. If no content field should change, write exactly: none",
                "Do not write JSON.",
                "Do not invent decisions or open questions.",
                "decision means the conversation settled or chose something.",
                "question means something is still unresolved.",
                "For an update, omit any field that should stay as-is.",
                "For a create, title is required.",
                "Use neutral titles for sensitive topics; privacy is handled by another card.",
                "Allowed field lines:",
                "title: short stable topic label",
                "summary: one concise sentence",
                "goal: current active goal, if any",
                "question: one unresolved question",
                "decision: one settled decision",
            ]
        )
        examples_block = "\n".join(
            [
                "Create for a fresh planning thread.",
                "title: Garden shed build",
                "summary: The user is planning to build a backyard shed.",
                "goal: Pick materials within budget.",
                "Create with one open issue and one settled choice.",
                "title: Trip booking",
                "question: Which dates work for the flights?",
                "decision: Book the seaside hotel.",
                "Update that changes only the goal.",
                "goal: Finish the draft before review.",
                "Nothing changed.",
                "none",
            ]
        )
        body = compose_card_prompt(
            instruction_head,
            examples_block,
            include_examples=self._include_examples,
        )
        lines = [
            body,
            f"conversation_id={conversation_id}",
            f"action={route.action.value}",
            f"target_id={route.target_id}",
        ]
        if existing_topic is not None:
            lines.extend(
                [
                    "<existing_target_topic>",
                    json_utils.dumps(existing_topic, indent=2, sort_keys=True),
                    "</existing_target_topic>",
                ]
            )
        lines.extend(
            [
                "<source_messages>",
                json_utils.dumps(
                    self._message_payload(source_messages),
                    indent=2,
                    sort_keys=True,
                ),
                "</source_messages>",
                "<full_existing_topic_snapshot>",
                json_utils.dumps(snapshot, indent=2, sort_keys=True),
                "</full_existing_topic_snapshot>",
            ]
        )
        return "\n".join(lines)

    def _build_target_boundary_prompt(
        self,
        *,
        conversation_id: str,
        messages: list[dict[str, Any]],
        route: _TopicRoute,
        content: _TopicContent,
    ) -> str:
        content_payload = {
            "title": content.title,
            "summary": content.summary,
            "active_goal": content.active_goal,
            "open_questions": list(content.open_questions),
            "decisions": list(content.decisions),
        }
        route_payload = {
            "action": route.action.value,
            "target_id": route.target_id,
            "source_message_ids": list(route.source_message_ids),
        }
        instruction_head = "\n".join(
            [
                "Decide how private one topic is. Pick the closest privacy label for it.",
                "Always write one line. Never write none.",
                "Do not write JSON.",
                "Use ordinary unless the topic itself is private romantic/intimate context, a stated relationship boundary, or ambiguous intimate context.",
                "If intimate context is present but the exact label is unclear, use ambiguous_intimate.",
                "Privacy labels:",
                "ordinary = an everyday topic with nothing private or intimate.",
                "romantic_private = a private romantic relationship matter.",
                "intimacy_private = a private intimate or sexual matter.",
                "intimacy_preference_private = a private personal preference about intimacy.",
                "intimacy_boundary = a personal limit the user wants respected.",
                "ambiguous_intimate = clearly intimate but you cannot tell which label fits, use when unsure.",
                "safety_blocked = content must be blocked for safety.",
                "privacy_level says how sensitive the topic is:",
                "0 = public, nothing private.",
                "1 = mildly private.",
                "2 = clearly private.",
                "3 = highly sensitive.",
                "For any non-ordinary privacy label, use at least 2 for how sensitive it is.",
                "Format: target_id privacy_label privacy_level confidence",
            ]
        )
        examples_block = "\n".join(
            [
                "An everyday topic about grocery shopping.",
                "tmp1 ordinary 0 0.7",
                "A private romantic relationship matter.",
                "tpc_private romantic_private 2 0.8",
                "A topic where the user states a personal limit they want respected.",
                "tmp2 intimacy_boundary 2 0.8",
                "Clearly intimate but the exact label is unclear.",
                "tpc_unclear ambiguous_intimate 2 0.7",
            ]
        )
        body = compose_card_prompt(
            instruction_head,
            examples_block,
            include_examples=self._include_examples,
        )
        return "\n".join(
            [
                body,
                f"conversation_id={conversation_id}",
                "<target_route>",
                json_utils.dumps(route_payload, indent=2, sort_keys=True),
                "</target_route>",
                "<target_content_draft>",
                json_utils.dumps(content_payload, indent=2, sort_keys=True),
                "</target_content_draft>",
                "<source_messages>",
                json_utils.dumps(
                    self._message_payload(self._messages_for_route(messages, route)),
                    indent=2,
                    sort_keys=True,
                ),
                "</source_messages>",
            ]
        )

    def _message_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
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

    def _messages_for_route(
        self,
        messages: list[dict[str, Any]],
        route: _TopicRoute,
    ) -> list[dict[str, Any]]:
        selected_ids = set(route.source_message_ids)
        return [
            message
            for message in messages
            if str(message.get("id") or "") in selected_ids
        ]

    @staticmethod
    def _messages_not_covered_by_routes(
        messages: list[dict[str, Any]],
        routes: tuple[_TopicRoute, ...],
    ) -> list[dict[str, Any]]:
        covered_message_ids = {
            message_id
            for route in routes
            for message_id in route.source_message_ids
        }
        return [
            message
            for message in messages
            if str(message.get("id") or "") not in covered_message_ids
        ]

    @staticmethod
    def _artifact_ids_from_route_messages(
        messages: list[dict[str, Any]],
        routes: tuple[_TopicRoute, ...],
    ) -> dict[str, tuple[str, ...]]:
        messages_by_id = {
            str(message["id"]): message
            for message in messages
            if message.get("id")
        }
        artifacts_by_target: dict[str, tuple[str, ...]] = {}
        for route in routes:
            artifact_ids: list[str] = []
            seen: set[str] = set()
            for message_id in route.source_message_ids:
                message = messages_by_id.get(message_id)
                if message is None:
                    continue
                for artifact_ref in TopicWorkingSetUpdater._message_artifact_refs(message):
                    artifact_id = str(artifact_ref.get("artifact_id") or "").strip()
                    if not artifact_id or artifact_id in seen:
                        continue
                    seen.add(artifact_id)
                    artifact_ids.append(artifact_id)
            if artifact_ids:
                artifacts_by_target[route.target_id] = tuple(artifact_ids)
        return artifacts_by_target

    @staticmethod
    def _message_text_for_topic_prompt(message: dict[str, Any]) -> str:
        if TopicWorkingSetUpdater._message_raw_text_allowed(message):
            return str(message["text"])
        placeholder = str(message.get("context_placeholder") or "").strip()
        if placeholder:
            return placeholder
        return (
            "[Message omitted from topic tracking | "
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


def _parse_route_card_output(
    text: str,
    *,
    valid_topic_ids: set[str],
    valid_message_ids: tuple[str, ...],
    conversation_id: str | None = None,
) -> tuple[_TopicRoute, ...]:
    valid_message_id_set = set(valid_message_ids)
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return ()
    routes: list[_TopicRoute] = []
    seen: set[tuple[str, str]] = set()
    create_aliases: set[str] = set()
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 2:
            continue
        action = _route_action_or_none(tokens[0])
        if action is None:
            continue
        target_id = tokens[1]
        if action is TopicUpdateActionType.CREATE:
            if not _is_temp_topic_target(target_id) or target_id in create_aliases:
                continue
            create_aliases.add(target_id)
        elif target_id not in valid_topic_ids:
            continue
        source_message_ids = _valid_message_ids_from_tokens(
            tokens[2:],
            valid_message_id_set=valid_message_id_set,
        )
        if not source_message_ids:
            logger.warning(
                "Dropping topic route line with no valid message ids "
                "(conversation_id=%s): %r",
                conversation_id,
                line,
            )
            continue
        key = (action.value, target_id)
        if key in seen:
            continue
        seen.add(key)
        routes.append(
            _TopicRoute(
                action=action,
                target_id=target_id,
                source_message_ids=tuple(source_message_ids),
            )
        )
        if len(routes) >= _MAX_TOPIC_CARD_ACTIONS:
            break
    return tuple(routes)


def _parse_new_topic_track_output(
    text: str,
    *,
    valid_message_ids: tuple[str, ...],
    conversation_id: str | None = None,
) -> tuple[_TopicRoute, ...]:
    lines = _card_lines(text)
    if not lines:
        return ()
    if all(
        line.strip("`*_.,;[](){}\"'").casefold() in {"ignore", "none"}
        for line in lines
    ):
        return ()
    valid_message_id_set = set(valid_message_ids)
    routes: list[_TopicRoute] = []
    for line in lines:
        tokens = _line_tokens(line)
        if not tokens or _clean_atom(tokens[0]) != "track":
            continue
        source_message_ids = _valid_message_ids_from_tokens(
            tokens[1:],
            valid_message_id_set=valid_message_id_set,
        )
        if not source_message_ids:
            logger.warning(
                "Dropping new-topic track line with no valid message ids "
                "(conversation_id=%s): %r",
                conversation_id,
                line,
            )
            continue
        routes.append(
            _TopicRoute(
                action=TopicUpdateActionType.CREATE,
                target_id=f"tmp{len(routes) + 1}",
                source_message_ids=tuple(source_message_ids),
            )
        )
        break
    return tuple(routes)


def _dedupe_routes(routes: list[_TopicRoute]) -> tuple[_TopicRoute, ...]:
    deduped: list[_TopicRoute] = []
    seen: set[tuple[str, str]] = set()
    for route in routes:
        key = (route.action.value, route.target_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return tuple(deduped)


def _parse_content_card_output(text: str) -> _TopicContent:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return _TopicContent()
    title: str | None = None
    summary: str | None = None
    active_goal: str | None = None
    open_questions: list[str] = []
    decisions: list[str] = []
    for line in lines:
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = _clean_atom(raw_key)
        value = _clean_text_value(raw_value)
        if not value:
            continue
        if key == "title":
            title = value
        elif key == "summary":
            summary = value
        elif key in {"goal", "active_goal"}:
            active_goal = value
        elif key in {"question", "open_question"}:
            open_questions.append(value)
        elif key == "decision":
            decisions.append(value)
    return _TopicContent(
        title=title,
        summary=summary,
        active_goal=active_goal,
        open_questions=tuple(_dedupe_texts(open_questions)),
        decisions=tuple(_dedupe_texts(decisions)),
    )


def _parse_boundary_card_output(
    text: str,
    *,
    valid_target_ids: set[str],
) -> dict[str, _TopicBoundary]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}
    boundaries: dict[str, _TopicBoundary] = {}
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 2:
            continue
        target_id = tokens[0]
        if target_id not in valid_target_ids or target_id in boundaries:
            continue
        boundary = normalize_intimacy_boundary(tokens[1])
        privacy_level = _int_or_none(tokens[2] if len(tokens) >= 3 else None)
        confidence = _float_or_none(tokens[3] if len(tokens) >= 4 else None)
        boundaries[target_id] = _TopicBoundary(
            boundary=boundary,
            privacy_level=_clamp_privacy_level(privacy_level),
            confidence=_clamp_confidence(confidence, default=0.7),
        )
    return boundaries


def _topic_card_plan_to_structured_plan(card_plan: _TopicCardPlan) -> TopicWorkingSetPlan:
    actions: list[TopicUpdateAction] = []
    for route in card_plan.routes:
        content = card_plan.contents.get(route.target_id, _TopicContent())
        boundary = card_plan.boundaries.get(route.target_id)
        artifact_ids = list(card_plan.artifacts.get(route.target_id, ()))
        source_message_ids = list(route.source_message_ids)
        if route.action is TopicUpdateActionType.CREATE:
            if content.title is None:
                continue
            boundary_value, privacy_level, boundary_confidence = _boundary_fields_for_action(
                route,
                boundary,
            )
            actions.append(
                TopicUpdateAction(
                    action=TopicUpdateActionType.CREATE,
                    title=content.title,
                    summary=content.summary or _TOPIC_STRING_SENTINEL,
                    active_goal=content.active_goal or _TOPIC_STRING_SENTINEL,
                    open_questions=list(content.open_questions),
                    decisions=list(content.decisions),
                    artifact_ids=artifact_ids,
                    source_message_ids=source_message_ids,
                    privacy_level=privacy_level,
                    intimacy_boundary=boundary_value,
                    intimacy_boundary_confidence=boundary_confidence,
                )
            )
            continue

        if route.action is TopicUpdateActionType.UPDATE:
            boundary_value, privacy_level, boundary_confidence = _boundary_fields_for_action(
                route,
                boundary,
            )
            actions.append(
                TopicUpdateAction(
                    action=TopicUpdateActionType.UPDATE,
                    topic_id=route.target_id,
                    title=content.title or _TOPIC_STRING_SENTINEL,
                    summary=content.summary or _TOPIC_STRING_SENTINEL,
                    active_goal=content.active_goal or _TOPIC_STRING_SENTINEL,
                    open_questions=list(content.open_questions),
                    decisions=list(content.decisions),
                    artifact_ids=artifact_ids,
                    source_message_ids=source_message_ids,
                    privacy_level=privacy_level,
                    intimacy_boundary=boundary_value,
                    intimacy_boundary_confidence=boundary_confidence,
                )
            )
            continue

        actions.append(
            TopicUpdateAction(
                action=route.action,
                topic_id=route.target_id,
                artifact_ids=artifact_ids,
                source_message_ids=source_message_ids,
            )
        )
    return TopicWorkingSetPlan(actions=actions, nothing_to_update=not actions)


def _boundary_fields_for_action(
    route: _TopicRoute,
    boundary: _TopicBoundary | None,
) -> tuple[str, int, float]:
    if boundary is None:
        if route.action is TopicUpdateActionType.CREATE:
            return IntimacyBoundary.ORDINARY.value, 0, 0.0
        return _TOPIC_BOUNDARY_SENTINEL, _TOPIC_INT_SENTINEL, _TOPIC_FLOAT_SENTINEL

    privacy_level = boundary.privacy_level
    if boundary.boundary is not IntimacyBoundary.ORDINARY:
        privacy_level = max(int(privacy_level or 0), 2)
    elif route.action is TopicUpdateActionType.UPDATE:
        privacy_level = _TOPIC_INT_SENTINEL
    elif privacy_level is None:
        privacy_level = 0
    return boundary.boundary.value, int(privacy_level), boundary.confidence


def _route_action_or_none(value: str) -> TopicUpdateActionType | None:
    try:
        action = TopicUpdateActionType(_clean_atom(value))
    except ValueError:
        return None
    if action is TopicUpdateActionType.NOOP:
        return None
    return action


def _topic_ids_from_snapshot(snapshot: dict[str, Any]) -> set[str]:
    return {
        str(topic.get("id"))
        for topic in [
            *(snapshot.get("active_topics") or []),
            *(snapshot.get("parked_topics") or []),
        ]
        if isinstance(topic, dict) and topic.get("id")
    }


def _topics_by_id_from_snapshot(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    topics: dict[str, dict[str, Any]] = {}
    for topic in [
        *(snapshot.get("active_topics") or []),
        *(snapshot.get("parked_topics") or []),
    ]:
        if isinstance(topic, dict) and topic.get("id"):
            topics[str(topic["id"])] = topic
    return topics


def _message_ids_from_messages(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(message["id"]) for message in messages if message.get("id"))


def _valid_message_ids_from_tokens(
    tokens: list[str],
    *,
    valid_message_id_set: set[str],
) -> tuple[str, ...]:
    ids: list[str] = []
    for token in tokens:
        if token not in valid_message_id_set or token in ids:
            continue
        ids.append(token)
    return tuple(ids)


def _is_temp_topic_target(value: str) -> bool:
    cleaned = value.strip()
    return cleaned.startswith("tmp") and all(
        character.isalnum() or character == "_" for character in cleaned
    )


def _card_lines(text: str) -> list[str]:
    normalized = (
        text.strip()
        .replace("<TAB>", " ")
        .replace("<tab>", " ")
        .replace("\\t", " ")
        .replace("\t", " ")
    )
    return [line.strip().strip("-* ").strip() for line in normalized.splitlines() if line.strip()]


def _lines_are_none(lines: list[str]) -> bool:
    return not lines or any(_clean_atom(line) == "none" for line in lines)


def _line_tokens(line: str) -> list[str]:
    normalized = line
    for separator in ("<TAB>", "<tab>", "\\t", "\t", "|", ",", ";", ":", "->"):
        normalized = normalized.replace(separator, " ")
    return [_clean_identifier(piece) for piece in normalized.split() if _clean_identifier(piece)]


def _clean_identifier(value: str) -> str:
    return value.strip().strip("`*_.,;[](){}\"'")


def _clean_atom(value: str) -> str:
    return _clean_identifier(value).casefold()


def _clean_text_value(value: str) -> str:
    text = value.strip().strip("` ").strip()
    if _clean_atom(text) in {"none", "null", "n/a", "na"}:
        return ""
    return text


def _dedupe_texts(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clamp_privacy_level(value: int | None) -> int | None:
    if value is None:
        return None
    return min(max(int(value), 0), 3)


def _clamp_confidence(value: float | None, *, default: float) -> float:
    if value is None:
        return default
    return min(max(float(value), 0.0), 1.0)


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
