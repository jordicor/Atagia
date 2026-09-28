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
from atagia.core.text_utils import strip_card_output_wrappers
from atagia.core.topic_repository import TopicRepository
from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.intimacy_boundary_policy import (
    normalize_intimacy_boundary,
    strongest_intimacy_boundary,
)
from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    known_intimacy_context_metadata,
)
from atagia.services.model_resolution import (
    component_id_for_llm_purpose,
    examples_enabled_for_component,
    parse_model_spec,
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

TopicContentField = Literal["title", "summary", "active_goal", "open_questions", "decisions"]
TopicWorkingSetCardName = Literal[
    "route",
    "content_title",
    "content_summary",
    "content_goal",
    "content_questions",
    "content_decisions",
    "boundary",
]

_CONTENT_FIELDS: tuple[TopicContentField, ...] = (
    "title",
    "summary",
    "active_goal",
    "open_questions",
    "decisions",
)
_CONTENT_CARD_NAMES: dict[TopicContentField, TopicWorkingSetCardName] = {
    "title": "content_title",
    "summary": "content_summary",
    "active_goal": "content_goal",
    "open_questions": "content_questions",
    "decisions": "content_decisions",
}
_CONTENT_DECISION_PURPOSES: dict[TopicContentField, str] = {
    "title": "topic_working_set_title_decision_card",
    "summary": "topic_working_set_summary_decision_card",
    "active_goal": "topic_working_set_goal_decision_card",
    "open_questions": "topic_working_set_questions_decision_card",
    "decisions": "topic_working_set_decisions_decision_card",
}
_CONTENT_FIELD_LABELS: dict[TopicContentField, str] = {
    "title": "title",
    "summary": "summary",
    "active_goal": "active goal",
    "open_questions": "open questions",
    "decisions": "settled decisions",
}

TOPIC_WORKING_SET_CARD_CONCURRENCY = 2

_CARD_PURPOSES: dict[TopicWorkingSetCardName, str] = {
    "route": "topic_working_set_route_card",
    "content_title": "topic_working_set_title_card",
    "content_summary": "topic_working_set_summary_card",
    "content_goal": "topic_working_set_goal_card",
    "content_questions": "topic_working_set_questions_card",
    "content_decisions": "topic_working_set_decisions_card",
    "boundary": "topic_working_set_boundary_card",
}
_CARD_MAX_OUTPUT_TOKENS: dict[TopicWorkingSetCardName, int] = {
    "route": 192,
    "content_title": 96,
    "content_summary": 160,
    "content_goal": 128,
    "content_questions": 256,
    "content_decisions": 256,
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
    open_questions: tuple[str, ...] | None = None
    decisions: tuple[str, ...] | None = None


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
    clear_summary: bool = False
    clear_active_goal: bool = False
    clear_open_questions: bool = False
    clear_decisions: bool = False
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
        if any(
            (
                self.clear_summary,
                self.clear_active_goal,
                self.clear_open_questions,
                self.clear_decisions,
            )
        ):
            if self.action is not TopicUpdateActionType.UPDATE:
                raise ValueError("content can only be cleared by an update action")
            if (
                (self.clear_summary and self.summary)
                or (self.clear_active_goal and self.active_goal)
                or (self.clear_open_questions and self.open_questions)
                or (self.clear_decisions and self.decisions)
            ):
                raise ValueError("a cleared content field cannot also have a replacement")
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
        self._update_mode = resolved_settings.topic_working_set_update_mode
        self._decision_models = (
            {
                field_name: resolve_component_model(
                    resolved_settings,
                    component_id_for_llm_purpose(purpose),
                )
                for field_name, purpose in _CONTENT_DECISION_PURPOSES.items()
            }
            if self._update_mode == "selective"
            else {}
        )

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
        update_routes = tuple(
            route for route in routes if route.action is TopicUpdateActionType.UPDATE
        )
        state: list[LLMMessage] = []
        if self._update_mode == "selective" and update_routes:
            context = []
            for route in update_routes:
                existing_topic = topics_by_id.get(route.target_id)
                if existing_topic is None:
                    raise ValueError(
                        f"Existing topic {route.target_id} is missing from the snapshot"
                    )
                context.append(
                    {
                        "topic_id": route.target_id,
                        "existing_topic": {
                            field_name: existing_topic.get(field_name)
                            for field_name in _CONTENT_FIELDS
                        },
                        "source_messages": self._message_payload(
                            self._messages_for_route(messages, route)
                        ),
                    }
                )
            state = [
                LLMMessage(
                    role="user",
                    content=json_utils.dumps(
                        {"conversation_id": conversation_id, "topics": context},
                        indent=2,
                        sort_keys=True,
                    ),
                )
            ]
        semaphore = asyncio.Semaphore(TOPIC_WORKING_SET_CARD_CONCURRENCY)
        failed = asyncio.Event()

        async def generate(
            route: _TopicRoute,
            field_name: TopicContentField,
        ) -> str | tuple[str, ...] | None:
            request = self._card_request(
                card_name=_CONTENT_CARD_NAMES[field_name],
                user_id=user_id,
                conversation_id=conversation_id,
                prompt=self._build_content_prompt(
                    field_name=field_name,
                    messages=messages,
                    route=route,
                    existing_topic=topics_by_id.get(route.target_id),
                ),
                snapshot=snapshot,
                target_id=route.target_id,
            )
            response = await self._llm_client.complete(request)
            return _parse_content_field_output(
                response.output_text,
                field_name=field_name,
                action=route.action,
            )

        async def choose_field(
            field_name: TopicContentField,
            selected_routes: tuple[_TopicRoute, ...],
        ) -> dict[str, str]:
            purpose = _CONTENT_DECISION_PURPOSES[field_name]
            questions = {
                route.target_id: self._build_content_decision_question(
                    field_name=field_name,
                    topic_id=route.target_id,
                )
                for route in selected_routes
            }
            answers = await self._llm_client.complete_choice_questions(
                model=self._decision_models[field_name],
                messages=state,
                questions=questions,
                metadata={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "purpose": purpose,
                    "topic_working_set_card": f"decision_{field_name}",
                    **self._intimacy_metadata_from_snapshot(snapshot),
                },
                concurrency=TOPIC_WORKING_SET_CARD_CONCURRENCY,
            )
            if answers.keys() != questions.keys():
                raise ValueError(f"{field_name} decisions did not cover every topic")
            return answers

        async def apply_choice(
            route: _TopicRoute,
            field_name: TopicContentField,
            decision: str,
        ) -> str | tuple[str, ...] | None:
            if decision == "keep":
                return None
            if decision == "clear" and field_name != "title":
                return () if field_name in {"open_questions", "decisions"} else ""
            if decision == "regenerate":
                return await generate(route, field_name)
            raise ValueError(f"Invalid {field_name} content decision: {decision}")

        async def run_field(
            route: _TopicRoute,
            field_name: TopicContentField,
            *,
            selective: bool,
        ) -> list[tuple[str, TopicContentField, str | tuple[str, ...] | None]]:
            async with semaphore:
                if failed.is_set():
                    raise asyncio.CancelledError()
                try:
                    if selective:
                        answers = await choose_field(field_name, (route,))
                        value = await apply_choice(
                            route, field_name, answers[route.target_id]
                        )
                    else:
                        value = await generate(route, field_name)
                    return [(route.target_id, field_name, value)]
                except BaseException:
                    failed.set()
                    raise

        async def run_typed_field(
            field_name: TopicContentField,
        ) -> list[tuple[str, TopicContentField, str | tuple[str, ...] | None]]:
            try:
                async with semaphore:
                    if failed.is_set():
                        raise asyncio.CancelledError()
                    answers = await choose_field(field_name, update_routes)

                async def resolve(
                    route: _TopicRoute,
                ) -> tuple[str, TopicContentField, str | tuple[str, ...] | None]:
                    decision = answers[route.target_id]
                    if decision == "regenerate":
                        async with semaphore:
                            if failed.is_set():
                                raise asyncio.CancelledError()
                            value = await apply_choice(route, field_name, decision)
                    else:
                        value = await apply_choice(route, field_name, decision)
                    return route.target_id, field_name, value

                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(resolve(route)) for route in update_routes]
                return [task.result() for task in tasks]
            except BaseException:
                failed.set()
                raise

        try:
            async with asyncio.TaskGroup() as group:
                tasks = []
                if self._update_mode == "direct":
                    tasks.extend(
                        group.create_task(run_field(route, field_name, selective=False))
                        for route in routes
                        for field_name in _CONTENT_FIELDS
                    )
                else:
                    for field_name in _CONTENT_FIELDS:
                        tasks.extend(
                            group.create_task(run_field(route, field_name, selective=False))
                            for route in routes
                            if route.action is TopicUpdateActionType.CREATE
                        )
                        if not update_routes:
                            continue
                        if parse_model_spec(
                            self._decision_models[field_name]
                        ).provider_slug == "typesafe":
                            tasks.append(group.create_task(run_typed_field(field_name)))
                        else:
                            tasks.extend(
                                group.create_task(
                                    run_field(route, field_name, selective=True)
                                )
                                for route in update_routes
                            )
        except BaseExceptionGroup as errors:
            raise errors.exceptions[0] from None
        content_fields: dict[str, dict[str, str | tuple[str, ...] | None]] = {
            route.target_id: {} for route in routes
        }
        for task in tasks:
            for target_id, field_name, value in task.result():
                content_fields[target_id][field_name] = value
        return {
            target_id: _TopicContent(**fields)
            for target_id, fields in content_fields.items()
        }

    @staticmethod
    def _build_content_decision_question(
        *,
        field_name: TopicContentField,
        topic_id: str,
    ) -> ChoiceQuestion:
        label = _CONTENT_FIELD_LABELS[field_name]
        criteria = {
            "keep": "The stored field remains accurate and complete; preserve it exactly.",
            "regenerate": (
                "New evidence or a correction requires a complete revised field. "
                "A separate content generator will write it."
            ),
        }
        if field_name != "title":
            criteria["clear"] = (
                "The stored field should be explicitly removed because no current "
                "supported value remains."
            )
        return ChoiceQuestion(
            instructions=(
                f"For existing topic {topic_id}, decide only what to do with its {label}. "
                "Use this topic's existing fields and its assigned source messages in the state. "
                "Treat corrections and resolved questions or decisions as changes when they "
                "make the stored field outdated. For a list, evaluate the entire current list. "
                "Do not write replacement content."
            ),
            criteria=criteria,
        )

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
        prompt: str | tuple[str, str],
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
        system_content = (
            "Keep track of the topics in this conversation, using the data below. "
            "Write only the requested plain-text lines. No JSON. No explanation."
        )
        if card_name in _CONTENT_CARD_NAMES.values():
            if not isinstance(prompt, tuple):
                raise ValueError("Content card prompt requires instructions and source context")
            instructions, source = prompt
            if not instructions.strip() or not source.strip():
                raise ValueError("Content card prompt requires instructions and source context")
            system_content = f"{system_content}\n{instructions}"
            user_content = source
        else:
            if not isinstance(prompt, str):
                raise ValueError("Non-content card prompt must be text")
            user_content = prompt
        return LLMCompletionRequest(
            model=self._card_models[card_name],
            messages=[
                LLMMessage(role="system", content=system_content),
                LLMMessage(role="user", content=user_content),
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
        field_name: TopicContentField,
        messages: list[dict[str, Any]],
        route: _TopicRoute,
        existing_topic: dict[str, Any] | None,
    ) -> tuple[str, str]:
        instructions = {
            "title": "Choose a short, stable title for this topic. Use a neutral title for sensitive topics.",
            "summary": "Write one concise sentence summarizing this topic, only when the messages support it.",
            "active_goal": "State the current active goal for this topic, only when the messages support one.",
            "open_questions": "List the currently unresolved questions for this topic. Include the complete updated list, one question per line. Do not invent questions.",
            "decisions": "List the decisions already settled for this topic. Include the complete updated list, one decision per line. Do not treat a mere proposal as settled.",
        }
        instruction_lines = [
            instructions[field_name],
            "Use only the source messages and existing topic values supplied in the user message.",
            "Answer only this requested content. Do not include a field label, topic ID, JSON, or explanation.",
        ]
        if route.action is TopicUpdateActionType.CREATE and field_name == "title":
            instruction_lines.append("A new topic requires a title. Write exactly one title line.")
        else:
            instruction_lines.append("Write exactly none when this value should not be created or changed.")
        if route.action is TopicUpdateActionType.UPDATE:
            if field_name != "title":
                instruction_lines.append(
                    "Write exactly clear only when the existing value should be removed."
                )
            instruction_lines.append("Otherwise, preserve the existing value by writing none.")
        instruction_lines.append(f"action={route.action.value}")
        instruction_head = "\n".join(instruction_lines)
        lines = []
        if existing_topic is not None:
            lines.extend(
                [
                    "<existing_target_topic>",
                    json_utils.dumps(
                        {
                            "title": existing_topic["title"],
                            "summary": existing_topic.get("summary"),
                            "active_goal": existing_topic.get("active_goal"),
                            "open_questions": existing_topic.get("open_questions"),
                            "decisions": existing_topic.get("decisions"),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    "</existing_target_topic>",
                ]
            )
        lines.extend(
            [
                "<source_messages>",
                json_utils.dumps(
                    self._message_payload(self._messages_for_route(messages, route)),
                    indent=2,
                    sort_keys=True,
                ),
                "</source_messages>",
            ]
        )
        return instruction_head, "\n".join(lines)

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
            "open_questions": list(content.open_questions or ()),
            "decisions": list(content.decisions or ()),
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
        summary_value = "" if action.clear_summary else _wire_string_to_none(action.summary)
        active_goal_value = "" if action.clear_active_goal else _wire_string_to_none(action.active_goal)
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
            open_questions=(
                []
                if action.clear_open_questions
                else action.open_questions if action.open_questions else None
            ),
            decisions=[] if action.clear_decisions else action.decisions if action.decisions else None,
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


def _parse_content_field_output(
    text: str,
    *,
    field_name: TopicContentField,
    action: TopicUpdateActionType,
) -> str | tuple[str, ...] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{field_name} card returned an empty answer")
    if len(lines) == 1 and lines[0].casefold() == "none":
        if field_name == "title" and action is TopicUpdateActionType.CREATE:
            raise ValueError("create actions require a title")
        return None
    if len(lines) == 1 and lines[0].casefold() == "clear":
        if action is not TopicUpdateActionType.UPDATE or field_name == "title":
            raise ValueError(f"{field_name} cannot be cleared for {action.value}")
        return () if field_name in {"open_questions", "decisions"} else ""
    if any(line.casefold() in {"none", "clear"} for line in lines):
        raise ValueError(f"{field_name} card mixed an instruction token with content")
    if field_name in {"open_questions", "decisions"}:
        return tuple(_dedupe_texts(lines))
    if len(lines) != 1:
        raise ValueError(f"{field_name} card returned multiple answers")
    return lines[0]


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
                raise ValueError("create actions require a title")
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
                    open_questions=list(content.open_questions or ()),
                    decisions=list(content.decisions or ()),
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
            content_updates: dict[str, Any] = {}
            for field_name in _CONTENT_FIELDS:
                value = getattr(content, field_name)
                if value is None:
                    continue
                if field_name != "title" and not value:
                    content_updates[f"clear_{field_name}"] = True
                    continue
                content_updates[field_name] = (
                    list(value) if field_name in {"open_questions", "decisions"} else value
                )
            actions.append(
                TopicUpdateAction(
                    action=TopicUpdateActionType.UPDATE,
                    topic_id=route.target_id,
                    **content_updates,
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
        strip_card_output_wrappers(text)
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
    return strip_card_output_wrappers(value).strip("`*_.,;[](){}\"'")


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
