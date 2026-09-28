"""Tests for offline Topic Working Set updates."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, cast

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.topic_working_set import (
    _CONTENT_CARD_NAMES,
    _CONTENT_FIELDS,
    _CONTENT_DECISION_PURPOSES,
    _TopicContent,
    _TopicRoute,
    _parse_content_field_output,
    _parse_route_card_output,
    TopicUpdateAction,
    TopicUpdateActionType,
    TopicContentField,
    TopicWorkingSetPlan,
    TopicWorkingSetUpdater,
)
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.services.llm_temperature import purpose_temperature
from atagia.services.model_resolution import component_id_for_llm_purpose
from benchmarks.topic_working_set_cards.compare import (
    BenchmarkCase,
    _build_artifact_prompt,
)
from tests.memory.card_leak_guard import assert_prompt_has_no_benchmark_leak

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"


def _prompt_only_updater(*, card_examples_enabled: bool = True) -> TopicWorkingSetUpdater:
    """Build an updater for pure prompt-construction assertions (no DB needed)."""
    provider = SequentialTopicProvider([])
    settings = replace(
        Settings.from_env(),
        card_examples_enabled=card_examples_enabled,
        llm_component_examples={},
    )
    return TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=settings,
    )


def _live_card_prompts(updater: TopicWorkingSetUpdater) -> dict[str, str]:
    """Render every live card prompt for a representative route/content/message set."""
    snapshot = {"active_topics": [], "parked_topics": []}
    messages = [{"id": "msg_1", "seq": 1, "role": "user", "text": "Plan a budget for the move."}]
    route = _TopicRoute(
        action=TopicUpdateActionType.CREATE,
        target_id="tmp1",
        source_message_ids=("msg_1",),
    )
    content = _TopicContent(title="Moving budget", summary="The user is planning a moving budget.")
    return {
        "existing_route": updater._build_existing_route_prompt(
            conversation_id="cnv_1", snapshot=snapshot, messages=messages
        ),
        "new_topic_track": updater._build_new_topic_track_prompt(
            conversation_id="cnv_1", snapshot=snapshot, messages=messages
        ),
        **{
            f"content_{field_name}": "\n".join(
                updater._build_content_prompt(
                    field_name=field_name,
                    messages=messages,
                    route=route,
                    existing_topic=None,
                )
            )
            for field_name in _CONTENT_FIELDS
        },
        "boundary": updater._build_target_boundary_prompt(
            conversation_id="cnv_1", messages=messages, route=route, content=content
        ),
    }


class SequentialTopicProvider(LLMProvider):
    name = "topic-working-set"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("No canned topic payload left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in topic working-set tests")


class ScriptedTopicProvider(SequentialTopicProvider):
    name = "openrouter"

    def __init__(
        self,
        *,
        routes: list[str],
        decisions: dict[tuple[str, str], str],
        content: dict[tuple[str, str], str] | None = None,
        boundaries: dict[str, str] | None = None,
    ) -> None:
        super().__init__([])
        self.routes = list(routes)
        self.decisions = decisions
        self.content = content or {}
        self.boundaries = boundaries or {}

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata["purpose"]
        target_id = request.metadata.get("stage") or request.metadata.get(
            "topic_working_set_target_id"
        )
        if purpose == "topic_working_set_route_card":
            answer = self.routes.pop(0)
        elif purpose.endswith("_decision_card"):
            answer = self.decisions[(purpose, target_id)]
        elif purpose == "topic_working_set_boundary_card":
            answer = self.boundaries.get(target_id, "none")
        else:
            answer = self.content[(purpose, target_id)]
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=answer,
        )


class ScriptedTypedTopicProvider(LLMProvider):
    name = "typesafe"
    supports_choices = True

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        assert request.choice_questions is not None
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            choice_answers={
                topic_id: ChoiceAnswer(
                    type="choice",
                    choice=self.answers[topic_id],
                    probabilities={self.answers[topic_id]: 1.0},
                    confidence=1.0,
                )
                for topic_id in request.choice_questions
            },
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in topic working-set tests")


async def _build_runtime(
    outputs: list[str],
    *,
    settings: Settings | None = None,
    provider: SequentialTopicProvider | None = None,
    extra_providers: list[LLMProvider] | None = None,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc))
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    topics = TopicRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-04-26T02:45:00+00:00', '2026-04-26T02:45:00+00:00')
        """
    )
    await connection.commit()
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other")
    provider = provider or SequentialTopicProvider(outputs)
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(
            provider_name=None if extra_providers else provider.name,
            providers=[provider, *(extra_providers or [])],
        ),
        clock=clock,
        topic_repository=topics,
        message_repository=messages,
        settings=settings,
    )
    return connection, clock, messages, topics, updater, provider


def _count_schema_key(value: object, key: str) -> int:
    if isinstance(value, dict):
        return int(key in value) + sum(_count_schema_key(child, key) for child in value.values())
    if isinstance(value, list):
        return sum(_count_schema_key(child, key) for child in value)
    return 0


def test_topic_working_set_plan_accepts_root_action_list_and_ignores_extra_fields() -> None:
    plan = TopicWorkingSetPlan.model_validate(
        [
            {
                "action": "create",
                "title": "Benchmark replay",
                "summary": "Compare evaluate-only runs from the same DB.",
                "source_message_ids": ["msg_1"],
                "rationale": "Provider-specific explanation field.",
            }
        ]
    )

    assert plan.nothing_to_update is False
    assert len(plan.actions) == 1
    assert plan.actions[0].title == "Benchmark replay"


def test_topic_working_set_schema_avoids_nullable_anyof_branches() -> None:
    schema = TopicWorkingSetPlan.model_json_schema()

    assert _count_schema_key(schema, "anyOf") == 0


def test_topic_update_action_normalizes_null_fields_to_wire_sentinels() -> None:
    action = TopicUpdateAction.model_validate(
        {
            "action": "update",
            "topic_id": "tpc_1",
            "title": None,
            "summary": None,
            "active_goal": None,
            "confidence": None,
            "privacy_level": None,
            "intimacy_boundary": None,
            "intimacy_boundary_confidence": None,
        }
    )

    assert action.title == ""
    assert action.summary == ""
    assert action.active_goal == ""
    assert action.confidence == -1.0
    assert action.privacy_level == -1
    assert action.intimacy_boundary == ""
    assert action.intimacy_boundary_confidence == -1.0
    assert action.clear_summary is False
    assert action.clear_active_goal is False


def test_topic_update_action_rejects_invalid_sentinel_replacements() -> None:
    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate({"action": "create", "title": "x", "confidence": -2.0})

    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate({"action": "create", "title": "x", "privacy_level": 7})

    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate(
            {"action": "create", "title": "x", "intimacy_boundary": "not_a_boundary"}
        )
    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate(
            {"action": "update", "topic_id": "tpc_1", "summary": "new", "clear_summary": True}
        )


def test_topic_content_cards_distinguish_none_clear_and_malformed_answers() -> None:
    assert _parse_content_field_output(
        "none", field_name="summary", action=TopicUpdateActionType.UPDATE
    ) is None
    assert _parse_content_field_output(
        "clear", field_name="active_goal", action=TopicUpdateActionType.UPDATE
    ) == ""
    assert _parse_content_field_output(
        "clear", field_name="open_questions", action=TopicUpdateActionType.UPDATE
    ) == ()
    assert _parse_content_field_output(
        "What remains?\nWho owns it?", field_name="open_questions", action=TopicUpdateActionType.UPDATE
    ) == ("What remains?", "Who owns it?")
    assert _parse_content_field_output(
        "Summary", field_name="title", action=TopicUpdateActionType.CREATE
    ) == "Summary"
    assert _parse_content_field_output(
        "[Atagia] API plan", field_name="title", action=TopicUpdateActionType.CREATE
    ) == "[Atagia] API plan"
    for field_name, action, answer in (
        ("title", TopicUpdateActionType.CREATE, "none"),
        ("title", TopicUpdateActionType.UPDATE, "clear"),
        ("summary", TopicUpdateActionType.UPDATE, ""),
        ("summary", TopicUpdateActionType.UPDATE, "First line\nSecond line"),
        ("decisions", TopicUpdateActionType.UPDATE, "none\nA settled choice"),
    ):
        with pytest.raises(ValueError):
            _parse_content_field_output(answer, field_name=field_name, action=action)


def test_topic_content_purposes_keep_model_and_temperature_routing() -> None:
    for purpose in (
        "topic_working_set_title_card",
        "topic_working_set_summary_card",
        "topic_working_set_goal_card",
        "topic_working_set_questions_card",
        "topic_working_set_decisions_card",
    ):
        assert component_id_for_llm_purpose(purpose) == "topic_working_set"
        assert purpose_temperature(purpose) == purpose_temperature("topic_working_set_update")


def test_topic_content_prompt_only_uses_routed_messages_and_target_snapshot() -> None:
    updater = _prompt_only_updater()
    route = _TopicRoute(
        action=TopicUpdateActionType.UPDATE,
        target_id="tpc_current",
        source_message_ids=("msg_current",),
    )
    instructions, source = updater._build_content_prompt(
        field_name="decisions",
        messages=[
            {"id": "msg_current", "role": "user", "text": "Yes, the second option."},
            {"id": "msg_other", "role": "user", "text": "Unrelated private discussion."},
        ],
        route=route,
        existing_topic={
            "title": "Garden planning",
            "summary": "The user is comparing two garden layouts.",
            "active_goal": "Choose the layout for the yard.",
            "open_questions": ["Which layout fits the yard?"],
            "decisions": ["Use a low-maintenance design."],
            "privacy_level": 2,
            "unrelated_detail": "Do not send this to the card.",
        },
    )

    assert "Yes, the second option." in source
    assert "Unrelated private discussion." not in source
    assert "The user is comparing two garden layouts." in source
    assert "Choose the layout for the yard." in source
    assert "Which layout fits the yard?" in source
    assert "Use a low-maintenance design." in source
    assert "Do not send this to the card." not in source
    assert "privacy_level" not in source
    assert "tpc_current" not in source
    assert "<full_existing_topic_snapshot>" not in source
    assert "Yes, the second option." not in instructions


@pytest.mark.parametrize("field_name", _CONTENT_FIELDS)
@pytest.mark.parametrize("action", (TopicUpdateActionType.CREATE, TopicUpdateActionType.UPDATE))
def test_topic_content_request_separates_instructions_from_source(
    field_name: TopicContentField,
    action: TopicUpdateActionType,
) -> None:
    updater = _prompt_only_updater(card_examples_enabled=True)
    route = _TopicRoute(
        action=action,
        target_id="tpc_current",
        source_message_ids=("msg_current",),
    )
    existing_topic = (
        {
            "title": "Willow onboarding",
            "summary": "The user is preparing a Willow onboarding guide.",
            "active_goal": "Draft the guide outline.",
            "open_questions": ["Who will review the draft?"],
            "decisions": ["Use the existing template."],
        }
        if action is TopicUpdateActionType.UPDATE
        else None
    )
    request = updater._card_request(
        card_name=_CONTENT_CARD_NAMES[field_name],
        user_id="usr_1",
        conversation_id="cnv_1",
        prompt=updater._build_content_prompt(
            field_name=field_name,
            messages=[
                {"id": "msg_current", "role": "user", "text": "Start a Willow onboarding guide."},
                {"id": "msg_other", "role": "user", "text": "Unrelated private discussion."},
            ],
            route=route,
            existing_topic=existing_topic,
        ),
        snapshot={"active_topics": [], "parked_topics": []},
        target_id=route.target_id,
    )

    system_message, user_message = request.messages
    assert system_message.role == "system"
    assert user_message.role == "user"
    assert "Use only the source messages" in system_message.content
    assert "Start a Willow onboarding guide." not in system_message.content
    assert "Willow onboarding" not in system_message.content
    assert f"action={action.value}" in system_message.content
    assert "Start a Willow onboarding guide." in user_message.content
    assert "Unrelated private discussion." not in user_message.content
    assert "action=" not in user_message.content
    assert "Use only the source messages" not in user_message.content
    assert "Examples:" not in system_message.content + user_message.content
    assert "Garden shed planning" not in system_message.content + user_message.content
    assert ("<existing_target_topic>" in user_message.content) is (
        action is TopicUpdateActionType.UPDATE
    )


@pytest.mark.asyncio
async def test_topic_content_failure_cancels_sibling_calls() -> None:
    class FailingProvider(LLMProvider):
        name = "topic-cancellation"

        def __init__(self) -> None:
            self.summary_started = asyncio.Event()
            self.summary_cancelled = asyncio.Event()
            self.requests: list[LLMCompletionRequest] = []

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            card = request.metadata["topic_working_set_card"]
            if card == "content_title":
                await self.summary_started.wait()
                raise ValueError("title card failed")
            if card == "content_summary":
                self.summary_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.summary_cancelled.set()
                    raise
            raise AssertionError(f"Unexpected content card started: {card}")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = FailingProvider()
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model="openai/test-model"),
    )
    route = _TopicRoute(
        action=TopicUpdateActionType.CREATE,
        target_id="tmp1",
        source_message_ids=("msg_1",),
    )
    with pytest.raises(ValueError, match="title card failed"):
        await asyncio.wait_for(
            updater._run_content_cards(
                user_id="usr_1",
                conversation_id="cnv_1",
                snapshot={"active_topics": [], "parked_topics": []},
                messages=[{"id": "msg_1", "role": "user", "text": "Plan a shed."}],
                routes=(route,),
            ),
            timeout=2,
        )
    assert provider.summary_cancelled.is_set()
    assert [request.metadata["topic_working_set_card"] for request in provider.requests] == [
        "content_title",
        "content_summary",
    ]


@pytest.mark.asyncio
async def test_topic_updater_creates_topic_and_links_only_valid_source_messages() -> None:
    outputs = [
        "none",
        "track msg_1 msg_other_user missing",
        "Benchmark observability",
        "Track retained DBs, manifests, and failed-question custody.",
        "Make benchmark comparisons reproducible.",
        "Which failures lacked sufficient evidence?",
        "Keep retrieval changes in shadow mode.",
        "tmp1 ordinary 0 0.84",
    ]
    connection, _clock, messages, topics, updater, provider = await _build_runtime(outputs)
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Let's add benchmark manifests and custody reports.",
            9,
            {
                "artifact_backed": True,
                "attachment_artifact_ids": ["art_manifest"],
                "attachments": [
                    {
                        "artifact_id": "art_manifest",
                        "artifact_type": "file",
                        "source_kind": "host_embedded",
                        "mime_type": "application/json",
                        "filename": "run-manifest.json",
                        "title": "Run manifest",
                        "privacy_level": 0,
                        "preserve_verbatim": False,
                        "requires_explicit_request": True,
                        "relevance_state": "active_work_material",
                        "summary_text": "Do not leak this summary text into the topic prompt.",
                    }
                ],
            },
        )
        await messages.create_message(
            "msg_other_user",
            "cnv_2",
            "user",
            1,
            "This should not be linked.",
            6,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        assert len(changed) == 1
        assert changed[0]["title"] == "Benchmark observability"
        assert changed[0]["open_questions_json"] == ["Which failures lacked sufficient evidence?"]
        assert changed[0]["artifact_ids_json"] == ["art_manifest"]
        sources = await topics.list_topic_sources(user_id="usr_1", topic_id=str(changed[0]["id"]))
        assert [
            (source["source_kind"], source["source_id"])
            for source in sources
        ] == [
            ("message", "msg_1"),
            ("artifact", "art_manifest"),
        ]
        route_prompt = provider.requests[0].messages[1].content
        new_topic_prompt = provider.requests[1].messages[1].content
        boundary_prompt = provider.requests[7].messages[1].content
        assert [
            request.metadata["purpose"]
            for request in provider.requests
        ] == [
            "topic_working_set_route_card",
            "topic_working_set_route_card",
            "topic_working_set_title_card",
            "topic_working_set_summary_card",
            "topic_working_set_goal_card",
            "topic_working_set_questions_card",
            "topic_working_set_decisions_card",
            "topic_working_set_boundary_card",
        ]
        assert "Never create a new topic in this card." in route_prompt
        assert "Default answer: track." in new_topic_prompt
        assert '"artifact_id": "art_manifest"' in route_prompt
        assert "run-manifest.json" in route_prompt
        assert '"relevance_state": "active_work_material"' in route_prompt
        assert "Do not leak this summary text" not in route_prompt
        assert "Do not write titles, summaries, goals" in route_prompt
        for boundary in IntimacyBoundary:
            assert boundary.value in boundary_prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_parks_existing_topic_from_model_plan() -> None:
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        ["park tpc_existing msg_2"]
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Qwen comparison",
            summary="Running model comparison.",
        )
        message = await messages.create_message(
            "msg_2",
            "cnv_1",
            "assistant",
            1,
            "The run should continue in the background.",
            8,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        assert len(changed) == 1
        assert changed[0]["id"] == "tpc_existing"
        assert changed[0]["status"] == "parked"
        assert changed[0]["summary"] == "Running model comparison."
        events = await topics.list_events(user_id="usr_1", conversation_id="cnv_1", topic_id="tpc_existing")
        assert [event["event_type"] for event in events] == ["created", "parked", "source_linked"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_uses_placeholders_for_raw_policy_restricted_messages() -> None:
    connection, _clock, messages, _topics, updater, provider = await _build_runtime(
        ["none", "ignore"]
    )
    try:
        message = await messages.create_message(
            "msg_restricted",
            "cnv_1",
            "user",
            1,
            "SECRET RAW ATTACHMENT TEXT",
            500,
            {
                "context_placeholder": "[Skipped attachment]",
            },
        )
        message["include_raw"] = False
        message["skip_by_default"] = True
        message["context_placeholder"] = "[Skipped attachment]"
        message["content_kind"] = "attachment"
        message["policy_reason"] = "heavy_content"

        await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        prompt = provider.requests[0].messages[1].content
        assert "SECRET RAW ATTACHMENT TEXT" not in prompt
        assert "[Skipped attachment]" in prompt
        assert '"raw_text_included": false' in prompt
        assert '"policy_reason": "heavy_content"' in prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_ignores_cross_conversation_topic_actions() -> None:
    connection, clock, messages, topics, updater, _provider = await _build_runtime(
        ["update tpc_other_conv msg_3", "ignore"]
    )
    try:
        conversations = ConversationRepository(connection, clock)
        await conversations.create_conversation("cnv_other", "usr_1", None, "coding_debug", "Other")
        await topics.create_topic(
            topic_id="tpc_other_conv",
            user_id="usr_1",
            conversation_id="cnv_other",
            title="Other conversation topic",
            summary="Original summary.",
        )
        message = await messages.create_message(
            "msg_3",
            "cnv_1",
            "user",
            1,
            "Please update the current topic only.",
            8,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )
        other_topic = await topics.get_topic("tpc_other_conv", "usr_1")
        snapshot = await topics.get_topic_snapshot(user_id="usr_1", conversation_id="cnv_1")

        assert changed == []
        assert other_topic is not None
        assert other_topic["summary"] == "Original summary."
        assert snapshot["freshness"]["last_processed_seq"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_does_not_clear_existing_fields_for_wire_sentinels() -> None:
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        ["update tpc_existing msg_5", *(["none"] * 5), "none"],
        settings=Settings.from_env({"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "direct"}),
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Existing private topic",
            summary="Keep this summary.",
            active_goal="Keep this goal.",
            open_questions=["Keep this question?"],
            decisions=["Keep this decision."],
            confidence=0.8,
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
        )
        message = await messages.create_message(
            "msg_5",
            "cnv_1",
            "user",
            1,
            "Still discussing this topic.",
            5,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        assert len(changed) == 1
        refreshed = await topics.get_topic("tpc_existing", "usr_1")
        assert refreshed is not None
        assert refreshed["title"] == "Existing private topic"
        assert refreshed["summary"] == "Keep this summary."
        assert refreshed["active_goal"] == "Keep this goal."
        assert refreshed["open_questions_json"] == ["Keep this question?"]
        assert refreshed["decisions_json"] == ["Keep this decision."]
        assert refreshed["confidence"] == 0.8
        assert refreshed["privacy_level"] == 2
        assert refreshed["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert refreshed["intimacy_boundary_confidence"] == 0.9
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_clears_only_explicitly_cleared_content() -> None:
    connection, _clock, messages, topics, updater, provider = await _build_runtime(
        ["update tpc_existing msg_8", "none", "clear", "clear", "clear", "clear", "none"],
        settings=Settings.from_env({"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "direct"}),
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Existing work",
            summary="Old summary.",
            active_goal="Old goal.",
            open_questions=["Old question?"],
            decisions=["Old decision."],
        )
        message = await messages.create_message(
            "msg_8", "cnv_1", "user", 1, "This topic is finished.", 5, {}
        )

        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )

        assert len(changed) == 1
        assert changed[0]["title"] == "Existing work"
        assert changed[0]["summary"] == ""
        assert changed[0]["active_goal"] == ""
        assert changed[0]["open_questions_json"] == []
        assert changed[0]["decisions_json"] == []
        assert [request.metadata["topic_working_set_card"] for request in provider.requests[1:6]] == [
            "content_title",
            "content_summary",
            "content_goal",
            "content_questions",
            "content_decisions",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("finite_enabled", [False, True])
async def test_selective_topic_update_keeps_all_content_without_generation(finite_enabled: bool) -> None:
    provider = ScriptedTopicProvider(
        routes=["update tpc_existing msg_1"],
        decisions={
            (purpose, "tpc_existing"): "keep"
            for purpose in _CONTENT_DECISION_PURPOSES.values()
        },
    )
    typed = ScriptedTypedTopicProvider({"tpc_existing": "keep"})
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env({
            "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": str(finite_enabled).lower(),
        }),
        provider=provider,
        extra_providers=[typed],
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Trip planning",
            summary="The user is planning a trip.",
            active_goal="Choose travel dates.",
            open_questions=["When should the trip start?"],
            decisions=["Travel by train."],
        )
        message = await messages.create_message(
            "msg_1", "cnv_1", "user", 1, "The travel plan is still current.", 7, {}
        )

        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )

        assert len(changed) == 1
        assert changed[0]["summary"] == "The user is planning a trip."
        assert changed[0]["open_questions_json"] == ["When should the trip start?"]
        assert changed[0]["decisions_json"] == ["Travel by train."]
        requests = [*provider.requests, *typed.requests]
        purposes = [request.metadata["purpose"] for request in requests]
        assert len(purposes) == 7  # route, five field decisions, boundary
        assert set(_CONTENT_DECISION_PURPOSES.values()).issubset(purposes)
        assert not any(request.metadata["topic_working_set_card"].startswith("content_") for request in provider.requests)
        assert len(typed.requests) == (5 if finite_enabled else 0)
        events = await topics.list_events(
            user_id="usr_1", conversation_id="cnv_1", topic_id="tpc_existing"
        )
        assert [event["event_type"] for event in events] == [
            "created", "updated", "source_linked"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_selective_topic_update_regenerates_correction_and_clears_resolved_list() -> None:
    choices = {
        "title": "keep",
        "summary": "regenerate",
        "active_goal": "keep",
        "open_questions": "clear",
        "decisions": "regenerate",
    }
    provider = ScriptedTopicProvider(
        routes=["update tpc_existing msg_2"],
        decisions={
            (_CONTENT_DECISION_PURPOSES[field_name], "tpc_existing"): choice
            for field_name, choice in choices.items()
        },
        content={
            ("topic_working_set_summary_card", "tpc_existing"): "The booking is for Friday, not Thursday.",
            ("topic_working_set_decisions_card", "tpc_existing"): "Travel on Friday.\nUse the early train.",
        },
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=provider,
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Train booking",
            summary="The booking is for Thursday.",
            active_goal="Finish the booking.",
            open_questions=["Which day is the trip?"],
            decisions=["Use the early train."],
        )
        message = await messages.create_message(
            "msg_2", "cnv_1", "user", 1, "Correction: Friday. The day is settled.", 9, {}
        )

        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )

        assert changed[0]["title"] == "Train booking"
        assert changed[0]["summary"] == "The booking is for Friday, not Thursday."
        assert changed[0]["active_goal"] == "Finish the booking."
        assert changed[0]["open_questions_json"] == []
        assert changed[0]["decisions_json"] == ["Travel on Friday.", "Use the early train."]
        generators = [
            request for request in provider.requests
            if request.metadata["topic_working_set_card"].startswith("content_")
        ]
        assert {request.metadata["purpose"] for request in generators} == {
            "topic_working_set_summary_card",
            "topic_working_set_decisions_card",
        }
        assert all(request.model == "openrouter/openai/gpt-6-luna" for request in generators)
        assert all("The booking is for Thursday." in request.messages[1].content for request in generators)
        assert all("Correction: Friday." in request.messages[1].content for request in generators)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_selective_topic_update_clears_fields_without_generation() -> None:
    clear_choices = {
        "title": "keep",
        "summary": "clear",
        "active_goal": "clear",
        "open_questions": "clear",
        "decisions": "clear",
    }
    provider = ScriptedTopicProvider(
        routes=["update tpc_existing msg_3"],
        decisions={
            (_CONTENT_DECISION_PURPOSES[field_name], "tpc_existing"): choice
            for field_name, choice in clear_choices.items()
        },
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=provider,
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Open work",
            summary="An open task.",
            active_goal="Finish the task.",
            open_questions=["Who owns it?"],
            decisions=["Try option A."],
        )
        message = await messages.create_message(
            "msg_3", "cnv_1", "user", 1, "Remove the obsolete details.", 6, {}
        )
        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )
        assert changed[0]["title"] == "Open work"
        assert changed[0]["summary"] == ""
        assert changed[0]["active_goal"] == ""
        assert changed[0]["open_questions_json"] == []
        assert changed[0]["decisions_json"] == []
        assert not any(request.metadata["topic_working_set_card"].startswith("content_") for request in provider.requests)
    finally:
        await connection.close()


def test_topic_title_decision_has_no_clear_option() -> None:
    title = TopicWorkingSetUpdater._build_content_decision_question(
        field_name="title", topic_id="tpc_existing"
    )
    questions = TopicWorkingSetUpdater._build_content_decision_question(
        field_name="open_questions", topic_id="tpc_existing"
    )
    assert set(title.criteria) == {"keep", "regenerate"}
    assert set(questions.criteria) == {"keep", "clear", "regenerate"}
    assert "tpc_existing" in title.instructions
    assert "resolved" in questions.instructions


@pytest.mark.asyncio
async def test_selective_mode_creates_topic_without_decision_filters() -> None:
    provider = ScriptedTopicProvider(
        routes=["none", "track msg_4"],
        decisions={},
        content={
            ("topic_working_set_title_card", "tmp1"): "Garden plan",
            ("topic_working_set_summary_card", "tmp1"): "Plan a small garden.",
            ("topic_working_set_goal_card", "tmp1"): "Choose plants.",
            ("topic_working_set_questions_card", "tmp1"): "Which plants fit?",
            ("topic_working_set_decisions_card", "tmp1"): "Use the sunny plot.",
        },
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=provider,
    )
    try:
        message = await messages.create_message(
            "msg_4", "cnv_1", "user", 1, "Let's plan the garden.", 5, {}
        )
        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )
        assert len(changed) == 1
        assert changed[0]["title"] == "Garden plan"
        assert changed[0]["decisions_json"] == ["Use the sunny plot."]
        assert len(provider.requests) == 8  # two route cards, five generators, boundary
        assert not any(
            request.metadata["purpose"].endswith("_decision_card")
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_selective_all_changed_runs_five_decisions_and_five_generators() -> None:
    provider = ScriptedTopicProvider(
        routes=["update tpc_existing msg_5"],
        decisions={
            (purpose, "tpc_existing"): "regenerate"
            for purpose in _CONTENT_DECISION_PURPOSES.values()
        },
        content={
            ("topic_working_set_title_card", "tpc_existing"): "Revised plan",
            ("topic_working_set_summary_card", "tpc_existing"): "The revised plan has new details.",
            ("topic_working_set_goal_card", "tpc_existing"): "none",
            ("topic_working_set_questions_card", "tpc_existing"): "Which option now fits?",
            ("topic_working_set_decisions_card", "tpc_existing"): "Use option B.",
        },
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=provider,
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing", user_id="usr_1", conversation_id="cnv_1",
            title="Old plan", summary="The old plan.", active_goal="Keep the original goal.",
        )
        message = await messages.create_message(
            "msg_5", "cnv_1", "user", 1, "Replace every part of the plan.", 7, {}
        )
        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message]
        )
        assert changed[0]["title"] == "Revised plan"
        assert changed[0]["active_goal"] == "Keep the original goal."
        assert changed[0]["open_questions_json"] == ["Which option now fits?"]
        assert len(provider.requests) == 12  # route + five filters + five generators + boundary
        assert sum(
            request.metadata["topic_working_set_card"].startswith("content_")
            for request in provider.requests
        ) == 5
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_selective_mixed_models_batch_only_same_field_and_isolate_topics() -> None:
    provider = ScriptedTopicProvider(
        routes=[
            "update tpc_a msg_a\nupdate tpc_b msg_b\nupdate tpc_other_user msg_b"
        ],
        decisions={
            (purpose, topic_id): "keep"
            for field_name, purpose in _CONTENT_DECISION_PURPOSES.items()
            if field_name != "summary"
            for topic_id in ("tpc_a", "tpc_b")
        },
        content={
            ("topic_working_set_summary_card", "tpc_b"): "The delivery date changed.",
        },
    )
    typed = ScriptedTypedTopicProvider({"tpc_a": "keep", "tpc_b": "regenerate"})
    settings = Settings.from_env(
        {
            "ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective",
            "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
            "ATAGIA_LLM_FINITE_DECISION_MODEL": "openrouter/openai/gpt-5.6-luna",
            "ATAGIA_LLM_MODEL__TOPIC_SUMMARY_DECISION": "typesafe/jev-1.13.0",
        }
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [], settings=settings, provider=provider, extra_providers=[typed]
    )
    try:
        for topic_id, user_id, conversation_id, title in (
            ("tpc_a", "usr_1", "cnv_1", "Travel"),
            ("tpc_b", "usr_1", "cnv_1", "Delivery"),
            ("tpc_other_user", "usr_2", "cnv_2", "Private other topic"),
        ):
            await topics.create_topic(
                topic_id=topic_id,
                user_id=user_id,
                conversation_id=conversation_id,
                title=title,
                summary=f"Old {title.lower()} summary.",
            )
        message_a = await messages.create_message(
            "msg_a", "cnv_1", "user", 1, "Travel remains the same.", 5, {}
        )
        message_b = await messages.create_message(
            "msg_b", "cnv_1", "user", 2, "Delivery is now Friday.", 5, {}
        )

        changed = await updater.update_from_messages(
            user_id="usr_1", conversation_id="cnv_1", messages=[message_a, message_b]
        )

        assert {topic["id"] for topic in changed} == {"tpc_a", "tpc_b"}
        assert (await topics.get_topic("tpc_a", "usr_1"))["summary"] == "Old travel summary."
        assert (await topics.get_topic("tpc_b", "usr_1"))["summary"] == "The delivery date changed."
        assert (await topics.get_topic("tpc_other_user", "usr_2"))["summary"] == "Old private other topic summary."
        assert len(typed.requests) == 1
        request = typed.requests[0]
        assert request.model == "jev-1.13.0"
        assert updater._decision_models["summary"] == "typesafe/jev-1.13.0"
        assert request.metadata["purpose"] == "topic_working_set_summary_decision_card"
        assert set(request.choice_questions) == {"tpc_a", "tpc_b"}
        assert all("summary" in question.instructions for question in request.choice_questions.values())
        state = json.loads(request.messages[0].content)
        assert state["conversation_id"] == "cnv_1"
        assert {
            topic["topic_id"]: [message["id"] for message in topic["source_messages"]]
            for topic in state["topics"]
        } == {"tpc_a": ["msg_a"], "tpc_b": ["msg_b"]}
        assert "Private other topic" not in request.messages[0].content
        assert all(
            request.metadata["purpose"] != "topic_working_set_summary_decision_card"
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_invalid_selective_decision_fails_before_persisting() -> None:
    provider = ScriptedTopicProvider(
        routes=["update tpc_existing msg_6"],
        decisions={
            (purpose, "tpc_existing"): "clear" if field_name == "title" else "keep"
            for field_name, purpose in _CONTENT_DECISION_PURPOSES.items()
        },
    )
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=provider,
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing", user_id="usr_1", conversation_id="cnv_1",
            title="Existing topic", summary="Original summary.",
        )
        message = await messages.create_message(
            "msg_6", "cnv_1", "user", 1, "More context.", 3, {}
        )
        with pytest.raises(LLMError, match="unknown option"):
            await updater.update_from_messages(
                user_id="usr_1", conversation_id="cnv_1", messages=[message]
            )
        assert (await topics.get_topic("tpc_existing", "usr_1"))["summary"] == "Original summary."
        events = await topics.list_events(
            user_id="usr_1", conversation_id="cnv_1", topic_id="tpc_existing"
        )
        assert [event["event_type"] for event in events] == ["created"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_selective_decision_failure_cancels_sibling_filter() -> None:
    class FailingDecisionProvider(LLMProvider):
        name = "openrouter"

        def __init__(self) -> None:
            self.summary_started = asyncio.Event()
            self.summary_cancelled = asyncio.Event()

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            purpose = request.metadata["purpose"]
            if purpose == "topic_working_set_title_decision_card":
                await self.summary_started.wait()
                raise ValueError("title decision failed")
            if purpose == "topic_working_set_summary_decision_card":
                self.summary_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.summary_cancelled.set()
                    raise
            raise AssertionError(f"Unexpected card started: {purpose}")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = FailingDecisionProvider()
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
    )
    with pytest.raises(ValueError, match="title decision failed"):
        await asyncio.wait_for(
            updater._run_content_cards(
                user_id="usr_1",
                conversation_id="cnv_1",
                snapshot={
                    "active_topics": [{"id": "tpc_a", "title": "Existing topic"}],
                    "parked_topics": [],
                },
                messages=[{"id": "msg_1", "role": "user", "text": "New detail."}],
                routes=(
                    _TopicRoute(
                        action=TopicUpdateActionType.UPDATE,
                        target_id="tpc_a",
                        source_message_ids=("msg_1",),
                    ),
                ),
            ),
            timeout=2,
        )
    assert provider.summary_cancelled.is_set()


@pytest.mark.asyncio
async def test_selective_filter_error_cancels_create_and_persists_nothing() -> None:
    class FailingMixedProvider(LLMProvider):
        name = "openrouter"

        def __init__(self) -> None:
            self.routes = ["update tpc_existing msg_update", "track msg_new"]
            self.create_started = asyncio.Event()
            self.create_cancelled = asyncio.Event()

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            purpose = request.metadata["purpose"]
            if purpose == "topic_working_set_route_card":
                answer = self.routes.pop(0)
            elif purpose == "topic_working_set_title_card":
                self.create_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.create_cancelled.set()
                    raise
            elif purpose == "topic_working_set_title_decision_card":
                await self.create_started.wait()
                answer = "clear"
            elif purpose.endswith("_decision_card"):
                answer = "keep"
            else:
                answer = "none"
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=answer
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = FailingMixedProvider()
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        [],
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
        provider=cast(Any, provider),
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing", user_id="usr_1", conversation_id="cnv_1",
            title="Existing topic", summary="Original summary.",
        )
        message_update = await messages.create_message(
            "msg_update", "cnv_1", "user", 1, "Update this topic.", 4, {}
        )
        message_new = await messages.create_message(
            "msg_new", "cnv_1", "user", 2, "Start another topic.", 4, {}
        )
        with pytest.raises(LLMError, match="unknown option"):
            await asyncio.wait_for(
                updater.update_from_messages(
                    user_id="usr_1", conversation_id="cnv_1",
                    messages=[message_update, message_new],
                ),
                timeout=2,
            )
        assert provider.create_cancelled.is_set()
        assert (await topics.get_topic("tpc_existing", "usr_1"))["summary"] == "Original summary."
        snapshot = await topics.get_topic_snapshot(
            user_id="usr_1", conversation_id="cnv_1"
        )
        assert {topic["id"] for topic in snapshot["active_topics"]} == {"tpc_existing"}
        events = await topics.list_events(user_id="usr_1", conversation_id="cnv_1")
        assert [event["event_type"] for event in events] == ["created"]
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_create", [True, False])
async def test_selective_generation_progresses_while_another_filter_waits(
    include_create: bool,
) -> None:
    class BlockingTopicProvider(LLMProvider):
        name = "openrouter"

        def __init__(self) -> None:
            self.title_filter_waiting = asyncio.Event()
            self.release_title_filter = asyncio.Event()
            self.summary_generation_started = asyncio.Event()
            self.active_requests = 0
            self.max_active_requests = 0

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests, self.active_requests
            )
            try:
                purpose = request.metadata["purpose"]
                target_id = request.metadata.get("topic_working_set_target_id")
                if purpose == "topic_working_set_title_decision_card":
                    self.title_filter_waiting.set()
                    await self.release_title_filter.wait()
                    answer = "keep"
                elif purpose == "topic_working_set_summary_decision_card":
                    answer = "keep" if include_create else "regenerate"
                elif purpose.endswith("_decision_card"):
                    answer = "keep"
                elif purpose == "topic_working_set_title_card":
                    answer = "New topic"
                elif purpose == "topic_working_set_summary_card":
                    assert target_id == ("tmp1" if include_create else "tpc_existing")
                    self.summary_generation_started.set()
                    answer = "New summary."
                else:
                    answer = "none"
                return LLMCompletionResponse(
                    provider=self.name, model=request.model, output_text=answer
                )
            finally:
                self.active_requests -= 1

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = BlockingTopicProvider()
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
    )
    routes = [
        _TopicRoute(
            action=TopicUpdateActionType.UPDATE,
            target_id="tpc_existing",
            source_message_ids=("msg_update",),
        )
    ]
    if include_create:
        routes.append(
            _TopicRoute(
                action=TopicUpdateActionType.CREATE,
                target_id="tmp1",
                source_message_ids=("msg_new",),
            )
        )
    task = asyncio.create_task(
        updater._run_content_cards(
            user_id="usr_1",
            conversation_id="cnv_1",
            snapshot={
                "active_topics": [
                    {"id": "tpc_existing", "title": "Old topic", "summary": "Old summary."}
                ],
                "parked_topics": [],
            },
            messages=[
                {"id": "msg_update", "role": "user", "text": "Update the old topic."},
                {"id": "msg_new", "role": "user", "text": "Start a new topic."},
            ],
            routes=tuple(routes),
        )
    )
    try:
        await asyncio.wait_for(provider.title_filter_waiting.wait(), timeout=2)
        await asyncio.wait_for(provider.summary_generation_started.wait(), timeout=2)
        assert task.done() is False
        assert provider.max_active_requests <= 2
    finally:
        provider.release_title_filter.set()
    contents = await asyncio.wait_for(task, timeout=2)
    assert contents["tmp1" if include_create else "tpc_existing"].summary == "New summary."
    assert provider.max_active_requests <= 2


@pytest.mark.asyncio
async def test_llm_summary_generation_does_not_wait_for_other_topic_summary_filter() -> None:
    class IndependentSummaryProvider(LLMProvider):
        name = "openrouter"

        def __init__(self) -> None:
            self.other_summary_waiting = asyncio.Event()
            self.release_other_summary = asyncio.Event()
            self.regeneration_started = asyncio.Event()

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            purpose = request.metadata["purpose"]
            target_id = request.metadata.get("stage") or request.metadata.get(
                "topic_working_set_target_id"
            )
            if purpose == "topic_working_set_summary_decision_card":
                if target_id == "tpc_a":
                    self.other_summary_waiting.set()
                    await self.release_other_summary.wait()
                    answer = "keep"
                else:
                    answer = "regenerate"
            elif purpose.endswith("_decision_card"):
                answer = "keep"
            elif purpose == "topic_working_set_summary_card":
                assert target_id == "tpc_b"
                self.regeneration_started.set()
                answer = "Updated B summary."
            else:
                raise AssertionError(f"Unexpected card: {purpose}")
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=answer
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = IndependentSummaryProvider()
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=Settings.from_env(
            {"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective"}
        ),
    )
    task = asyncio.create_task(
        updater._run_content_cards(
            user_id="usr_1",
            conversation_id="cnv_1",
            snapshot={
                "active_topics": [
                    {"id": "tpc_a", "title": "Topic A", "summary": "Old A."},
                    {"id": "tpc_b", "title": "Topic B", "summary": "Old B."},
                ],
                "parked_topics": [],
            },
            messages=[
                {"id": "msg_a", "role": "user", "text": "A remains unchanged."},
                {"id": "msg_b", "role": "user", "text": "B has a revision."},
            ],
            routes=(
                _TopicRoute(TopicUpdateActionType.UPDATE, "tpc_a", ("msg_a",)),
                _TopicRoute(TopicUpdateActionType.UPDATE, "tpc_b", ("msg_b",)),
            ),
        )
    )
    try:
        await asyncio.wait_for(provider.other_summary_waiting.wait(), timeout=2)
        await asyncio.wait_for(provider.regeneration_started.wait(), timeout=2)
        assert task.done() is False
    finally:
        provider.release_other_summary.set()
    contents = await asyncio.wait_for(task, timeout=2)
    assert contents["tpc_a"].summary is None
    assert contents["tpc_b"].summary == "Updated B summary."


@pytest.mark.asyncio
async def test_topic_updater_allows_explicit_ordinary_boundary_update() -> None:
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        ["update tpc_existing msg_6", *(["none"] * 5), "tpc_existing ordinary 0 0.6"],
        settings=Settings.from_env({"ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "direct"}),
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Existing boundary topic",
            summary="Keep this summary.",
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.91,
        )
        message = await messages.create_message(
            "msg_6",
            "cnv_1",
            "user",
            1,
            "This topic is now ordinary, but keep the stored privacy level.",
            6,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        assert len(changed) == 1
        refreshed = await topics.get_topic("tpc_existing", "usr_1")
        assert refreshed is not None
        assert refreshed["title"] == "Existing boundary topic"
        assert refreshed["summary"] == "Keep this summary."
        assert refreshed["privacy_level"] == 2
        assert refreshed["intimacy_boundary"] == IntimacyBoundary.ORDINARY.value
        assert refreshed["intimacy_boundary_confidence"] == 0.6
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_accepts_canonical_non_ordinary_boundary_values() -> None:
    outputs = [
        "none",
        "track msg_7",
        "Private boundary discussion",
        "Discusses a private relationship boundary.",
        "none",
        "none",
        "none",
        "tmp1 romantic_private 0 0.82",
    ]
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(outputs)
    try:
        message = await messages.create_message(
            "msg_7",
            "cnv_1",
            "user",
            1,
            "Let's keep this relationship boundary private.",
            7,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )

        assert len(changed) == 1
        refreshed = await topics.get_topic(str(changed[0]["id"]), "usr_1")
        assert refreshed is not None
        assert refreshed["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert refreshed["intimacy_boundary_confidence"] == 0.82
        assert refreshed["privacy_level"] == 2
        assert refreshed["active_goal"] is None
        assert refreshed["open_questions_json"] == []
        assert refreshed["decisions_json"] == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_records_progress_when_batch_only_closes_topic() -> None:
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        ["close tpc_existing msg_4"]
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Current thread",
            summary="Open work.",
        )
        message = await messages.create_message(
            "msg_4",
            "cnv_1",
            "assistant",
            1,
            "Done, that thread is closed.",
            8,
            {},
        )

        changed = await updater.update_from_messages(
            user_id="usr_1",
            conversation_id="cnv_1",
            messages=[message],
        )
        snapshot = await topics.get_topic_snapshot(user_id="usr_1", conversation_id="cnv_1")

        assert changed[0]["status"] == "closed"
        assert snapshot["active_topics"] == []
        assert snapshot["parked_topics"] == []
        assert snapshot["freshness"]["last_processed_seq"] == 1
        assert snapshot["freshness"]["lag_message_count"] == 0
    finally:
        await connection.close()


def test_live_card_prompts_include_examples_except_content_cards() -> None:
    prompts = _live_card_prompts(_prompt_only_updater(card_examples_enabled=True))

    assert set(prompts) == {
        "existing_route",
        "new_topic_track",
        "content_title",
        "content_summary",
        "content_active_goal",
        "content_open_questions",
        "content_decisions",
        "boundary",
    }
    for name, prompt in prompts.items():
        assert ("Examples:" in prompt) is (not name.startswith("content_"))


def test_live_card_prompts_omit_examples_when_disabled() -> None:
    prompts = _live_card_prompts(_prompt_only_updater(card_examples_enabled=False))

    for prompt in prompts.values():
        assert "Examples:" not in prompt


def _artifact_harness_prompt() -> str:
    # The artifact card is HARNESS-ONLY: there is no engine LLM counterpart
    # (the engine derives artifact links deterministically in
    # ``_artifact_ids_from_route_messages``), so the 4-engine-card guard above
    # cannot see it. It is still on the default-graded shadow path, so its
    # baked-in few-shot example is a real leak surface and must be guarded too.
    # Render it against a synthetic dummy case so the only benchmark-derived text
    # that could appear is a few-shot example, not a real case message echoed
    # back as input (which would be a false positive).
    dummy_case = BenchmarkCase(
        case_id="leak_guard_dummy",
        conversation_id="cnv_leak_guard_dummy",
        snapshot={"active_topics": [], "parked_topics": [], "freshness": {"status": "missing"}},
        messages=[
            {
                "id": "msg_leak_guard_dummy",
                "seq": 1,
                "role": "user",
                "text": "PLACEHOLDER_QUERY_TOKEN",
                "metadata_json": {
                    "attachments": [
                        {
                            "artifact_id": "art_leak_guard_dummy",
                            "artifact_type": "document",
                            "source_kind": "upload",
                        }
                    ]
                },
            }
        ],
        expected={"actions": []},
    )
    dummy_route = _TopicRoute(
        action=TopicUpdateActionType.CREATE,
        target_id="tmp1",
        source_message_ids=("msg_leak_guard_dummy",),
    )
    return _build_artifact_prompt(dummy_case, routes=(dummy_route,))


def test_topic_card_prompts_do_not_leak_shadow_benchmark_content() -> None:
    # The topic working-set cards have their own shadow benchmark; their few-shot
    # examples must not reuse a benchmark case message or distinctive answer token,
    # so the benchmark keeps measuring generalization rather than recall of the key.
    prompts = _live_card_prompts(_prompt_only_updater(card_examples_enabled=True))
    combined_prompt = "\n".join(prompts.values())
    assert_prompt_has_no_benchmark_leak(
        combined_prompt, "benchmarks/topic_working_set_cards/cases.jsonl"
    )


def test_artifact_card_prompt_does_not_leak_shadow_benchmark_content() -> None:
    # Guard the harness-only artifact card separately (see _artifact_harness_prompt).
    assert_prompt_has_no_benchmark_leak(
        _artifact_harness_prompt(), "benchmarks/topic_working_set_cards/cases.jsonl"
    )


def test_boundary_prompt_glosses_every_intimacy_boundary_value() -> None:
    prompts = _live_card_prompts(_prompt_only_updater(card_examples_enabled=False))
    boundary_prompt = prompts["boundary"]

    for boundary in IntimacyBoundary:
        assert boundary.value in boundary_prompt


def test_route_card_drops_line_with_zero_valid_message_ids() -> None:
    routes = _parse_route_card_output(
        "update tpc_known msg_missing\nupdate tpc_known msg_real",
        valid_topic_ids={"tpc_known"},
        valid_message_ids=("msg_real",),
        conversation_id="cnv_1",
    )

    assert len(routes) == 1
    assert routes[0].target_id == "tpc_known"
    assert routes[0].source_message_ids == ("msg_real",)


def test_route_card_drops_only_line_when_no_valid_message_ids() -> None:
    routes = _parse_route_card_output(
        "update tpc_known msg_missing",
        valid_topic_ids={"tpc_known"},
        valid_message_ids=("msg_real",),
        conversation_id="cnv_1",
    )

    assert routes == ()


@pytest.mark.asyncio
async def test_typed_batch_releases_independent_topic_generations() -> None:
    started: set[str] = set()
    both_started = asyncio.Event()
    release = asyncio.Event()

    class GenerationProvider(LLMProvider):
        name = "openrouter"

        async def complete(self, request):
            purpose = request.metadata["purpose"]
            if purpose.endswith("_decision_card"):
                answer = "keep"
            else:
                assert purpose == "topic_working_set_summary_card"
                target = request.metadata["topic_working_set_target_id"]
                started.add(target)
                if len(started) == 2:
                    both_started.set()
                await release.wait()
                answer = f"Revised {target}."
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=answer
            )

    typed = ScriptedTypedTopicProvider({"tpc_a": "regenerate", "tpc_b": "regenerate"})
    client = LLMClient(providers=[GenerationProvider(), typed])
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(datetime(2026, 4, 26, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=Settings.from_env({
            "ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE": "selective",
            "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
            "ATAGIA_LLM_FINITE_DECISION_MODEL": "openrouter/openai/gpt-5.6-luna",
            "ATAGIA_LLM_MODEL__TOPIC_SUMMARY_DECISION": "typesafe/jev-1.13.0",
        }),
    )
    task = asyncio.create_task(updater._run_content_cards(
        user_id="usr_1", conversation_id="cnv_1",
        snapshot={"active_topics": [
            {"id": "tpc_a", "title": "A", "summary": "Old A."},
            {"id": "tpc_b", "title": "B", "summary": "Old B."},
        ], "parked_topics": []},
        messages=[{"id": "msg_1", "role": "user", "text": "Revise both summaries."}],
        routes=tuple(
            _TopicRoute(TopicUpdateActionType.UPDATE, target, ("msg_1",))
            for target in ("tpc_a", "tpc_b")
        ),
    ))
    try:
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert not task.done()
    finally:
        release.set()
        contents = await asyncio.wait_for(task, timeout=2)
        await client.aclose()
    assert contents["tpc_a"].summary == "Revised tpc_a."
    assert contents["tpc_b"].summary == "Revised tpc_b."
    assert len(typed.requests) == 1
