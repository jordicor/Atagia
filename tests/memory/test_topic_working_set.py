"""Tests for offline Topic Working Set updates."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.topic_working_set import (
    _TopicContent,
    _TopicRoute,
    _parse_route_card_output,
    TopicUpdateAction,
    TopicUpdateActionType,
    TopicWorkingSetPlan,
    TopicWorkingSetUpdater,
)
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from benchmarks.topic_working_set_cards.compare import (
    BenchmarkCase,
    _build_artifact_prompt,
)
from tests.memory.card_leak_guard import assert_prompt_has_no_benchmark_leak

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


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
        "content": updater._build_content_prompt(
            conversation_id="cnv_1",
            snapshot=snapshot,
            messages=messages,
            route=route,
            existing_topic=None,
        ),
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


async def _build_runtime(outputs: list[str]):
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
    provider = SequentialTopicProvider(outputs)
    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        topic_repository=topics,
        message_repository=messages,
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


def test_topic_update_action_rejects_invalid_sentinel_replacements() -> None:
    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate({"action": "create", "title": "x", "confidence": -2.0})

    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate({"action": "create", "title": "x", "privacy_level": 7})

    with pytest.raises(ValueError):
        TopicUpdateAction.model_validate(
            {"action": "create", "title": "x", "intimacy_boundary": "not_a_boundary"}
        )


@pytest.mark.asyncio
async def test_topic_updater_creates_topic_and_links_only_valid_source_messages() -> None:
    outputs = [
        "none",
        "track msg_1 msg_other_user missing",
        (
            "title: Benchmark observability\n"
            "summary: Track retained DBs, manifests, and failed-question custody.\n"
            "goal: Make benchmark comparisons reproducible.\n"
            "question: Which failures lacked sufficient evidence?\n"
            "decision: Keep retrieval changes in shadow mode."
        ),
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
        boundary_prompt = provider.requests[3].messages[1].content
        assert [
            request.metadata["purpose"]
            for request in provider.requests
        ] == [
            "topic_working_set_route_card",
            "topic_working_set_route_card",
            "topic_working_set_content_card",
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
        ["update tpc_existing msg_5", "none", "none"]
    )
    try:
        await topics.create_topic(
            topic_id="tpc_existing",
            user_id="usr_1",
            conversation_id="cnv_1",
            title="Existing private topic",
            summary="Keep this summary.",
            active_goal="Keep this goal.",
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
        assert refreshed["confidence"] == 0.8
        assert refreshed["privacy_level"] == 2
        assert refreshed["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert refreshed["intimacy_boundary_confidence"] == 0.9
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_allows_explicit_ordinary_boundary_update() -> None:
    connection, _clock, messages, topics, updater, _provider = await _build_runtime(
        ["update tpc_existing msg_6", "none", "tpc_existing ordinary 0 0.6"]
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
        (
            "title: Private boundary discussion\n"
            "summary: Discusses a private relationship boundary."
        ),
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


def test_live_card_prompts_include_examples_by_default() -> None:
    prompts = _live_card_prompts(_prompt_only_updater(card_examples_enabled=True))

    assert set(prompts) == {"existing_route", "new_topic_track", "content", "boundary"}
    for prompt in prompts.values():
        assert "Examples:" in prompt


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
