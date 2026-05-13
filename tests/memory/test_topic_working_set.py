"""Tests for offline Topic Working Set updates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.topic_working_set import (
    TopicUpdateAction,
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

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class SequentialTopicProvider(LLMProvider):
    name = "topic-working-set"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("No canned topic payload left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payloads.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in topic working-set tests")


async def _build_runtime(payloads: list[dict[str, object]]):
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
    provider = SequentialTopicProvider(payloads)
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
    payload = {
        "actions": [
            {
                "action": "create",
                "title": "Benchmark observability",
                "summary": "Track retained DBs, manifests, and failed-question custody.",
                "active_goal": "Make benchmark comparisons reproducible.",
                "open_questions": ["Which failures lacked sufficient evidence?"],
                "decisions": ["Keep retrieval changes in shadow mode."],
                "artifact_ids": ["art_manifest", "art_hallucinated"],
                "source_message_ids": ["msg_1", "msg_other_user", "missing"],
                "confidence": 0.84,
                "privacy_level": 0,
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, provider = await _build_runtime([payload])
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
        prompt = provider.requests[0].messages[1].content
        assert "Do not create dataset-specific or benchmark-specific topics." in prompt
        assert '"artifact_id": "art_manifest"' in prompt
        assert "run-manifest.json" in prompt
        assert '"relevance_state": "active_work_material"' in prompt
        assert "Do not leak this summary text" not in prompt
        assert "omit any field other than action" in prompt
        for boundary in IntimacyBoundary:
            assert boundary.value in prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_parks_existing_topic_from_model_plan() -> None:
    payload = {
        "actions": [
            {
                "action": "park",
                "topic_id": "tpc_existing",
                "summary": "Paused until the Qwen benchmark run finishes.",
                "source_message_ids": ["msg_2"],
                "confidence": 0.7,
                "privacy_level": 0,
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
        assert changed[0]["summary"] == "Paused until the Qwen benchmark run finishes."
        events = await topics.list_events(user_id="usr_1", conversation_id="cnv_1", topic_id="tpc_existing")
        assert [event["event_type"] for event in events] == ["created", "parked", "source_linked"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_updater_uses_placeholders_for_raw_policy_restricted_messages() -> None:
    payload = {"actions": [], "nothing_to_update": True}
    connection, _clock, messages, _topics, updater, provider = await _build_runtime([payload])
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
    payload = {
        "actions": [
            {
                "action": "update",
                "topic_id": "tpc_other_conv",
                "summary": "This should not be applied across conversations.",
                "source_message_ids": ["msg_3"],
            }
        ],
        "nothing_to_update": False,
    }
    connection, clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
    payload = {
        "actions": [
            {
                "action": "update",
                "topic_id": "tpc_existing",
                "title": "",
                "summary": "",
                "active_goal": "",
                "confidence": -1.0,
                "privacy_level": -1,
                "intimacy_boundary": "",
                "intimacy_boundary_confidence": -1.0,
                "source_message_ids": ["msg_5"],
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
    payload = {
        "actions": [
            {
                "action": "update",
                "topic_id": "tpc_existing",
                "intimacy_boundary": "ordinary",
                "intimacy_boundary_confidence": 0.6,
                "source_message_ids": ["msg_6"],
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
    payload = {
        "actions": [
            {
                "action": "create",
                "title": "Private boundary discussion",
                "summary": "Discusses a private relationship boundary.",
                "intimacy_boundary": "romantic_private",
                "intimacy_boundary_confidence": 0.82,
                "privacy_level": 0,
                "source_message_ids": ["msg_7"],
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
    payload = {
        "actions": [
            {
                "action": "close",
                "topic_id": "tpc_existing",
                "summary": "Closed after the user finished the thread.",
                "source_message_ids": ["msg_4"],
            }
        ],
        "nothing_to_update": False,
    }
    connection, _clock, messages, topics, updater, _provider = await _build_runtime([payload])
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
