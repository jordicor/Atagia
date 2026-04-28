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
from atagia.memory.topic_working_set import TopicWorkingSetPlan, TopicWorkingSetUpdater
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
