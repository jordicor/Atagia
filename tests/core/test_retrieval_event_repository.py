"""Integration tests for retrieval logging repositories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.retrieval_event_repository import MemoryFeedbackRepository, RetrievalEventRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    feedback = MemoryFeedbackRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "User Two")
    await messages.create_message("msg_1", "cnv_1", "user", 1, "Need help with retries", 5, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Let's inspect the queue", 6, {})
    await messages.create_message("msg_3", "cnv_2", "user", 1, "Other user prompt", 4, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User is debugging retry behavior.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        memory_id="mem_1",
    )
    await memories.create_memory_object(
        user_id="usr_2",
        conversation_id="cnv_2",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User two memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.7,
        privacy_level=0,
        memory_id="mem_2",
    )
    return connection, clock, events, feedback


@pytest.mark.asyncio
async def test_retrieval_event_round_trip_create_get_and_list() -> None:
    connection, _clock, events, _feedback = await _build_runtime()
    try:
        created = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "response_message_id": "msg_2",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["retry queue"], "skip_retrieval": False},
                "selected_memory_ids_json": [],
                "context_view_json": {
                    "selected_memory_ids": [],
                    "total_tokens_estimate": 48,
                    "items_included": 0,
                    "items_dropped": 0,
                },
                "outcome_json": {"zero_candidates": True},
            }
        )

        fetched = await events.get_event(created["id"], "usr_1")
        listed = await events.list_events("usr_1", "cnv_1", limit=10, offset=0)

        assert created["selected_memory_ids_json"] == []
        assert fetched["retrieval_plan_json"]["fts_queries"] == ["retry queue"]
        assert listed[0]["id"] == created["id"]
        assert listed[0]["context_view_json"]["total_tokens_estimate"] == 48
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_retrieval_events_are_isolated_by_user_id() -> None:
    connection, _clock, events, _feedback = await _build_runtime()
    try:
        first = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "response_message_id": "msg_2",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["retry"]},
                "selected_memory_ids_json": ["mem_1"],
                "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
                "outcome_json": {},
            }
        )
        second = await events.create_event(
            {
                "user_id": "usr_2",
                "conversation_id": "cnv_2",
                "request_message_id": "msg_3",
                "response_message_id": None,
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["other"]},
                "selected_memory_ids_json": [],
                "context_view_json": {"selected_memory_ids": [], "items_included": 0, "items_dropped": 0},
                "outcome_json": {},
            }
        )

        assert await events.get_event(first["id"], "usr_2") is None
        assert await events.get_event(second["id"], "usr_1") is None
        assert [event["id"] for event in await events.list_events("usr_1", None, 10, 0)] == [first["id"]]
        assert [event["id"] for event in await events.list_events("usr_2", None, 10, 0)] == [second["id"]]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_feedback_round_trip() -> None:
    connection, _clock, events, feedback = await _build_runtime()
    try:
        event = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "response_message_id": "msg_2",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["retry"]},
                "selected_memory_ids_json": ["mem_1"],
                "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
                "outcome_json": {},
            }
        )

        created = await feedback.create_feedback(
            retrieval_event_id=event["id"],
            memory_id="mem_1",
            user_id="usr_1",
            feedback_type="useful",
            score=0.9,
            metadata={"source": "manual"},
        )
        listed = await feedback.list_feedback("mem_1", "usr_1")

        assert created["feedback_type"] == "useful"
        assert created["metadata_json"] == {"source": "manual"}
        assert [item["id"] for item in listed] == [created["id"]]
        assert await feedback.list_feedback("mem_1", "usr_2") == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_feedback_rejects_memory_owned_by_another_user() -> None:
    connection, _clock, events, feedback = await _build_runtime()
    try:
        event = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "response_message_id": "msg_2",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["retry"]},
                "selected_memory_ids_json": [],
                "context_view_json": {"selected_memory_ids": [], "items_included": 0, "items_dropped": 0},
                "outcome_json": {},
            }
        )

        with pytest.raises(ValueError, match="Memory object mem_2 does not belong to user usr_1"):
            await feedback.create_feedback(
                retrieval_event_id=event["id"],
                memory_id="mem_2",
                user_id="usr_1",
                feedback_type="irrelevant",
                score=0.1,
                metadata={},
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_feedback_rejects_same_user_memory_not_selected_in_event() -> None:
    connection, _clock, events, feedback = await _build_runtime()
    try:
        event = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "response_message_id": "msg_2",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {"fts_queries": ["retry"]},
                "selected_memory_ids_json": [],
                "context_view_json": {"selected_memory_ids": [], "items_included": 0, "items_dropped": 0},
                "outcome_json": {},
            }
        )

        with pytest.raises(
            ValueError,
            match="Memory object mem_1 was not selected in retrieval event",
        ):
            await feedback.create_feedback(
                retrieval_event_id=event["id"],
                memory_id="mem_1",
                user_id="usr_1",
                feedback_type="irrelevant",
                score=0.1,
                metadata={},
            )
    finally:
        await connection.close()
