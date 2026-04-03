"""Tests for conversation dataset export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.services.dataset_exporter import DatasetExporter

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other Chat")
    await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "Need retry help",
        3,
        {},
        "2023-05-08T13:56:00",
    )
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Try the guard", 4, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Retry memory",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id="mem_1",
    )
    await events.create_event(
        {
            "id": "ret_1",
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry"]},
            "selected_memory_ids_json": ["mem_1"],
            "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
            "outcome_json": {
                "detected_needs": ["follow_up_failure"],
                "scored_candidates": [{"memory_id": "mem_1", "final_score": 0.9}],
            },
            "created_at": "2026-04-05T14:00:00+00:00",
        }
    )
    return connection, clock


@pytest.mark.asyncio
async def test_export_conversation_with_and_without_retrieval_traces() -> None:
    connection, clock = await _build_runtime()
    try:
        exporter = DatasetExporter(connection, clock)

        with_traces = await exporter.export_conversation("cnv_1", "usr_1", include_retrieval_traces=True)
        without_traces = await exporter.export_conversation("cnv_1", "usr_1", include_retrieval_traces=False)

        assert [message.seq for message in with_traces.messages] == [1, 2]
        assert with_traces.messages[0].occurred_at == "2023-05-08T13:56:00"
        assert with_traces.messages[1].occurred_at == with_traces.messages[1].created_at
        assert with_traces.retrieval_traces is not None
        assert with_traces.retrieval_traces[0].retrieval_event_id == "ret_1"
        assert without_traces.retrieval_traces is None
        json.dumps(with_traces.model_dump(mode="json"))
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_export_conversation_verifies_user_ownership() -> None:
    connection, clock = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="Conversation not found for user"):
            await DatasetExporter(connection, clock).export_conversation("cnv_1", "usr_2")
    finally:
        await connection.close()
