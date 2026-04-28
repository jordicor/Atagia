"""Tests for retrieval service orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.topic_repository import TopicRepository
from atagia.models.schemas_memory import RetrievalTrace
from atagia.services.retrieval_service import RetrievalService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@dataclass(slots=True)
class _Runtime:
    clock: FrozenClock


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 26, 2, 35, tzinfo=timezone.utc))
    return connection, clock


async def _insert_assistant_mode(connection: aiosqlite.Connection, mode_id: str = "coding_debug") -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (mode_id, "Coding Debug", "hash_1", "{}", "2026-04-26T02:35:00+00:00", "2026-04-26T02:35:00+00:00"),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_attach_topic_snapshot_populates_trace_without_changing_retrieval_behavior() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        topics = TopicRepository(connection, clock)
        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")
        active_topic = await topics.create_topic(
            topic_id="tpc_active",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Active topic",
            summary="Trace-only active topic.",
            open_questions=["Which candidate was filtered?"],
            last_touched_seq=5,
        )
        await topics.link_source(
            user_id="usr_a",
            topic_id=str(active_topic["id"]),
            source_kind="message",
            source_id="msg_1",
        )
        await topics.create_topic(
            topic_id="tpc_parked",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Parked topic",
            status="parked",
            summary="Trace-only parked topic.",
        )
        await topics.create_topic(
            topic_id="tpc_other_user",
            user_id="usr_b",
            conversation_id="cnv_b",
            title="Other user topic",
        )
        trace = RetrievalTrace(
            query_text="What changed?",
            user_id="usr_a",
            conversation_id="cnv_a",
            timestamp_iso=clock.now().isoformat(),
        )

        await RetrievalService(_Runtime(clock))._attach_topic_snapshot(
            connection,
            user_id="usr_a",
            conversation_id="cnv_a",
            trace=trace,
        )

        assert [topic.id for topic in trace.topic_snapshot.active_topics] == ["tpc_active"]
        assert [topic.id for topic in trace.topic_snapshot.parked_topics] == ["tpc_parked"]
        assert trace.topic_snapshot.active_topics[0].open_questions == [
            "Which candidate was filtered?"
        ]
        assert trace.topic_snapshot.active_topics[0].source_counts == {"message": 1}
        assert trace.topic_snapshot.active_topics[0].source_refs == [
            {
                "source_kind": "message",
                "source_id": "msg_1",
                "relation_kind": "evidence",
            }
        ]
    finally:
        await connection.close()
