"""Tests for conversation topic working-set persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.topic_repository import TopicRepository

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 26, 2, 20, tzinfo=timezone.utc))
    return connection, clock


async def _insert_assistant_mode(connection: aiosqlite.Connection, mode_id: str = "coding_debug") -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (mode_id, "Coding Debug", "hash_1", "{}", "2026-04-26T02:20:00+00:00", "2026-04-26T02:20:00+00:00"),
    )
    await connection.commit()


async def _seed_conversations(connection: aiosqlite.Connection, clock: FrozenClock) -> None:
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    await users.create_user("usr_a")
    await users.create_user("usr_b")
    await _insert_assistant_mode(connection)
    await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
    await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")


@pytest.mark.asyncio
async def test_create_topic_round_trips_json_fields_and_respects_user_isolation() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_conversations(connection, clock)
        repository = TopicRepository(connection, clock)

        topic = await repository.create_topic(
            topic_id="tpc_alpha",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Benchmark follow-up",
            summary="Track LoCoMo retrieval gaps before optimizing ranking.",
            active_goal="Identify failed evidence custody paths.",
            open_questions=["Which failures had enough candidates?"],
            decisions=["Keep SQLite as source of truth."],
            artifact_ids=["art_report"],
            source_message_start_seq=3,
            source_message_end_seq=7,
            last_touched_seq=7,
            confidence=0.72,
            privacy_level=1,
        )
        await repository.create_topic(
            topic_id="tpc_other_user",
            user_id="usr_b",
            conversation_id="cnv_b",
            title="Other user topic",
        )

        assert topic["id"] == "tpc_alpha"
        assert topic["status"] == "active"
        assert topic["open_questions_json"] == ["Which failures had enough candidates?"]
        assert topic["decisions_json"] == ["Keep SQLite as source of truth."]
        assert topic["artifact_ids_json"] == ["art_report"]
        assert topic["source_message_start_seq"] == 3
        assert topic["source_message_end_seq"] == 7
        assert topic["last_touched_at"] == "2026-04-26T02:20:00+00:00"

        assert await repository.get_topic("tpc_alpha", "usr_b") is None
        assert [item["id"] for item in await repository.list_topics(user_id="usr_a", conversation_id="cnv_a")] == [
            "tpc_alpha"
        ]

        events = await repository.list_events(user_id="usr_a", conversation_id="cnv_a", topic_id="tpc_alpha")
        assert len(events) == 1
        assert events[0]["event_type"] == "created"
        assert events[0]["payload_json"] == {"status": "active", "title": "Benchmark follow-up"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_update_topic_and_snapshot_keep_active_and_parked_work_sets_compact() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_conversations(connection, clock)
        repository = TopicRepository(connection, clock)

        first = await repository.create_topic(
            topic_id="tpc_first",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="First active topic",
            summary="Older active work.",
            last_touched_seq=2,
        )
        clock.advance(seconds=300)
        second = await repository.create_topic(
            topic_id="tpc_second",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Second active topic",
            summary="Newer active work.",
            last_touched_seq=4,
        )

        parked = await repository.update_topic(
            topic_id=str(first["id"]),
            user_id="usr_a",
            status="parked",
            summary="Paused until custody reports finish.",
            open_questions=["Which benchmark shard regressed?"],
            event_type="parked",
            event_payload={"reason": "waiting_for_benchmark"},
        )

        assert parked is not None
        assert parked["status"] == "parked"
        assert parked["summary"] == "Paused until custody reports finish."
        assert parked["open_questions_json"] == ["Which benchmark shard regressed?"]
        assert await repository.update_topic(topic_id="missing", user_id="usr_a", title="No-op") is None

        snapshot = await repository.get_topic_snapshot(
            user_id="usr_a",
            conversation_id="cnv_a",
            active_limit=1,
            parked_limit=1,
        )
        assert snapshot["active_topics"] == [
            {
                "id": second["id"],
                "status": "active",
                "title": "Second active topic",
                "summary": "Newer active work.",
                "active_goal": None,
                "open_questions": [],
                "decisions": [],
                "artifact_ids": [],
                "source_counts": {},
                "source_refs": [],
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "last_touched_seq": 4,
                "last_touched_at": "2026-04-26T02:25:00+00:00",
                "confidence": 0.5,
                "privacy_level": 0,
                "intimacy_boundary": "ordinary",
                "intimacy_boundary_confidence": 0.0,
            }
        ]
        assert snapshot["freshness"]["status"] == "fresh"
        assert snapshot["freshness"]["last_processed_seq"] == 4
        assert snapshot["parked_topics"][0]["id"] == "tpc_first"
        assert snapshot["parked_topics"][0]["open_questions"] == ["Which benchmark shard regressed?"]

        events = await repository.list_events(user_id="usr_a", conversation_id="cnv_a", topic_id="tpc_first")
        assert [event["event_type"] for event in events] == ["created", "parked"]
        assert events[-1]["payload_json"] == {"reason": "waiting_for_benchmark"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_snapshot_reports_freshness_lag_against_messages() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_conversations(connection, clock)
        messages = MessageRepository(connection, clock)
        topics = TopicRepository(connection, clock)
        for seq in range(1, 7):
            await messages.create_message(
                f"msg_{seq}",
                "cnv_a",
                "user" if seq % 2 else "assistant",
                seq,
                f"Message {seq}",
                10,
                {},
            )
        await topics.create_topic(
            topic_id="tpc_active",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Active topic",
            last_touched_seq=2,
            source_message_start_seq=1,
            source_message_end_seq=2,
        )

        snapshot = await topics.get_topic_snapshot(
            user_id="usr_a",
            conversation_id="cnv_a",
            refresh_message_threshold=2,
            stale_message_threshold=4,
            refresh_token_threshold=100,
            stale_token_threshold=200,
        )

        assert snapshot["active_topics"][0]["source_message_start_seq"] == 1
        assert snapshot["active_topics"][0]["source_message_end_seq"] == 2
        assert snapshot["freshness"] == {
            "status": "stale",
            "last_processed_seq": 2,
            "last_processed_message_id": "msg_2",
            "latest_message_seq": 6,
            "lag_message_count": 4,
            "lag_token_count": 40,
            "refresh_message_threshold": 2,
            "stale_message_threshold": 4,
            "refresh_token_threshold": 100,
            "stale_token_threshold": 200,
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_link_source_is_idempotent_and_user_scoped() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_conversations(connection, clock)
        repository = TopicRepository(connection, clock)
        await repository.create_topic(
            topic_id="tpc_alpha",
            user_id="usr_a",
            conversation_id="cnv_a",
            title="Artifact grounding",
        )

        first_link = await repository.link_source(
            user_id="usr_a",
            topic_id="tpc_alpha",
            source_kind="message",
            source_id="msg_7",
            relation_kind="evidence",
        )
        duplicate_link = await repository.link_source(
            user_id="usr_a",
            topic_id="tpc_alpha",
            source_kind="message",
            source_id="msg_7",
            relation_kind="evidence",
        )

        assert first_link["id"] == duplicate_link["id"]
        sources = await repository.list_topic_sources(user_id="usr_a", topic_id="tpc_alpha")
        assert len(sources) == 1
        assert sources[0]["source_kind"] == "message"
        assert sources[0]["source_id"] == "msg_7"
        assert sources[0]["relation_kind"] == "evidence"

        events = await repository.list_events(user_id="usr_a", conversation_id="cnv_a", topic_id="tpc_alpha")
        assert [event["event_type"] for event in events] == ["created", "source_linked"]
        assert events[-1]["payload_json"] == {
            "source_kind": "message",
            "source_id": "msg_7",
            "relation_kind": "evidence",
        }

        with pytest.raises(ValueError, match="does not belong to user"):
            await repository.link_source(
                user_id="usr_b",
                topic_id="tpc_alpha",
                source_kind="message",
                source_id="msg_7",
            )
    finally:
        await connection.close()
