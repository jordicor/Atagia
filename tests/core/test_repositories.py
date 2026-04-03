"""Integration tests for the Step 3 repository layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.storage_backend import InProcessBackend
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    return connection, clock


async def _insert_assistant_mode(connection: aiosqlite.Connection, mode_id: str = "coding_debug") -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (mode_id, "Coding Debug", "hash_1", "{}", "2026-03-30T12:00:00+00:00", "2026-03-30T12:00:00+00:00"),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_user_and_workspace_crud_round_trip() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock)

        user = await users.create_user("usr_a", "external_a")
        workspace = await workspaces.create_workspace("wrk_a", "usr_a", "Main", {"kind": "repo"})

        assert user["id"] == "usr_a"
        assert workspace["metadata_json"] == {"kind": "repo"}
        assert await workspaces.get_workspace("wrk_a", "usr_a") is not None
        assert await workspaces.get_workspace("wrk_a", "usr_b") is None
        assert [item["id"] for item in await workspaces.list_workspaces("usr_a")] == ["wrk_a"]

        clock.advance(seconds=60)
        await users.delete_user("usr_a")
        deleted = await users.get_user("usr_a")
        assert deleted["deleted_at"] == "2026-03-30T12:01:00+00:00"
        assert deleted["updated_at"] == "2026-03-30T12:01:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_crud_and_workspace_filtering() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await workspaces.create_workspace("wrk_a", "usr_a", "Main", {})
        await workspaces.create_workspace("wrk_b", "usr_a", "Secondary", {})

        first = await conversations.create_conversation("cnv_a", "usr_a", "wrk_a", "coding_debug", "First")
        clock.advance(seconds=30)
        second = await conversations.create_conversation("cnv_b", "usr_a", "wrk_b", "coding_debug", "Second")

        assert first["workspace_id"] == "wrk_a"
        assert second["workspace_id"] == "wrk_b"
        assert await conversations.get_conversation("cnv_a", "usr_b") is None
        assert [item["id"] for item in await conversations.list_conversations("usr_a")] == ["cnv_b", "cnv_a"]
        assert [item["id"] for item in await conversations.list_conversations("usr_a", workspace_id="wrk_a")] == ["cnv_a"]

        clock.advance(seconds=30)
        await conversations.update_conversation_status("cnv_a", "usr_a", "archived")
        archived = await conversations.get_conversation("cnv_a", "usr_a")
        assert archived["status"] == "archived"
        assert archived["updated_at"] == "2026-03-30T12:01:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_message_round_trip_ordering_and_user_isolation() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")

        await messages.create_message("msg_2", "cnv_a", "assistant", 2, "Second", 1, {})
        await messages.create_message("msg_1", "cnv_a", "user", 1, "First", 1, {"source": "user"})
        await messages.create_message("msg_3", "cnv_b", "user", 1, "Other user", 1, {})

        ordered = await messages.get_messages("cnv_a", "usr_a", limit=10, offset=0)
        assert [item["id"] for item in ordered] == ["msg_1", "msg_2"]
        assert ordered[0]["metadata_json"] == {"source": "user"}
        assert await messages.get_messages("cnv_a", "usr_b", limit=10, offset=0) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_message_persists_occurred_at_and_defaults_to_created_at() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        explicit = await messages.create_message(
            "msg_explicit",
            "cnv_a",
            "user",
            1,
            "Historical message",
            2,
            {},
            "2023-05-08T13:56:00",
        )
        defaulted = await messages.create_message(
            "msg_defaulted",
            "cnv_a",
            "assistant",
            2,
            "Live reply",
            2,
            {},
            "   ",
        )

        assert explicit["occurred_at"] == "2023-05-08T13:56:00"
        assert explicit["created_at"] == "2026-03-30T12:00:00+00:00"
        assert defaulted["occurred_at"] == defaulted["created_at"] == "2026-03-30T12:00:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_fts_search_filters_by_user_before_ranking() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")

        await messages.create_message("msg_a1", "cnv_a", "user", 1, "vector cache invalidation", 3, {})
        await messages.create_message("msg_b1", "cnv_b", "user", 1, "vector cache invalidation", 3, {})

        user_a_hits = await messages.search_messages("usr_a", "vector", limit=10)
        user_b_hits = await messages.search_messages("usr_b", "vector", limit=10)

        assert [item["id"] for item in user_a_hits] == ["msg_a1"]
        assert [item["id"] for item in user_b_hits] == ["msg_b1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_object_search_filters_by_user_before_ranking() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                canonical_text, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES
                (?, ?, NULL, NULL, ?, 'evidence', 'conversation', ?, '{}', 'extracted', 0.5, 0.5, 0.0, 0.0, 0, NULL, NULL, 'active', ?, ?),
                (?, ?, NULL, NULL, ?, 'evidence', 'conversation', ?, '{}', 'extracted', 0.5, 0.5, 0.0, 0.0, 0, NULL, NULL, 'active', ?, ?)
            """,
            (
                "mem_a",
                "usr_a",
                "coding_debug",
                "Alice likes concise answers",
                "2026-03-30T12:00:00+00:00",
                "2026-03-30T12:00:00+00:00",
                "mem_b",
                "usr_b",
                "coding_debug",
                "Bob likes concise answers",
                "2026-03-30T12:00:00+00:00",
                "2026-03-30T12:00:00+00:00",
            ),
        )
        await connection.commit()

        user_a_hits = await memories.search_memory_objects("usr_a", "concise", limit=10)
        user_b_hits = await memories.search_memory_objects("usr_b", "concise", limit=10)

        assert [item["id"] for item in user_a_hits] == ["mem_a"]
        assert [item["id"] for item in user_b_hits] == ["mem_b"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_workspace_delete_triggers_schema_cascade_and_orphaning() -> None:
    connection, clock = await _connection_and_clock()
    cache_backend = InProcessBackend()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock, storage_backend=cache_backend)
        conversations = ConversationRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await workspaces.create_workspace("wrk_a", "usr_a", "Main", {})
        await conversations.create_conversation("cnv_a", "usr_a", "wrk_a", "coding_debug", "Chat A")
        await cache_backend.set_context_view(
            "ctx:1",
            {"user_id": "usr_a", "conversation_id": "cnv_a"},
            ttl_seconds=60,
        )
        await cache_backend.set_context_view(
            "ctx:2",
            {"user_id": "usr_b", "conversation_id": "cnv_b"},
            ttl_seconds=60,
        )
        await connection.execute(
            """
            INSERT INTO summary_views(
                id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, created_at
            )
            VALUES (?, NULL, ?, 0, 0, 'workspace_rollup', 'summary', '[]', 1.5, ?)
            """,
            ("sum_a", "wrk_a", "2026-03-30T12:00:00+00:00"),
        )
        await connection.commit()

        await workspaces.delete_workspace("wrk_a", "usr_a")

        conversation = await conversations.get_conversation("cnv_a", "usr_a")
        assert conversation["workspace_id"] is None

        cursor = await connection.execute(
            "SELECT workspace_id FROM summary_views WHERE id = ?",
            ("sum_a",),
        )
        summary_row = await cursor.fetchone()
        assert summary_row["workspace_id"] is None
        assert await cache_backend.get_context_view("ctx:1") is None
        assert await cache_backend.get_context_view("ctx:2") == {
            "user_id": "usr_b",
            "conversation_id": "cnv_b",
        }
    finally:
        await cache_backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_create_message_assigns_sequences_atomically_across_connections(tmp_path: Path) -> None:
    database_path = tmp_path / "atagia-messages.db"
    connection_a = await initialize_database(str(database_path), MIGRATIONS_DIR)
    connection_b = await initialize_database(str(database_path), MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection_a, clock)
        conversations = ConversationRepository(connection_a, clock)
        await users.create_user("usr_a")
        await _insert_assistant_mode(connection_a)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        messages_a = MessageRepository(connection_a, clock)
        messages_b = MessageRepository(connection_b, clock)
        created = await asyncio.gather(
            messages_a.create_message("msg_1", "cnv_a", "user", None, "First", 1, {}),
            messages_b.create_message("msg_2", "cnv_a", "assistant", None, "Second", 1, {}),
        )

        ordered = await messages_a.get_messages("cnv_a", "usr_a", limit=10, offset=0)

        assert {item["id"] for item in created} == {"msg_1", "msg_2"}
        assert [item["seq"] for item in ordered] == [1, 2]
    finally:
        await connection_a.close()
        await connection_b.close()
