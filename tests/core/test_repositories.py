"""Integration tests for the Step 3 repository layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_provenance import MemoryProvenanceWriter
from atagia.core.storage_backend import InProcessBackend
from atagia.core.repositories import (
    RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ConversationRepository,
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    SpaceBoundaryMode,
    SummaryViewKind,
)
from atagia.services.errors import ConversationNotActiveError

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


async def _create_test_memory(
    memories: MemoryObjectRepository,
    *,
    user_id: str,
    memory_id: str,
    canonical_text: str,
) -> dict[str, Any]:
    return await memories.create_memory_object(
        user_id=user_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.USER,
        canonical_text=canonical_text,
        payload={},
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        memory_id=memory_id,
    )


async def _count_table(connection: aiosqlite.Connection, table_name: str) -> int:
    cursor = await connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}")
    row = await cursor.fetchone()
    return int(row["count"])


@pytest.mark.asyncio
async def test_memory_evidence_repository_round_trips_packets_and_enforces_user_scope() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        evidence = MemoryEvidenceRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")
        await messages.create_message("msg_trigger", "cnv_a", "assistant", 1, "What's your fave?", 4, {})
        await messages.create_message(
            "msg_source",
            "cnv_a",
            "user",
            2,
            "Yeah, me too! Contemporary dance really speaks to me.",
            9,
            {},
            occurred_at="2023-01-20T16:04:00+00:00",
        )
        await messages.create_message("msg_other_user", "cnv_b", "user", 1, "Other user's evidence.", 4, {})
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_gina",
            canonical_text="Gina's favorite dance style is contemporary.",
        )

        packet = await evidence.create_support_edge_with_spans(
            user_id="usr_a",
            memory_id="mem_gina",
            support_kind="contextual_direct",
            evidence_polarity="supports",
            speaker_relation_to_subject="self_report",
            confidence=0.91,
            rationale="Gina answers the favorite-dance question.",
            spans=[
                {
                    "span_role": "source",
                    "message_id": "msg_source",
                    "quote_text": "Contemporary dance really speaks to me.",
                },
                {
                    "span_role": "trigger",
                    "message_id": "msg_trigger",
                    "quote_text": "What's your fave?",
                },
            ],
        )

        assert packet["support_kind"] == "contextual_direct"
        assert [span["span_role"] for span in packet["spans"]] == ["source", "trigger"]
        packets = await evidence.list_packets_for_memory_ids(
            user_id="usr_a",
            memory_ids=["mem_gina"],
        )
        assert packets["mem_gina"][0]["speaker_relation_to_subject"] == "self_report"
        assert await evidence.list_packets_for_memory_ids(
            user_id="usr_b",
            memory_ids=["mem_gina"],
        ) == {}

        with pytest.raises(ValueError, match="same user"):
            await evidence.create_support_edge_with_spans(
                user_id="usr_a",
                memory_id="mem_gina",
                spans=[
                    {
                        "span_role": "source",
                        "message_id": "msg_other_user",
                        "quote_text": "Other user's evidence.",
                    }
                ],
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_evidence_spans_cascade_when_memory_is_deleted() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        evidence = MemoryEvidenceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await messages.create_message("msg_source", "cnv_a", "user", 1, "I like contemporary.", 4, {})
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_gina",
            canonical_text="Gina likes contemporary dance.",
        )
        await evidence.create_support_edge_with_spans(
            user_id="usr_a",
            memory_id="mem_gina",
            spans=[
                {
                    "span_role": "source",
                    "message_id": "msg_source",
                    "quote_text": "I like contemporary.",
                }
            ],
        )

        assert await _count_table(connection, "memory_support_edges") == 1
        assert await _count_table(connection, "memory_evidence_spans") == 1
        await connection.execute("DELETE FROM memory_objects WHERE id = ?", ("mem_gina",))
        await connection.commit()

        assert await _count_table(connection, "memory_support_edges") == 0
        assert await _count_table(connection, "memory_evidence_spans") == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_evidence_repository_preserves_quote_whitespace() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        evidence = MemoryEvidenceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await messages.create_message("msg_source", "cnv_a", "user", 1, "prefix\n\nexact quote", 4, {})
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_gina",
            canonical_text="Gina has multiline evidence.",
        )

        packet = await evidence.create_support_edge_with_spans(
            user_id="usr_a",
            memory_id="mem_gina",
            spans=[
                {
                    "span_role": "source",
                    "message_id": "msg_source",
                    "quote_text": "prefix\n\nexact quote",
                    "char_start": 0,
                    "char_end": 19,
                }
            ],
        )

        span = packet["spans"][0]
        assert span["quote_text"] == "prefix\n\nexact quote"
        assert span["char_start"] == 0
        assert span["char_end"] == 19
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_provenance_writer_falls_back_to_exact_message_text() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        message_text = "Gina: I won regionals.\n\n[Attachments omitted]\nimage attachment"

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await messages.create_message("msg_source", "cnv_a", "user", 1, message_text, 8, {})
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_gina",
            canonical_text="Gina won regionals.",
        )

        writer = MemoryProvenanceWriter(connection, clock)
        packet = await writer.create_packet_from_source_messages(
            user_id="usr_a",
            memory_id="mem_gina",
            source_message_ids=["msg_source"],
            writer_kind="test",
            support_kind="direct",
            source_quote_by_message_id={
                "msg_source": "Gina: I won regionals. [Attachments omitted] image attachment"
            },
        )

        assert packet is not None
        span = packet["spans"][0]
        assert span["quote_text"] == message_text
        assert span["char_start"] == 0
        assert span["char_end"] == len(message_text)
        assert span["metadata_json"]["quote_fallback"] == "full_message_exact"
    finally:
        await connection.close()


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
        assert await users.get_active_user("usr_a") is None
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
        clock.advance(seconds=30)
        temporary = await conversations.create_conversation(
            "cnv_temp",
            "usr_a",
            None,
            "coding_debug",
            "Temporary",
            temporary=True,
            temporary_ttl_seconds=900,
            purge_on_close=True,
        )
        clock.advance(seconds=30)
        isolated = await conversations.create_conversation(
            "cnv_isolated",
            "usr_a",
            None,
            "coding_debug",
            "Isolated",
            isolated_mode=True,
        )

        assert first["workspace_id"] == "wrk_a"
        assert second["workspace_id"] == "wrk_b"
        assert temporary["temporary"] == 1
        assert temporary["temporary_ttl_seconds"] == 900
        assert temporary["purge_on_close"] == 1
        assert temporary["last_activity_at"] == temporary["created_at"]
        assert isolated["isolated_mode"] == 1
        assert await conversations.get_conversation("cnv_a", "usr_b") is None
        assert [item["id"] for item in await conversations.list_conversations("usr_a")] == [
            "cnv_isolated",
            "cnv_b",
            "cnv_a",
        ]
        assert [
            item["id"]
            for item in await conversations.list_conversations("usr_a", include_temporary=True)
        ] == ["cnv_isolated", "cnv_temp", "cnv_b", "cnv_a"]
        assert [item["id"] for item in await conversations.list_conversations("usr_a", workspace_id="wrk_a")] == ["cnv_a"]

        clock.advance(seconds=30)
        await conversations.update_conversation_status("cnv_a", "usr_a", "archived")
        archived = await conversations.get_conversation("cnv_a", "usr_a")
        assert archived["status"] == "archived"
        assert archived["updated_at"] == "2026-03-30T12:02:00+00:00"
        assert [item["id"] for item in await conversations.list_conversations("usr_a")] == [
            "cnv_isolated",
            "cnv_b",
        ]
        assert [
            item["id"]
            for item in await conversations.list_conversations("usr_a", include_archived=True)
        ] == ["cnv_a", "cnv_isolated", "cnv_b"]
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
        conversation = await conversations.get_conversation("cnv_a", "usr_a")
        assert conversation["last_activity_at"] == "2026-03-30T12:00:00+00:00"

        ordered = await messages.get_messages("cnv_a", "usr_a", limit=10, offset=0)
        assert [item["id"] for item in ordered] == ["msg_1", "msg_2"]
        assert ordered[0]["metadata_json"] == {"source": "user"}
        assert await messages.get_messages("cnv_a", "usr_b", limit=10, offset=0) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_message_rejects_non_active_conversation_atomically() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await conversations.update_conversation_status("cnv_a", "usr_a", "closed")

        with pytest.raises(ConversationNotActiveError):
            await messages.create_message("msg_closed", "cnv_a", "user", None, "Nope", 1, {})

        rows = await messages.get_messages("cnv_a", "usr_a", limit=10, offset=0)
        conversation = await conversations.get_conversation("cnv_a", "usr_a")
        assert rows == []
        assert conversation["last_activity_at"] == "2026-03-30T12:00:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_temporary_conversation_messages_are_hidden_from_other_conversations() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_active", "usr_a", None, "coding_debug", "Active")
        await conversations.create_conversation(
            "cnv_temp",
            "usr_a",
            None,
            "coding_debug",
            "Temp",
            temporary=True,
            purge_on_close=True,
        )
        await conversations.create_conversation(
            "cnv_isolated",
            "usr_a",
            None,
            "coding_debug",
            "Isolated",
            isolated_mode=True,
        )
        await conversations.create_conversation("cnv_pending", "usr_a", None, "coding_debug", "Pending")
        await messages.create_message("msg_active", "cnv_active", "user", None, "visible kiwi", 1, {})
        await messages.create_message("msg_temp", "cnv_temp", "user", None, "secret kiwi", 1, {})
        await messages.create_message("msg_isolated", "cnv_isolated", "user", None, "isolated kiwi", 1, {})
        await messages.create_message("msg_pending", "cnv_pending", "user", None, "pending kiwi", 1, {})
        await conversations.update_conversation_status("cnv_pending", "usr_a", "pending_deletion")

        visible_to_active = await messages.search_messages_with_privacy(
            user_id="usr_a",
            query="kiwi",
            privacy_ceiling=3,
            limit=10,
            allow_conversation_id="cnv_active",
        )
        visible_to_temp = await messages.search_messages_with_privacy(
            user_id="usr_a",
            query="kiwi",
            privacy_ceiling=3,
            limit=10,
            allow_conversation_id="cnv_temp",
        )
        visible_to_isolated = await messages.search_messages_with_privacy(
            user_id="usr_a",
            query="kiwi",
            privacy_ceiling=3,
            limit=10,
            allow_conversation_id="cnv_isolated",
        )

        assert [row["id"] for row in visible_to_active] == ["msg_active"]
        assert {row["id"] for row in visible_to_temp} == {"msg_active", "msg_temp"}
        assert [row["id"] for row in visible_to_isolated] == ["msg_isolated"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_recent_messages_returns_newest_rows_sorted_ascending() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        for seq in range(1, 13):
            await messages.create_message(
                f"msg_{seq}",
                "cnv_a",
                "user" if seq % 2 else "assistant",
                seq,
                f"Message {seq}",
                2,
                {},
            )

        recent = await messages.get_recent_messages("cnv_a", "usr_a", limit=5)

        assert [item["seq"] for item in recent] == [8, 9, 10, 11, 12]
        assert [item["id"] for item in recent] == ["msg_8", "msg_9", "msg_10", "msg_11", "msg_12"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_messages_from_seq_returns_tail_in_order() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        for seq in range(1, 8):
            await messages.create_message(
                f"msg_tail_{seq}",
                "cnv_a",
                "user" if seq % 2 else "assistant",
                seq,
                f"Message {seq}",
                2,
                {},
            )

        tail = await messages.get_messages_from_seq("cnv_a", "usr_a", start_seq=4)

        assert [item["seq"] for item in tail] == [4, 5, 6, 7]
        assert [item["id"] for item in tail] == [
            "msg_tail_4",
            "msg_tail_5",
            "msg_tail_6",
            "msg_tail_7",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_message_persists_optional_occurred_at_without_created_at_fallback() -> None:
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
        assert defaulted["occurred_at"] is None
        assert defaulted["created_at"] == "2026-03-30T12:00:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_message_derives_context_policy_columns_from_metadata_and_size() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        normal = await messages.create_message(
            "msg_normal",
            "cnv_a",
            "user",
            1,
            "Short text",
            3,
            {},
        )
        explicit = await messages.create_message(
            "msg_explicit",
            "cnv_a",
            "assistant",
            2,
            "Attached PDF payload",
            6,
            {
                "content_kind": "pdf",
                "include_raw": False,
            },
        )
        mechanical = await messages.create_message(
            "msg_mechanical",
            "cnv_a",
            "user",
            3,
            "x" * 5000,
            4096,
            {},
        )
        auto_mechanical = await messages.create_message(
            "msg_auto_mechanical",
            "cnv_a",
            "assistant",
            None,
            "y" * 17000,
            None,
            {},
        )
        long_but_below_token_threshold = await messages.create_message(
            "msg_long_below_token_threshold",
            "cnv_a",
            "user",
            None,
            "z" * 5000,
            None,
            {},
        )

        assert normal["content_kind"] == "text"
        assert normal["include_raw"] == 1
        assert normal["skip_by_default"] == 0
        assert normal["heavy_content"] == 0
        assert normal["context_placeholder"] is None
        assert normal["policy_reason"] is None

        assert explicit["content_kind"] == "pdf"
        assert explicit["include_raw"] == 0
        assert explicit["skip_by_default"] == 1
        assert explicit["heavy_content"] == 0
        assert explicit["requires_explicit_request"] == 1
        assert explicit["policy_reason"] == "skip_by_default"
        assert explicit["context_placeholder"] is not None
        assert "msg_explicit" in explicit["context_placeholder"]

        assert mechanical["content_kind"] == "attachment"
        assert mechanical["include_raw"] == 0
        assert mechanical["skip_by_default"] == 1
        assert mechanical["heavy_content"] == 1
        assert mechanical["requires_explicit_request"] == 1
        assert mechanical["policy_reason"] == "mechanical_size_threshold"
        assert mechanical["context_placeholder"] is not None
        assert "msg_mechanical" in mechanical["context_placeholder"]

        assert auto_mechanical["seq"] == 4
        assert auto_mechanical["context_placeholder"] is not None
        assert "seq=4" in auto_mechanical["context_placeholder"]
        assert "msg_auto_mechanical" in auto_mechanical["context_placeholder"]
        assert long_but_below_token_threshold["heavy_content"] == 0
        assert long_but_below_token_threshold["include_raw"] == 1
        assert long_but_below_token_threshold["skip_by_default"] == 0
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
                (?, ?, NULL, NULL, ?, 'evidence', 'chat', ?, '{}', 'extracted', 0.5, 0.5, 0.0, 0.0, 0, NULL, NULL, 'active', ?, ?),
                (?, ?, NULL, NULL, ?, 'evidence', 'chat', ?, '{}', 'extracted', 0.5, 0.5, 0.0, 0.0, 0, NULL, NULL, 'active', ?, ?)
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
async def test_create_memory_object_persists_temporal_fields() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)

        created = await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User is traveling next week.",
            payload={"temporal_confidence": 0.82},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            valid_from="2026-04-06T00:00:00+00:00",
            valid_to="2026-04-12T23:59:59.999999+00:00",
            temporal_type="bounded",
            memory_id="mem_temporal",
        )

        assert created["temporal_type"] == "bounded"
        assert created["valid_from"] == "2026-04-06T00:00:00+00:00"
        assert created["valid_to"] == "2026-04-12T23:59:59.999999+00:00"
        assert created["payload_json"]["temporal_confidence"] == pytest.approx(0.82)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_memory_object_persists_index_text() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)

        created = await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers terse debugging help.",
            index_text="Context: this preference was stated during a production incident debugging discussion.",
            payload={},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_index_text",
        )

        assert created["index_text"] == (
            "Context: this preference was stated during a production incident debugging discussion."
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_object_search_matches_index_text() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)

        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User preference memory",
            index_text="This memory came from a discussion about websocket backoff troubleshooting.",
            payload={},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_index_search",
        )

        hits = await memories.search_memory_objects("usr_a", "websocket", limit=10)

        assert [item["id"] for item in hits] == ["mem_index_search"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_count_for_context_counts_only_active_memories() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        for memory_id, status in (
            ("mem_active", MemoryStatus.ACTIVE),
            ("mem_review", MemoryStatus.REVIEW_REQUIRED),
            ("mem_pending", MemoryStatus.PENDING_USER_CONFIRMATION),
            ("mem_declined", MemoryStatus.DECLINED),
        ):
            await memories.create_memory_object(
                user_id="usr_a",
                assistant_mode_id="coding_debug",
                conversation_id="cnv_a",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text=f"{memory_id} memory",
                payload={"source_message_ids": ["msg_1"]},
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                status=status,
                memory_id=memory_id,
            )

        count = await memories.count_for_context(
            "usr_a",
            [MemoryScope.CONVERSATION],
            workspace_id=None,
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
        )

        assert count == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_context_helpers_apply_phase7_namespace_platform_and_sensitivity_gates() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await workspaces.create_workspace("wrk_a", "usr_a", "Character")
        await conversations.create_conversation("cnv_a", "usr_a", "wrk_a", "coding_debug", "Chat A")
        await conversations.create_conversation("cnv_other", "usr_a", "wrk_a", "coding_debug", "Other")

        seed_kwargs = {
            "user_id": "usr_a",
            "assistant_mode_id": "coding_debug",
            "object_type": MemoryObjectType.EVIDENCE,
            "source_kind": MemorySourceKind.EXTRACTED,
            "confidence": 0.9,
            "privacy_level": 0,
            "status": MemoryStatus.ACTIVE,
            "user_persona_id": "persona_a",
            "platform_id": "default",
            "sensitivity": MemorySensitivity.PUBLIC,
        }
        await memories.create_memory_object(
            **seed_kwargs,
            memory_id="mem_chat",
            conversation_id="cnv_a",
            workspace_id="wrk_a",
            scope=MemoryScope.CONVERSATION,
            scope_canonical=MemoryScope.CHAT.value,
            canonical_text="chat context",
        )
        await memories.create_memory_object(
            **seed_kwargs,
            memory_id="mem_character",
            workspace_id="wrk_a",
            conversation_id=None,
            character_id="char_a",
            scope=MemoryScope.WORKSPACE,
            scope_canonical=MemoryScope.CHARACTER.value,
            canonical_text="character context",
        )
        await memories.create_memory_object(
            **seed_kwargs,
            memory_id="mem_user",
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.GLOBAL_USER,
            scope_canonical=MemoryScope.USER.value,
            canonical_text="user context",
        )
        await memories.create_memory_object(
            **seed_kwargs,
            memory_id="mem_other_chat",
            workspace_id="wrk_a",
            conversation_id="cnv_other",
            scope=MemoryScope.CONVERSATION,
            scope_canonical=MemoryScope.CHAT.value,
            canonical_text="other chat context",
        )
        await memories.create_memory_object(
            **{**seed_kwargs, "sensitivity": MemorySensitivity.PRIVATE},
            memory_id="mem_private",
            workspace_id="wrk_a",
            conversation_id="cnv_a",
            scope=MemoryScope.CONVERSATION,
            scope_canonical=MemoryScope.CHAT.value,
            canonical_text="private context",
        )
        await memories.create_memory_object(
            **{**seed_kwargs, "platform_id": "ios"},
            memory_id="mem_wrong_platform",
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.GLOBAL_USER,
            scope_canonical=MemoryScope.USER.value,
            canonical_text="wrong platform context",
            platform_locked=True,
            platform_id_lock="ios",
        )

        rows = await memories.list_eligible_for_context(
            "usr_a",
            [MemoryScope.CONVERSATION, MemoryScope.WORKSPACE, MemoryScope.GLOBAL_USER],
            workspace_id="wrk_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
        )
        count = await memories.count_for_context(
            "usr_a",
            [MemoryScope.CONVERSATION, MemoryScope.WORKSPACE, MemoryScope.GLOBAL_USER],
            workspace_id="wrk_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
        )

        assert {row["id"] for row in rows} == {"mem_chat", "mem_character", "mem_user"}
        assert count == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_repository_helpers_default_to_retrieval_eligible_statuses() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
        await messages.create_message("msg_1", "cnv_a", "user", 1, "first", 1, {})
        await messages.create_message("msg_2", "cnv_a", "user", 2, "second", 1, {})

        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Active credential memory",
            payload={"source_message_ids": ["msg_1"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_active",
        )
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Pending credential memory",
            payload={"source_message_ids": ["msg_2"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=3,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            memory_id="mem_pending",
        )
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Declined credential memory",
            payload={"source_message_ids": ["msg_2"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=3,
            status=MemoryStatus.DECLINED,
            memory_id="mem_declined",
        )

        default_rows = await memories.list_for_user("usr_a")
        all_rows = await memories.list_for_user("usr_a", statuses=None)
        default_hits = await memories.search_memory_objects("usr_a", "credential", limit=10)
        all_hits = await memories.search_memory_objects("usr_a", "credential", limit=10, statuses=None)
        default_has_msg_1 = await memories.has_memory_for_source_message(
            user_id="usr_a",
            object_type=MemoryObjectType.EVIDENCE,
            source_message_id="msg_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
        )
        default_has_msg_2 = await memories.has_memory_for_source_message(
            user_id="usr_a",
            object_type=MemoryObjectType.EVIDENCE,
            source_message_id="msg_2",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
        )
        all_has_msg_2 = await memories.has_memory_for_source_message(
            user_id="usr_a",
            object_type=MemoryObjectType.EVIDENCE,
            source_message_id="msg_2",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            statuses=None,
        )
        default_msg_2_rows = await memories.list_for_source_message(
            user_id="usr_a",
            source_message_id="msg_2",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
        )
        all_msg_2_rows = await memories.list_for_source_message(
            user_id="usr_a",
            source_message_id="msg_2",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            statuses=None,
        )

        assert RETRIEVAL_ELIGIBLE_MEMORY_STATUSES == (MemoryStatus.ACTIVE,)
        assert [row["id"] for row in default_rows] == ["mem_active"]
        assert {row["id"] for row in all_rows} == {"mem_active", "mem_pending", "mem_declined"}
        assert [row["id"] for row in default_hits] == ["mem_active"]
        assert {row["id"] for row in all_hits} == {"mem_active", "mem_pending", "mem_declined"}
        assert default_has_msg_1 is True
        assert default_has_msg_2 is False
        assert all_has_msg_2 is True
        assert default_msg_2_rows == []
        assert {row["id"] for row in all_msg_2_rows} == {"mem_pending", "mem_declined"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_state_snapshot_excludes_expired_ephemeral_rows() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")

        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="User is working from home.",
            payload={"status": "working"},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_state_user",
        )
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            scope=MemoryScope.CONVERSATION,
            canonical_text="User is at the airport.",
            payload={"status": "airport"},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            temporal_type="ephemeral",
            valid_from="2026-03-28T09:00:00+00:00",
            memory_id="mem_state_expired_ephemeral",
        )

        snapshot = await memories.get_state_snapshot(
            "usr_a",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
        )

        assert snapshot == {"status": "working"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_state_snapshot_applies_space_boundaries() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await conversations.create_conversation(
            "cnv_a",
            "usr_a",
            None,
            "coding_debug",
            "Chat A",
            platform_id="default",
        )

        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="User is in normal work mode.",
            payload={"status": "global"},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_state_global",
            user_persona_id=None,
            platform_id="default",
            sensitivity=MemorySensitivity.PUBLIC,
            scope_canonical=MemoryScope.USER.value,
        )
        clock.advance(seconds=1)
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="User is inside a private vault.",
            payload={"status": "vault"},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_state_vault",
            user_persona_id=None,
            platform_id="default",
            sensitivity=MemorySensitivity.PUBLIC,
            scope_canonical=MemoryScope.USER.value,
            space_id="space_vault",
            space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT.value,
        )
        clock.advance(seconds=1)
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="User is inside a severed room.",
            payload={"status": "severed"},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_state_severed",
            user_persona_id=None,
            platform_id="default",
            sensitivity=MemorySensitivity.PUBLIC,
            scope_canonical=MemoryScope.USER.value,
            space_id="space_severed",
            space_boundary_mode=SpaceBoundaryMode.SEVERANCE.value,
        )

        outside = await memories.get_state_snapshot(
            "usr_a",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            platform_id="default",
        )
        inside_vault = await memories.get_state_snapshot(
            "usr_a",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            platform_id="default",
            active_space_id="space_vault",
            active_space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
        )
        inside_severance = await memories.get_state_snapshot(
            "usr_a",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            platform_id="default",
            active_space_id="space_severed",
            active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
        )

        assert outside == {"status": "global"}
        assert inside_vault == {"status": "vault"}
        assert inside_severance == {"status": "severed"}
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
                id, user_id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, created_at
            )
            VALUES (?, ?, NULL, ?, NULL, NULL, 'workspace_rollup', 'summary', '[]', 1.5, ?)
            """,
            ("sum_a", "usr_a", "wrk_a", "2026-03-30T12:00:00+00:00"),
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


@pytest.mark.asyncio
async def test_upsert_summary_mirror_is_deterministic_and_updates_in_place() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)

        first = await memories.upsert_summary_mirror(
            user_id="usr_a",
            summary_view_id="sum_episode_1",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=1,
            summary_text="First episode summary",
            source_object_ids=["mem_1"],
            created_at="2026-03-30T12:00:00+00:00",
            scope=MemoryScope.GLOBAL_USER,
            privacy_level=1,
            language_codes=["EN", "jp", "zz"],
        )
        surface = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id=first["id"],
            surface_type="alias",
            surface_text="first summary surface",
        )
        second = await memories.upsert_summary_mirror(
            user_id="usr_a",
            summary_view_id="sum_episode_1",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=1,
            summary_text="Updated episode summary",
            source_object_ids=["mem_1", "mem_2"],
            created_at="2026-03-30T12:00:00+00:00",
            updated_at="2026-03-30T12:05:00+00:00",
            scope=MemoryScope.GLOBAL_USER,
            privacy_level=2,
            language_codes=["es", "en", "jp"],
        )

        rows = await memories.list_for_user("usr_a")

        assert first["id"] == second["id"] == "sum_mem_sum_episode_1"
        assert len(rows) == 1
        assert rows[0]["canonical_text"] == "Updated episode summary"
        assert rows[0]["payload_json"]["summary_view_id"] == "sum_episode_1"
        assert rows[0]["payload_json"]["source_object_ids"] == ["mem_1", "mem_2"]
        assert rows[0]["privacy_level"] == 2
        assert rows[0]["language_codes_json"] == ["en", "es"]
        surface_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id=first["id"],
        )
        assert [row["id"] for row in surface_rows] == [surface["id"]]
        assert surface_rows[0]["status"] == "stale"
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="summary",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_retrieval_surface_upsert_is_deterministic_and_user_isolated() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await _insert_assistant_mode(connection)
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_a",
            canonical_text="Ben moved to 44 Pine Lane.",
        )
        await _create_test_memory(
            memories,
            user_id="usr_b",
            memory_id="mem_b",
            canonical_text="Bob moved to 55 Cedar Street.",
        )

        first = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_a",
            surface_type="alias",
            surface_text="nuevo apartamento",
            alias_kind="translation",
            language_code="ES",
            confidence=0.6,
            derivation={"version": 1},
        )
        second = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_a",
            surface_type="alias",
            surface_text="  nuevo   apartamento  ",
            alias_kind="translation",
            language_code="es",
            confidence=0.9,
            derivation={"version": 2},
        )
        other_user = await surfaces.upsert_surface(
            user_id="usr_b",
            memory_id="mem_b",
            surface_type="alias",
            surface_text="nuevo apartamento",
            alias_kind="translation",
            language_code="es",
        )

        user_a_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        )
        user_b_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_b",
            memory_id="mem_b",
        )

        assert first["id"] == second["id"]
        assert other_user["id"] != first["id"]
        assert len(user_a_rows) == 1
        assert len(user_b_rows) == 1
        assert user_a_rows[0]["surface_text"] == "nuevo apartamento"
        assert user_a_rows[0]["confidence"] == pytest.approx(0.9)
        assert user_a_rows[0]["derivation_json"] == {"version": 2}
        assert user_a_rows[0]["language_code"] == "es"
        assert await surfaces.list_surfaces_for_memory(
            user_id="usr_b",
            memory_id="mem_a",
        ) == []

        user_a_hits = await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="apartamento",
        )
        user_b_hits = await surfaces.search_active_surfaces(
            user_id="usr_b",
            fts_query="apartamento",
        )

        assert [row["id"] for row in user_a_hits] == [first["id"]]
        assert [row["id"] for row in user_b_hits] == [other_user["id"]]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_retrieval_surface_rejects_blank_and_evidential_rows() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_a",
            canonical_text="Rosa takes amlodipine.",
        )

        with pytest.raises(ValueError, match="surface_text must be non-empty"):
            await surfaces.upsert_surface(
                user_id="usr_a",
                memory_id="mem_a",
                surface_type="alias",
                surface_text="  ",
            )

        with pytest.raises(ValueError, match="must be non_evidential"):
            await surfaces.upsert_surface(
                user_id="usr_a",
                memory_id="mem_a",
                surface_type="alias",
                surface_text="amlodipino",
                non_evidential=False,
            )

        with pytest.raises(ValueError, match="memory_id must belong to user_id"):
            await surfaces.upsert_surface(
                user_id="usr_b",
                memory_id="mem_a",
                surface_type="alias",
                surface_text="amlodipino",
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_retrieval_surface_stale_and_delete_behavior() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_a",
            canonical_text="Rosa takes amlodipine 10 mg.",
        )

        active = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_a",
            surface_type="alias",
            surface_text="amlodipino",
            alias_kind="translation",
            language_code="es",
        )
        assert [row["id"] for row in await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="amlodipino",
        )] == [active["id"]]

        assert await surfaces.mark_surfaces_stale_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        ) == 1
        stale_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        )
        assert stale_rows[0]["status"] == "stale"
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="amlodipino",
        ) == []

        restored = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_a",
            surface_type="alias",
            surface_text="amlodipino",
            alias_kind="translation",
            language_code="es",
        )
        assert restored["id"] == active["id"]
        assert restored["status"] == "active"

        assert await surfaces.mark_surfaces_deleted_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        ) == 1
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="amlodipino",
        ) == []

        assert await surfaces.delete_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        ) == 1
        assert await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_a",
        ) == []
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="amlodipino",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_status_changes_stale_or_delete_retrieval_surfaces() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, clock)

        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_archive",
            canonical_text="Archive lifecycle memory.",
        )
        await _create_test_memory(
            memories,
            user_id="usr_a",
            memory_id="mem_delete",
            canonical_text="Delete lifecycle memory.",
        )
        archive_surface = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_archive",
            surface_type="alias",
            surface_text="archivelifecycle",
        )
        delete_surface = await surfaces.upsert_surface(
            user_id="usr_a",
            memory_id="mem_delete",
            surface_type="alias",
            surface_text="deletelifecycle",
        )

        assert await memories.archive_memory_object("mem_archive", "usr_a")
        deleted = await memories.update_memory_object_status(
            memory_id="mem_delete",
            user_id="usr_a",
            status=MemoryStatus.DELETED,
        )

        assert deleted is not None
        archive_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_archive",
        )
        delete_rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_a",
            memory_id="mem_delete",
        )
        assert [(row["id"], row["status"]) for row in archive_rows] == [
            (archive_surface["id"], "stale")
        ]
        assert [(row["id"], row["status"]) for row in delete_rows] == [
            (delete_surface["id"], "deleted")
        ]
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="archivelifecycle",
        ) == []
        assert await surfaces.search_active_surfaces(
            user_id="usr_a",
            fts_query="deletelifecycle",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_user_memory_preferences_default_true_and_round_trip() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        created = await users.create_user(user_id="usr_pref")
        assert created["remember_across_chats"] == 1
        assert created["remember_across_devices"] == 1
        assert created["memory_privacy_mode"] == "balanced"

        prefs = await users.get_memory_preferences("usr_pref")
        assert prefs == {
            "remember_across_chats": True,
            "remember_across_devices": True,
            "memory_privacy_mode": "balanced",
        }

        updated = await users.update_memory_preferences(
            "usr_pref",
            remember_across_chats=False,
        )
        assert updated == {
            "remember_across_chats": False,
            "remember_across_devices": True,
            "memory_privacy_mode": "balanced",
        }

        # Partial update preserves the unspecified flag.
        updated_again = await users.update_memory_preferences(
            "usr_pref",
            remember_across_devices=False,
            memory_privacy_mode="trusted_private",
        )
        assert updated_again == {
            "remember_across_chats": False,
            "remember_across_devices": False,
            "memory_privacy_mode": "trusted_private",
        }

        # No-op update returns the current preferences without writing.
        passthrough = await users.update_memory_preferences("usr_pref")
        assert passthrough == updated_again

        # Erased / unknown user returns None.
        assert await users.get_memory_preferences("missing") is None
        assert await users.update_memory_preferences("missing", remember_across_chats=True) is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_conversation_populates_redesign_identity_columns() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        await users.create_user(user_id="usr_ident")
        await _insert_assistant_mode(connection, mode_id="companion")
        workspaces = WorkspaceRepository(connection, clock)
        await workspaces.create_workspace(
            workspace_id="ws_ident",
            user_id="usr_ident",
            name="Workspace identity",
        )

        conversations = ConversationRepository(connection, clock)
        # Caller does not pass new identity fields -> values mirror legacy
        # ones so retrieval that reads the canonical columns works for
        # legacy-only callers without a Phase 4 cut.
        legacy_only = await conversations.create_conversation(
            conversation_id="cnv_legacy",
            user_id="usr_ident",
            workspace_id="ws_ident",
            assistant_mode_id="companion",
            title="legacy",
        )
        assert legacy_only["mode"] == "companion"
        assert legacy_only["character_id"] == "ws_ident"
        assert legacy_only["incognito"] == 0
        # ``platform_id`` stays None for legacy-only callers; Phase 4 makes
        # it required at the public boundary.
        assert legacy_only["platform_id"] is None
        assert legacy_only["user_persona_id"] is None

        # Caller passes new identity fields explicitly -> they win.
        canonical = await conversations.create_conversation(
            conversation_id="cnv_canon",
            user_id="usr_ident",
            workspace_id="ws_ident",
            assistant_mode_id="companion",
            title="canonical",
            user_persona_id="alter",
            platform_id="sillytavern_desktop",
            character_id="galactic_explorer",
            mode="brainstorm",
            incognito=True,
        )
        assert canonical["user_persona_id"] == "alter"
        assert canonical["platform_id"] == "sillytavern_desktop"
        assert canonical["character_id"] == "galactic_explorer"
        assert canonical["mode"] == "brainstorm"
        assert canonical["incognito"] == 1
        assert canonical["isolated_mode"] == 1
    finally:
        await connection.close()


def test_namespace_scope_clauses_emits_chat_only_when_incognito() -> None:
    clauses, parameters = MemoryObjectRepository.namespace_scope_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id=None,
        character_id="ch1",
        conversation_id="cnv_1",
        remember_across_chats=True,
        incognito=True,
    )
    assert len(clauses) == 1
    assert "scope_canonical = 'chat'" in clauses[0]
    assert parameters == [None, "cnv_1"]


def test_namespace_scope_clauses_emits_chat_only_when_cross_chat_disabled() -> None:
    clauses, _ = MemoryObjectRepository.namespace_scope_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id="alter",
        character_id="ch1",
        conversation_id="cnv_1",
        remember_across_chats=False,
        incognito=False,
    )
    assert len(clauses) == 1
    assert "scope_canonical = 'chat'" in clauses[0]


def test_namespace_scope_clauses_drops_character_when_id_missing() -> None:
    clauses, _ = MemoryObjectRepository.namespace_scope_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id=None,
        character_id=None,
        conversation_id="cnv_1",
        remember_across_chats=True,
        incognito=False,
    )
    assert len(clauses) == 2
    assert "scope_canonical = 'character'" not in " ".join(clauses)
    assert any("scope_canonical = 'chat'" in c for c in clauses)
    assert any("scope_canonical = 'user'" in c for c in clauses)


def test_namespace_scope_clauses_emits_all_three_when_eligible() -> None:
    clauses, parameters = MemoryObjectRepository.namespace_scope_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id="alter",
        character_id="ch1",
        conversation_id="cnv_1",
        remember_across_chats=True,
        incognito=False,
    )
    assert len(clauses) == 3
    assert "scope_canonical = 'chat'" in clauses[0]
    assert "scope_canonical = 'character'" in clauses[1]
    assert "scope_canonical = 'user'" in clauses[2]
    # Each clause produces the parameters expected by the OR-combination.
    assert parameters == ["alter", "cnv_1", "alter", "ch1", "alter"]


def test_sensitivity_filter_clause_default_is_fail_closed() -> None:
    clause = MemoryObjectRepository.sensitivity_filter_clause(gates_enabled=False)
    assert "sensitivity = 'public'" in clause
    open_clause = MemoryObjectRepository.sensitivity_filter_clause(gates_enabled=True)
    assert "private" in open_clause and "secret" in open_clause
    assert "unknown" not in open_clause


def test_platform_lock_clause_handles_cross_device_off() -> None:
    on_clause, on_params = MemoryObjectRepository.platform_lock_clause(
        platform_id="p1", remember_across_devices=True
    )
    assert "platform_locked = 0" in on_clause
    assert on_params == ["p1"]
    off_clause, off_params = MemoryObjectRepository.platform_lock_clause(
        platform_id="p1", remember_across_devices=False
    )
    assert "platform_id = ?" in off_clause
    assert off_params == ["p1", "p1"]


@pytest.mark.asyncio
async def test_create_memory_object_populates_redesign_fields_with_strictest_wins() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        await users.create_user(user_id="usr_mem")
        await _insert_assistant_mode(connection, mode_id="companion")
        memories = MemoryObjectRepository(connection, clock)

        # privacy_level=1 with a high-risk medication category must escalate
        # to ``private`` (strictest-wins) even though privacy_level alone
        # would map to ``public``.
        med = await memories.create_memory_object(
            user_id="usr_mem",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="user takes sertraline 50mg",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=1,
            memory_category=MemoryCategory.MEDICATION,
        )
        assert med["sensitivity"] == "private"
        assert med["scope_canonical"] == "user"
        assert med["themes_json"] == []
        assert med["auto_expires"] == 0
        assert med["platform_locked"] == 0

        # Ephemeral session scope auto-expires.
        eph_conv = await ConversationRepository(connection, clock).create_conversation(
            conversation_id="cnv_eph",
            user_id="usr_mem",
            workspace_id=None,
            assistant_mode_id="companion",
            title="ephemeral",
        )
        eph = await memories.create_memory_object(
            user_id="usr_mem",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.EPHEMERAL_SESSION,
            canonical_text="ephemeral note",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.5,
            privacy_level=0,
            conversation_id=eph_conv["id"],
            assistant_mode_id="companion",
        )
        assert eph["scope_canonical"] == "chat"
        assert eph["auto_expires"] == 1
        assert eph["sensitivity"] == "public"

        # PIN/password category always escalates to secret.
        pin = await memories.create_memory_object(
            user_id="usr_mem",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="user shared a PIN earlier",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
        )
        assert pin["sensitivity"] == "secret"

        # Explicit caller-supplied sensitivity wins over the derived value.
        forced = await memories.create_memory_object(
            user_id="usr_mem",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="explicit secret",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            sensitivity=MemorySensitivity.SECRET,
            themes=["security", "credentials"],
            platform_locked=True,
            platform_id_lock="sillytavern_desktop",
            user_persona_id="alter",
            character_id="ch1",
            platform_id="sillytavern_desktop",
        )
        assert forced["sensitivity"] == "secret"
        assert forced["themes_json"] == ["security", "credentials"]
        assert forced["platform_locked"] == 1
        assert forced["platform_id_lock"] == "sillytavern_desktop"
        assert forced["user_persona_id"] == "alter"
        assert forced["character_id"] == "ch1"
        assert forced["platform_id"] == "sillytavern_desktop"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_set_conversation_incognito_is_reversible_and_mirrors_isolated_mode() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        await users.create_user(user_id="usr_inc")
        await _insert_assistant_mode(connection, mode_id="companion")
        conversations = ConversationRepository(connection, clock)
        created = await conversations.create_conversation(
            conversation_id="cnv_inc",
            user_id="usr_inc",
            workspace_id=None,
            assistant_mode_id="companion",
            title="incognito test",
        )
        assert created["incognito"] == 0
        assert created["isolated_mode"] == 0

        toggled_on = await conversations.set_conversation_incognito(
            "cnv_inc", "usr_inc", True
        )
        assert toggled_on["incognito"] == 1
        # Legacy isolated_mode stays in sync so legacy retrieval still
        # treats the conversation as isolated.
        assert toggled_on["isolated_mode"] == 1

        toggled_off = await conversations.set_conversation_incognito(
            "cnv_inc", "usr_inc", False
        )
        assert toggled_off["incognito"] == 0
        assert toggled_off["isolated_mode"] == 0

        # Toggling again works (truly reversible, not one-way).
        toggled_on_again = await conversations.set_conversation_incognito(
            "cnv_inc", "usr_inc", True
        )
        assert toggled_on_again["incognito"] == 1
    finally:
        await connection.close()
