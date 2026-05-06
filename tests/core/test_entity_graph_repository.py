"""Tests for SQLite entity graph persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.entity_graph_repository import EntityGraphRepository
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.models.schemas_memory import ConversationStatus, IntimacyBoundary, MemoryScope, MemorySensitivity

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc))
    return connection, clock


async def _insert_assistant_mode(connection: aiosqlite.Connection, mode_id: str = "coding_debug") -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (mode_id, "Coding Debug", "hash_1", "{}", "2026-05-02T12:00:00+00:00", "2026-05-02T12:00:00+00:00"),
    )
    await connection.commit()


async def _seed_two_users(connection: aiosqlite.Connection, clock: FrozenClock) -> None:
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    await users.create_user("usr_a")
    await users.create_user("usr_b")
    await _insert_assistant_mode(connection)
    await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Chat A")
    await conversations.create_conversation("cnv_a2", "usr_a", None, "coding_debug", "Chat A2")
    await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Chat B")
    await messages.create_message(
        "msg_a",
        "cnv_a",
        "user",
        1,
        "Maria is my sister and Alba is her daughter.",
        12,
        {},
    )
    await messages.create_message(
        "msg_b",
        "cnv_b",
        "user",
        1,
        "Maria works with another user.",
        8,
        {},
    )


async def _count(connection: aiosqlite.Connection, table_name: str) -> int:
    cursor = await connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}")
    row = await cursor.fetchone()
    return int(row["count"])


@pytest.mark.asyncio
async def test_graph_repository_round_trips_relationships_and_enforces_user_scope() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)

        maria = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Maria",
            confidence=0.92,
        )
        alba = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Alba",
            confidence=0.91,
        )
        other_user_entity = await repository.create_entity(
            user_id="usr_b",
            conversation_id="cnv_b",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Maria",
            confidence=0.9,
        )

        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(maria["id"]),
            surface_text="my sister",
            confidence=0.88,
        )
        relationship = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(maria["id"]),
            target_entity_id=str(alba["id"]),
            predicate="person.parent_of",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            confidence=0.9,
        )
        await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(relationship["id"]),
            source_kind="message",
            source_id="msg_a",
            conversation_id="cnv_a",
            message_id="msg_a",
            evidence_quote="Alba is her daughter",
        )

        assert await repository.get_entity(str(maria["id"]), "usr_b") is None
        assert await repository.find_aliases(user_id="usr_a", surface_text="my sister")
        assert await repository.find_aliases(user_id="usr_b", surface_text="my sister") == []
        assert await repository.list_relationships_for_entity(user_id="usr_a", entity_id=str(maria["id"]))
        assert await repository.list_relationships_for_entity(user_id="usr_b", entity_id=str(maria["id"])) == []

        with pytest.raises(ValueError, match="Target entity"):
            await repository.upsert_relationship(
                user_id="usr_a",
                source_entity_id=str(maria["id"]),
                target_entity_id=str(other_user_entity["id"]),
                predicate="person.knows",
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_workspace_delete_does_not_promote_graph_entities_to_broader_scope() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        await users.create_user("usr_a")
        await _insert_assistant_mode(connection)
        await workspaces.create_workspace("wrk_a", "usr_a", "Scoped workspace")
        await repository.create_entity(
            user_id="usr_a",
            workspace_id="wrk_a",
            assistant_mode_id="coding_debug",
            entity_type="project",
            display_name="Workspace-only entity",
        )

        await workspaces.delete_workspace("wrk_a", "usr_a")
        cards = await repository.list_entity_cards(
            user_id="usr_a",
            allowed_scopes=[MemoryScope.ASSISTANT_MODE, MemoryScope.GLOBAL_USER],
            workspace_id=None,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            cross_chat_allowed=True,
            privacy_ceiling=3,
            allow_intimacy_context=True,
            limit=20,
        )

        assert cards == []
        assert await _count(connection, "graph_entities") == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_mentions_and_relationship_sources_are_idempotent() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        maria = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Maria",
        )
        alba = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Alba",
        )

        first_mention = await repository.upsert_mention(
            user_id="usr_a",
            entity_id=str(maria["id"]),
            source_kind="message",
            source_id="msg_a",
            surface_text="Maria",
            evidence_quote="Maria is my sister",
            conversation_id="cnv_a",
            message_id="msg_a",
        )
        second_mention = await repository.upsert_mention(
            user_id="usr_a",
            entity_id=str(maria["id"]),
            source_kind="message",
            source_id="msg_a",
            surface_text="Maria",
            evidence_quote="Maria is my sister",
            conversation_id="cnv_a",
            message_id="msg_a",
        )
        first_relationship = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(maria["id"]),
            target_entity_id=str(alba["id"]),
            predicate="person.parent_of",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
        )
        second_relationship = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(maria["id"]),
            target_entity_id=str(alba["id"]),
            predicate="person.parent_of",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
        )
        first_source = await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(first_relationship["id"]),
            source_kind="message",
            source_id="msg_a",
            evidence_quote="Alba is her daughter",
            conversation_id="cnv_a",
            message_id="msg_a",
        )
        second_source = await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(first_relationship["id"]),
            source_kind="message",
            source_id="msg_a",
            evidence_quote="Alba is her daughter",
            conversation_id="cnv_a",
            message_id="msg_a",
        )

        assert first_mention["id"] == second_mention["id"]
        assert first_relationship["id"] == second_relationship["id"]
        assert first_source["id"] == second_source["id"]
        assert await _count(connection, "graph_entity_mentions") == 1
        assert await _count(connection, "graph_relationships") == 1
        assert await _count(connection, "graph_relationship_sources") == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_entity_cards_keep_conversation_scope_to_active_conversation_when_cross_chat_allowed() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Current Conversation",
            confidence=0.9,
        )
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a2",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Other Conversation",
            confidence=0.9,
        )

        cards = await repository.list_entity_cards(
            user_id="usr_a",
            allowed_scopes=[MemoryScope.CONVERSATION],
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            cross_chat_allowed=True,
        )

        assert [card["display_name"] for card in cards] == ["Current Conversation"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_relationship_upsert_preserves_strongest_intimacy_boundary() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        jordi = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Jordi",
        )
        maria = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Maria",
        )

        first = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(jordi["id"]),
            target_entity_id=str(maria["id"]),
            predicate="person.knows",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            confidence=0.55,
            privacy_level=0,
            intimacy_boundary=IntimacyBoundary.ORDINARY,
            intimacy_boundary_confidence=0.0,
        )
        upgraded = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(jordi["id"]),
            target_entity_id=str(maria["id"]),
            predicate="person.knows",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            confidence=0.8,
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.72,
        )
        later_ordinary = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(jordi["id"]),
            target_entity_id=str(maria["id"]),
            predicate="person.knows",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            confidence=0.9,
            privacy_level=0,
            intimacy_boundary=IntimacyBoundary.ORDINARY,
            intimacy_boundary_confidence=0.1,
        )

        assert first["id"] == upgraded["id"] == later_ordinary["id"]
        assert upgraded["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert upgraded["privacy_level"] == 2
        assert upgraded["intimacy_boundary_confidence"] == 0.72
        assert later_ordinary["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert later_ordinary["privacy_level"] == 2
        assert later_ordinary["intimacy_boundary_confidence"] == 0.72
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_relationship_dedupe_keeps_conversation_scopes_separate() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        maria = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Maria",
        )
        alba = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Alba",
        )

        first = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(maria["id"]),
            target_entity_id=str(alba["id"]),
            predicate="person.parent_of",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
        )
        second = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(maria["id"]),
            target_entity_id=str(alba["id"]),
            predicate="person.parent_of",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_a2",
            assistant_mode_id="coding_debug",
        )

        assert first["id"] != second["id"]
        assert await _count(connection, "graph_relationships") == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_entity_cards_apply_policy_and_conversation_visibility_filters() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        await conversations.create_conversation(
            "cnv_temp",
            "usr_a",
            None,
            "coding_debug",
            "Temporary",
            temporary=True,
        )
        await conversations.create_conversation("cnv_archived", "usr_a", None, "coding_debug", "Archived")
        await conversations.update_conversation_status(
            "cnv_archived",
            "usr_a",
            ConversationStatus.ARCHIVED.value,
        )
        visible = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Visible",
            privacy_level=1,
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(visible["id"]),
            surface_text="Visible alias",
            status="active",
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(visible["id"]),
            surface_text="Review alias",
            status="review_required",
        )
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Too private",
            privacy_level=2,
        )
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Intimate",
            privacy_level=1,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
        )
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_temp",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Other temporary",
        )
        await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_archived",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Archived",
        )

        cards = await repository.list_entity_cards(
            user_id="usr_a",
            allowed_scopes=[MemoryScope.CONVERSATION],
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            cross_chat_allowed=True,
            privacy_ceiling=1,
            allow_intimacy_context=False,
            limit=20,
        )

        assert [card["display_name"] for card in cards] == ["Visible"]
        assert cards[0]["aliases"] == ["Visible alias"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_entity_cards_phase7_filters_alias_side_table_metadata() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        entity = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Visible",
            privacy_level=0,
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(entity["id"]),
            surface_text="Visible public alias",
            status="active",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(entity["id"]),
            surface_text="Legacy unknown alias",
            status="active",
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(entity["id"]),
            surface_text="Private alias",
            status="active",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PRIVATE,
        )
        await repository.upsert_alias(
            user_id="usr_a",
            entity_id=str(entity["id"]),
            surface_text="Wrong persona alias",
            status="active",
            user_persona_id="persona_b",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )

        cards = await repository.list_entity_cards(
            user_id="usr_a",
            allowed_scopes=[MemoryScope.CONVERSATION],
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            cross_chat_allowed=True,
            privacy_ceiling=1,
            allow_intimacy_context=False,
            limit=20,
        )

        assert [card["display_name"] for card in cards] == ["Visible"]
        assert cards[0]["aliases"] == ["Visible public alias"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_relationship_sources_apply_namespace_and_sensitivity_gates() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_two_users(connection, clock)
        repository = EntityGraphRepository(connection, clock)
        source = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Source",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        target = await repository.create_entity(
            user_id="usr_a",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            entity_type="person",
            display_name="Target",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        relationship = await repository.upsert_relationship(
            user_id="usr_a",
            source_entity_id=str(source["id"]),
            target_entity_id=str(target["id"]),
            predicate="person.knows",
            conversation_id="cnv_a",
            assistant_mode_id="coding_debug",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
            scope_canonical=MemoryScope.CHAT.value,
        )
        await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(relationship["id"]),
            source_kind="message",
            source_id="msg_a",
            evidence_quote="visible quote",
            conversation_id="cnv_a",
            message_id="msg_a",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(relationship["id"]),
            source_kind="message",
            source_id="msg_b",
            evidence_quote="wrong persona quote",
            conversation_id="cnv_a",
            user_persona_id="persona_b",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PUBLIC,
        )
        await repository.link_relationship_source(
            user_id="usr_a",
            relationship_id=str(relationship["id"]),
            source_kind="message",
            source_id="msg_c",
            evidence_quote="private quote",
            conversation_id="cnv_a",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
            sensitivity=MemorySensitivity.PRIVATE,
        )

        sources = await repository.list_relationship_sources(
            user_id="usr_a",
            relationship_id=str(relationship["id"]),
            conversation_id="cnv_a",
            user_persona_id="persona_a",
            platform_id="default",
            character_id="char_a",
        )

        assert [source["evidence_quote"] for source in sources] == ["visible quote"]
    finally:
        await connection.close()
