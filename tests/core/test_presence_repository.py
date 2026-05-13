"""Tests for durable Presence attribution storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.presence_repository import PresenceRepository, presence_snapshot
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import PresenceKind
from atagia.services.presence_resolution import ensure_conversation_active_presence

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


@pytest.mark.asyncio
async def test_presence_ids_are_scoped_by_owner_user() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        await users.create_user("usr_1")
        await users.create_user("usr_2")

        presences = PresenceRepository(connection, clock)
        first = await presences.resolve_active_presence(
            owner_user_id="usr_1",
            active_presence_id=None,
            character_id="shared_character",
        )
        second = await presences.resolve_active_presence(
            owner_user_id="usr_2",
            active_presence_id=None,
            character_id="shared_character",
        )

        assert first["id"] == "shared_character"
        assert second["id"] == "shared_character"
        assert first["owner_user_id"] == "usr_1"
        assert second["owner_user_id"] == "usr_2"
        assert first["kind"] == PresenceKind.OWNED_FACET.value
        assert second["kind"] == PresenceKind.OWNED_FACET.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ensure_conversation_active_presence_backfills_legacy_conversation() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
        conversations = ConversationRepository(connection, clock)
        conversation = await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Chat",
            character_id="assistant_alpha",
        )
        await connection.execute(
            """
            UPDATE conversations
            SET active_presence_id = NULL
            WHERE id = ?
            """,
            ("cnv_1",),
        )
        await connection.commit()
        conversation = await conversations.get_conversation("cnv_1", "usr_1")
        assert conversation is not None
        assert conversation["active_presence_id"] is None

        updated, active_presence = await ensure_conversation_active_presence(
            connection,
            clock,
            conversation=conversation,
        )

        assert updated["active_presence_id"] == "assistant_alpha"
        assert active_presence.presence_id == "assistant_alpha"
        assert active_presence.kind is PresenceKind.OWNED_FACET
        row = await PresenceRepository(connection, clock).get_presence(
            owner_user_id="usr_1",
            presence_id="assistant_alpha",
        )
        assert row is not None
        assert presence_snapshot(row).presence_id == "assistant_alpha"
    finally:
        await connection.close()
