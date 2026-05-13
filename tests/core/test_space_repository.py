"""Tests for Space boundary persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository, WorkspaceRepository
from atagia.core.space_repository import SpaceRepository, space_snapshot
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import SpaceBoundaryMode
from atagia.services.space_resolution import ensure_conversation_active_space

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


@pytest.mark.asyncio
async def test_space_repository_resolves_workspace_space() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        spaces = SpaceRepository(connection, clock)

        row = await spaces.resolve_active_space(
            owner_user_id="usr_1",
            space_id=None,
            workspace_id="wrk_alpha",
            boundary_mode=SpaceBoundaryMode.SEVERANCE,
            display_name="Alpha",
        )

        assert row is not None
        snapshot = space_snapshot(row)
        assert snapshot.space_id == "wrk_alpha"
        assert snapshot.boundary_mode is SpaceBoundaryMode.SEVERANCE
        assert snapshot.display_name == "Alpha"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ensure_conversation_active_space_backfills_conversation() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
        await WorkspaceRepository(connection, clock).create_workspace(
            "wrk_alpha",
            "usr_1",
            "Alpha",
        )
        conversations = ConversationRepository(connection, clock)
        conversation = await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            "wrk_alpha",
            "coding_debug",
            "Chat",
        )

        updated, snapshot = await ensure_conversation_active_space(
            connection,
            clock,
            conversation=conversation,
            workspace_id="wrk_alpha",
            boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
            display_name="Vault",
        )

        assert snapshot is not None
        assert snapshot.space_id == "wrk_alpha"
        assert snapshot.boundary_mode is SpaceBoundaryMode.PRIVACY_VAULT
        assert updated["active_space_id"] == "wrk_alpha"
        assert updated["active_space_boundary_mode"] == "privacy_vault"

        stored = await conversations.get_conversation("cnv_1", "usr_1")
        assert stored is not None
        assert stored["active_space_id"] == "wrk_alpha"
    finally:
        await connection.close()
