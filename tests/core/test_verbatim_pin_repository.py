"""Integration tests for verbatim pin persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import UserRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryScope,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    return connection, clock


@pytest.mark.asyncio
async def test_verbatim_pin_repository_crud_and_user_isolation() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        pins = VerbatimPinRepository(connection, clock)
        await users.create_user("usr_a")
        await users.create_user("usr_b")

        created = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_1",
            canonical_text="Bank card PIN: 4512",
            index_text="bank card PIN",
            privacy_level=3,
            created_by="usr_a",
            reason="keep handy",
            expires_at="2026-04-01T00:00:00+00:00",
            payload_json={"source_message_id": "msg_1"},
        )

        assert created["status"] == VerbatimPinStatus.ACTIVE.value
        assert created["payload_json"]["source_message_id"] == "msg_1"
        assert await pins.get_verbatim_pin(created["id"], "usr_a") is not None
        assert await pins.get_verbatim_pin(created["id"], "usr_b") is None
        assert [row["id"] for row in await pins.list_verbatim_pins("usr_a")] == [created["id"]]
        assert await pins.list_verbatim_pins("usr_b") == []

        clock.advance(seconds=30)
        archived = await pins.update_verbatim_pin(
            created["id"],
            "usr_a",
            status=VerbatimPinStatus.ARCHIVED,
            reason="no longer needed",
        )
        assert archived is not None
        assert archived["status"] == VerbatimPinStatus.ARCHIVED.value
        assert archived["reason"] == "no longer needed"
        assert archived["updated_at"] == "2026-03-30T12:00:30+00:00"

        deleted = await pins.delete_verbatim_pin(created["id"], "usr_a")
        assert deleted is not None
        assert deleted["status"] == VerbatimPinStatus.DELETED.value
        assert deleted["deleted_at"] == "2026-03-30T12:00:30+00:00"
        assert await pins.list_verbatim_pins("usr_a") == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pin_repository_search_excludes_expired_and_deleted_pins() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        pins = VerbatimPinRepository(connection, clock)
        await users.create_user("usr_a")

        active = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_active",
            canonical_text="Bank card PIN: 4512",
            index_text="bank card PIN",
            privacy_level=3,
            created_by="usr_a",
            expires_at="2026-04-01T00:00:00+00:00",
        )
        expired = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_expired",
            canonical_text="Old card PIN: 9999",
            index_text="old card PIN",
            privacy_level=3,
            created_by="usr_a",
            expires_at="2026-03-29T00:00:00+00:00",
            status=VerbatimPinStatus.EXPIRED,
        )
        deleted = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_deleted",
            canonical_text="Deleted card PIN: 0000",
            index_text="deleted card PIN",
            privacy_level=3,
            created_by="usr_a",
        )
        await pins.delete_verbatim_pin(deleted["id"], "usr_a")

        active_rows = await pins.search_active_verbatim_pins(
            user_id="usr_a",
            query="bank card PIN",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.GLOBAL_USER],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            limit=10,
            as_of="2026-03-30T12:00:00+00:00",
        )
        assert [row["id"] for row in active_rows] == [active["id"]]

        assert await pins.search_active_verbatim_pins(
            user_id="usr_a",
            query="4512",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.GLOBAL_USER],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            limit=10,
            as_of="2026-03-30T12:00:00+00:00",
        ) == []
        assert expired["status"] == VerbatimPinStatus.EXPIRED.value
        expired_rows = await pins.list_verbatim_pins(
            "usr_a",
            status_filter=[VerbatimPinStatus.EXPIRED],
        )
        assert expired_rows[0]["id"] == expired["id"]

        await pins.delete_verbatim_pin(active["id"], "usr_a")
        assert await pins.search_active_verbatim_pins(
            user_id="usr_a",
            query="bank card PIN",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.GLOBAL_USER],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            limit=10,
            as_of="2026-03-30T12:00:00+00:00",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pin_search_filters_intimacy_boundary_until_authorized() -> None:
    connection, clock = await _connection_and_clock()
    try:
        users = UserRepository(connection, clock)
        pins = VerbatimPinRepository(connection, clock)
        await users.create_user("usr_a")

        ordinary = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_ordinary",
            canonical_text="ordinary pottery note",
            index_text="ordinary pottery note",
            privacy_level=0,
            created_by="usr_a",
        )
        restricted = await pins.create_verbatim_pin(
            user_id="usr_a",
            scope=MemoryScope.GLOBAL_USER,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_private",
            canonical_text="private pottery note",
            index_text="private pottery note",
            privacy_level=0,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
            created_by="usr_a",
        )

        ordinary_rows = await pins.search_active_verbatim_pins(
            user_id="usr_a",
            query="pottery",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.GLOBAL_USER],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            limit=10,
            as_of="2026-03-30T12:00:00+00:00",
        )
        authorized_rows = await pins.search_active_verbatim_pins(
            user_id="usr_a",
            query="pottery",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.GLOBAL_USER],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            limit=10,
            allow_intimacy_context=True,
            as_of="2026-03-30T12:00:00+00:00",
        )

        assert [row["id"] for row in ordinary_rows] == [ordinary["id"]]
        assert {row["id"] for row in authorized_rows} == {ordinary["id"], restricted["id"]}
        assert restricted["privacy_level"] == 2
    finally:
        await connection.close()
