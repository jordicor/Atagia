"""Integration tests for the consent profile repository."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.consent_repository import MemoryConsentProfileRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import UserRepository
from atagia.models.schemas_memory import MemoryCategory

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@pytest.mark.asyncio
async def test_upsert_and_get_consent_profile_are_user_scoped() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 9, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        repository = MemoryConsentProfileRepository(connection, clock)

        await users.create_user("usr_1")
        await users.create_user("usr_2")

        first = await repository.upsert_profile(
            user_id="usr_1",
            category=MemoryCategory.PIN_OR_PASSWORD,
            confirmed_count=1,
            declined_count=0,
            last_confirmed_at="2026-04-05T08:59:00+00:00",
        )
        clock.advance(seconds=60)
        second = await repository.upsert_profile(
            user_id="usr_1",
            category=MemoryCategory.PIN_OR_PASSWORD,
            confirmed_count=2,
            declined_count=1,
            last_confirmed_at="2026-04-05T09:00:30+00:00",
            last_declined_at="2026-04-05T09:00:45+00:00",
        )
        other = await repository.upsert_profile(
            user_id="usr_2",
            category=MemoryCategory.PIN_OR_PASSWORD,
            confirmed_count=0,
            declined_count=2,
            last_declined_at="2026-04-05T09:01:00+00:00",
        )

        loaded = await repository.get_profile("usr_1", MemoryCategory.PIN_OR_PASSWORD)

        assert first["confirmed_count"] == 1
        assert second["confirmed_count"] == 2
        assert second["declined_count"] == 1
        assert second["updated_at"] == "2026-04-05T09:01:00+00:00"
        assert loaded is not None
        assert loaded["user_id"] == "usr_1"
        assert loaded["confirmed_count"] == 2
        assert loaded["declined_count"] == 1
        assert other["user_id"] == "usr_2"
        assert await repository.get_profile("usr_1", MemoryCategory.FINANCIAL) is None
    finally:
        await connection.close()
