"""Conversation-activity ownership migration and read-boundary tests."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

import aiosqlite
import pytest

from atagia.core.db_sqlite import MigrationManager, initialize_database


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)


async def _seed_owners_and_activity(connection: aiosqlite.Connection) -> None:
    timestamp = "2026-07-13T10:00:00+00:00"
    await connection.executemany(
        """
        INSERT INTO users(id, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        [
            ("usr_activity_owner_1", timestamp, timestamp),
            ("usr_activity_owner_2", timestamp, timestamp),
        ],
    )
    await connection.execute(
        """
        INSERT INTO conversations(id, user_id, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            "cnv_activity_owner",
            "usr_activity_owner_1",
            timestamp,
            timestamp,
        ),
    )
    await connection.execute(
        """
        INSERT INTO conversation_activity_stats(
            user_id,
            conversation_id,
            updated_at
        )
        VALUES (?, ?, ?)
        """,
        (
            "usr_activity_owner_1",
            "cnv_activity_owner",
            timestamp,
        ),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_migration_0065_purges_preexisting_owner_mismatches(
    tmp_path: Path,
) -> None:
    legacy_migrations = tmp_path / "migrations_through_0064"
    legacy_migrations.mkdir()
    for migration in MigrationManager(MIGRATIONS_DIR).discover():
        if migration.version <= 64:
            copy2(migration.path, legacy_migrations / migration.path.name)

    database_path = tmp_path / "activity_owner_upgrade.db"
    legacy = await initialize_database(str(database_path), legacy_migrations)
    try:
        await _seed_owners_and_activity(legacy)
        await legacy.execute(
            """
            UPDATE conversations
            SET user_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "usr_activity_owner_2",
                "2026-07-13T10:01:00+00:00",
                "cnv_activity_owner",
            ),
        )
        await legacy.commit()
    finally:
        await legacy.close()

    upgraded = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        cursor = await upgraded.execute(
            """
            SELECT user_id
            FROM conversation_activity_stats
            WHERE conversation_id = ?
            """,
            ("cnv_activity_owner",),
        )
        assert await cursor.fetchall() == []
        cursor = await upgraded.execute("PRAGMA foreign_key_check")
        assert await cursor.fetchall() == []
    finally:
        await upgraded.close()


@pytest.mark.asyncio
async def test_activity_owner_move_cleanup_is_atomic_and_guards_direct_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity_owner_atomic.db"
    connection = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        await _seed_owners_and_activity(connection)

        await connection.execute("BEGIN IMMEDIATE")
        await connection.execute(
            "UPDATE conversations SET user_id = ? WHERE id = ?",
            ("usr_activity_owner_2", "cnv_activity_owner"),
        )
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM conversation_activity_stats"
        )
        assert (await cursor.fetchone())[0] == 0
        await connection.rollback()

        cursor = await connection.execute(
            "SELECT user_id FROM conversations WHERE id = ?",
            ("cnv_activity_owner",),
        )
        assert (await cursor.fetchone())[0] == "usr_activity_owner_1"
        cursor = await connection.execute(
            "SELECT user_id FROM conversation_activity_stats"
        )
        assert (await cursor.fetchone())[0] == "usr_activity_owner_1"

        with pytest.raises(aiosqlite.IntegrityError, match="owner mismatch"):
            await connection.execute(
                """
                INSERT INTO conversation_activity_stats(
                    user_id,
                    conversation_id,
                    updated_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    "usr_activity_owner_2",
                    "cnv_activity_owner",
                    "2026-07-13T10:02:00+00:00",
                ),
            )
        await connection.rollback()

        with pytest.raises(aiosqlite.IntegrityError, match="owner mismatch"):
            await connection.execute(
                """
                UPDATE conversation_activity_stats
                SET user_id = ?
                WHERE conversation_id = ?
                """,
                ("usr_activity_owner_2", "cnv_activity_owner"),
            )
        await connection.rollback()

        await connection.execute(
            "UPDATE conversations SET user_id = ? WHERE id = ?",
            ("usr_activity_owner_2", "cnv_activity_owner"),
        )
        await connection.commit()
    finally:
        await connection.close()

    reopened = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        cursor = await reopened.execute(
            "SELECT user_id FROM conversations WHERE id = ?",
            ("cnv_activity_owner",),
        )
        assert (await cursor.fetchone())[0] == "usr_activity_owner_2"
        cursor = await reopened.execute(
            "SELECT COUNT(*) FROM conversation_activity_stats"
        )
        assert (await cursor.fetchone())[0] == 0
        cursor = await reopened.execute("PRAGMA foreign_key_check")
        assert await cursor.fetchall() == []
    finally:
        await reopened.close()
