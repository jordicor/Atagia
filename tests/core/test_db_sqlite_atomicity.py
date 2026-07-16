"""Concurrency and failure-injection tests for manager-owned migrations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any

import aiosqlite
import pytest

from atagia.core.config import default_resource_path
from atagia.core.db_sqlite import (
    Migration,
    MigrationManager,
    close_connection,
    initialize_database,
    open_connection,
)

MIGRATIONS_DIR = Path(default_resource_path("migrations"))


class _SnapshotBarrierMigrationManager(MigrationManager):
    """Force two managers to use the same pre-lock applied-version snapshot."""

    def __init__(
        self,
        migrations_path: Path,
        barrier: asyncio.Barrier,
    ) -> None:
        super().__init__(migrations_path)
        self._barrier = barrier

    async def applied_versions(self, connection: aiosqlite.Connection) -> set[int]:
        versions = await super().applied_versions(connection)
        await self._barrier.wait()
        return versions


def _migration(version: int) -> Migration:
    return next(
        migration
        for migration in MigrationManager(MIGRATIONS_DIR).discover()
        if migration.version == version
    )


def _copy_migrations(destination: Path, *, through_version: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for migration in MigrationManager(MIGRATIONS_DIR).discover():
        if migration.version <= through_version:
            shutil.copy2(migration.path, destination / migration.path.name)


def _install_modified_migration(
    destination: Path,
    *,
    version: int,
    anchor: str,
    injected_sql: str,
) -> None:
    migration = _migration(version)
    sql = migration.sql
    assert sql.count(anchor) == 1, (migration.path.name, anchor)
    modified = sql.replace(anchor, f"{anchor}\n{injected_sql}", 1)
    (destination / migration.path.name).write_text(modified, encoding="utf-8")


def _normalized_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


def _logical_fingerprint(database_path: Path) -> str:
    """Return deterministic schema and row state, ignoring apply timestamps."""
    connection = sqlite3.connect(database_path)
    try:
        schema_rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        table_names = [str(row[1]) for row in schema_rows if row[0] == "table"]
        table_rows: dict[str, list[list[Any]]] = {}
        for table_name in table_names:
            quoted_name = table_name.replace('"', '""')
            if table_name == "schema_migrations":
                rows = connection.execute(
                    "SELECT version, name FROM schema_migrations"
                ).fetchall()
            else:
                rows = connection.execute(
                    f'SELECT * FROM "{quoted_name}"'  # noqa: S608 - schema names
                ).fetchall()
            normalized = [[_normalized_value(value) for value in row] for row in rows]
            normalized.sort(
                key=lambda row: json.dumps(row, sort_keys=True, default=str)
            )
            table_rows[table_name] = normalized
        return json.dumps(
            {"schema": schema_rows, "rows": table_rows},
            sort_keys=True,
            default=str,
        )
    finally:
        connection.close()


async def _prepare_pre_migration_snapshot(
    tmp_path: Path,
    *,
    version: int,
) -> tuple[Path, Path]:
    migrations_dir = tmp_path / "snapshot_migrations"
    _copy_migrations(migrations_dir, through_version=version - 1)
    database_path = tmp_path / "snapshot.db"
    connection = await initialize_database(str(database_path), migrations_dir)
    timestamp = "2026-07-13T00:00:00+00:00"
    await connection.execute(
        """
        INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
        VALUES ('usr_atomicity', NULL, ?, ?, NULL)
        """,
        (timestamp, timestamp),
    )
    await connection.execute(
        """
        INSERT INTO assistant_modes(
            id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
        )
        VALUES ('mode_atomicity', 'Atomicity', NULL, '{}', ?, ?)
        """,
        (timestamp, timestamp),
    )
    await connection.execute(
        """
        INSERT INTO conversations(
            id, user_id, assistant_mode_id, status, created_at, updated_at
        )
        VALUES (
            'cnv_atomicity', 'usr_atomicity', 'mode_atomicity',
            'active', ?, ?
        )
        """,
        (timestamp, timestamp),
    )
    memory_scope = "chat" if version > 34 else "conversation"
    await connection.execute(
        """
        INSERT INTO memory_objects(
            id, user_id, conversation_id, assistant_mode_id,
            object_type, scope, canonical_text, source_kind, created_at, updated_at
        )
        VALUES (
            'mem_atomicity', 'usr_atomicity', 'cnv_atomicity', 'mode_atomicity',
            'evidence', ?, 'durable atomicity probe', 'extracted', ?, ?
        )
        """,
        (memory_scope, timestamp, timestamp),
    )
    if version == 22:
        await connection.execute(
            """
            INSERT INTO artifacts(
                id, user_id, conversation_id, artifact_type, source_kind,
                status, created_at, updated_at
            )
            VALUES (
                'art_atomicity', 'usr_atomicity', 'cnv_atomicity', 'file',
                'external_ref', 'ready', ?, ?
            )
            """,
            (timestamp, timestamp),
        )
        await connection.execute(
            """
            INSERT INTO artifact_blobs(
                artifact_id, storage_kind, storage_uri, byte_size,
                sha256, created_at, updated_at
            )
            VALUES (
                'art_atomicity', 'external_ref', 'probe://atomicity', 1,
                'probe-sha', ?, ?
            )
            """,
            (timestamp, timestamp),
        )
    if version == 29:
        await connection.execute(
            """
            INSERT INTO graph_entities(
                id, user_id, conversation_id, assistant_mode_id,
                entity_type, display_name, created_at, updated_at
            )
            VALUES (
                'ent_atomicity', 'usr_atomicity', 'cnv_atomicity',
                'mode_atomicity', 'person', 'Atomicity Entity', ?, ?
            )
            """,
            (timestamp, timestamp),
        )
        await connection.execute(
            """
            INSERT INTO graph_relationships(
                id, user_id, source_entity_id, target_value_json,
                predicate, scope, conversation_id, assistant_mode_id,
                dedupe_key, created_at, updated_at
            )
            VALUES (
                'rel_atomicity', 'usr_atomicity', 'ent_atomicity', '{}',
                'mentions', 'conversation', 'cnv_atomicity', 'mode_atomicity',
                'atomicity-dedupe', ?, ?
            )
            """,
            (timestamp, timestamp),
        )
    await connection.commit()
    await close_connection(connection)
    return database_path, migrations_dir


async def _assert_foreign_keys_enabled(
    connection: aiosqlite.Connection,
) -> None:
    cursor = await connection.execute("PRAGMA foreign_keys;")
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1


async def _assert_concurrent_migration_is_idempotent(
    tmp_path: Path,
    *,
    migration_sql: str,
    probe_table: str,
) -> None:
    migrations_dir = tmp_path / "concurrent_migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_concurrent_probe.sql").write_text(
        migration_sql,
        encoding="utf-8",
    )
    database_path = tmp_path / "concurrent.db"

    setup = await open_connection(str(database_path))
    try:
        await MigrationManager(migrations_dir).ensure_schema_table(setup)
    finally:
        await close_connection(setup)

    first = await open_connection(str(database_path))
    second = await open_connection(str(database_path))
    barrier = asyncio.Barrier(2)
    first_manager = _SnapshotBarrierMigrationManager(migrations_dir, barrier)
    second_manager = _SnapshotBarrierMigrationManager(migrations_dir, barrier)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                first_manager.apply_all(first),
                second_manager.apply_all(second),
            ),
            timeout=5.0,
        )

        assert sorted(len(result) for result in results) == [0, 1]
        assert [migration.version for result in results for migration in result] == [1]
        cursor = await first.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 1
        cursor = await first.execute(f'SELECT COUNT(*) FROM "{probe_table}"')
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 1
        await _assert_foreign_keys_enabled(first)
        await _assert_foreign_keys_enabled(second)
    finally:
        await close_connection(second)
        await close_connection(first)


@pytest.mark.asyncio
async def test_concurrent_normal_migration_rechecks_after_writer_lock(
    tmp_path: Path,
) -> None:
    await _assert_concurrent_migration_is_idempotent(
        tmp_path,
        migration_sql="""
CREATE TABLE concurrent_normal_probe (
    id INTEGER PRIMARY KEY
);
INSERT INTO concurrent_normal_probe(id) VALUES (1);
""",
        probe_table="concurrent_normal_probe",
    )


@pytest.mark.asyncio
async def test_concurrent_fk_off_migration_rechecks_after_writer_lock(
    tmp_path: Path,
) -> None:
    await _assert_concurrent_migration_is_idempotent(
        tmp_path,
        migration_sql="""-- atagia:foreign_keys_off
CREATE TABLE concurrent_fk_off_parent (
    id INTEGER PRIMARY KEY
);
CREATE TABLE concurrent_fk_off_probe (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL REFERENCES concurrent_fk_off_parent(id)
);
INSERT INTO concurrent_fk_off_parent(id) VALUES (1);
INSERT INTO concurrent_fk_off_probe(id, parent_id) VALUES (1, 1);
""",
        probe_table="concurrent_fk_off_probe",
    )


@pytest.mark.asyncio
async def test_normal_migration_failure_rolls_back_script_and_version(
    tmp_path: Path,
) -> None:
    migrations_dir = tmp_path / "normal_failure_migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_normal_failure.sql").write_text(
        """
CREATE TABLE normal_failure_probe (id INTEGER PRIMARY KEY);
INSERT INTO normal_failure_probe(id) VALUES (1);
SELECT * FROM atagia_forced_missing_table;
""",
        encoding="utf-8",
    )
    connection = await open_connection(str(tmp_path / "normal_failure.db"))
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            await MigrationManager(migrations_dir).apply_all(connection)

        cursor = await connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'normal_failure_probe'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 0
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 0
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
async def test_normal_migration_rejects_transaction_control_before_execution(
    tmp_path: Path,
) -> None:
    migrations_dir = tmp_path / "normal_transaction_control_migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_normal_transaction_control.sql").write_text(
        """
CREATE TABLE escaped_transaction_probe (id INTEGER PRIMARY KEY);
COMMIT;
SELECT * FROM atagia_forced_missing_table;
""",
        encoding="utf-8",
    )
    connection = await open_connection(str(tmp_path / "normal_transaction_control.db"))
    try:
        with pytest.raises(
            ValueError,
            match="MigrationManager owns the transaction",
        ):
            await MigrationManager(migrations_dir).apply_all(connection)

        cursor = await connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'escaped_transaction_probe'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 0
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 0
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)


async def _assert_clean_retry(
    database_path: Path,
    migrations_dir: Path,
    *,
    version: int,
) -> None:
    migration = _migration(version)
    shutil.copy2(migration.path, migrations_dir / migration.path.name)
    connection = await open_connection(str(database_path))
    try:
        applied = await MigrationManager(migrations_dir).apply_all(connection)
        assert [item.version for item in applied] == [version]
        cursor = await connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = ?",
            (version,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert int(row[0]) == 1
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)


_DESTRUCTIVE_FAILPOINTS = (
    (13, "DROP TABLE IF EXISTS memory_objects_fts;", "0013-drop-fts"),
    (13, "DROP TABLE memory_objects;", "0013-rebuild-memory"),
    (
        13,
        "INSERT INTO memory_objects_fts(memory_objects_fts) VALUES('rebuild');",
        "0013-rebuild-fts",
    ),
    (22, "DROP TABLE artifact_blobs;", "0022-rebuild-artifact-blobs"),
    (29, "DROP TABLE graph_relationships;", "0029-rebuild-relationships"),
    (29, "DROP TABLE graph_entities;", "0029-rebuild-entities"),
    (34, "DROP TABLE IF EXISTS memory_objects_fts;", "0034-drop-memory-fts"),
    (34, "DROP TABLE memory_objects_old;", "0034-rebuild-memory"),
    (34, "DROP TABLE IF EXISTS verbatim_pins_fts;", "0034-drop-pin-fts"),
    (34, "DROP TABLE verbatim_pins_old;", "0034-rebuild-pins"),
    (
        34,
        "DROP TABLE contract_dimensions_current_old;",
        "0034-rebuild-contract",
    ),
    (
        34,
        "DROP TABLE memory_consent_profile_old;",
        "0034-rebuild-consent",
    ),
    (
        34,
        "DROP TABLE graph_relationships_old;",
        "0034-rebuild-relationships",
    ),
    (34, "DROP TABLE memory_links_old;", "0034-rebuild-links"),
    (34, "DROP TABLE conversations_old;", "0034-rebuild-conversations"),
    (
        34,
        "DROP TABLE conversation_activity_stats_old;",
        "0034-rebuild-activity",
    ),
    (34, "DROP TABLE retrieval_events_old;", "0034-rebuild-retrieval-events"),
    (34, "DROP TABLE phase11_dropped_memory_ids;", "0034-cleanup"),
    (45, "DROP TABLE conversations_old;", "0045-rebuild-conversations"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "anchor", "_case_name"),
    _DESTRUCTIVE_FAILPOINTS,
    ids=[case_name for _, _, case_name in _DESTRUCTIVE_FAILPOINTS],
)
async def test_fk_off_migration_failure_is_fully_atomic_and_retryable(
    tmp_path: Path,
    version: int,
    anchor: str,
    _case_name: str,
) -> None:
    database_path, migrations_dir = await _prepare_pre_migration_snapshot(
        tmp_path,
        version=version,
    )
    before = _logical_fingerprint(database_path)
    _install_modified_migration(
        migrations_dir,
        version=version,
        anchor=anchor,
        injected_sql="SELECT atagia_test_fail();",
    )

    connection = await open_connection(str(database_path))

    def fail() -> None:
        raise RuntimeError("deterministic migration failpoint")

    await connection.create_function("atagia_test_fail", 0, fail)
    try:
        with pytest.raises(sqlite3.OperationalError):
            await MigrationManager(migrations_dir).apply_all(connection)
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)

    assert _logical_fingerprint(database_path) == before
    await _assert_clean_retry(
        database_path,
        migrations_dir,
        version=version,
    )


@pytest.mark.asyncio
async def test_fk_check_failure_rolls_back_script_and_version(tmp_path: Path) -> None:
    version = 22
    database_path, migrations_dir = await _prepare_pre_migration_snapshot(
        tmp_path,
        version=version,
    )
    before = _logical_fingerprint(database_path)
    migration = _migration(version)
    invalid_fk_sql = """
CREATE TABLE atagia_fk_violation_probe (
    user_id TEXT NOT NULL REFERENCES users(id)
);
INSERT INTO atagia_fk_violation_probe(user_id) VALUES ('missing-user');
"""
    (migrations_dir / migration.path.name).write_text(
        f"{migration.sql.rstrip()}\n{invalid_fk_sql}",
        encoding="utf-8",
    )

    connection = await open_connection(str(database_path))
    try:
        with pytest.raises(RuntimeError, match="Foreign key violations"):
            await MigrationManager(migrations_dir).apply_all(connection)
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)

    assert _logical_fingerprint(database_path) == before
    await _assert_clean_retry(database_path, migrations_dir, version=version)


@pytest.mark.asyncio
async def test_metadata_insert_failure_rolls_back_script_and_version(
    tmp_path: Path,
) -> None:
    version = 22
    database_path, migrations_dir = await _prepare_pre_migration_snapshot(
        tmp_path,
        version=version,
    )
    migration = _migration(version)
    shutil.copy2(migration.path, migrations_dir / migration.path.name)

    connection = await open_connection(str(database_path))
    await connection.execute(
        """
        CREATE TRIGGER fail_migration_metadata
        BEFORE INSERT ON schema_migrations
        WHEN NEW.version = 22
        BEGIN
            SELECT RAISE(ABORT, 'deterministic metadata failure');
        END;
        """
    )
    await connection.commit()
    await close_connection(connection)
    before = _logical_fingerprint(database_path)

    connection = await open_connection(str(database_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="metadata failure"):
            await MigrationManager(migrations_dir).apply_all(connection)
        await _assert_foreign_keys_enabled(connection)
    finally:
        await close_connection(connection)

    assert _logical_fingerprint(database_path) == before
    connection = await open_connection(str(database_path))
    await connection.execute("DROP TRIGGER fail_migration_metadata;")
    await connection.commit()
    await close_connection(connection)
    await _assert_clean_retry(database_path, migrations_dir, version=version)


@pytest.mark.asyncio
async def test_cancellation_rolls_back_and_survives_close_reopen(
    tmp_path: Path,
) -> None:
    version = 22
    database_path, migrations_dir = await _prepare_pre_migration_snapshot(
        tmp_path,
        version=version,
    )
    before = _logical_fingerprint(database_path)
    _install_modified_migration(
        migrations_dir,
        version=version,
        anchor="DROP TABLE artifact_blobs;",
        injected_sql="SELECT atagia_test_block();",
    )
    entered = threading.Event()
    release = threading.Event()

    def block() -> int:
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("migration cancellation failpoint timed out")
        return 0

    connection = await open_connection(str(database_path))
    await connection.create_function("atagia_test_block", 0, block)
    task = asyncio.create_task(MigrationManager(migrations_dir).apply_all(connection))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_foreign_keys_enabled(connection)
    await close_connection(connection)

    assert _logical_fingerprint(database_path) == before
    await _assert_clean_retry(database_path, migrations_dir, version=version)


@pytest.mark.asyncio
async def test_fresh_database_failure_matches_pre_target_snapshot(
    tmp_path: Path,
) -> None:
    version = 13
    expected_migrations = tmp_path / "expected_migrations"
    _copy_migrations(expected_migrations, through_version=version - 1)
    expected_database = tmp_path / "expected.db"
    connection = await initialize_database(
        str(expected_database),
        expected_migrations,
    )
    await close_connection(connection)

    failing_migrations = tmp_path / "failing_migrations"
    _copy_migrations(failing_migrations, through_version=version - 1)
    _install_modified_migration(
        failing_migrations,
        version=version,
        anchor="DROP TABLE memory_objects;",
        injected_sql="SELECT * FROM atagia_forced_missing_table;",
    )
    failed_database = tmp_path / "failed.db"
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        await initialize_database(str(failed_database), failing_migrations)

    assert _logical_fingerprint(failed_database) == _logical_fingerprint(
        expected_database
    )
    await _assert_clean_retry(
        failed_database,
        failing_migrations,
        version=version,
    )


def test_all_packaged_migrations_use_manager_owned_transactions() -> None:
    migrations = MigrationManager(MIGRATIONS_DIR).discover()
    marked = [
        migration
        for migration in migrations
        if MigrationManager._requires_foreign_keys_off(migration)
    ]
    marked_versions = {migration.version for migration in marked}
    assert {13, 22, 29, 34, 45}.issubset(marked_versions)
    for migration in migrations:
        MigrationManager._validate_manager_owned_migration(migration)


@pytest.mark.parametrize(
    "statement",
    (
        "BEGIN;",
        "BEGIN TRANSACTION;",
        "BEGIN IMMEDIATE;",
        "BEGIN EXCLUSIVE TRANSACTION;",
        "COMMIT;",
        "COMMIT TRANSACTION;",
        "END TRANSACTION;",
        "ROLLBACK;",
        "ROLLBACK TRANSACTION TO SAVEPOINT prior;",
        "SAVEPOINT nested;",
        "RELEASE SAVEPOINT nested;",
        "SELECT 1; COMMIT;",
        "BEGIN\nIMMEDIATE\n;",
    ),
)
def test_manager_owned_migration_rejects_top_level_transaction_control(
    tmp_path: Path,
    statement: str,
) -> None:
    path = tmp_path / "9999_invalid.sql"
    path.write_text(
        f"{statement}\nSELECT 1;\n",
        encoding="utf-8",
    )
    migration = Migration(version=9999, name="invalid", path=path)

    with pytest.raises(ValueError, match="MigrationManager owns the transaction"):
        MigrationManager._validate_manager_owned_migration(migration)


def test_manager_owned_transaction_validator_allows_trigger_body(
    tmp_path: Path,
) -> None:
    path = tmp_path / "9999_trigger.sql"
    path.write_text(
        """CREATE TEMP TRIGGER valid_trigger
BEFORE INSERT ON example
BEGIN
    SELECT 1;
END;
""",
        encoding="utf-8",
    )
    migration = Migration(version=9999, name="trigger", path=path)

    MigrationManager._validate_manager_owned_migration(migration)
