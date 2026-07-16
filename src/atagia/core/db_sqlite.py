"""SQLite connection and migration helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import TypeVar
from uuid import uuid4

import aiosqlite

_MIGRATION_PATTERN = re.compile(r"^(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")
_FOREIGN_KEYS_OFF_MARKER = "-- atagia:foreign_keys_off"
_TRANSACTION_CONTROL_KEYWORDS = frozenset(
    {"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"}
)
SQLITE_BUSY_TIMEOUT_MS = 60_000
_T = TypeVar("_T")


def _iter_sql_statements(script: str) -> list[str]:
    """Split trusted migration SQL without breaking trigger bodies."""
    statements: list[str] = []
    buffer: list[str] = []
    for character in script:
        buffer.append(character)
        if character != ";":
            continue
        candidate = "".join(buffer)
        if sqlite3.complete_statement(candidate):
            statements.append(candidate)
            buffer.clear()
    remainder = "".join(buffer)
    if remainder.strip():
        statements.append(remainder)
    return statements


def _strip_leading_sql_comments(statement: str) -> str:
    remaining = statement.lstrip()
    while remaining:
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            if newline < 0:
                return ""
            remaining = remaining[newline + 1 :].lstrip()
            continue
        if remaining.startswith("/*"):
            comment_end = remaining.find("*/", 2)
            if comment_end < 0:
                return ""
            remaining = remaining[comment_end + 2 :].lstrip()
            continue
        return remaining
    return ""


def _transaction_control_statement(script: str) -> str | None:
    for statement in _iter_sql_statements(script):
        normalized = _strip_leading_sql_comments(statement)
        keyword_match = re.match(r"[A-Za-z]+", normalized)
        if (
            keyword_match is not None
            and keyword_match.group(0).upper() in _TRANSACTION_CONTROL_KEYWORDS
        ):
            return " ".join(normalized.split())
    return None


def is_in_memory_database(database_path: str) -> bool:
    return (
        database_path == ":memory:"
        or "mode=memory" in database_path
        or "vfs=memdb" in database_path
    )


def _should_use_uri(database_path: str) -> bool:
    return database_path.startswith("file:")


def resolve_runtime_database_path(database_path: str) -> str:
    """Normalize special database paths for multi-connection runtimes."""
    if database_path == ":memory:":
        # SQLite's shared-cache in-memory URI returns SQLITE_LOCKED_SHAREDCACHE
        # immediately when independent runtime connections contend, bypassing
        # busy_timeout. The memdb VFS keeps the database entirely in memory but
        # uses normal database locking, so a concurrent writer waits and the
        # runtime's busy timeout can do its job.
        return f"file:/atagia-{uuid4().hex}?vfs=memdb"
    return database_path


def _ensure_parent_directory(database_path: str) -> None:
    if is_in_memory_database(database_path) or _should_use_uri(database_path):
        return
    Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


async def apply_startup_pragmas(
    connection: aiosqlite.Connection,
    database_path: str,
    *,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> None:
    """Apply the recommended SQLite startup pragmas."""
    await connection.execute(f"PRAGMA busy_timeout = {max(1, busy_timeout_ms)};")
    if not is_in_memory_database(database_path):
        await connection.execute("PRAGMA journal_mode = WAL;")
    await connection.execute("PRAGMA synchronous = NORMAL;")
    await connection.execute("PRAGMA foreign_keys = ON;")
    await connection.execute("PRAGMA temp_store = MEMORY;")


async def _finish_even_if_cancelled(awaitable: Awaitable[_T]) -> _T:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            return await task
        finally:
            raise


async def close_connection(connection: aiosqlite.Connection) -> None:
    """Close an aiosqlite connection even if the owning task is cancelled."""
    await _finish_even_if_cancelled(connection.close())


async def _connect_tracked(
    database_path: str, connections: list[aiosqlite.Connection]
) -> aiosqlite.Connection:
    pending_connection = aiosqlite.connect(
        database_path, uri=_should_use_uri(database_path)
    )
    task = asyncio.ensure_future(pending_connection)
    try:
        connection = await asyncio.shield(task)
    except asyncio.CancelledError:
        connection = await task
        await close_connection(connection)
        raise
    connections.append(connection)
    return connection


async def open_connection(
    database_path: str,
    *,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> aiosqlite.Connection:
    """Open an SQLite connection with startup pragmas and row factory."""
    _ensure_parent_directory(database_path)
    connections: list[aiosqlite.Connection] = []
    try:
        connection = await _connect_tracked(database_path, connections)
        connection.row_factory = aiosqlite.Row
        await apply_startup_pragmas(
            connection,
            database_path,
            busy_timeout_ms=busy_timeout_ms,
        )
        return connection
    except BaseException:
        for connection in connections:
            await close_connection(connection)
        raise


@dataclass(frozen=True, slots=True)
class Migration:
    """Filesystem-backed SQL migration."""

    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


class MigrationManager:
    """Discovers and applies numbered SQL migrations."""

    def __init__(self, migrations_path: str | Path) -> None:
        self._migrations_path = Path(migrations_path)

    def discover(self) -> list[Migration]:
        if not self._migrations_path.exists():
            raise FileNotFoundError(
                f"Missing migrations directory: {self._migrations_path}"
            )
        migrations: list[Migration] = []
        for path in sorted(self._migrations_path.glob("*.sql")):
            match = _MIGRATION_PATTERN.match(path.name)
            if match is None:
                raise ValueError(f"Invalid migration filename: {path.name}")
            migrations.append(
                Migration(
                    version=int(match.group("version")),
                    name=match.group("name"),
                    path=path,
                )
            )
        return migrations

    async def ensure_schema_table(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        await connection.commit()

    async def applied_versions(self, connection: aiosqlite.Connection) -> set[int]:
        cursor = await connection.execute("SELECT version FROM schema_migrations")
        rows = await cursor.fetchall()
        return {int(row["version"]) for row in rows}

    @staticmethod
    async def _migration_is_applied(
        connection: aiosqlite.Connection,
        migration: Migration,
    ) -> bool:
        cursor = await connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (migration.version,),
        )
        return await cursor.fetchone() is not None

    @staticmethod
    async def _execute_migration_sql(
        connection: aiosqlite.Connection,
        migration: Migration,
    ) -> None:
        # executescript() commits an existing transaction before running the
        # script. Execute complete statements individually so the writer lock,
        # migration SQL, and metadata insert remain one atomic transaction.
        for statement in _iter_sql_statements(migration.sql):
            await connection.execute(statement)

    async def apply_all(self, connection: aiosqlite.Connection) -> list[Migration]:
        await self.ensure_schema_table(connection)
        applied_versions = await self.applied_versions(connection)
        migrations = self.discover()
        for migration in migrations:
            self._validate_manager_owned_migration(migration)
        pending = [
            migration
            for migration in migrations
            if migration.version not in applied_versions
        ]
        applied: list[Migration] = []
        for migration in pending:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            if self._requires_foreign_keys_off(migration):
                was_applied = await self._apply_with_foreign_keys_disabled(
                    connection, migration, timestamp
                )
            else:
                was_applied = await self._apply_with_foreign_keys_enabled(
                    connection, migration, timestamp
                )
            if was_applied:
                applied.append(migration)
        return applied

    async def _apply_with_foreign_keys_enabled(
        self,
        connection: aiosqlite.Connection,
        migration: Migration,
        timestamp: str,
    ) -> bool:
        try:
            await connection.execute("BEGIN IMMEDIATE;")
            if await self._migration_is_applied(connection, migration):
                await connection.commit()
                return False
            await self._execute_migration_sql(connection, migration)
            # Migration files are trusted local SQL, but the metadata insert should
            # still use a parameterized statement to follow the repository rule.
            await connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.name, timestamp),
            )
            await connection.commit()
            return True
        except BaseException:
            await _finish_even_if_cancelled(connection.rollback())
            raise

    @staticmethod
    def _requires_foreign_keys_off(migration: Migration) -> bool:
        return _FOREIGN_KEYS_OFF_MARKER in migration.sql

    @staticmethod
    def _validate_manager_owned_migration(migration: Migration) -> None:
        """Reject any migration script that can escape its manager transaction."""
        transaction_statement = _transaction_control_statement(migration.sql)
        if transaction_statement is not None:
            raise ValueError(
                "Migration "
                f"{migration.version}_{migration.name} contains top-level transaction "
                f"control ({transaction_statement}); MigrationManager owns the transaction"
            )

    @staticmethod
    async def _set_foreign_keys(
        connection: aiosqlite.Connection,
        *,
        enabled: bool,
    ) -> None:
        expected = 1 if enabled else 0
        await connection.execute(f"PRAGMA foreign_keys = {'ON' if enabled else 'OFF'};")
        cursor = await connection.execute("PRAGMA foreign_keys;")
        row = await cursor.fetchone()
        actual = int(row[0]) if row is not None else -1
        if actual != expected:
            state = "enabled" if enabled else "disabled"
            raise RuntimeError(f"Failed to leave SQLite foreign keys {state}")

    async def _apply_with_foreign_keys_disabled(
        self,
        connection: aiosqlite.Connection,
        migration: Migration,
        timestamp: str,
    ) -> bool:
        await connection.commit()
        try:
            await self._set_foreign_keys(connection, enabled=False)
            await connection.execute("BEGIN IMMEDIATE;")
            if await self._migration_is_applied(connection, migration):
                await connection.commit()
                return False
            await self._execute_migration_sql(connection, migration)
            cursor = await connection.execute("PRAGMA foreign_key_check;")
            violations = await cursor.fetchall()
            if violations:
                raise RuntimeError(
                    f"Foreign key violations after migration {migration.version}_{migration.name}"
                )
            await connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.name, timestamp),
            )
            await connection.commit()
            return True
        except BaseException:
            await _finish_even_if_cancelled(connection.rollback())
            raise
        finally:
            # foreign_keys cannot be changed while a transaction is active, so
            # restoration must happen after either COMMIT or ROLLBACK.
            await _finish_even_if_cancelled(
                self._set_foreign_keys(connection, enabled=True)
            )


async def initialize_database(
    database_path: str,
    migrations_path: str | Path,
) -> aiosqlite.Connection:
    """Open a connection and apply all pending migrations."""
    connection = await open_connection(database_path)
    manager = MigrationManager(migrations_path)
    try:
        await manager.apply_all(connection)
    except BaseException:
        await close_connection(connection)
        raise
    return connection
