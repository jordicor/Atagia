"""SQLite connection and migration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

import aiosqlite

_MIGRATION_PATTERN = re.compile(r"^(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")
_FOREIGN_KEYS_OFF_MARKER = "-- atagia:foreign_keys_off"


def is_in_memory_database(database_path: str) -> bool:
    return database_path == ":memory:" or "mode=memory" in database_path


def _should_use_uri(database_path: str) -> bool:
    return database_path.startswith("file:")


def resolve_runtime_database_path(database_path: str) -> str:
    """Normalize special database paths for multi-connection runtimes."""
    if database_path == ":memory:":
        return f"file:atagia-{uuid4().hex}?mode=memory&cache=shared"
    return database_path


def _ensure_parent_directory(database_path: str) -> None:
    if is_in_memory_database(database_path) or _should_use_uri(database_path):
        return
    Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


async def apply_startup_pragmas(
    connection: aiosqlite.Connection,
    database_path: str,
) -> None:
    """Apply the recommended SQLite startup pragmas."""
    if not is_in_memory_database(database_path):
        await connection.execute("PRAGMA journal_mode = WAL;")
    await connection.execute("PRAGMA synchronous = NORMAL;")
    await connection.execute("PRAGMA foreign_keys = ON;")
    await connection.execute("PRAGMA busy_timeout = 5000;")
    await connection.execute("PRAGMA temp_store = MEMORY;")


async def open_connection(database_path: str) -> aiosqlite.Connection:
    """Open an SQLite connection with startup pragmas and row factory."""
    _ensure_parent_directory(database_path)
    connection = await aiosqlite.connect(database_path, uri=_should_use_uri(database_path))
    connection.row_factory = aiosqlite.Row
    await apply_startup_pragmas(connection, database_path)
    return connection


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
            raise FileNotFoundError(f"Missing migrations directory: {self._migrations_path}")
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

    async def apply_all(self, connection: aiosqlite.Connection) -> list[Migration]:
        await self.ensure_schema_table(connection)
        applied_versions = await self.applied_versions(connection)
        pending = [migration for migration in self.discover() if migration.version not in applied_versions]
        for migration in pending:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            try:
                if self._requires_foreign_keys_off(migration):
                    await self._apply_with_foreign_keys_disabled(connection, migration, timestamp)
                else:
                    # Migration files are trusted local SQL, but the metadata insert should
                    # still use a parameterized statement to follow the repository rule.
                    await connection.executescript(f"BEGIN;\n{migration.sql.rstrip()}\n")
                    await connection.execute(
                        """
                        INSERT INTO schema_migrations(version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (migration.version, migration.name, timestamp),
                    )
                    await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return pending

    @staticmethod
    def _requires_foreign_keys_off(migration: Migration) -> bool:
        return _FOREIGN_KEYS_OFF_MARKER in migration.sql

    async def _apply_with_foreign_keys_disabled(
        self,
        connection: aiosqlite.Connection,
        migration: Migration,
        timestamp: str,
    ) -> None:
        await connection.commit()
        await connection.execute("PRAGMA foreign_keys = OFF;")
        try:
            await connection.executescript(migration.sql.rstrip() + "\n")
            cursor = await connection.execute("PRAGMA foreign_key_check;")
            violations = await cursor.fetchall()
            if violations:
                raise RuntimeError(
                    f"Foreign key violations after migration {migration.version}_{migration.name}"
                )
            await connection.execute("PRAGMA foreign_keys = ON;")
            await connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.name, timestamp),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            await connection.execute("PRAGMA foreign_keys = ON;")
            raise


async def initialize_database(
    database_path: str,
    migrations_path: str | Path,
) -> aiosqlite.Connection:
    """Open a connection and apply all pending migrations."""
    connection = await open_connection(database_path)
    manager = MigrationManager(migrations_path)
    try:
        await manager.apply_all(connection)
    except Exception:
        await connection.close()
        raise
    return connection
