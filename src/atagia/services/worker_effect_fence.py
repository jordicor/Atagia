"""SQLite statement fences for durable worker domain effects."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import sqlite3
from typing import AsyncIterator

import aiosqlite

from atagia.core.clock import Clock
from atagia.models.schemas_jobs import ClaimedJob

_CONTEXT_TABLE = "_atagia_job_effect_context"
_STALE_FENCE_ERROR = "ATAGIA_STALE_JOB_EFFECT_FENCE"
_FENCE_TRIGGER_PREFIX = "_atagia_effect_fence_"
_FENCED_OPERATIONS = ("INSERT", "UPDATE", "DELETE")
_EXCLUDED_TABLES = {
    "schema_migrations",
    "worker_job_runs",
}


class StaleJobEffectFenceError(RuntimeError):
    """A domain statement attempted to use an obsolete execution fence."""


class WorkerEffectFence:
    """Fence every main-table write made by one worker-domain connection.

    TEMP triggers live only on the dedicated worker connection. Each domain
    statement checks the durable execution owner, fence, lifecycle epoch,
    derivation revision, and lease inside the same SQLite write transaction. A
    takeover, derivation invalidation, or lifecycle revocation therefore
    linearizes before or after the domain effect; it can never race between an
    application-side check and the SQL statement.
    """

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    @asynccontextmanager
    async def activate(self, claim: ClaimedJob) -> AsyncIterator[None]:
        await self._ensure_initialized()
        await self.assert_write_coverage()
        await self._connection.execute(f"DELETE FROM temp.{_CONTEXT_TABLE}")
        await self._connection.execute(
            f"""
            INSERT INTO temp.{_CONTEXT_TABLE}(
                singleton,
                job_id,
                user_id,
                execution_owner,
                execution_fence,
                lifecycle_epoch,
                derivation_revision
            ) VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.envelope.job_id,
                claim.envelope.user_id,
                claim.owner_id,
                claim.execution_fence,
                claim.lifecycle_epoch,
                claim.derivation_revision,
            ),
        )
        await self._connection.commit()
        caught: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            caught = exc
        finally:
            if self._connection.in_transaction:
                await self._connection.rollback()
            await self._connection.execute(f"DELETE FROM temp.{_CONTEXT_TABLE}")
            await self._connection.commit()
        if caught is not None:
            if _is_stale_fence_exception(caught):
                raise StaleJobEffectFenceError(
                    f"Domain effect fence lost for durable job {claim.envelope.job_id}"
                ) from caught
            raise caught

    async def assert_write_coverage(self) -> None:
        """Fail fast when a writable main table has no fence trigger.

        Every ordinary main-schema table must carry all three fence triggers.
        The exemptions are mechanical, never name lists: ``virtual`` and
        ``shadow`` rows in ``pragma_table_list`` cannot carry triggers at all,
        and a plain table prefixed by an enumerated virtual table's name plus
        ``_`` is that module's backing storage (for example sqlite-vec's
        ``*_vector_chunksNN``, which its ``xShadowName`` does not flag) whose
        writes are already fenced in-transaction by the embedding backend.
        ``sqlite_``-internal tables and ``_EXCLUDED_TABLES`` are the fence's
        own bookkeeping.

        The virtual-prefix exemption assumes tables appear only through
        reviewed migrations: a migration that named a real domain table
        ``<virtual_table>_<anything>`` would be exempted silently. Before the
        embedding backend ships enabled by default, tighten this to the
        module's exact backing-table suffixes.
        """

        cursor = await self._connection.execute(
            """
            SELECT name, type
            FROM pragma_table_list
            WHERE schema = 'main'
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        virtual_prefixes = tuple(
            f"{row['name']}_" for row in rows if str(row["type"]) == "virtual"
        )
        required = sorted(
            str(row["name"])
            for row in rows
            if str(row["type"]) == "table"
            and not str(row["name"]).startswith("sqlite_")
            and str(row["name"]) not in _EXCLUDED_TABLES
            and not str(row["name"]).startswith(virtual_prefixes)
        )
        cursor = await self._connection.execute(
            """
            SELECT tbl_name, sql
            FROM sqlite_temp_master
            WHERE type = 'trigger'
              AND name GLOB ?
            """,
            (f"{_FENCE_TRIGGER_PREFIX}*",),
        )
        fenced_operations: dict[str, set[str]] = {}
        for row in await cursor.fetchall():
            operations = fenced_operations.setdefault(str(row["tbl_name"]), set())
            trigger_sql = str(row["sql"] or "")
            for operation in _FENCED_OPERATIONS:
                if f"BEFORE {operation} ON" in trigger_sql:
                    operations.add(operation)
        await cursor.close()
        unfenced = [
            table_name
            for table_name in required
            if fenced_operations.get(table_name, set()) != set(_FENCED_OPERATIONS)
        ]
        if unfenced:
            raise RuntimeError(
                "Worker effect fence does not cover writable tables: "
                + ", ".join(unfenced)
                + ". Create domain tables before worker fence initialization "
                "or document a mechanical exemption in worker_effect_fence.py."
            )

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await self._connection.create_function(
                "atagia_effect_now",
                0,
                lambda: self._clock.now().isoformat(),
                deterministic=False,
            )
            await self._connection.execute(
                f"""
                CREATE TEMP TABLE IF NOT EXISTS {_CONTEXT_TABLE}(
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    job_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    execution_owner TEXT NOT NULL,
                    execution_fence INTEGER NOT NULL,
                    lifecycle_epoch TEXT NOT NULL,
                    derivation_revision INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            cursor = await self._connection.execute(
                """
                SELECT name
                FROM pragma_table_list
                WHERE schema = 'main'
                  AND type = 'table'
                ORDER BY name ASC
                """
            )
            table_names = [str(row["name"]) for row in await cursor.fetchall()]
            await cursor.close()
            trigger_index = 0
            for table_name in table_names:
                if table_name.startswith("sqlite_") or table_name in _EXCLUDED_TABLES:
                    continue
                quoted_table = _quote_identifier(table_name)
                for operation in _FENCED_OPERATIONS:
                    trigger_index += 1
                    await self._connection.execute(
                        f"""
                        CREATE TEMP TRIGGER IF NOT EXISTS
                            {_FENCE_TRIGGER_PREFIX}{trigger_index}
                        BEFORE {operation} ON main.{quoted_table}
                        WHEN EXISTS (SELECT 1 FROM temp.{_CONTEXT_TABLE})
                        BEGIN
                            SELECT CASE WHEN NOT EXISTS (
                                SELECT 1
                                FROM main.worker_job_runs AS job
                                JOIN main.user_lifecycles AS lifecycle
                                  ON lifecycle.user_id = job.user_id
                                 AND lifecycle.lifecycle_epoch = job.lifecycle_epoch
                                JOIN temp.{_CONTEXT_TABLE} AS context
                                  ON context.job_id = job.job_id
                                 AND context.user_id = job.user_id
                                 AND context.execution_owner = job.execution_owner
                                 AND context.execution_fence = job.execution_fence
                                 AND context.lifecycle_epoch = job.lifecycle_epoch
                                 AND context.derivation_revision = job.derivation_revision
                                LEFT JOIN main.users AS user ON user.id = job.user_id
                                WHERE job.status = 'running'
                                  AND job.execution_lease_expires_at > atagia_effect_now()
                                  AND lifecycle.derivation_revision = job.derivation_revision
                                  AND lifecycle.state = 'active'
                                  AND lifecycle.erasure_cleanup_id IS NULL
                                  AND (
                                      job.user_id = 'atagia_system'
                                      OR (user.id IS NOT NULL AND user.deleted_at IS NULL)
                                  )
                                  AND (
                                      (
                                          job.maintenance_operation_id IS NULL
                                          AND NOT EXISTS (
                                              SELECT 1
                                              FROM main.admin_maintenance_operations
                                                  AS active_operation
                                              WHERE (
                                                  active_operation.status =
                                                      'remediation_required'
                                                  OR (
                                                      active_operation.status = 'active'
                                                      AND (
                                                          active_operation.phase = 'dirty'
                                                          OR julianday(
                                                              active_operation.lease_expires_at
                                                          ) > julianday('now')
                                                      )
                                                  )
                                              )
                                                AND (
                                                    active_operation.scope_kind = 'global'
                                                    OR active_operation.user_id = job.user_id
                                                )
                                          )
                                      )
                                      OR EXISTS (
                                          SELECT 1
                                          FROM main.admin_maintenance_operations
                                              AS owned_operation
                                          WHERE owned_operation.id =
                                              job.maintenance_operation_id
                                            AND owned_operation.status = 'active'
                                            AND julianday(
                                                owned_operation.lease_expires_at
                                            ) > julianday('now')
                                            AND owned_operation.scope_kind = 'user'
                                            AND owned_operation.user_id = job.user_id
                                            AND owned_operation.lifecycle_epoch =
                                                job.lifecycle_epoch
                                            AND owned_operation.derivation_revision =
                                                job.derivation_revision
                                      )
                                  )
                            ) THEN RAISE(ABORT, '{_STALE_FENCE_ERROR}') END;
                        END
                        """
                    )
            await self._connection.commit()
            self._initialized = True


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _is_stale_fence_exception(exc: BaseException) -> bool:
    return isinstance(exc, (sqlite3.DatabaseError, aiosqlite.Error)) and (
        _STALE_FENCE_ERROR in str(exc)
    )
