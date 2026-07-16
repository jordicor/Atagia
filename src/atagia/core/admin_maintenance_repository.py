"""Durable fences for long-running admin mutations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository
from atagia.core.transcript_rebuild_repository import (
    TranscriptRebuildRepository,
    UserAvailabilitySnapshot,
)
from atagia.services.errors import TranscriptRebuildInProgressError

DEFAULT_MAINTENANCE_LEASE_SECONDS = 120.0
EMBEDDING_DELETE_EFFECT = "delete_embedding"


@dataclass(slots=True)
class AdminMaintenanceOperation:
    """One durable ownership fence and its exact canonical source identity."""

    operation_id: str
    operation_kind: str
    recovery_key: str
    scope_kind: str
    user_id: str | None
    lifecycle_epoch: str | None
    derivation_revision: int | None
    owner_token: str
    lease_seconds: float
    phase: str = "prepared"
    resumed: bool = False

    @property
    def availability_snapshot(self) -> UserAvailabilitySnapshot:
        if self.scope_kind != "user":
            raise ValueError("A global maintenance operation has no user snapshot")
        assert self.lifecycle_epoch is not None
        assert self.derivation_revision is not None
        return UserAvailabilitySnapshot(
            lifecycle_epoch=self.lifecycle_epoch,
            derivation_revision=self.derivation_revision,
        )


class AdminMaintenanceRepository(BaseRepository):
    """Acquire, validate, advance, and durably release maintenance fences."""

    async def acquire_user(
        self,
        *,
        user_id: str,
        operation_kind: str,
        recovery_key: str | None = None,
        lease_seconds: float = DEFAULT_MAINTENANCE_LEASE_SECONDS,
    ) -> AdminMaintenanceOperation:
        rebuilds = TranscriptRebuildRepository(self._connection, self._clock)
        normalized_recovery_key = recovery_key or operation_kind
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await self.recover_expired(commit=False)
            operation = await self._resume_remediation(
                scope_kind="user",
                user_id=user_id,
                operation_kind=operation_kind,
                recovery_key=normalized_recovery_key,
                lease_seconds=lease_seconds,
            )
            if operation is None:
                snapshot = await rebuilds.capture_user_availability_snapshot(user_id)
                await rebuilds.require_user_availability_snapshot(user_id, snapshot)
                operation = AdminMaintenanceOperation(
                    operation_id=generate_prefixed_id("admop"),
                    operation_kind=operation_kind,
                    recovery_key=normalized_recovery_key,
                    scope_kind="user",
                    user_id=user_id,
                    lifecycle_epoch=snapshot.lifecycle_epoch,
                    derivation_revision=snapshot.derivation_revision,
                    owner_token=generate_prefixed_id("admown"),
                    lease_seconds=lease_seconds,
                )
                await self._insert_active(operation)
            else:
                await rebuilds.require_user_availability_snapshot(
                    user_id,
                    operation.availability_snapshot,
                    allowed_maintenance_operation_id=operation.operation_id,
                )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return operation

    async def acquire_global(
        self,
        *,
        operation_kind: str,
        recovery_key: str | None = None,
        lease_seconds: float = DEFAULT_MAINTENANCE_LEASE_SECONDS,
    ) -> AdminMaintenanceOperation:
        normalized_recovery_key = recovery_key or operation_kind
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await self.recover_expired(commit=False)
            operation = await self._resume_remediation(
                scope_kind="global",
                user_id=None,
                operation_kind=operation_kind,
                recovery_key=normalized_recovery_key,
                lease_seconds=lease_seconds,
            )
            rebuilds = TranscriptRebuildRepository(self._connection, self._clock)
            if operation is None:
                await rebuilds.require_scope_available()
                operation = AdminMaintenanceOperation(
                    operation_id=generate_prefixed_id("admop"),
                    operation_kind=operation_kind,
                    recovery_key=normalized_recovery_key,
                    scope_kind="global",
                    user_id=None,
                    lifecycle_epoch=None,
                    derivation_revision=None,
                    owner_token=generate_prefixed_id("admown"),
                    lease_seconds=lease_seconds,
                )
                await self._insert_active(operation)
            else:
                await rebuilds.require_scope_available(
                    allowed_maintenance_operation_id=operation.operation_id
                )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return operation

    async def _resume_remediation(
        self,
        *,
        scope_kind: str,
        user_id: str | None,
        operation_kind: str,
        recovery_key: str,
        lease_seconds: float,
    ) -> AdminMaintenanceOperation | None:
        row = await self._fetch_one(
            """
            SELECT *
            FROM admin_maintenance_operations
            WHERE status = 'remediation_required'
              AND scope_kind = ?
              AND user_id IS ?
              AND operation_kind = ?
              AND recovery_key = ?
            LIMIT 1
            """,
            (scope_kind, user_id, operation_kind, recovery_key),
        )
        if row is None:
            return None
        owner_token = generate_prefixed_id("admown")
        lease_now = _lease_now()
        cursor = await self._connection.execute(
            """
            UPDATE admin_maintenance_operations
            SET status = 'active', owner_token = ?, heartbeat_at = ?,
                lease_expires_at = ?, error_class = NULL, error_message = NULL,
                completed_at = NULL, updated_at = ?
            WHERE id = ? AND status = 'remediation_required'
            """,
            (
                owner_token,
                lease_now,
                _lease_deadline(lease_seconds),
                self._timestamp(),
                str(row["id"]),
            ),
        )
        if int(cursor.rowcount or 0) != 1:
            raise TranscriptRebuildInProgressError(
                "Admin maintenance remediation ownership changed"
            )
        return AdminMaintenanceOperation(
            operation_id=str(row["id"]),
            operation_kind=operation_kind,
            recovery_key=recovery_key,
            scope_kind=scope_kind,
            user_id=user_id,
            lifecycle_epoch=(
                str(row["lifecycle_epoch"])
                if row.get("lifecycle_epoch") is not None
                else None
            ),
            derivation_revision=(
                int(row["derivation_revision"])
                if row.get("derivation_revision") is not None
                else None
            ),
            owner_token=owner_token,
            lease_seconds=lease_seconds,
            phase=str(row["phase"]),
            resumed=True,
        )

    async def _insert_active(self, operation: AdminMaintenanceOperation) -> None:
        timestamp = self._timestamp()
        lease_now = _lease_now()
        lease_expires_at = _lease_deadline(operation.lease_seconds)
        try:
            await self._connection.execute(
                """
                INSERT INTO admin_maintenance_operations(
                    id, operation_kind, recovery_key, scope_kind, user_id, lifecycle_epoch,
                    derivation_revision, owner_token, heartbeat_at,
                    lease_expires_at, phase, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    operation.operation_id,
                    operation.operation_kind,
                    operation.recovery_key,
                    operation.scope_kind,
                    operation.user_id,
                    operation.lifecycle_epoch,
                    operation.derivation_revision,
                    operation.owner_token,
                    lease_now,
                    lease_expires_at,
                    operation.phase,
                    timestamp,
                    timestamp,
                ),
            )
        except aiosqlite.IntegrityError as exc:
            raise TranscriptRebuildInProgressError(
                "Another admin maintenance operation is already active for this scope"
            ) from exc

    async def require_current(self, operation: AdminMaintenanceOperation) -> None:
        """Require active ownership and the exact operation source revision."""

        row = await self._fetch_one(
            """
            SELECT phase
            FROM admin_maintenance_operations
            WHERE id = ?
              AND status = 'active'
              AND operation_kind = ?
              AND recovery_key = ?
              AND scope_kind = ?
              AND user_id IS ?
              AND lifecycle_epoch IS ?
              AND derivation_revision IS ?
              AND owner_token = ?
              AND julianday(lease_expires_at) > julianday('now')
            """,
            (
                operation.operation_id,
                operation.operation_kind,
                operation.recovery_key,
                operation.scope_kind,
                operation.user_id,
                operation.lifecycle_epoch,
                operation.derivation_revision,
                operation.owner_token,
            ),
        )
        if row is None:
            raise TranscriptRebuildInProgressError(
                "Admin maintenance ownership or source revision changed"
            )
        operation.phase = str(row["phase"])
        rebuilds = TranscriptRebuildRepository(self._connection, self._clock)
        if operation.scope_kind == "global":
            await rebuilds.require_scope_available(
                allowed_maintenance_operation_id=operation.operation_id
            )
            return
        assert operation.user_id is not None
        await rebuilds.require_user_availability_snapshot(
            operation.user_id,
            operation.availability_snapshot,
            allowed_maintenance_operation_id=operation.operation_id,
        )

    async def mark_dirty(self, operation: AdminMaintenanceOperation) -> None:
        """Durably mark that expiration now requires an explicit same-op retry."""

        await self.require_current(operation)
        if operation.phase == "dirty":
            return
        cursor = await self._connection.execute(
            """
            UPDATE admin_maintenance_operations
            SET phase = 'dirty', updated_at = ?
            WHERE id = ?
              AND status = 'active'
              AND owner_token = ?
              AND phase = 'prepared'
              AND julianday(lease_expires_at) > julianday('now')
            """,
            (
                self._timestamp(),
                operation.operation_id,
                operation.owner_token,
            ),
        )
        if int(cursor.rowcount or 0) != 1:
            raise TranscriptRebuildInProgressError(
                "Admin maintenance ownership changed before its first effect"
            )
        operation.phase = "dirty"

    async def enqueue_effects(
        self,
        operation: AdminMaintenanceOperation,
        *,
        effect_kind: str,
        target_ids: Iterable[str],
    ) -> None:
        """Persist idempotent post-commit work in the mutation transaction."""

        if effect_kind != EMBEDDING_DELETE_EFFECT:
            raise ValueError(f"Unsupported maintenance effect: {effect_kind}")
        await self.require_current(operation)
        timestamp = self._timestamp()
        rows = [
            (
                operation.operation_id,
                effect_kind,
                target_id,
                timestamp,
                timestamp,
            )
            for target_id in dict.fromkeys(str(item).strip() for item in target_ids)
            if target_id
        ]
        if not rows:
            return
        await self._connection.executemany(
            """
            INSERT OR IGNORE INTO admin_maintenance_effects(
                operation_id, effect_kind, target_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, 'pending', ?, ?)
            """,
            rows,
        )

    async def list_pending_effect_targets(
        self,
        operation: AdminMaintenanceOperation,
        *,
        effect_kind: str,
    ) -> list[str]:
        """Return durable effects still owed by the current owner."""

        if effect_kind != EMBEDDING_DELETE_EFFECT:
            raise ValueError(f"Unsupported maintenance effect: {effect_kind}")
        await self.require_current(operation)
        cursor = await self._connection.execute(
            """
            SELECT target_id
            FROM admin_maintenance_effects
            WHERE operation_id = ?
              AND effect_kind = ?
              AND status = 'pending'
            ORDER BY target_id ASC
            """,
            (operation.operation_id, effect_kind),
        )
        return [str(row["target_id"]) for row in await cursor.fetchall()]

    async def complete_effect(
        self,
        operation: AdminMaintenanceOperation,
        *,
        effect_kind: str,
        target_id: str,
    ) -> None:
        """Checkpoint one idempotent external effect after it succeeds."""

        if self._connection.in_transaction:
            await self._connection.rollback()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await self.require_current(operation)
            cursor = await self._connection.execute(
                """
                UPDATE admin_maintenance_effects
                SET status = 'completed', updated_at = ?, completed_at = ?
                WHERE operation_id = ?
                  AND effect_kind = ?
                  AND target_id = ?
                  AND status = 'pending'
                """,
                (
                    self._timestamp(),
                    self._timestamp(),
                    operation.operation_id,
                    effect_kind,
                    target_id,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise TranscriptRebuildInProgressError(
                    "Admin maintenance effect was no longer pending"
                )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def advance_derivation_revision(
        self,
        operation: AdminMaintenanceOperation,
        *,
        new_revision: int,
    ) -> None:
        """Move a user operation to the revision created by its own mutation."""

        if operation.scope_kind != "user" or operation.derivation_revision is None:
            raise ValueError("Only a user maintenance operation can advance revision")
        if new_revision <= operation.derivation_revision:
            raise ValueError("Maintenance revision must advance monotonically")
        cursor = await self._connection.execute(
            """
            UPDATE admin_maintenance_operations
            SET derivation_revision = ?, updated_at = ?
            WHERE id = ?
              AND status = 'active'
              AND owner_token = ?
              AND julianday(lease_expires_at) > julianday('now')
              AND derivation_revision = ?
            """,
            (
                new_revision,
                self._timestamp(),
                operation.operation_id,
                operation.owner_token,
                operation.derivation_revision,
            ),
        )
        if int(cursor.rowcount or 0) != 1:
            raise TranscriptRebuildInProgressError(
                "Admin maintenance source revision changed concurrently"
            )
        operation.derivation_revision = new_revision

    async def succeed(self, operation: AdminMaintenanceOperation) -> None:
        await self._finish(operation, status="succeeded")

    async def heartbeat(self, operation: AdminMaintenanceOperation) -> bool:
        """Extend a live lease without allowing an expired owner to revive."""

        if not self._connection.in_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            now = _lease_now()
            cursor = await self._connection.execute(
                """
                UPDATE admin_maintenance_operations
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE id = ?
                  AND status = 'active'
                  AND owner_token = ?
                  AND julianday(lease_expires_at) > julianday(?)
                """,
                (
                    now,
                    _lease_deadline(operation.lease_seconds),
                    self._timestamp(),
                    operation.operation_id,
                    operation.owner_token,
                    now,
                ),
            )
            await self._connection.commit()
            return int(cursor.rowcount or 0) == 1
        except BaseException:
            await self._connection.rollback()
            raise

    async def recover_expired(self, *, commit: bool = True) -> int:
        """Fail expired owners and cancel their stranded jobs atomically."""

        now = _lease_now()
        await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = 'cancelled',
                finished_at = ?,
                last_heartbeat_at = ?,
                error_class = 'AdminMaintenanceLeaseExpired',
                error_message = NULL,
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL,
                execution_fence = execution_fence + 1
            WHERE maintenance_operation_id IN (
                SELECT id
                FROM admin_maintenance_operations
                WHERE status = 'active'
                  AND julianday(lease_expires_at) <= julianday(?)
            )
              AND status IN (
                  'queued', 'awaiting_claim', 'running', 'retrying', 'deferred'
              )
            """,
            (now, now, now),
        )
        cursor = await self._connection.execute(
            """
            UPDATE admin_maintenance_operations
            SET status = CASE
                    WHEN phase = 'dirty' THEN 'remediation_required'
                    ELSE 'failed'
                END,
                error_class = 'AdminMaintenanceLeaseExpired',
                error_message = 'Maintenance owner lease expired',
                updated_at = ?,
                completed_at = CASE WHEN phase = 'dirty' THEN NULL ELSE ? END
            WHERE status = 'active'
              AND julianday(lease_expires_at) <= julianday(?)
            """,
            (self._timestamp(), self._timestamp(), now),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def fail(
        self,
        operation: AdminMaintenanceOperation,
        error: BaseException,
    ) -> None:
        await self._finish(
            operation,
            status="failed",
            error_class=error.__class__.__name__,
            error_message=str(error)[:1000],
            require_current=False,
        )

    async def _finish(
        self,
        operation: AdminMaintenanceOperation,
        *,
        status: str,
        error_class: str | None = None,
        error_message: str | None = None,
        require_current: bool = True,
    ) -> None:
        if self._connection.in_transaction:
            await self._connection.rollback()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            state_cursor = await self._connection.execute(
                """
                SELECT status, owner_token, phase
                FROM admin_maintenance_operations
                WHERE id = ?
                """,
                (operation.operation_id,),
            )
            state = await state_cursor.fetchone()
            if state is None and not require_current:
                await self._connection.commit()
                return
            if state is None or str(state["owner_token"]) != operation.owner_token:
                raise TranscriptRebuildInProgressError(
                    "Admin maintenance ownership was lost"
                )
            current_status = str(state["status"])
            final_status = (
                "remediation_required"
                if status == "failed" and str(state["phase"]) == "dirty"
                else status
            )
            if current_status == final_status:
                await self._connection.commit()
                return
            if current_status != "active":
                raise TranscriptRebuildInProgressError(
                    "Admin maintenance operation is already terminal"
                )
            if require_current:
                await self.require_current(operation)
                cursor = await self._connection.execute(
                    """
                    SELECT 1
                    FROM worker_job_runs
                    WHERE maintenance_operation_id = ?
                      AND status IN (
                          'queued', 'awaiting_claim', 'running', 'retrying', 'deferred'
                      )
                    LIMIT 1
                    """,
                    (operation.operation_id,),
                )
                if await cursor.fetchone() is not None:
                    raise TranscriptRebuildInProgressError(
                        "Admin maintenance still owns unfinished durable jobs"
                    )
                cursor = await self._connection.execute(
                    """
                    SELECT 1
                    FROM admin_maintenance_effects
                    WHERE operation_id = ? AND status = 'pending'
                    LIMIT 1
                    """,
                    (operation.operation_id,),
                )
                if await cursor.fetchone() is not None:
                    raise TranscriptRebuildInProgressError(
                        "Admin maintenance still owns unfinished external effects"
                    )
            else:
                await self._connection.execute(
                    """
                    UPDATE worker_job_runs
                    SET status = 'cancelled',
                        finished_at = ?,
                        last_heartbeat_at = ?,
                        error_class = 'AdminMaintenanceFailed',
                        error_message = NULL,
                        recovery_envelope_json = NULL,
                        envelope_schema_version = NULL,
                        dispatch_token = NULL,
                        dispatch_visibility_deadline = NULL,
                        execution_owner = NULL,
                        execution_lease_expires_at = NULL,
                        deferred_until = NULL,
                        execution_fence = execution_fence + 1
                    WHERE maintenance_operation_id = ?
                      AND status IN (
                          'queued', 'awaiting_claim', 'running', 'retrying', 'deferred'
                      )
                    """,
                    (
                        self._timestamp(),
                        self._timestamp(),
                        operation.operation_id,
                    ),
                )
            cursor = await self._connection.execute(
                """
                UPDATE admin_maintenance_operations
                SET status = ?, error_class = ?, error_message = ?,
                    updated_at = ?, completed_at = ?
                WHERE id = ? AND status = 'active' AND owner_token = ?
                """,
                (
                    final_status,
                    error_class,
                    error_message,
                    self._timestamp(),
                    (
                        None
                        if final_status == "remediation_required"
                        else self._timestamp()
                    ),
                    operation.operation_id,
                    operation.owner_token,
                ),
            )
            if int(cursor.rowcount or 0) not in {0, 1}:
                raise RuntimeError("Unexpected maintenance release cardinality")
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise


@asynccontextmanager
async def admin_maintenance_operation(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    operation_kind: str,
    user_id: str | None,
    recovery_key: str | None = None,
    heartbeat_connection_factory: Callable[[], Awaitable[aiosqlite.Connection]]
    | None = None,
    lease_seconds: float = DEFAULT_MAINTENANCE_LEASE_SECONDS,
) -> AsyncIterator[AdminMaintenanceOperation]:
    """Own and always durably release one admin maintenance fence."""

    repository = AdminMaintenanceRepository(connection, clock)
    operation = (
        await repository.acquire_global(
            operation_kind=operation_kind,
            recovery_key=recovery_key,
            lease_seconds=lease_seconds,
        )
        if user_id is None
        else await repository.acquire_user(
            user_id=user_id,
            operation_kind=operation_kind,
            recovery_key=recovery_key,
            lease_seconds=lease_seconds,
        )
    )
    stop_heartbeat = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    if heartbeat_connection_factory is not None:
        heartbeat_task = asyncio.create_task(
            _heartbeat_operation(
                operation,
                connection_factory=heartbeat_connection_factory,
                stop=stop_heartbeat,
                clock=clock,
            )
        )
    try:
        yield operation
        if heartbeat_task is not None and heartbeat_task.done():
            heartbeat_task.result()
        await repository.succeed(operation)
    except BaseException as exc:
        try:
            await repository.fail(operation, exc)
        except Exception as release_error:
            raise release_error from exc
        raise
    finally:
        stop_heartbeat.set()
        if heartbeat_task is not None:
            await asyncio.gather(heartbeat_task, return_exceptions=True)


async def _heartbeat_operation(
    operation: AdminMaintenanceOperation,
    *,
    connection_factory: Callable[[], Awaitable[aiosqlite.Connection]],
    stop: asyncio.Event,
    clock: Clock,
) -> None:
    interval = max(0.1, operation.lease_seconds / 3.0)
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        connection = await connection_factory()
        try:
            if not await AdminMaintenanceRepository(
                connection,
                clock,
            ).heartbeat(operation):
                raise TranscriptRebuildInProgressError(
                    "Admin maintenance heartbeat lost ownership"
                )
        finally:
            await connection.close()


def _lease_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lease_deadline(lease_seconds: float) -> str:
    if lease_seconds <= 0:
        raise ValueError("Maintenance lease must be positive")
    return (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
