"""Durable execution of external user-erasure cleanup checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any

import aiosqlite

from atagia.core.canonical import canonical_json_hash
from atagia.core.clock import Clock
from atagia.core.storage_backend import LegacyTransientPurgeResult, StorageBackend
from atagia.core.user_erasure_repository import (
    UserErasureConflictError,
    UserErasureRepository,
)
from atagia.models.schemas_jobs import WORKER_GROUP_NAME

logger = logging.getLogger(__name__)

_MAX_CAS_RELOADS = 4


@dataclass(frozen=True, slots=True)
class ErasureRecoveryResult:
    """Count-only startup recovery outcome."""

    completed: int = 0
    pending: int = 0


@dataclass(slots=True)
class UserErasureCleanupService:
    """Drive one durable cleanup record to its verified terminal state."""

    connection: aiosqlite.Connection
    clock: Clock
    storage_backend: StorageBackend
    storage_backend_name: str

    async def complete_cleanup(self, cleanup_id: str) -> dict[str, Any] | None:
        """Complete a cleanup, reloading boundedly when another owner wins a CAS."""

        repository = UserErasureRepository(self.connection, self.clock)
        last_conflict: UserErasureConflictError | None = None
        for _attempt in range(_MAX_CAS_RELOADS):
            cleanup = await repository.get_cleanup(cleanup_id)
            if cleanup is None:
                return None
            try:
                return await self._complete_snapshot(repository, cleanup)
            except UserErasureConflictError as exc:
                last_conflict = exc
                continue
        assert last_conflict is not None
        raise last_conflict

    async def _complete_snapshot(
        self,
        repository: UserErasureRepository,
        cleanup: dict[str, Any],
    ) -> dict[str, Any]:
        cleanup_id = str(cleanup["cleanup_id"])
        record_version = int(cleanup["record_version"])
        try:
            targets = await repository.list_cleanup_targets(cleanup_id)
            for target in targets:
                if target["checkpoint_state"] != "pending":
                    continue
                evidence_hash, evidence_reference = await self._execute_target(
                    cleanup,
                    target,
                    expected_cleanup_version=record_version,
                )
                record_version = await repository.checkpoint_target(
                    cleanup_id=cleanup_id,
                    target_id=str(target["target_id"]),
                    expected_cleanup_version=record_version,
                    expected_target_version=int(target["row_version"]),
                    checkpoint_state="verified",
                    evidence_sha256=evidence_hash,
                    evidence_reference=evidence_reference,
                )

            targets = await repository.list_cleanup_targets(cleanup_id)
            transient_target = next(
                (
                    target
                    for target in targets
                    if target["target_kind"] == "transient_backend"
                    and target["checkpoint_state"] in {"verified", "decommissioned"}
                ),
                None,
            )
            if cleanup["cleanup_kind"] == "current" and transient_target is None:
                raise RuntimeError(
                    "Current erasure has no verified transient-backend checkpoint"
                )

            for revoked_job in await repository.list_revoked_jobs(cleanup_id):
                if revoked_job["purge_state"] != "pending":
                    continue
                assert transient_target is not None
                record_version = await repository.checkpoint_revoked_job(
                    cleanup_id=cleanup_id,
                    job_id=str(revoked_job["job_id"]),
                    expected_cleanup_version=record_version,
                    expected_job_version=int(revoked_job["row_version"]),
                    purge_state="verified",
                    evidence_sha256=str(transient_target["evidence_sha256"]),
                    evidence_reference=str(transient_target["evidence_reference"]),
                )

            targets = await repository.list_cleanup_targets(cleanup_id)
            revoked_jobs = await repository.list_revoked_jobs(cleanup_id)
            evidence_manifest = {
                "schema_version": 1,
                "cleanup_kind": str(cleanup["cleanup_kind"]),
                "protocol_version": int(cleanup["protocol_version"]),
                "targets": [
                    {
                        "target_kind": str(target["target_kind"]),
                        "backend_name": str(target["backend_name"]),
                        "checkpoint_state": str(target["checkpoint_state"]),
                        "evidence_sha256": target["evidence_sha256"],
                    }
                    for target in targets
                ],
                "revoked_jobs": [
                    {
                        "job_id": str(job["job_id"]),
                        "purge_state": str(job["purge_state"]),
                        "invalidated_execution_fence": int(
                            job["invalidated_execution_fence"]
                        ),
                        "evidence_sha256": job["evidence_sha256"],
                    }
                    for job in revoked_jobs
                ],
            }
            references = ["atagia:user-erasure:v1"]
            references.extend(
                sorted(
                    {
                        str(target["evidence_reference"])
                        for target in targets
                        if target["evidence_reference"]
                    }
                )[:31]
            )
            return await repository.finalize_cleanup(
                cleanup_id=cleanup_id,
                expected_record_version=record_version,
                evidence_manifest_sha256=canonical_json_hash(evidence_manifest),
                evidence_references=references,
            )
        except UserErasureConflictError:
            raise
        except Exception as exc:
            await self._record_error(repository, cleanup_id, record_version, exc)
            raise

    async def _execute_target(
        self,
        cleanup: dict[str, Any],
        target: dict[str, Any],
        *,
        expected_cleanup_version: int,
    ) -> tuple[str, str]:
        target_kind = str(target["target_kind"])
        if target_kind != "transient_backend":
            raise RuntimeError(
                f"Cleanup target {target_kind!r} requires explicit external evidence"
            )
        if cleanup["cleanup_kind"] != "current":
            raise RuntimeError(
                "Legacy destinations must be checkpointed from an explicit historical inventory"
            )
        backend_name = str(target["backend_name"])
        if backend_name != self.storage_backend_name:
            raise RuntimeError(
                "Erasure cleanup targets a different transient backend "
                f"({backend_name!r} != {self.storage_backend_name!r})"
            )
        lifecycle_epoch = str(cleanup["lifecycle_epoch"])
        lifecycle_cleanup_key = str(cleanup["lifecycle_cleanup_key"])
        if target.get("lifecycle_epoch") != lifecycle_epoch:
            raise RuntimeError(
                "Erasure target lifecycle epoch does not match its cleanup"
            )
        if str(target["target_key"]) != lifecycle_cleanup_key:
            raise RuntimeError("Erasure target does not name the captured cleanup key")

        indexed_notifications_purged = (
            await self.storage_backend.revoke_lifecycle_and_purge_notifications(
                lifecycle_cleanup_key,
                lifecycle_epoch,
                group_name=WORKER_GROUP_NAME,
            )
        )
        legacy_purge = await self._purge_legacy_transient_state_under_cleanup_fence(
            cleanup,
            target,
            expected_cleanup_version=expected_cleanup_version,
            database_path=await self._database_path(),
        )
        evidence = {
            "schema_version": 2,
            "operation": "lifecycle_revoke_and_legacy_cutover_purge",
            "backend_name": backend_name,
            "lifecycle_epoch": lifecycle_epoch,
            "indexed_notifications_purged": indexed_notifications_purged,
            "legacy_transient_purge": {
                **asdict(legacy_purge),
                "total_deleted": legacy_purge.total_deleted,
            },
        }
        return (
            canonical_json_hash(evidence),
            f"{backend_name}:lifecycle-revoke-and-purge:v2",
        )

    async def _purge_legacy_transient_state_under_cleanup_fence(
        self,
        cleanup: dict[str, Any],
        target: dict[str, Any],
        *,
        expected_cleanup_version: int,
        database_path: str,
    ) -> LegacyTransientPurgeResult:
        """Run the broad cutover purge only while this cleanup owns SQLite."""

        if self.connection.in_transaction:
            raise RuntimeError("Legacy transient purge requires a clean transaction")
        await self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.connection.execute(
                """
                SELECT 1
                FROM user_erasure_cleanups AS cleanup
                JOIN user_erasure_cleanup_targets AS target
                  ON target.cleanup_id = cleanup.cleanup_id
                JOIN user_lifecycles AS lifecycle
                  ON lifecycle.user_id = cleanup.candidate_user_id
                 AND lifecycle.lifecycle_epoch = cleanup.lifecycle_epoch
                 AND lifecycle.lifecycle_cleanup_key = cleanup.lifecycle_cleanup_key
                 AND lifecycle.erasure_cleanup_id = cleanup.cleanup_id
                JOIN deletion_tombstones AS tombstone
                  ON tombstone.id = cleanup.tombstone_id
                WHERE cleanup.cleanup_id = ?
                  AND cleanup.record_version = ?
                  AND cleanup.cleanup_kind = 'current'
                  AND cleanup.candidate_user_id = ?
                  AND cleanup.lifecycle_epoch = ?
                  AND cleanup.lifecycle_cleanup_key = ?
                  AND cleanup.canonical_deleted_at IS NOT NULL
                  AND target.target_id = ?
                  AND target.row_version = ?
                  AND target.target_kind = 'transient_backend'
                  AND target.backend_name = ?
                  AND target.target_key = cleanup.lifecycle_cleanup_key
                  AND target.lifecycle_epoch = cleanup.lifecycle_epoch
                  AND target.checkpoint_state = 'pending'
                  AND lifecycle.state = 'cleanup_pending'
                  AND tombstone.entity_type = 'user'
                  AND tombstone.deletion_reason = 'right_to_erasure'
                  AND tombstone.erasure_cleanup_state = 'pending'
                  AND tombstone.erasure_protocol_version = cleanup.protocol_version
                  AND tombstone.erasure_lifecycle_epoch = cleanup.lifecycle_epoch
                  AND NOT EXISTS (
                      SELECT 1
                      FROM users
                      WHERE users.id = cleanup.candidate_user_id
                  )
                LIMIT 1
                """,
                (
                    cleanup["cleanup_id"],
                    expected_cleanup_version,
                    cleanup["candidate_user_id"],
                    cleanup["lifecycle_epoch"],
                    cleanup["lifecycle_cleanup_key"],
                    target["target_id"],
                    target["row_version"],
                    target["backend_name"],
                ),
            )
            current = await cursor.fetchone()
            await cursor.close()
            if current is None:
                raise UserErasureConflictError(
                    "Legacy transient purge no longer owns the captured cleanup"
                )
            result = await self.storage_backend.purge_legacy_transient_state(
                database_path,
                str(cleanup["candidate_user_id"]),
            )
            if not result.clean:
                raise RuntimeError(
                    "Legacy transient cleanup found malformed candidates that "
                    "cannot be attributed safely"
                )
            await self.connection.commit()
            return result
        except BaseException:
            await self.connection.rollback()
            raise

    async def _database_path(self) -> str:
        cursor = await self.connection.execute("PRAGMA database_list")
        rows = await cursor.fetchall()
        await cursor.close()
        for row in rows:
            if str(row["name"]) == "main":
                path = str(row["file"] or "")
                return path or ":memory:"
        raise RuntimeError("SQLite main database path is unavailable")

    @staticmethod
    async def _record_error(
        repository: UserErasureRepository,
        cleanup_id: str,
        record_version: int,
        exc: Exception,
    ) -> None:
        try:
            await repository.record_cleanup_error(
                cleanup_id,
                expected_record_version=record_version,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            logger.warning(
                "Could not persist user-erasure cleanup diagnostics for %s",
                cleanup_id,
                exc_info=True,
            )


async def recover_pending_user_erasures(
    connection: aiosqlite.Connection,
    clock: Clock,
    storage_backend: StorageBackend,
    *,
    storage_backend_name: str,
    limit: int = 100,
) -> ErasureRecoveryResult:
    """Attempt bounded startup recovery without making canonical data accessible."""

    repository = UserErasureRepository(connection, clock)
    cleanups = await repository.list_resumable_cleanups(limit=limit)
    service = UserErasureCleanupService(
        connection,
        clock,
        storage_backend,
        storage_backend_name,
    )
    completed = 0
    pending = 0
    for cleanup in cleanups:
        cleanup_id = str(cleanup["cleanup_id"])
        try:
            await service.complete_cleanup(cleanup_id)
            completed += 1
        except Exception:
            pending += 1
            logger.warning(
                "User-erasure cleanup remains pending after startup recovery: %s",
                cleanup_id,
                exc_info=True,
            )
    return ErasureRecoveryResult(completed=completed, pending=pending)
