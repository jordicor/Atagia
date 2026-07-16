"""Durable SQLite coordination for lifecycle-fenced user erasure."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from atagia.core import json_utils
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, user_erasure_marker_hash


ERASURE_PROTOCOL_VERSION = 1
_NONTERMINAL_JOB_STATUSES = (
    "queued",
    "awaiting_claim",
    "running",
    "retrying",
    "deferred",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_TARGETS = 1_024
_MAX_EVIDENCE_REFERENCES = 32
_MAX_EVIDENCE_REFERENCE_LENGTH = 512
_MAX_TARGET_KEY_LENGTH = 2_048
_MAX_TARGET_COMPONENT_LENGTH = 128


class UserErasureConflictError(RuntimeError):
    """The requested erasure operation no longer owns the expected SQLite state."""


class UserErasureNotReadyError(RuntimeError):
    """A cleanup cannot be finalized because durable checkpoints remain open."""


class LegacyErasureReconciliationError(RuntimeError):
    """Legacy cleanup proof is incomplete or does not match the retained marker."""


@dataclass(frozen=True, slots=True)
class ErasureCleanupTargetSpec:
    """One concrete external or derived destination that must be verified clean."""

    target_kind: str
    target_key: str
    backend_name: str = ""
    lifecycle_epoch: str | None = None


@dataclass(frozen=True, slots=True)
class ErasurePreparation:
    """Coordinates committed by the canonical erasure transaction."""

    cleanup_id: str
    tombstone_id: str
    lifecycle_epoch: str | None
    lifecycle_cleanup_key: str
    record_version: int
    revoked_job_count: int
    resumed: bool = False


CanonicalDelete = Callable[[], Awaitable[None]]
ScopeCountsProvider = Callable[[], Awaitable[Mapping[str, int]]]
TransactionPrecondition = Callable[[], Awaitable[None]]
Failpoint = Callable[[str], None]


class UserErasureRepository(BaseRepository):
    """Prepare, resume, checkpoint, finalize, and retire durable erasures.

    Current-protocol preparation owns one ``BEGIN IMMEDIATE`` transaction.  An
    optional transaction precondition runs after that writer lock is acquired
    and before preparation reads or writes canonical state.  ``canonical_delete``
    then runs inside the same transaction and must remove the canonical
    ``users`` row after any caller-owned child/audit cleanup.  If it returns
    while the row remains, the whole preparation rolls back.
    """

    async def prepare_current_erasure(
        self,
        *,
        user_id: str,
        scope_counts: Mapping[str, int] | None,
        target_specs: Sequence[ErasureCleanupTargetSpec],
        canonical_delete: CanonicalDelete,
        scope_counts_provider: ScopeCountsProvider | None = None,
        transaction_precondition: TransactionPrecondition | None = None,
        expected_lifecycle_epoch: str | None = None,
        cleanup_id: str | None = None,
        tombstone_id: str | None = None,
        failpoint: Failpoint | None = None,
    ) -> ErasurePreparation:
        """Delete canonical state, revoke old work, and persist resume evidence."""

        self._require_no_open_transaction()
        resolved_cleanup_id = cleanup_id or generate_prefixed_id("erc")
        resolved_tombstone_id = tombstone_id or generate_prefixed_id("tmb")
        validated_targets = self._validate_target_specs(target_specs)
        timestamp = self._timestamp()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            if transaction_precondition is not None:
                await transaction_precondition()
            existing = await self._cleanup_for_candidate(user_id)
            if existing is not None:
                if existing.get("canonical_deleted_at") is None:
                    raise UserErasureConflictError(
                        "A cleanup record exists without sealed canonical deletion"
                    )
                await self._connection.commit()
                return ErasurePreparation(
                    cleanup_id=str(existing["cleanup_id"]),
                    tombstone_id=str(existing["tombstone_id"]),
                    lifecycle_epoch=_optional_text(existing.get("lifecycle_epoch")),
                    lifecycle_cleanup_key=str(existing["lifecycle_cleanup_key"]),
                    record_version=int(existing["record_version"]),
                    revoked_job_count=await self._revoked_job_count(
                        str(existing["cleanup_id"])
                    ),
                    resumed=True,
                )
            await self._raise_for_retained_marker(user_id)
            lifecycle = await self._active_lifecycle(user_id)
            if lifecycle is None:
                raise UserErasureConflictError(
                    "Cannot prepare erasure without one active user lifecycle"
                )
            lifecycle_epoch = str(lifecycle["lifecycle_epoch"])
            lifecycle_cleanup_key = str(lifecycle["lifecycle_cleanup_key"])
            if (
                expected_lifecycle_epoch is not None
                and lifecycle_epoch != expected_lifecycle_epoch
            ):
                raise UserErasureConflictError(
                    "The active user lifecycle changed before erasure preparation"
                )
            if scope_counts is not None and scope_counts_provider is not None:
                raise ValueError(
                    "Provide either scope_counts or scope_counts_provider, not both"
                )
            resolved_scope_counts = (
                await scope_counts_provider()
                if scope_counts_provider is not None
                else scope_counts or {}
            )
            scope_summary = self._scope_summary(user_id, resolved_scope_counts)
            await self._connection.execute(
                """
                INSERT INTO deletion_tombstones(
                    id,
                    entity_type,
                    deleted_at,
                    deletion_reason,
                    deleted_by,
                    scope_summary,
                    erasure_protocol_version,
                    erasure_cleanup_state,
                    cleanup_verified_at,
                    cleanup_evidence_manifest_sha256,
                    cleanup_evidence_references_json,
                    erasure_lifecycle_epoch,
                    legacy_reconciled,
                    erasure_row_version
                ) VALUES (?, 'user', ?, 'right_to_erasure', 'system', ?, ?,
                          'pending', NULL, NULL, '[]', ?, 0, 0)
                """,
                (
                    resolved_tombstone_id,
                    timestamp,
                    json_utils.dumps(scope_summary, sort_keys=True),
                    ERASURE_PROTOCOL_VERSION,
                    lifecycle_epoch,
                ),
            )
            await self._connection.execute(
                """
                INSERT INTO user_erasure_cleanups(
                    cleanup_id,
                    tombstone_id,
                    cleanup_kind,
                    candidate_user_id,
                    user_id_sha256,
                    lifecycle_epoch,
                    lifecycle_cleanup_key,
                    protocol_version,
                    inventory_manifest_sha256,
                    canonical_deleted_at,
                    attempt_count,
                    last_attempt_at,
                    last_error,
                    record_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, 'current', ?, ?, ?, ?, ?, NULL, NULL, 0, NULL,
                          NULL, 0, ?, ?)
                """,
                (
                    resolved_cleanup_id,
                    resolved_tombstone_id,
                    user_id,
                    user_erasure_marker_hash(user_id),
                    lifecycle_epoch,
                    lifecycle_cleanup_key,
                    ERASURE_PROTOCOL_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            await self._insert_targets(
                cleanup_id=resolved_cleanup_id,
                target_specs=validated_targets,
                default_lifecycle_epoch=lifecycle_epoch,
                timestamp=timestamp,
            )
            self._call_failpoint(failpoint, "cleanup_prepared")
            revoked_job_count = await self._revoke_nonterminal_jobs(
                cleanup_id=resolved_cleanup_id,
                user_id=user_id,
                lifecycle_epoch=lifecycle_epoch,
                timestamp=timestamp,
            )
            lifecycle_cursor = await self._connection.execute(
                """
                UPDATE user_lifecycles
                SET state = 'cleanup_pending',
                    cache_revision = cache_revision + 1,
                    source_revision = source_revision + 1,
                    erasure_cleanup_id = ?,
                    revoked_at = ?,
                    cleanup_completed_at = NULL,
                    last_cleanup_error = NULL,
                    updated_at = ?
                WHERE user_id = ?
                  AND lifecycle_epoch = ?
                  AND state = 'active'
                  AND erasure_cleanup_id IS NULL
                """,
                (
                    resolved_cleanup_id,
                    timestamp,
                    timestamp,
                    user_id,
                    lifecycle_epoch,
                ),
            )
            if int(lifecycle_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError(
                    "The active lifecycle was lost while preparing erasure"
                )
            self._call_failpoint(failpoint, "jobs_revoked")
            await canonical_delete()
            remaining = await self._fetch_raw_one(
                "SELECT 1 AS found FROM users WHERE id = ? LIMIT 1",
                (user_id,),
            )
            if remaining is not None:
                raise UserErasureConflictError(
                    "Canonical erasure callback returned without deleting the user"
                )
            await self._assert_revocations_durable(
                cleanup_id=resolved_cleanup_id,
                user_id=user_id,
                lifecycle_epoch=lifecycle_epoch,
                expected_count=revoked_job_count,
            )
            cleanup_cursor = await self._connection.execute(
                """
                UPDATE user_erasure_cleanups
                SET canonical_deleted_at = ?,
                    record_version = record_version + 1,
                    updated_at = ?
                WHERE cleanup_id = ?
                  AND record_version = 0
                  AND canonical_deleted_at IS NULL
                """,
                (timestamp, timestamp, resolved_cleanup_id),
            )
            if int(cleanup_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError(
                    "Erasure cleanup preparation lost its record-version fence"
                )
            self._call_failpoint(failpoint, "canonical_delete_sealed")
            await self._connection.commit()
            return ErasurePreparation(
                cleanup_id=resolved_cleanup_id,
                tombstone_id=resolved_tombstone_id,
                lifecycle_epoch=lifecycle_epoch,
                lifecycle_cleanup_key=lifecycle_cleanup_key,
                record_version=1,
                revoked_job_count=revoked_job_count,
            )
        except BaseException:
            await self._connection.rollback()
            raise

    async def prepare_legacy_reconciliation(
        self,
        *,
        tombstone_id: str,
        candidate_user_id: str,
        inventory_manifest_sha256: str,
        historical_inventory_complete: bool,
        target_specs: Sequence[ErasureCleanupTargetSpec],
        cleanup_id: str | None = None,
    ) -> ErasurePreparation:
        """Open conservative cleanup only for a hash-matching legacy marker."""

        self._require_no_open_transaction()
        if not historical_inventory_complete:
            raise LegacyErasureReconciliationError(
                "Legacy reconciliation requires a complete historical inventory"
            )
        inventory_hash = self._validate_sha256(
            inventory_manifest_sha256,
            field_name="inventory_manifest_sha256",
        )
        validated_targets = self._validate_target_specs(target_specs)
        if not validated_targets:
            raise LegacyErasureReconciliationError(
                "Legacy reconciliation requires cleanup or decommission targets"
            )
        resolved_cleanup_id = cleanup_id or generate_prefixed_id("erc")
        cleanup_key = generate_prefixed_id("ulk")
        timestamp = self._timestamp()
        marker_hash = user_erasure_marker_hash(candidate_user_id)
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            marker = await self._fetch_raw_one(
                """
                SELECT *
                FROM deletion_tombstones
                WHERE id = ?
                  AND entity_type = 'user'
                  AND deletion_reason = 'right_to_erasure'
                """,
                (tombstone_id,),
            )
            if marker is None or marker["erasure_cleanup_state"] != "legacy_unknown":
                raise LegacyErasureReconciliationError(
                    "The selected tombstone is not an unreconciled legacy marker"
                )
            scope = self._decode_object(marker.get("scope_summary"))
            if scope.get("user_id_sha256") != marker_hash:
                raise LegacyErasureReconciliationError(
                    "Candidate identifier does not match the legacy marker hash"
                )
            if (
                await self._fetch_raw_one(
                    "SELECT 1 AS found FROM users WHERE id = ? LIMIT 1",
                    (candidate_user_id,),
                )
                is not None
            ):
                raise LegacyErasureReconciliationError(
                    "Legacy reconciliation cannot target a current canonical user"
                )
            if (
                await self._fetch_raw_one(
                    "SELECT 1 AS found FROM user_lifecycles WHERE user_id = ? LIMIT 1",
                    (candidate_user_id,),
                )
                is not None
            ):
                raise LegacyErasureReconciliationError(
                    "Legacy reconciliation found an unresolved SQLite lifecycle"
                )
            existing = await self._fetch_raw_one(
                """
                SELECT *
                FROM user_erasure_cleanups
                WHERE tombstone_id = ?
                   OR candidate_user_id = ?
                LIMIT 1
                """,
                (tombstone_id, candidate_user_id),
            )
            if existing is not None:
                if (
                    str(existing["tombstone_id"]) != tombstone_id
                    or str(existing["user_id_sha256"]) != marker_hash
                ):
                    raise LegacyErasureReconciliationError(
                        "A different legacy reconciliation already owns this identity"
                    )
                await self._connection.commit()
                return ErasurePreparation(
                    cleanup_id=str(existing["cleanup_id"]),
                    tombstone_id=tombstone_id,
                    lifecycle_epoch=None,
                    lifecycle_cleanup_key=str(existing["lifecycle_cleanup_key"]),
                    record_version=int(existing["record_version"]),
                    revoked_job_count=0,
                    resumed=True,
                )
            await self._connection.execute(
                """
                INSERT INTO user_erasure_cleanups(
                    cleanup_id,
                    tombstone_id,
                    cleanup_kind,
                    candidate_user_id,
                    user_id_sha256,
                    lifecycle_epoch,
                    lifecycle_cleanup_key,
                    protocol_version,
                    inventory_manifest_sha256,
                    canonical_deleted_at,
                    attempt_count,
                    last_attempt_at,
                    last_error,
                    record_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, 'legacy_reconciliation', ?, ?, NULL, ?, ?, ?, ?,
                          0, NULL, NULL, 0, ?, ?)
                """,
                (
                    resolved_cleanup_id,
                    tombstone_id,
                    candidate_user_id,
                    marker_hash,
                    cleanup_key,
                    ERASURE_PROTOCOL_VERSION,
                    inventory_hash,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            await self._insert_targets(
                cleanup_id=resolved_cleanup_id,
                target_specs=validated_targets,
                default_lifecycle_epoch=None,
                timestamp=timestamp,
            )
            await self._connection.commit()
            return ErasurePreparation(
                cleanup_id=resolved_cleanup_id,
                tombstone_id=tombstone_id,
                lifecycle_epoch=None,
                lifecycle_cleanup_key=cleanup_key,
                record_version=0,
                revoked_job_count=0,
            )
        except BaseException:
            await self._connection.rollback()
            raise

    async def get_cleanup(self, cleanup_id: str) -> dict[str, Any] | None:
        return await self._cleanup_by_id(cleanup_id)

    async def get_erasure_state_for_candidate(
        self,
        candidate_user_id: str,
    ) -> dict[str, Any] | None:
        marker_hash = user_erasure_marker_hash(candidate_user_id)
        return await self._fetch_raw_one(
            """
            SELECT
                tombstone.id AS tombstone_id,
                tombstone.deleted_at,
                tombstone.erasure_protocol_version,
                tombstone.erasure_cleanup_state,
                tombstone.cleanup_verified_at,
                tombstone.erasure_lifecycle_epoch,
                tombstone.legacy_reconciled,
                tombstone.erasure_row_version,
                tombstone.scope_summary,
                cleanup.cleanup_id,
                cleanup.cleanup_kind,
                cleanup.record_version,
                cleanup.last_error
            FROM deletion_tombstones AS tombstone
            LEFT JOIN user_erasure_cleanups AS cleanup
              ON cleanup.tombstone_id = tombstone.id
            WHERE tombstone.entity_type = 'user'
              AND tombstone.deletion_reason = 'right_to_erasure'
              AND json_extract(tombstone.scope_summary, '$.user_id_sha256') = ?
            ORDER BY tombstone.deleted_at DESC, tombstone.id ASC
            LIMIT 1
            """,
            (marker_hash,),
        )

    async def list_resumable_cleanups(
        self, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return await self._fetch_raw_all(
            """
            SELECT *
            FROM user_erasure_cleanups
            WHERE canonical_deleted_at IS NOT NULL
            ORDER BY updated_at ASC, cleanup_id ASC
            LIMIT ?
            """,
            (min(limit, 1_000),),
        )

    async def list_cleanup_targets(self, cleanup_id: str) -> list[dict[str, Any]]:
        return await self._fetch_raw_all(
            """
            SELECT *
            FROM user_erasure_cleanup_targets
            WHERE cleanup_id = ?
            ORDER BY target_kind ASC, backend_name ASC, target_id ASC
            """,
            (cleanup_id,),
        )

    async def list_revoked_jobs(self, cleanup_id: str) -> list[dict[str, Any]]:
        return await self._fetch_raw_all(
            """
            SELECT *
            FROM user_erasure_revoked_jobs
            WHERE cleanup_id = ?
            ORDER BY job_id ASC
            """,
            (cleanup_id,),
        )

    async def checkpoint_target(
        self,
        *,
        cleanup_id: str,
        target_id: str,
        expected_cleanup_version: int,
        expected_target_version: int,
        checkpoint_state: str,
        evidence_sha256: str,
        evidence_reference: str,
    ) -> int:
        """CAS one external target checkpoint and return the new cleanup version."""

        if checkpoint_state not in {"verified", "decommissioned"}:
            raise ValueError("checkpoint_state must be verified or decommissioned")
        evidence_hash = self._validate_sha256(
            evidence_sha256, field_name="evidence_sha256"
        )
        reference = self._validate_evidence_reference(evidence_reference)
        timestamp = self._timestamp()
        self._require_no_open_transaction()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            target_cursor = await self._connection.execute(
                """
                UPDATE user_erasure_cleanup_targets
                SET checkpoint_state = ?,
                    evidence_sha256 = ?,
                    evidence_reference = ?,
                    verified_at = ?,
                    row_version = row_version + 1,
                    updated_at = ?
                WHERE target_id = ?
                  AND cleanup_id = ?
                  AND checkpoint_state = 'pending'
                  AND row_version = ?
                """,
                (
                    checkpoint_state,
                    evidence_hash,
                    reference,
                    timestamp,
                    timestamp,
                    target_id,
                    cleanup_id,
                    expected_target_version,
                ),
            )
            if int(target_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError("Cleanup target checkpoint lost its CAS")
            new_version = await self._bump_cleanup_version(
                cleanup_id,
                expected_cleanup_version=expected_cleanup_version,
                timestamp=timestamp,
            )
            await self._connection.commit()
            return new_version
        except BaseException:
            await self._connection.rollback()
            raise

    async def checkpoint_revoked_job(
        self,
        *,
        cleanup_id: str,
        job_id: str,
        expected_cleanup_version: int,
        expected_job_version: int,
        purge_state: str,
        evidence_sha256: str,
        evidence_reference: str,
    ) -> int:
        """CAS exact notification-purge proof and return the new cleanup version."""

        if purge_state not in {"verified", "decommissioned"}:
            raise ValueError("purge_state must be verified or decommissioned")
        evidence_hash = self._validate_sha256(
            evidence_sha256, field_name="evidence_sha256"
        )
        reference = self._validate_evidence_reference(evidence_reference)
        timestamp = self._timestamp()
        self._require_no_open_transaction()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            job_cursor = await self._connection.execute(
                """
                UPDATE user_erasure_revoked_jobs
                SET purge_state = ?,
                    evidence_sha256 = ?,
                    evidence_reference = ?,
                    purged_at = ?,
                    row_version = row_version + 1
                WHERE cleanup_id = ?
                  AND job_id = ?
                  AND purge_state = 'pending'
                  AND row_version = ?
                """,
                (
                    purge_state,
                    evidence_hash,
                    reference,
                    timestamp,
                    cleanup_id,
                    job_id,
                    expected_job_version,
                ),
            )
            if int(job_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError("Revoked-job checkpoint lost its CAS")
            new_version = await self._bump_cleanup_version(
                cleanup_id,
                expected_cleanup_version=expected_cleanup_version,
                timestamp=timestamp,
            )
            await self._connection.commit()
            return new_version
        except BaseException:
            await self._connection.rollback()
            raise

    async def record_cleanup_error(
        self,
        cleanup_id: str,
        *,
        expected_record_version: int,
        error_message: str,
    ) -> int:
        """Persist bounded retry diagnostics without losing the resume record."""

        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE user_erasure_cleanups
            SET attempt_count = attempt_count + 1,
                last_attempt_at = ?,
                last_error = ?,
                record_version = record_version + 1,
                updated_at = ?
            WHERE cleanup_id = ?
              AND record_version = ?
            RETURNING record_version
            """,
            (
                timestamp,
                str(error_message)[:500],
                timestamp,
                cleanup_id,
                expected_record_version,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            await self._connection.rollback()
            raise UserErasureConflictError("Cleanup error update lost its CAS")
        await self._connection.commit()
        return int(row["record_version"])

    async def finalize_cleanup(
        self,
        *,
        cleanup_id: str,
        expected_record_version: int,
        evidence_manifest_sha256: str,
        evidence_references: Sequence[str],
        failpoint: Failpoint | None = None,
    ) -> dict[str, Any]:
        """Atomically verify the tombstone and remove all temporary resume data."""

        evidence_hash = self._validate_sha256(
            evidence_manifest_sha256,
            field_name="evidence_manifest_sha256",
        )
        references_json = json_utils.dumps(
            self._validate_evidence_references(evidence_references),
            sort_keys=True,
        )
        self._require_no_open_transaction()
        timestamp = self._timestamp()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            cleanup = await self._cleanup_by_id(cleanup_id)
            if (
                cleanup is None
                or int(cleanup["record_version"]) != expected_record_version
            ):
                raise UserErasureConflictError(
                    "Cleanup finalization lost its record CAS"
                )
            if cleanup.get("canonical_deleted_at") is None:
                raise UserErasureNotReadyError("Canonical user deletion is not sealed")
            if (
                cleanup["cleanup_kind"] == "legacy_reconciliation"
                and cleanup.get("inventory_manifest_sha256") is None
            ):
                raise UserErasureNotReadyError(
                    "Legacy historical-inventory evidence is missing"
                )
            if (
                await self._fetch_raw_one(
                    "SELECT 1 AS found FROM users WHERE id = ? LIMIT 1",
                    (cleanup["candidate_user_id"],),
                )
                is not None
            ):
                raise UserErasureNotReadyError("Canonical user data is still present")
            marker = await self._fetch_raw_one(
                "SELECT * FROM deletion_tombstones WHERE id = ?",
                (cleanup["tombstone_id"],),
            )
            if marker is None:
                raise UserErasureConflictError("Cleanup tombstone no longer exists")
            expected_marker_state = (
                "pending" if cleanup["cleanup_kind"] == "current" else "legacy_unknown"
            )
            if marker["erasure_cleanup_state"] != expected_marker_state:
                raise UserErasureConflictError("Cleanup tombstone state changed")
            await self._assert_all_checkpoints_complete(cleanup_id)
            await self._assert_no_recoverable_jobs(cleanup)
            if cleanup["cleanup_kind"] == "current":
                lifecycle = await self._fetch_raw_one(
                    """
                    SELECT 1 AS found
                    FROM user_lifecycles
                    WHERE user_id = ?
                      AND lifecycle_epoch = ?
                      AND lifecycle_cleanup_key = ?
                      AND erasure_cleanup_id = ?
                      AND state = 'cleanup_pending'
                    LIMIT 1
                    """,
                    (
                        cleanup["candidate_user_id"],
                        cleanup["lifecycle_epoch"],
                        cleanup["lifecycle_cleanup_key"],
                        cleanup_id,
                    ),
                )
                if lifecycle is None:
                    raise UserErasureConflictError(
                        "Cleanup no longer owns the captured user lifecycle"
                    )
            elif (
                await self._fetch_raw_one(
                    "SELECT 1 AS found FROM user_lifecycles WHERE user_id = ? LIMIT 1",
                    (cleanup["candidate_user_id"],),
                )
                is not None
            ):
                raise UserErasureNotReadyError(
                    "Legacy reconciliation still has an unresolved lifecycle"
                )
            self._call_failpoint(failpoint, "checkpoint_validation_complete")
            marker_cursor = await self._connection.execute(
                """
                UPDATE deletion_tombstones
                SET erasure_protocol_version = ?,
                    erasure_cleanup_state = 'verified',
                    cleanup_verified_at = ?,
                    cleanup_evidence_manifest_sha256 = ?,
                    cleanup_evidence_references_json = ?,
                    erasure_lifecycle_epoch = ?,
                    legacy_reconciled = ?,
                    erasure_row_version = erasure_row_version + 1
                WHERE id = ?
                  AND erasure_row_version = ?
                  AND erasure_cleanup_state = ?
                """,
                (
                    ERASURE_PROTOCOL_VERSION,
                    timestamp,
                    evidence_hash,
                    references_json,
                    cleanup["lifecycle_epoch"],
                    1 if cleanup["cleanup_kind"] == "legacy_reconciliation" else 0,
                    cleanup["tombstone_id"],
                    marker["erasure_row_version"],
                    expected_marker_state,
                ),
            )
            if int(marker_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError("Tombstone verification lost its CAS")
            self._call_failpoint(failpoint, "tombstone_verified")
            if cleanup["cleanup_kind"] == "current":
                await self._connection.execute(
                    """
                    DELETE FROM worker_job_runs
                    WHERE user_id = ?
                      AND lifecycle_epoch = ?
                    """,
                    (cleanup["candidate_user_id"], cleanup["lifecycle_epoch"]),
                )
            cleanup_cursor = await self._connection.execute(
                """
                DELETE FROM user_erasure_cleanups
                WHERE cleanup_id = ?
                  AND record_version = ?
                """,
                (cleanup_id, expected_record_version),
            )
            if int(cleanup_cursor.rowcount or 0) != 1:
                raise UserErasureConflictError("Cleanup deletion lost its CAS")
            self._call_failpoint(failpoint, "cleanup_record_deleted")
            if cleanup["cleanup_kind"] == "current":
                lifecycle_cursor = await self._connection.execute(
                    """
                    DELETE FROM user_lifecycles
                    WHERE user_id = ?
                      AND lifecycle_epoch = ?
                      AND erasure_cleanup_id = ?
                      AND state = 'cleanup_pending'
                    """,
                    (
                        cleanup["candidate_user_id"],
                        cleanup["lifecycle_epoch"],
                        cleanup_id,
                    ),
                )
                if int(lifecycle_cursor.rowcount or 0) != 1:
                    raise UserErasureConflictError("Lifecycle retirement lost its CAS")
            await self._connection.commit()
            verified = await self._fetch_raw_one(
                "SELECT * FROM deletion_tombstones WHERE id = ?",
                (cleanup["tombstone_id"],),
            )
            if verified is None:  # pragma: no cover - guarded by the transaction above.
                raise RuntimeError("Verified erasure tombstone disappeared")
            return verified
        except BaseException:
            await self._connection.rollback()
            raise

    async def list_legacy_unknown_tombstones(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return await self._fetch_raw_all(
            """
            SELECT tombstone.*
            FROM deletion_tombstones AS tombstone
            WHERE tombstone.entity_type = 'user'
              AND tombstone.deletion_reason = 'right_to_erasure'
              AND tombstone.erasure_cleanup_state = 'legacy_unknown'
              AND NOT EXISTS (
                  SELECT 1
                  FROM user_erasure_cleanups AS cleanup
                  WHERE cleanup.tombstone_id = tombstone.id
              )
            ORDER BY tombstone.deleted_at ASC, tombstone.id ASC
            LIMIT ?
            """,
            (min(limit, 1_000),),
        )

    async def list_retention_eligible_tombstones(
        self,
        *,
        deleted_before: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return await self._fetch_raw_all(
            self._retention_eligible_sql(select_clause="tombstone.*")
            + " ORDER BY tombstone.deleted_at ASC, tombstone.id ASC LIMIT ?",
            (deleted_before, min(limit, 1_000)),
        )

    async def retire_tombstone(
        self,
        tombstone_id: str,
        *,
        expected_row_version: int,
        deleted_before: str,
    ) -> bool:
        """Retire only a cleanup-proven marker whose retention period elapsed."""

        cursor = await self._connection.execute(
            """
            DELETE FROM deletion_tombstones
            WHERE id IN (
                SELECT tombstone.id
                FROM deletion_tombstones AS tombstone
                WHERE tombstone.id = ?
                  AND tombstone.erasure_row_version = ?
                  AND tombstone.deleted_at < ?
                  AND tombstone.entity_type = 'user'
                  AND tombstone.deletion_reason = 'right_to_erasure'
                  AND tombstone.erasure_protocol_version = ?
                  AND tombstone.erasure_cleanup_state = 'verified'
                  AND tombstone.cleanup_verified_at IS NOT NULL
                  AND tombstone.cleanup_evidence_manifest_sha256 IS NOT NULL
                  AND (
                      tombstone.erasure_lifecycle_epoch IS NOT NULL
                      OR tombstone.legacy_reconciled = 1
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM user_erasure_cleanups AS cleanup
                      WHERE cleanup.tombstone_id = tombstone.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM user_lifecycles AS lifecycle
                      WHERE tombstone.erasure_lifecycle_epoch IS NOT NULL
                        AND lifecycle.lifecycle_epoch = tombstone.erasure_lifecycle_epoch
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM worker_job_runs AS job
                      WHERE tombstone.erasure_lifecycle_epoch IS NOT NULL
                        AND job.lifecycle_epoch = tombstone.erasure_lifecycle_epoch
                        AND (
                            job.status IN (
                                'queued', 'awaiting_claim', 'running', 'retrying', 'deferred'
                            )
                            OR job.recovery_envelope_json IS NOT NULL
                        )
                  )
            )
            """,
            (
                tombstone_id,
                expected_row_version,
                deleted_before,
                ERASURE_PROTOCOL_VERSION,
            ),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0) == 1

    async def _revoke_nonterminal_jobs(
        self,
        *,
        cleanup_id: str,
        user_id: str,
        lifecycle_epoch: str,
        timestamp: str,
    ) -> int:
        placeholders = ", ".join("?" for _ in _NONTERMINAL_JOB_STATUSES)
        await self._connection.execute(
            f"""
            INSERT INTO user_erasure_revoked_jobs(
                cleanup_id,
                job_id,
                conversation_id,
                stream_name,
                target_backend,
                dispatch_token,
                prior_status,
                invalidated_execution_fence,
                purge_state,
                evidence_sha256,
                evidence_reference,
                invalidated_at,
                purged_at,
                row_version
            )
            SELECT
                ?,
                job_id,
                conversation_id,
                stream_name,
                target_backend,
                dispatch_token,
                status,
                execution_fence + 1,
                CASE WHEN dispatch_token IS NULL THEN 'not_required' ELSE 'pending' END,
                NULL,
                CASE WHEN dispatch_token IS NULL THEN 'not_dispatched' ELSE NULL END,
                ?,
                CASE WHEN dispatch_token IS NULL THEN ? ELSE NULL END,
                0
            FROM worker_job_runs
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND status IN ({placeholders})
            """,
            (
                cleanup_id,
                timestamp,
                timestamp,
                user_id,
                lifecycle_epoch,
                *_NONTERMINAL_JOB_STATUSES,
            ),
        )
        count = await self._revoked_job_count(cleanup_id)
        cursor = await self._connection.execute(
            f"""
            UPDATE worker_job_runs
            SET status = 'cancelled',
                conversation_id = NULL,
                finished_at = ?,
                last_heartbeat_at = ?,
                error_class = 'LifecycleRevoked',
                error_message = 'captured user lifecycle was revoked by erasure',
                terminal_diagnostics_json = json_object(
                    'reason', 'user_erasure',
                    'cleanup_id', ?,
                    'revoked_execution_fence', execution_fence + 1
                ),
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL,
                execution_fence = execution_fence + 1
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND status IN ({placeholders})
            """,
            (
                timestamp,
                timestamp,
                cleanup_id,
                user_id,
                lifecycle_epoch,
                *_NONTERMINAL_JOB_STATUSES,
            ),
        )
        if int(cursor.rowcount or 0) != count:
            raise UserErasureConflictError(
                "Durable job revocation changed while capturing purge coordinates"
            )
        return count

    async def _assert_all_checkpoints_complete(self, cleanup_id: str) -> None:
        open_target = await self._fetch_raw_one(
            """
            SELECT 1 AS found
            FROM user_erasure_cleanup_targets
            WHERE cleanup_id = ?
              AND (
                  checkpoint_state = 'pending'
                  OR evidence_sha256 IS NULL
                  OR evidence_reference IS NULL
                  OR verified_at IS NULL
              )
            LIMIT 1
            """,
            (cleanup_id,),
        )
        if open_target is not None:
            raise UserErasureNotReadyError("External cleanup targets remain pending")
        open_job = await self._fetch_raw_one(
            """
            SELECT 1 AS found
            FROM user_erasure_revoked_jobs
            WHERE cleanup_id = ?
              AND (
                  purge_state = 'pending'
                  OR purged_at IS NULL
                  OR (
                      purge_state IN ('verified', 'decommissioned')
                      AND (
                          evidence_sha256 IS NULL
                          OR evidence_reference IS NULL
                      )
                  )
              )
            LIMIT 1
            """,
            (cleanup_id,),
        )
        if open_job is not None:
            raise UserErasureNotReadyError("Revoked job notifications remain pending")

    async def _assert_revocations_durable(
        self,
        *,
        cleanup_id: str,
        user_id: str,
        lifecycle_epoch: str,
        expected_count: int,
    ) -> None:
        row = await self._fetch_raw_one(
            """
            SELECT COUNT(*) AS count
            FROM user_erasure_revoked_jobs AS revoked
            JOIN worker_job_runs AS job
              ON job.job_id = revoked.job_id
             AND job.user_id = ?
             AND job.lifecycle_epoch = ?
             AND job.status = 'cancelled'
             AND job.execution_fence = revoked.invalidated_execution_fence
             AND job.recovery_envelope_json IS NULL
             AND job.envelope_schema_version IS NULL
             AND job.dispatch_token IS NULL
             AND job.execution_owner IS NULL
             AND job.execution_lease_expires_at IS NULL
            WHERE revoked.cleanup_id = ?
            """,
            (user_id, lifecycle_epoch, cleanup_id),
        )
        durable_count = int(row["count"]) if row is not None else 0
        if durable_count != expected_count:
            raise UserErasureConflictError(
                "Canonical deletion removed or changed a durably revoked job"
            )

    async def _assert_no_recoverable_jobs(self, cleanup: Mapping[str, Any]) -> None:
        lifecycle_epoch = cleanup.get("lifecycle_epoch")
        if lifecycle_epoch is None:
            return
        placeholders = ", ".join("?" for _ in _NONTERMINAL_JOB_STATUSES)
        recoverable = await self._fetch_raw_one(
            f"""
            SELECT 1 AS found
            FROM worker_job_runs
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND (
                  status IN ({placeholders})
                  OR recovery_envelope_json IS NOT NULL
                  OR envelope_schema_version IS NOT NULL
                  OR dispatch_token IS NOT NULL
                  OR execution_owner IS NOT NULL
                  OR execution_lease_expires_at IS NOT NULL
              )
            LIMIT 1
            """,
            (
                cleanup["candidate_user_id"],
                lifecycle_epoch,
                *_NONTERMINAL_JOB_STATUSES,
            ),
        )
        if recoverable is not None:
            raise UserErasureNotReadyError(
                "An old-lifecycle job still has recoverable or executable state"
            )

    async def _insert_targets(
        self,
        *,
        cleanup_id: str,
        target_specs: Sequence[ErasureCleanupTargetSpec],
        default_lifecycle_epoch: str | None,
        timestamp: str,
    ) -> None:
        for spec in target_specs:
            await self._connection.execute(
                """
                INSERT INTO user_erasure_cleanup_targets(
                    target_id,
                    cleanup_id,
                    target_kind,
                    backend_name,
                    target_key,
                    lifecycle_epoch,
                    checkpoint_state,
                    evidence_sha256,
                    evidence_reference,
                    verified_at,
                    row_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, 0, ?, ?)
                """,
                (
                    generate_prefixed_id("ert"),
                    cleanup_id,
                    spec.target_kind,
                    spec.backend_name,
                    spec.target_key,
                    spec.lifecycle_epoch
                    if spec.lifecycle_epoch is not None
                    else default_lifecycle_epoch,
                    timestamp,
                    timestamp,
                ),
            )

    async def _bump_cleanup_version(
        self,
        cleanup_id: str,
        *,
        expected_cleanup_version: int,
        timestamp: str,
    ) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE user_erasure_cleanups
            SET record_version = record_version + 1,
                last_attempt_at = ?,
                last_error = NULL,
                updated_at = ?
            WHERE cleanup_id = ?
              AND record_version = ?
            RETURNING record_version
            """,
            (timestamp, timestamp, cleanup_id, expected_cleanup_version),
        )
        row = await cursor.fetchone()
        if row is None:
            raise UserErasureConflictError("Cleanup record checkpoint lost its CAS")
        return int(row["record_version"])

    async def _active_lifecycle(self, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_raw_one(
            """
            SELECT lifecycle.*
            FROM user_lifecycles AS lifecycle
            JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND users.deleted_at IS NULL
            LIMIT 1
            """,
            (user_id,),
        )

    async def _cleanup_for_candidate(self, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_raw_one(
            """
            SELECT *
            FROM user_erasure_cleanups
            WHERE candidate_user_id = ?
            LIMIT 1
            """,
            (user_id,),
        )

    async def _cleanup_by_id(self, cleanup_id: str) -> dict[str, Any] | None:
        return await self._fetch_raw_one(
            "SELECT * FROM user_erasure_cleanups WHERE cleanup_id = ?",
            (cleanup_id,),
        )

    async def _raise_for_retained_marker(self, user_id: str) -> None:
        state = await self.get_erasure_state_for_candidate(user_id)
        if state is None:
            return
        marker_state = str(state["erasure_cleanup_state"])
        raise UserErasureConflictError(
            f"User identifier is retained by an erasure tombstone ({marker_state})"
        )

    async def _revoked_job_count(self, cleanup_id: str) -> int:
        row = await self._fetch_raw_one(
            """
            SELECT COUNT(*) AS count
            FROM user_erasure_revoked_jobs
            WHERE cleanup_id = ?
            """,
            (cleanup_id,),
        )
        return int(row["count"]) if row is not None else 0

    async def _fetch_raw_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        await cursor.close()
        return None if row is None else dict(row)

    async def _fetch_raw_all(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    @staticmethod
    def _decode_object(value: Any) -> dict[str, Any]:
        if not isinstance(value, str):
            return {}
        try:
            decoded = json_utils.loads(value)
        except json_utils.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _scope_summary(user_id: str, scope_counts: Mapping[str, int]) -> dict[str, Any]:
        summary: dict[str, Any] = {"user_id_sha256": user_erasure_marker_hash(user_id)}
        for key, value in scope_counts.items():
            normalized_key = str(key).strip()
            if not normalized_key or len(normalized_key) > 64:
                raise ValueError("Erasure scope count keys must be 1..64 characters")
            if normalized_key == "user_id_sha256":
                raise ValueError("user_id_sha256 is owned by the erasure repository")
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError("Erasure scope counts must be non-negative integers")
            summary[normalized_key] = int(value)
        return summary

    @classmethod
    def _validate_target_specs(
        cls,
        target_specs: Sequence[ErasureCleanupTargetSpec],
    ) -> tuple[ErasureCleanupTargetSpec, ...]:
        if len(target_specs) > _MAX_TARGETS:
            raise ValueError(
                f"An erasure cleanup supports at most {_MAX_TARGETS} targets"
            )
        validated: list[ErasureCleanupTargetSpec] = []
        identities: set[tuple[str, str, str]] = set()
        for spec in target_specs:
            target_kind = cls._bounded_component(
                spec.target_kind, field_name="target_kind"
            )
            backend_name = cls._bounded_component(
                spec.backend_name,
                field_name="backend_name",
                allow_empty=True,
            )
            target_key = str(spec.target_key)
            if not target_key or len(target_key) > _MAX_TARGET_KEY_LENGTH:
                raise ValueError(
                    f"target_key must be 1..{_MAX_TARGET_KEY_LENGTH} characters"
                )
            identity = (target_kind, backend_name, target_key)
            if identity in identities:
                raise ValueError("Duplicate erasure cleanup target")
            identities.add(identity)
            validated.append(
                ErasureCleanupTargetSpec(
                    target_kind=target_kind,
                    backend_name=backend_name,
                    target_key=target_key,
                    lifecycle_epoch=_optional_text(spec.lifecycle_epoch),
                )
            )
        return tuple(validated)

    @staticmethod
    def _bounded_component(
        value: str,
        *,
        field_name: str,
        allow_empty: bool = False,
    ) -> str:
        normalized = str(value).strip()
        if (not normalized and not allow_empty) or len(
            normalized
        ) > _MAX_TARGET_COMPONENT_LENGTH:
            qualifier = "0" if allow_empty else "1"
            raise ValueError(
                f"{field_name} must be {qualifier}..{_MAX_TARGET_COMPONENT_LENGTH} characters"
            )
        return normalized

    @classmethod
    def _validate_evidence_references(cls, references: Sequence[str]) -> list[str]:
        if not references or len(references) > _MAX_EVIDENCE_REFERENCES:
            raise ValueError(
                f"evidence_references must contain 1..{_MAX_EVIDENCE_REFERENCES} items"
            )
        return [cls._validate_evidence_reference(reference) for reference in references]

    @staticmethod
    def _validate_evidence_reference(reference: str) -> str:
        normalized = str(reference).strip()
        if not normalized or len(normalized) > _MAX_EVIDENCE_REFERENCE_LENGTH:
            raise ValueError(
                "evidence_reference must be "
                f"1..{_MAX_EVIDENCE_REFERENCE_LENGTH} characters"
            )
        return normalized

    @staticmethod
    def _validate_sha256(value: str, *, field_name: str) -> str:
        normalized = str(value).strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        return normalized

    @staticmethod
    def _retention_eligible_sql(*, select_clause: str) -> str:
        return f"""
            SELECT {select_clause}
            FROM deletion_tombstones AS tombstone
            WHERE tombstone.deleted_at < ?
              AND tombstone.entity_type = 'user'
              AND tombstone.deletion_reason = 'right_to_erasure'
              AND tombstone.erasure_protocol_version = {ERASURE_PROTOCOL_VERSION}
              AND tombstone.erasure_cleanup_state = 'verified'
              AND tombstone.cleanup_verified_at IS NOT NULL
              AND tombstone.cleanup_evidence_manifest_sha256 IS NOT NULL
              AND (
                  tombstone.erasure_lifecycle_epoch IS NOT NULL
                  OR tombstone.legacy_reconciled = 1
              )
              AND NOT EXISTS (
                  SELECT 1 FROM user_erasure_cleanups AS cleanup
                  WHERE cleanup.tombstone_id = tombstone.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM user_lifecycles AS lifecycle
                  WHERE tombstone.erasure_lifecycle_epoch IS NOT NULL
                    AND lifecycle.lifecycle_epoch = tombstone.erasure_lifecycle_epoch
              )
              AND NOT EXISTS (
                  SELECT 1 FROM worker_job_runs AS job
                  WHERE tombstone.erasure_lifecycle_epoch IS NOT NULL
                    AND job.lifecycle_epoch = tombstone.erasure_lifecycle_epoch
                    AND (
                        job.status IN (
                            'queued', 'awaiting_claim', 'running', 'retrying', 'deferred'
                        )
                        OR job.recovery_envelope_json IS NOT NULL
                    )
              )
        """

    def _require_no_open_transaction(self) -> None:
        if self._connection.in_transaction:
            raise RuntimeError("UserErasureRepository requires transaction ownership")

    @staticmethod
    def _call_failpoint(failpoint: Failpoint | None, name: str) -> None:
        if failpoint is not None:
            failpoint(name)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
