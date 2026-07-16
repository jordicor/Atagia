"""Durable worker-job tracking repository."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
import hashlib
from typing import Any

from atagia.core import json_utils
from atagia.core.ids import derive_child_job_id
from atagia.core.repositories import BaseRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    REVISE_STREAM_NAME,
    ClaimedJob,
    CompactionJobKind,
    CompactionJobPayload,
    DurableJobNotification,
    GraphProjectionJobPayload,
    InitialContextPackageRefreshJobPayload,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobRunStatus,
    JobType,
    MessageJobPayload,
    RevisionJobPayload,
)

NONTERMINAL_JOB_STATUSES: tuple[JobRunStatus, ...] = (
    JobRunStatus.QUEUED,
    JobRunStatus.AWAITING_CLAIM,
    JobRunStatus.RUNNING,
    JobRunStatus.RETRYING,
    JobRunStatus.DEFERRED,
)
TERMINAL_JOB_STATUSES: tuple[JobRunStatus, ...] = (
    JobRunStatus.SUCCEEDED,
    JobRunStatus.SKIPPED,
    JobRunStatus.FAILED,
    JobRunStatus.DEAD_LETTERED,
    JobRunStatus.CANCELLED,
)
ROOT_JOB_TYPES: tuple[JobType, ...] = (
    JobType.EXTRACT_MEMORY_CANDIDATES,
    JobType.PROJECT_CONTRACT,
)
REGENERATABLE_AGGREGATE_STREAMS: dict[JobType, str] = {
    JobType.REVISE_BELIEFS: REVISE_STREAM_NAME,
    JobType.COMPACT_SUMMARIES: COMPACT_STREAM_NAME,
    JobType.SYNC_GRAPH: GRAPH_STREAM_NAME,
    JobType.REFRESH_INITIAL_CONTEXT_PACKAGE: INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
}


@dataclass(frozen=True, slots=True)
class JobNamespaceFilter:
    """Namespace policy used for non-admin job-status views."""

    user_persona_id: str | None
    platform_id: str
    character_id: str | None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True


class JobRunRepository(BaseRepository):
    """Persistence operations for background worker job runs."""

    async def create_durable_job(
        self,
        *,
        stream_name: str,
        target_backend: str,
        envelope: JobEnvelope,
        source_token_estimate: int | None,
        size_bucket: str | None,
        queued_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito_snapshot: bool = False,
        remember_across_chats_snapshot: bool = True,
        remember_across_devices_snapshot: bool = True,
        temporary_snapshot: bool = False,
        purge_on_close_snapshot: bool = False,
        policy_snapshot: dict[str, Any] | None = None,
        parent_claim: ClaimedJob | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        serialized_envelope = json_utils.dumps(
            envelope.model_dump(mode="json"),
            sort_keys=True,
        )
        started_transaction = (
            parent_claim is not None and not self._connection.in_transaction
        )
        if started_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            timestamp = queued_at or self._timestamp()
            cursor = await self._connection.execute(
                """
            INSERT INTO worker_job_runs(
                job_id,
                stream_name,
                target_backend,
                job_type,
                user_id,
                conversation_id,
                parent_job_id,
                transcript_rebuild_id,
                maintenance_operation_id,
                source_message_ids_json,
                status,
                source_token_estimate,
                size_bucket,
                queued_at,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                temporary_snapshot,
                purge_on_close_snapshot,
                policy_snapshot_json,
                envelope_schema_version,
                recovery_envelope_json,
                lifecycle_epoch,
                derivation_revision,
                lifecycle_cleanup_key
            )
            SELECT
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                lifecycle.lifecycle_epoch,
                lifecycle.derivation_revision,
                lifecycle.lifecycle_cleanup_key
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users
              ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND (
                  (
                      ? IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS active_operation
                          WHERE (
                              active_operation.status = 'remediation_required'
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
                                OR active_operation.user_id = lifecycle.user_id
                            )
                      )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM admin_maintenance_operations AS owned_operation
                      WHERE owned_operation.id = ?
                        AND owned_operation.status = 'active'
                        AND julianday(owned_operation.lease_expires_at) >
                            julianday('now')
                        AND owned_operation.scope_kind = 'user'
                        AND owned_operation.user_id = lifecycle.user_id
                        AND owned_operation.lifecycle_epoch =
                            lifecycle.lifecycle_epoch
                        AND owned_operation.derivation_revision =
                            lifecycle.derivation_revision
                  )
              )
              AND (
                  ? IS NULL
                  OR EXISTS (
                      SELECT 1
                      FROM worker_job_runs AS parent
                      WHERE parent.job_id = ?
                        AND parent.user_id = lifecycle.user_id
                        AND parent.status = ?
                        AND parent.execution_owner = ?
                        AND parent.execution_fence = ?
                        AND parent.lifecycle_epoch = ?
                        AND parent.derivation_revision = ?
                        AND parent.maintenance_operation_id IS ?
                        AND parent.derivation_revision = lifecycle.derivation_revision
                        AND parent.execution_lease_expires_at > ?
                  )
              )
            ON CONFLICT(job_id) DO NOTHING
            """,
                (
                    envelope.job_id,
                    stream_name,
                    target_backend,
                    envelope.job_type.value,
                    envelope.user_id,
                    envelope.conversation_id,
                    envelope.parent_job_id,
                    envelope.transcript_rebuild_id,
                    envelope.maintenance_operation_id,
                    json_utils.dumps(
                        [str(item) for item in envelope.message_ids], sort_keys=True
                    ),
                    JobRunStatus.QUEUED.value,
                    source_token_estimate,
                    size_bucket,
                    timestamp,
                    json_utils.dumps(metadata or {}, sort_keys=True),
                    user_persona_id,
                    platform_id,
                    character_id,
                    1 if incognito_snapshot else 0,
                    1 if remember_across_chats_snapshot else 0,
                    1 if remember_across_devices_snapshot else 0,
                    1 if temporary_snapshot else 0,
                    1 if purge_on_close_snapshot else 0,
                    json_utils.dumps(policy_snapshot or {}, sort_keys=True),
                    envelope.schema_version,
                    serialized_envelope,
                    envelope.user_id,
                    envelope.maintenance_operation_id,
                    envelope.maintenance_operation_id,
                    envelope.parent_job_id,
                    parent_claim.envelope.job_id if parent_claim is not None else None,
                    JobRunStatus.RUNNING.value,
                    parent_claim.owner_id if parent_claim is not None else None,
                    parent_claim.execution_fence if parent_claim is not None else None,
                    parent_claim.lifecycle_epoch if parent_claim is not None else None,
                    (
                        parent_claim.derivation_revision
                        if parent_claim is not None
                        else None
                    ),
                    envelope.maintenance_operation_id,
                    self._timestamp(),
                ),
            )
            if commit:
                await self._connection.commit()
            row = await self.get_job(envelope.job_id)
            if row is None:
                raise RuntimeError(
                    f"Cannot create durable job {envelope.job_id}: user lifecycle is not active"
                )
            if int(cursor.rowcount or 0) == 0:
                active_lifecycle_epoch = await self._active_lifecycle_epoch(
                    envelope.user_id
                )
                if (
                    active_lifecycle_epoch is None
                    or str(row["lifecycle_epoch"]) != active_lifecycle_epoch
                ):
                    raise RuntimeError(
                        f"Cannot reuse durable job {envelope.job_id}: user lifecycle is not active"
                    )
                if parent_claim is not None and not await self.claim_is_current(
                    parent_claim
                ):
                    raise RuntimeError(
                        f"Cannot reuse durable job {envelope.job_id}: parent execution fence is not current"
                    )
                existing_envelope = row.get("recovery_envelope_json")
                if existing_envelope is None:
                    if str(row["status"]) not in {
                        status.value for status in TERMINAL_JOB_STATUSES
                    } or not self._matches_persisted_job_identity(
                        row,
                        stream_name=stream_name,
                        target_backend=target_backend,
                        envelope=envelope,
                    ):
                        raise ValueError(
                            f"Durable job_id {envelope.job_id} already belongs to a different job identity"
                        )
                elif existing_envelope != envelope.model_dump(mode="json"):
                    raise ValueError(
                        f"Durable job_id {envelope.job_id} already belongs to a different envelope"
                    )
            return row
        except BaseException:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def _active_lifecycle_epoch(self, user_id: str) -> str | None:
        cursor = await self._connection.execute(
            """
            SELECT lifecycle.lifecycle_epoch
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
            LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return None if row is None else str(row["lifecycle_epoch"])

    @staticmethod
    def _matches_persisted_job_identity(
        row: dict[str, Any],
        *,
        stream_name: str,
        target_backend: str,
        envelope: JobEnvelope,
    ) -> bool:
        """Compare only immutable, non-content identity after terminal cleanup."""

        source_message_ids = row.get("source_message_ids_json")
        return (
            str(row.get("job_id")) == envelope.job_id
            and str(row.get("stream_name")) == stream_name
            and str(row.get("target_backend")) == target_backend
            and str(row.get("job_type")) == envelope.job_type.value
            and str(row.get("user_id")) == envelope.user_id
            and row.get("conversation_id") == envelope.conversation_id
            and row.get("parent_job_id") == envelope.parent_job_id
            and row.get("transcript_rebuild_id") == envelope.transcript_rebuild_id
            and row.get("maintenance_operation_id") == envelope.maintenance_operation_id
            and source_message_ids == [str(item) for item in envelope.message_ids]
        )

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM worker_job_runs
            WHERE job_id = ?
            """,
            (job_id,),
        )

    async def recover_expired_execution_leases(self, *, commit: bool = True) -> int:
        """Return expired execution owners to the durable dispatch queue."""

        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                last_heartbeat_at = ?,
                error_class = 'ExecutionLeaseExpired',
                error_message = 'worker execution lease expired before a terminal transition'
            WHERE status IN (?, ?)
              AND execution_lease_expires_at IS NOT NULL
              AND execution_lease_expires_at <= ?
              AND recovery_envelope_json IS NOT NULL
            """,
            (
                JobRunStatus.QUEUED.value,
                timestamp,
                JobRunStatus.RUNNING.value,
                JobRunStatus.RETRYING.value,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def cancel_jobs_with_inactive_lifecycle(self, *, commit: bool = True) -> int:
        """Fence durable work whose captured lifecycle is no longer active."""

        timestamp = self._timestamp()
        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                finished_at = ?,
                last_heartbeat_at = ?,
                error_class = 'LifecycleRevoked',
                error_message = 'captured user lifecycle is no longer active',
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                execution_fence = execution_fence + 1
            WHERE status IN ({placeholders})
              AND NOT EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            """.format(placeholders=placeholders),
            (
                JobRunStatus.CANCELLED.value,
                timestamp,
                timestamp,
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def cancel_jobs_with_stale_derivation(
        self,
        *,
        job_id: str | None = None,
        dispatch_token: str | None = None,
        commit: bool = True,
    ) -> int:
        """Terminalize frozen-payload work from an older source revision.

        Durable envelopes capture source text and other derivation inputs when
        they are enqueued. A later canonical edit cannot make that payload
        current again, so queued, deferred, retried, awaiting-claim, and
        running rows retain their enqueue revision and are cancelled instead
        of being rebased at dispatch or claim time.
        """

        timestamp = self._timestamp()
        status_placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        identity_clauses: list[str] = []
        identity_parameters: list[Any] = []
        if job_id is not None:
            identity_clauses.append("worker_job_runs.job_id = ?")
            identity_parameters.append(job_id)
        if dispatch_token is not None:
            identity_clauses.append("worker_job_runs.dispatch_token = ?")
            identity_parameters.append(dispatch_token)
        identity_where = (
            " AND " + " AND ".join(identity_clauses) if identity_clauses else ""
        )
        cursor = await self._connection.execute(
            f"""
            UPDATE worker_job_runs
            SET status = ?,
                finished_at = ?,
                last_heartbeat_at = ?,
                error_class = 'DerivationRevisionChanged',
                error_message = 'captured derivation revision is no longer current',
                terminal_diagnostics_json = ?,
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL,
                execution_fence = execution_fence + 1
            WHERE status IN ({status_placeholders})
              {identity_where}
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision !=
                        worker_job_runs.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            """,
            (
                JobRunStatus.CANCELLED.value,
                timestamp,
                timestamp,
                json_utils.dumps(
                    {"reason": "derivation_revision_changed"},
                    sort_keys=True,
                ),
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
                *identity_parameters,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def reconcile_stale_root_jobs_after_revision_bump(
        self,
        user_id: str,
        previous_revision: int,
        new_revision: int,
        excluded_source_message_ids: Iterable[str] = (),
        excluded_conversation_ids: Iterable[str] = (),
        allow_validated_requeue: bool = True,
    ) -> dict[str, int]:
        """Requeue only frozen root payloads proven unchanged after a bump.

        The caller owns the source-mutation transaction; this method never
        commits. A root extraction/contract job keeps its durable envelope and
        may move to ``queued`` at ``new_revision`` only when every canonical
        source coordinate represented by that envelope still matches SQLite.
        Excluded or unverifiable roots are cancelled. A stale aggregate/child
        attempt is never rebased in place: when its intent can be rebuilt from
        current SQLite state, a new parentless job is persisted at the new
        revision before the old attempt is cancelled. This also preserves
        aggregate work whose producing root is already terminal.
        """

        if not self._connection.in_transaction:
            raise RuntimeError(
                "stale root reconciliation requires a caller-owned transaction"
            )
        if previous_revision < 0 or new_revision <= previous_revision:
            raise ValueError("new_revision must be greater than previous_revision")
        excluded_messages = {
            str(item).strip()
            for item in excluded_source_message_ids
            if str(item).strip()
        }
        excluded_conversations = {
            str(item).strip() for item in excluded_conversation_ids if str(item).strip()
        }
        lifecycle_cursor = await self._connection.execute(
            """
            SELECT
                lifecycle.lifecycle_epoch,
                lifecycle.lifecycle_cleanup_key
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users AS active_user ON active_user.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.derivation_revision = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (active_user.id IS NOT NULL AND active_user.deleted_at IS NULL)
              )
            LIMIT 1
            """,
            (user_id, new_revision),
        )
        lifecycle_is_current = await lifecycle_cursor.fetchone()
        await lifecycle_cursor.close()
        if lifecycle_is_current is None:
            raise RuntimeError(
                "new derivation revision is not the active user lifecycle revision"
            )
        lifecycle_epoch = str(lifecycle_is_current["lifecycle_epoch"])
        lifecycle_cleanup_key = str(lifecycle_is_current["lifecycle_cleanup_key"])

        status_placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        rows = await self._fetch_all(
            f"""
            SELECT *
            FROM worker_job_runs
            WHERE user_id = ?
              AND derivation_revision = ?
              AND lifecycle_epoch = ?
              AND lifecycle_cleanup_key = ?
              AND status IN ({status_placeholders})
            ORDER BY queued_at ASC, job_id ASC
            """,
            (
                user_id,
                previous_revision,
                lifecycle_epoch,
                lifecycle_cleanup_key,
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
            ),
        )
        requeue_ids: list[str] = []
        cancel_root_ids: list[str] = []
        cancel_aggregate_ids: list[str] = []
        root_types = {job_type.value for job_type in ROOT_JOB_TYPES}
        for row in rows:
            job_id = str(row["job_id"])
            source_ids = {
                str(item)
                for item in (row.get("source_message_ids_json") or [])
                if str(item)
            }
            conversation_id = _optional_identifier(row.get("conversation_id"))
            is_root = (
                str(row.get("job_type")) in root_types
                and row.get("parent_job_id") is None
            )
            if row.get("maintenance_operation_id") is not None:
                if is_root:
                    cancel_root_ids.append(job_id)
                else:
                    cancel_aggregate_ids.append(job_id)
                continue
            if not is_root:
                cancel_aggregate_ids.append(job_id)
                replacement = await self._regenerated_aggregate_envelope(
                    row,
                    new_revision=new_revision,
                    lifecycle_epoch=lifecycle_epoch,
                    excluded_source_message_ids=excluded_messages,
                    excluded_conversation_ids=excluded_conversations,
                    allow_frozen_payload_reuse=allow_validated_requeue,
                )
                if replacement is not None:
                    await self._persist_regenerated_aggregate(
                        row,
                        replacement,
                        new_revision=new_revision,
                    )
                continue
            excluded = bool(source_ids & excluded_messages) or (
                conversation_id is not None
                and conversation_id in excluded_conversations
            )
            if (
                not allow_validated_requeue
                or excluded
                or not await self._frozen_message_root_is_current(row)
            ):
                cancel_root_ids.append(job_id)
                continue
            requeue_ids.append(job_id)

        requeued = await self._requeue_job_ids_at_revision(
            requeue_ids,
            previous_revision=previous_revision,
            new_revision=new_revision,
        )
        cancelled = await self._cancel_stale_job_ids(
            cancel_root_ids,
            previous_revision=previous_revision,
            reason="source_excluded_or_payload_changed",
        )
        aggregate_cancelled = await self._cancel_stale_job_ids(
            cancel_aggregate_ids,
            previous_revision=previous_revision,
            reason="aggregate_requires_regeneration",
        )
        return {
            "requeued": requeued,
            "cancelled": cancelled,
            "aggregate_cancelled": aggregate_cancelled,
        }

    async def _regenerated_aggregate_envelope(
        self,
        row: dict[str, Any],
        *,
        new_revision: int,
        lifecycle_epoch: str,
        excluded_source_message_ids: set[str],
        excluded_conversation_ids: set[str],
        allow_frozen_payload_reuse: bool,
    ) -> JobEnvelope | None:
        envelope_data = row.get("recovery_envelope_json")
        if not isinstance(envelope_data, dict):
            return None
        try:
            envelope = JobEnvelope.model_validate(envelope_data)
        except (TypeError, ValueError):
            return None
        expected_stream = REGENERATABLE_AGGREGATE_STREAMS.get(envelope.job_type)
        if expected_stream is None:
            return None
        if not self._aggregate_row_matches_envelope(
            row,
            envelope=envelope,
            expected_stream=expected_stream,
        ):
            return None
        # Transcript-rebuild-owned work is regenerated by its workflow, which
        # has a separate stage fence and target inventory.
        if (
            envelope.transcript_rebuild_id is not None
            or envelope.maintenance_operation_id is not None
        ):
            return None

        replacement_job_id = derive_child_job_id(
            envelope.job_id,
            envelope.job_type.value,
            f"derivation-regeneration:{new_revision}",
        )
        if envelope.job_type is JobType.REFRESH_INITIAL_CONTEXT_PACKAGE:
            return await self._regenerated_icp_envelope(
                envelope,
                replacement_job_id=replacement_job_id,
                lifecycle_epoch=lifecycle_epoch,
                new_revision=new_revision,
            )
        if envelope.job_type is JobType.COMPACT_SUMMARIES:
            return await self._regenerated_compaction_envelope(
                envelope,
                replacement_job_id=replacement_job_id,
            )

        source_ids = set(envelope.message_ids)
        source_is_excluded = bool(source_ids & excluded_source_message_ids)
        conversation_is_excluded = (
            envelope.conversation_id is not None
            and envelope.conversation_id in excluded_conversation_ids
        )
        if (
            not allow_frozen_payload_reuse
            or source_is_excluded
            or conversation_is_excluded
        ):
            return None
        if envelope.job_type is JobType.SYNC_GRAPH:
            try:
                payload = GraphProjectionJobPayload.model_validate(envelope.payload)
            except (TypeError, ValueError):
                return None
            if not await self._frozen_message_payload_is_current(
                row,
                envelope=envelope,
                payload=payload,
            ):
                return None
            memory_ids = {
                *payload.source_memory_ids,
                *(
                    memory_id
                    for chunk in payload.chunks
                    for memory_id in chunk.source_memory_ids
                ),
            }
            if not await self._active_memory_ids_exist(
                envelope.user_id,
                memory_ids,
            ):
                return None
            return envelope.model_copy(
                update={
                    "job_id": replacement_job_id,
                    "parent_job_id": None,
                    "created_at": self._clock.now(),
                }
            )
        if envelope.job_type is JobType.REVISE_BELIEFS:
            try:
                payload = RevisionJobPayload.model_validate(envelope.payload)
            except (TypeError, ValueError):
                return None
            if not await self._frozen_revision_payload_is_current(
                row,
                envelope=envelope,
                payload=payload,
            ):
                return None
            return envelope.model_copy(
                update={
                    "job_id": replacement_job_id,
                    "parent_job_id": None,
                    "created_at": self._clock.now(),
                }
            )
        return None

    @staticmethod
    def _aggregate_row_matches_envelope(
        row: dict[str, Any],
        *,
        envelope: JobEnvelope,
        expected_stream: str,
    ) -> bool:
        return (
            str(row.get("job_id")) == envelope.job_id
            and str(row.get("job_type")) == envelope.job_type.value
            and str(row.get("user_id")) == envelope.user_id
            and row.get("conversation_id") == envelope.conversation_id
            and row.get("parent_job_id") == envelope.parent_job_id
            and row.get("transcript_rebuild_id") == envelope.transcript_rebuild_id
            and row.get("maintenance_operation_id") == envelope.maintenance_operation_id
            and row.get("source_message_ids_json") == envelope.message_ids
            and str(row.get("stream_name")) == expected_stream
        )

    async def _regenerated_icp_envelope(
        self,
        envelope: JobEnvelope,
        *,
        replacement_job_id: str,
        lifecycle_epoch: str,
        new_revision: int,
    ) -> JobEnvelope | None:
        try:
            payload = InitialContextPackageRefreshJobPayload.model_validate(
                envelope.payload
            )
        except (TypeError, ValueError):
            return None
        if (
            payload.user_id != envelope.user_id
            or payload.conversation_id != envelope.conversation_id
            or payload.source_message_ids != envelope.message_ids
        ):
            return None
        if payload.conversation_id is not None and (
            await self._active_conversation_snapshot(
                envelope.user_id,
                payload.conversation_id,
            )
            is None
        ):
            return None
        current_source_ids = await self._current_source_message_ids(
            envelope.user_id,
            payload.source_message_ids,
        )
        generation = await UserLifecycleRepository(
            self._connection,
            self._clock,
        ).reserve_icp_refresh_generation(
            envelope.user_id,
            expected_lifecycle_epoch=lifecycle_epoch,
            commit=False,
        )
        if generation is None:
            return None
        discriminator = hashlib.sha256(
            (
                f"{envelope.job_id}\x1f{replacement_job_id}\x1f"
                f"{new_revision}\x1f{generation}"
            ).encode("utf-8")
        ).hexdigest()
        replacement_payload = payload.model_copy(
            update={
                "reason": InitialContextPackageRefreshReason.SOURCE_CHANGED,
                "refresh_generation": generation,
                "refresh_dedupe_key": (
                    f"initial_context_package_refresh:source_changed:{discriminator}"
                ),
                "source_message_ids": current_source_ids,
            }
        )
        return envelope.model_copy(
            update={
                "job_id": replacement_job_id,
                "parent_job_id": None,
                "message_ids": current_source_ids,
                "payload": replacement_payload.model_dump(mode="json"),
                "created_at": self._clock.now(),
            }
        )

    async def _regenerated_compaction_envelope(
        self,
        envelope: JobEnvelope,
        *,
        replacement_job_id: str,
    ) -> JobEnvelope | None:
        try:
            payload = CompactionJobPayload.model_validate(envelope.payload)
        except (TypeError, ValueError):
            return None
        if payload.user_id != envelope.user_id:
            return None
        if (
            payload.job_kind is CompactionJobKind.CONVERSATION_CHUNK
            and payload.conversation_id != envelope.conversation_id
        ):
            return None
        if (
            envelope.conversation_id is not None
            and payload.conversation_id != envelope.conversation_id
        ):
            return None
        preferences = await self._active_user_preferences(envelope.user_id)
        if preferences is None:
            return None
        updates: dict[str, Any] = {
            "remember_across_chats": preferences["remember_across_chats"],
            "remember_across_devices": preferences["remember_across_devices"],
        }
        if payload.conversation_id is not None:
            conversation = await self._active_conversation_snapshot(
                envelope.user_id,
                payload.conversation_id,
            )
            if conversation is None:
                return None
            updates.update(
                {
                    "workspace_id": conversation["workspace_id"],
                    "user_persona_id": conversation["user_persona_id"],
                    "platform_id": conversation["platform_id"],
                    "character_id": conversation["character_id"],
                    "mode": conversation["mode"],
                    "incognito": conversation["incognito"],
                    "temporary": conversation["temporary"],
                    "temporary_ttl_seconds": conversation["temporary_ttl_seconds"],
                    "purge_on_close": conversation["purge_on_close"],
                }
            )
        replacement_payload = payload.model_copy(update=updates)
        current_source_ids = await self._current_source_message_ids(
            envelope.user_id,
            envelope.message_ids,
        )
        return envelope.model_copy(
            update={
                "job_id": replacement_job_id,
                "parent_job_id": None,
                "message_ids": current_source_ids,
                "payload": replacement_payload.model_dump(mode="json"),
                "created_at": self._clock.now(),
            }
        )

    async def _persist_regenerated_aggregate(
        self,
        stale_row: dict[str, Any],
        replacement: JobEnvelope,
        *,
        new_revision: int,
    ) -> None:
        stream_name = REGENERATABLE_AGGREGATE_STREAMS[replacement.job_type]
        policy_snapshot = self._policy_snapshot_for_envelope(replacement)
        existing_metadata = stale_row.get("metadata_json")
        metadata = (
            dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
        )
        metadata.update(
            {
                "regenerated_from_job_id": str(stale_row["job_id"]),
                "regenerated_for_derivation_revision": new_revision,
            }
        )
        if replacement.job_type is JobType.REFRESH_INITIAL_CONTEXT_PACKAGE:
            metadata["reason"] = InitialContextPackageRefreshReason.SOURCE_CHANGED.value
        stored = await self.create_durable_job(
            stream_name=stream_name,
            target_backend=str(stale_row["target_backend"]),
            envelope=replacement,
            source_token_estimate=stale_row.get("source_token_estimate"),
            size_bucket=_optional_identifier(stale_row.get("size_bucket")),
            queued_at=self._timestamp(),
            metadata=metadata,
            user_persona_id=_optional_identifier(
                policy_snapshot.get("user_persona_id")
            ),
            platform_id=str(policy_snapshot.get("platform_id") or "default"),
            character_id=_optional_identifier(policy_snapshot.get("character_id")),
            incognito_snapshot=bool(policy_snapshot.get("incognito")),
            remember_across_chats_snapshot=bool(
                policy_snapshot.get("remember_across_chats", True)
            ),
            remember_across_devices_snapshot=bool(
                policy_snapshot.get("remember_across_devices", True)
            ),
            temporary_snapshot=bool(policy_snapshot.get("temporary")),
            purge_on_close_snapshot=bool(policy_snapshot.get("purge_on_close")),
            policy_snapshot=policy_snapshot,
            parent_claim=None,
            commit=False,
        )
        if int(stored["derivation_revision"]) != new_revision or str(
            stored["status"]
        ) not in {status.value for status in NONTERMINAL_JOB_STATUSES}:
            raise RuntimeError(
                "regenerated aggregate did not capture the active derivation revision"
            )

    @staticmethod
    def _policy_snapshot_for_envelope(envelope: JobEnvelope) -> dict[str, Any]:
        payload = envelope.payload
        return {
            "user_persona_id": payload.get("user_persona_id"),
            "platform_id": str(payload.get("platform_id") or "default"),
            "character_id": payload.get("character_id"),
            "conversation_id": envelope.conversation_id,
            "mode": payload.get("mode") or payload.get("assistant_mode_id"),
            "incognito": bool(payload.get("incognito", False)),
            "remember_across_chats": bool(payload.get("remember_across_chats", True)),
            "remember_across_devices": bool(
                payload.get("remember_across_devices", True)
            ),
            "memory_privacy_mode": str(
                payload.get("memory_privacy_mode") or "balanced"
            ),
            "temporary": bool(payload.get("temporary", False)),
            "temporary_ttl_seconds": payload.get("temporary_ttl_seconds"),
            "purge_on_close": bool(payload.get("purge_on_close", False)),
            "valid_to": payload.get("valid_to"),
        }

    async def _active_user_preferences(
        self,
        user_id: str,
    ) -> dict[str, Any] | None:
        row = await self._fetch_one(
            """
            SELECT
                remember_across_chats,
                remember_across_devices,
                memory_privacy_mode
            FROM users
            WHERE id = ? AND deleted_at IS NULL
            LIMIT 1
            """,
            (user_id,),
        )
        if row is None:
            return None
        return {
            "remember_across_chats": bool(row["remember_across_chats"]),
            "remember_across_devices": bool(row["remember_across_devices"]),
            "memory_privacy_mode": str(row["memory_privacy_mode"] or "balanced"),
        }

    async def _active_conversation_snapshot(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        row = await self._fetch_one(
            """
            SELECT
                conversation.workspace_id,
                conversation.assistant_mode_id,
                conversation.user_persona_id,
                conversation.platform_id,
                conversation.character_id,
                conversation.mode,
                conversation.incognito,
                conversation.isolated_mode,
                conversation.temporary,
                conversation.temporary_ttl_seconds,
                conversation.purge_on_close
            FROM conversations AS conversation
            JOIN users AS active_user
              ON active_user.id = conversation.user_id
             AND active_user.deleted_at IS NULL
            WHERE conversation.id = ?
              AND conversation.user_id = ?
              AND conversation.status = 'active'
            LIMIT 1
            """,
            (conversation_id, user_id),
        )
        if row is None:
            return None
        return {
            "workspace_id": _optional_identifier(row["workspace_id"]),
            "assistant_mode_id": str(row["assistant_mode_id"]),
            "user_persona_id": _optional_identifier(row["user_persona_id"]),
            "platform_id": str(row["platform_id"] or "default"),
            "character_id": _optional_identifier(
                row["character_id"] or row["workspace_id"]
            ),
            "mode": str(row["mode"] or row["assistant_mode_id"]),
            "incognito": bool(row["incognito"] or row["isolated_mode"]),
            "temporary": bool(row["temporary"]),
            "temporary_ttl_seconds": row["temporary_ttl_seconds"],
            "purge_on_close": bool(row["purge_on_close"]),
        }

    async def _current_source_message_ids(
        self,
        user_id: str,
        message_ids: Iterable[str],
    ) -> list[str]:
        stable_ids = list(dict.fromkeys(str(item) for item in message_ids if str(item)))
        if not stable_ids:
            return []
        placeholders = ", ".join("?" for _ in stable_ids)
        rows = await self._fetch_all(
            f"""
            SELECT message.id
            FROM messages AS message
            JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            JOIN users AS active_user
              ON active_user.id = conversation.user_id
             AND active_user.deleted_at IS NULL
            WHERE conversation.user_id = ?
              AND conversation.status = 'active'
              AND message.id IN ({placeholders})
            """,
            (user_id, *stable_ids),
        )
        current = {str(row["id"]) for row in rows}
        return [message_id for message_id in stable_ids if message_id in current]

    async def _active_memory_ids_exist(
        self,
        user_id: str,
        memory_ids: Iterable[str],
    ) -> bool:
        stable_ids = {str(item) for item in memory_ids if str(item)}
        if not stable_ids:
            return True
        placeholders = ", ".join("?" for _ in stable_ids)
        cursor = await self._connection.execute(
            f"""
            SELECT COUNT(DISTINCT id) AS count
            FROM memory_objects
            WHERE user_id = ?
              AND status = 'active'
              AND id IN ({placeholders})
            """,
            (user_id, *sorted(stable_ids)),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None and int(row["count"]) == len(stable_ids)

    async def _frozen_revision_payload_is_current(
        self,
        row: dict[str, Any],
        *,
        envelope: JobEnvelope,
        payload: RevisionJobPayload,
    ) -> bool:
        if (
            payload.user_id != envelope.user_id
            or payload.conversation_id != envelope.conversation_id
            or envelope.message_ids != [payload.source_message_id]
            or row.get("source_message_ids_json") != [payload.source_message_id]
        ):
            return False
        source = await self._canonical_message_root(payload.source_message_id)
        if source is None or str(source["conversation_status"]) != "active":
            return False
        if (
            str(source["user_id"]) != envelope.user_id
            or str(source["conversation_id"]) != envelope.conversation_id
        ):
            return False
        expected: dict[str, Any] = {
            "assistant_mode_id": str(source["assistant_mode_id"]),
            "workspace_id": _optional_identifier(source["workspace_id"]),
            "conversation_id": str(source["conversation_id"]),
            "user_persona_id": _optional_identifier(source["user_persona_id"]),
            "platform_id": str(source["platform_id"] or "default"),
            "character_id": _optional_identifier(
                source["character_id"] or source["workspace_id"]
            ),
            "active_mind_id": _optional_identifier(
                source["message_active_mind_id"]
                or source["conversation_active_mind_id"]
            ),
            "source_mind_id": _optional_identifier(
                source["message_source_mind_id"]
                or source["message_active_mind_id"]
                or source["conversation_active_mind_id"]
            ),
            "mind_topology": str(source["mind_topology"] or "unimind"),
            "active_embodiment_id": _optional_identifier(
                source["message_active_embodiment_id"]
                or source["conversation_active_embodiment_id"]
            ),
            "cross_embodiment_mode": str(
                source["cross_embodiment_mode"] or "direct_if_same_body"
            ),
            "active_realm_id": _optional_identifier(
                source["message_active_realm_id"]
                or source["conversation_active_realm_id"]
            ),
            "cross_realm_mode": str(source["cross_realm_mode"] or "none"),
            "mode": str(source["mode"] or source["assistant_mode_id"]),
            "incognito": bool(source["incognito"] or source["isolated_mode"]),
            "remember_across_chats": bool(source["remember_across_chats"]),
            "remember_across_devices": bool(source["remember_across_devices"]),
            "temporary": bool(source["temporary"]),
            "temporary_ttl_seconds": source["temporary_ttl_seconds"],
            "purge_on_close": bool(source["purge_on_close"]),
            "isolated_mode": bool(source["isolated_mode"]),
        }
        payload_data = payload.model_dump(mode="json")
        if any(payload_data.get(key) != value for key, value in expected.items()):
            return False
        memory_ids = set(payload.evidence_memory_ids)
        if payload.belief_id:
            memory_ids.add(payload.belief_id)
        if not await self._active_memory_ids_exist(envelope.user_id, memory_ids):
            return False
        policy_snapshot = row.get("policy_snapshot_json")
        if not isinstance(policy_snapshot, dict):
            return False
        expected_snapshot = self._policy_snapshot_for_envelope(envelope)
        return all(
            policy_snapshot.get(key) == value
            for key, value in expected_snapshot.items()
        )

    async def _frozen_message_root_is_current(self, row: dict[str, Any]) -> bool:
        envelope_data = row.get("recovery_envelope_json")
        if not isinstance(envelope_data, dict):
            return False
        try:
            envelope = JobEnvelope.model_validate(envelope_data)
            payload = MessageJobPayload.model_validate(envelope.payload)
        except (TypeError, ValueError):
            return False
        if (
            envelope.job_type not in ROOT_JOB_TYPES
            or envelope.parent_job_id is not None
        ):
            return False
        return await self._frozen_message_payload_is_current(
            row,
            envelope=envelope,
            payload=payload,
        )

    async def _frozen_message_payload_is_current(
        self,
        row: dict[str, Any],
        *,
        envelope: JobEnvelope,
        payload: MessageJobPayload,
    ) -> bool:
        """Verify one immutable message payload against canonical SQLite state."""

        if envelope.conversation_id is None:
            return False
        if (
            str(row.get("job_id")) != envelope.job_id
            or str(row.get("job_type")) != envelope.job_type.value
            or str(row.get("user_id")) != envelope.user_id
            or row.get("conversation_id") != envelope.conversation_id
            or row.get("parent_job_id") != envelope.parent_job_id
            or row.get("transcript_rebuild_id") != envelope.transcript_rebuild_id
            or row.get("maintenance_operation_id") != envelope.maintenance_operation_id
        ):
            return False
        if envelope.message_ids != [payload.message_id]:
            return False
        if row.get("source_message_ids_json") != [payload.message_id]:
            return False
        source = await self._canonical_message_root(payload.message_id)
        if source is None:
            return False
        if str(source["user_id"]) != envelope.user_id:
            return False
        if str(source["conversation_id"]) != envelope.conversation_id:
            return False
        if str(source["conversation_status"]) != "active":
            return False
        expected_scalars: dict[str, object] = {
            "message_text": str(source["message_text"]),
            "role": str(source["message_role"]),
            "message_occurred_at": _optional_identifier(source["message_occurred_at"]),
            "assistant_mode_id": str(source["assistant_mode_id"]),
            "workspace_id": _optional_identifier(source["workspace_id"]),
            "user_persona_id": _optional_identifier(source["user_persona_id"]),
            "platform_id": str(source["platform_id"] or "default"),
            "character_id": _optional_identifier(
                source["character_id"] or source["workspace_id"]
            ),
            "mode": str(source["mode"] or source["assistant_mode_id"]),
            "active_presence_id": _optional_identifier(
                source["message_active_presence_id"]
                or source["conversation_active_presence_id"]
            ),
            "source_presence_id": _optional_identifier(
                source["message_source_presence_id"]
                or source["message_active_presence_id"]
                or source["conversation_active_presence_id"]
            ),
            "active_space_id": _optional_identifier(
                source["message_space_id"] or source["conversation_active_space_id"]
            ),
            "active_mind_id": _optional_identifier(
                source["message_active_mind_id"]
                or source["conversation_active_mind_id"]
            ),
            "source_mind_id": _optional_identifier(
                source["message_source_mind_id"]
                or source["message_active_mind_id"]
                or source["conversation_active_mind_id"]
            ),
            "active_presence_kind": str(source["active_presence_kind"] or "unknown"),
            "active_presence_display_name": _optional_identifier(
                source["active_presence_display_name"]
            ),
            "source_presence_kind": str(source["source_presence_kind"] or "unknown"),
            "source_presence_display_name": _optional_identifier(
                source["source_presence_display_name"]
            ),
            "active_space_boundary_mode": str(
                source["active_space_boundary_mode"] or "focus"
            ),
            "active_space_display_name": _optional_identifier(
                source["active_space_display_name"]
            ),
            "active_mind_display_name": _optional_identifier(
                source["active_mind_display_name"]
            ),
            "active_embodiment_id": _optional_identifier(
                source["message_active_embodiment_id"]
                or source["conversation_active_embodiment_id"]
            ),
            "active_embodiment_display_name": _optional_identifier(
                source["active_embodiment_display_name"]
            ),
            "cross_embodiment_mode": str(
                source["cross_embodiment_mode"] or "direct_if_same_body"
            ),
            "active_realm_id": _optional_identifier(
                source["message_active_realm_id"]
                or source["conversation_active_realm_id"]
            ),
            "active_realm_display_name": _optional_identifier(
                source["active_realm_display_name"]
            ),
            "cross_realm_mode": str(source["cross_realm_mode"] or "none"),
            "mind_topology": str(source["mind_topology"] or "unimind"),
        }
        coordinate_records = (
            ("active_presence_id", "active_presence_record_id"),
            ("source_presence_id", "source_presence_record_id"),
            ("active_space_id", "active_space_record_id"),
            ("active_mind_id", "active_mind_record_id"),
            ("source_mind_id", "source_mind_record_id"),
            ("active_embodiment_id", "active_embodiment_record_id"),
            ("active_realm_id", "active_realm_record_id"),
        )
        for payload_key, record_key in coordinate_records:
            if expected_scalars[payload_key] is not None and source[record_key] is None:
                return False
        payload_data = payload.model_dump(mode="json")
        if any(
            payload_data.get(key) != value for key, value in expected_scalars.items()
        ):
            return False
        expected_policy = {
            "incognito": bool(source["incognito"] or source["isolated_mode"]),
            "remember_across_chats": bool(source["remember_across_chats"]),
            "remember_across_devices": bool(source["remember_across_devices"]),
            "memory_privacy_mode": str(source["memory_privacy_mode"] or "balanced"),
            "temporary": bool(source["temporary"]),
            "temporary_ttl_seconds": source["temporary_ttl_seconds"],
            "purge_on_close": bool(source["purge_on_close"]),
            "isolated_mode": bool(source["isolated_mode"]),
        }
        if any(
            payload_data.get(key) != value for key, value in expected_policy.items()
        ):
            return False
        policy_snapshot = row.get("policy_snapshot_json")
        if not isinstance(policy_snapshot, dict):
            return False
        if policy_snapshot.get("conversation_id") != envelope.conversation_id:
            return False
        for key in (
            "user_persona_id",
            "platform_id",
            "character_id",
            "mode",
            "incognito",
            "remember_across_chats",
            "remember_across_devices",
            "memory_privacy_mode",
            "temporary",
            "temporary_ttl_seconds",
            "purge_on_close",
            "valid_to",
        ):
            if policy_snapshot.get(key) != payload_data.get(key):
                return False
        return await self._recent_messages_are_current(
            envelope.conversation_id,
            payload.recent_messages,
        )

    async def _canonical_message_root(
        self,
        message_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT
                message.id AS message_id,
                message.conversation_id AS conversation_id,
                message.role AS message_role,
                message.text AS message_text,
                message.occurred_at AS message_occurred_at,
                message.active_presence_id AS message_active_presence_id,
                message.source_presence_id AS message_source_presence_id,
                message.space_id AS message_space_id,
                message.active_mind_id AS message_active_mind_id,
                message.source_mind_id AS message_source_mind_id,
                message.active_embodiment_id AS message_active_embodiment_id,
                message.active_realm_id AS message_active_realm_id,
                conversation.user_id AS user_id,
                conversation.status AS conversation_status,
                conversation.workspace_id AS workspace_id,
                conversation.assistant_mode_id AS assistant_mode_id,
                conversation.user_persona_id AS user_persona_id,
                conversation.platform_id AS platform_id,
                conversation.character_id AS character_id,
                conversation.mode AS mode,
                conversation.incognito AS incognito,
                conversation.isolated_mode AS isolated_mode,
                conversation.temporary AS temporary,
                conversation.temporary_ttl_seconds AS temporary_ttl_seconds,
                conversation.purge_on_close AS purge_on_close,
                conversation.active_presence_id AS conversation_active_presence_id,
                conversation.active_space_id AS conversation_active_space_id,
                conversation.active_mind_id AS conversation_active_mind_id,
                conversation.mind_topology AS mind_topology,
                conversation.active_embodiment_id
                    AS conversation_active_embodiment_id,
                conversation.active_realm_id AS conversation_active_realm_id,
                active_presence.id AS active_presence_record_id,
                active_presence.kind AS active_presence_kind,
                active_presence.display_name AS active_presence_display_name,
                source_presence.id AS source_presence_record_id,
                source_presence.kind AS source_presence_kind,
                source_presence.display_name AS source_presence_display_name,
                active_space.id AS active_space_record_id,
                active_space.boundary_mode AS active_space_boundary_mode,
                active_space.display_name AS active_space_display_name,
                active_mind.id AS active_mind_record_id,
                active_mind.display_name AS active_mind_display_name,
                source_mind.id AS source_mind_record_id,
                active_embodiment.id AS active_embodiment_record_id,
                active_embodiment.display_name
                    AS active_embodiment_display_name,
                active_embodiment.cross_embodiment_mode
                    AS cross_embodiment_mode,
                active_realm.id AS active_realm_record_id,
                active_realm.display_name AS active_realm_display_name,
                active_realm.cross_realm_mode AS cross_realm_mode,
                active_user.remember_across_chats AS remember_across_chats,
                active_user.remember_across_devices AS remember_across_devices,
                active_user.memory_privacy_mode AS memory_privacy_mode
            FROM messages AS message
            JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            JOIN users AS active_user
              ON active_user.id = conversation.user_id
             AND active_user.deleted_at IS NULL
            LEFT JOIN presences AS active_presence
              ON active_presence.owner_user_id = conversation.user_id
             AND active_presence.id = COALESCE(
                    message.active_presence_id,
                    conversation.active_presence_id
                 )
            LEFT JOIN presences AS source_presence
              ON source_presence.owner_user_id = conversation.user_id
             AND source_presence.id = COALESCE(
                    message.source_presence_id,
                    message.active_presence_id,
                    conversation.active_presence_id
                 )
            LEFT JOIN spaces AS active_space
              ON active_space.owner_user_id = conversation.user_id
             AND active_space.id = COALESCE(
                    message.space_id,
                    conversation.active_space_id
                 )
            LEFT JOIN minds AS active_mind
              ON active_mind.owner_user_id = conversation.user_id
             AND active_mind.id = COALESCE(
                    message.active_mind_id,
                    conversation.active_mind_id
                 )
            LEFT JOIN minds AS source_mind
              ON source_mind.owner_user_id = conversation.user_id
             AND source_mind.id = COALESCE(
                    message.source_mind_id,
                    message.active_mind_id,
                    conversation.active_mind_id
                 )
            LEFT JOIN embodiments AS active_embodiment
              ON active_embodiment.owner_user_id = conversation.user_id
             AND active_embodiment.id = COALESCE(
                    message.active_embodiment_id,
                    conversation.active_embodiment_id
                 )
            LEFT JOIN realms AS active_realm
              ON active_realm.owner_user_id = conversation.user_id
             AND active_realm.id = COALESCE(
                    message.active_realm_id,
                    conversation.active_realm_id
                 )
            WHERE message.id = ?
            LIMIT 1
            """,
            (message_id,),
        )

    async def _recent_messages_are_current(
        self,
        conversation_id: str,
        recent_messages: list[Any],
    ) -> bool:
        for recent in recent_messages:
            message_id = _optional_identifier(getattr(recent, "id", None))
            if message_id is None:
                return False
            row = await self._fetch_one(
                """
                SELECT
                    id,
                    role,
                    text,
                    seq,
                    occurred_at,
                    include_raw,
                    skip_by_default,
                    context_placeholder,
                    content_kind,
                    policy_reason,
                    artifact_backed,
                    verbatim_required,
                    heavy_content
                FROM messages
                WHERE id = ?
                  AND conversation_id = ?
                LIMIT 1
                """,
                (message_id, conversation_id),
            )
            if row is None:
                return False
            if (
                str(row["role"]) != recent.role
                or _recent_message_content(row) != recent.content
                or int(row["seq"]) != recent.seq
                or _optional_identifier(row["occurred_at"])
                != _optional_identifier(recent.occurred_at)
            ):
                return False
        return True

    async def _requeue_job_ids_at_revision(
        self,
        job_ids: list[str],
        *,
        previous_revision: int,
        new_revision: int,
    ) -> int:
        if not job_ids:
            return 0
        timestamp = self._timestamp()
        changed = 0
        for job_id in job_ids:
            cursor = await self._connection.execute(
                """
                UPDATE worker_job_runs
                SET status = ?,
                    derivation_revision = ?,
                    execution_fence = execution_fence + 1,
                    execution_owner = NULL,
                    execution_lease_expires_at = NULL,
                    dispatch_token = NULL,
                    dispatch_visibility_deadline = NULL,
                    deferred_until = NULL,
                    finished_at = NULL,
                    last_heartbeat_at = ?,
                    error_class = NULL,
                    error_message = NULL,
                    terminal_diagnostics_json = '{}'
                WHERE job_id = ?
                  AND derivation_revision = ?
                  AND status IN (?, ?, ?, ?, ?)
                  AND recovery_envelope_json IS NOT NULL
                """,
                (
                    JobRunStatus.QUEUED.value,
                    new_revision,
                    timestamp,
                    job_id,
                    previous_revision,
                    *(status.value for status in NONTERMINAL_JOB_STATUSES),
                ),
            )
            changed += int(cursor.rowcount or 0)
        return changed

    async def _cancel_stale_job_ids(
        self,
        job_ids: list[str],
        *,
        previous_revision: int,
        reason: str,
    ) -> int:
        if not job_ids:
            return 0
        timestamp = self._timestamp()
        changed = 0
        diagnostics = json_utils.dumps({"reason": reason}, sort_keys=True)
        for job_id in job_ids:
            cursor = await self._connection.execute(
                """
                UPDATE worker_job_runs
                SET status = ?,
                    finished_at = ?,
                    last_heartbeat_at = ?,
                    error_class = 'DerivationRevisionChanged',
                    error_message = 'captured derivation source is no longer current',
                    terminal_diagnostics_json = ?,
                    recovery_envelope_json = NULL,
                    envelope_schema_version = NULL,
                    dispatch_token = NULL,
                    dispatch_visibility_deadline = NULL,
                    execution_owner = NULL,
                    execution_lease_expires_at = NULL,
                    deferred_until = NULL,
                    execution_fence = execution_fence + 1
                WHERE job_id = ?
                  AND derivation_revision = ?
                  AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    JobRunStatus.CANCELLED.value,
                    timestamp,
                    timestamp,
                    diagnostics,
                    job_id,
                    previous_revision,
                    *(status.value for status in NONTERMINAL_JOB_STATUSES),
                ),
            )
            changed += int(cursor.rowcount or 0)
        return changed

    async def claim_dispatchable_jobs(
        self,
        *,
        target_backend: str,
        limit: int,
        visibility_seconds: float,
    ) -> list[dict[str, Any]]:
        """CAS eligible rows to awaiting-claim before any transient publish."""

        if limit <= 0:
            return []
        now = self._clock.now()
        timestamp = now.isoformat()
        visibility_deadline = (
            now + timedelta(seconds=max(0.1, visibility_seconds))
        ).isoformat()
        cursor = await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                dispatch_token = 'dsp_' || lower(hex(randomblob(16))),
                dispatch_visibility_deadline = ?,
                dispatch_attempt_count = dispatch_attempt_count + 1,
                deferred_until = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL
            WHERE _rowid IN (
                SELECT candidate._rowid
                FROM worker_job_runs AS candidate
                WHERE candidate.target_backend = ?
                  AND candidate.recovery_envelope_json IS NOT NULL
                  AND (
                      candidate.status = ?
                      OR (
                          candidate.status IN (?, ?)
                          AND (candidate.deferred_until IS NULL OR candidate.deferred_until <= ?)
                      )
                      OR (
                          candidate.status = ?
                          AND candidate.dispatch_visibility_deadline IS NOT NULL
                          AND candidate.dispatch_visibility_deadline <= ?
                      )
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM user_lifecycles AS lifecycle
                      LEFT JOIN users ON users.id = lifecycle.user_id
                      WHERE lifecycle.user_id = candidate.user_id
                        AND lifecycle.lifecycle_epoch = candidate.lifecycle_epoch
                        AND lifecycle.derivation_revision =
                            candidate.derivation_revision
                        AND lifecycle.state = 'active'
                        AND lifecycle.erasure_cleanup_id IS NULL
                        AND (
                            lifecycle.user_id = 'atagia_system'
                            OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                      )
                  )
                  AND (
                      (
                          candidate.maintenance_operation_id IS NULL
                          AND NOT EXISTS (
                              SELECT 1
                              FROM admin_maintenance_operations AS active_operation
                              WHERE (
                                  active_operation.status = 'remediation_required'
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
                                    OR active_operation.user_id = candidate.user_id
                                )
                          )
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS owned_operation
                          WHERE owned_operation.id =
                              candidate.maintenance_operation_id
                            AND owned_operation.status = 'active'
                            AND julianday(owned_operation.lease_expires_at) >
                                julianday('now')
                            AND owned_operation.scope_kind = 'user'
                            AND owned_operation.user_id = candidate.user_id
                            AND owned_operation.lifecycle_epoch =
                                candidate.lifecycle_epoch
                            AND owned_operation.derivation_revision =
                                candidate.derivation_revision
                      )
                  )
                  AND (
                      NOT EXISTS (
                          SELECT 1
                          FROM conversation_transcript_selections AS selection
                          WHERE selection.user_id = candidate.user_id
                            AND selection.state IN (
                                'rebuilding',
                                'remediation_required'
                            )
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM transcript_rebuild_workflows AS workflow
                          JOIN conversation_transcript_selections AS selection
                            ON selection.current_workflow_id = workflow.id
                          WHERE workflow.id = candidate.transcript_rebuild_id
                            AND workflow.user_id = candidate.user_id
                            AND selection.user_id = candidate.user_id
                            AND selection.state = 'rebuilding'
                            AND workflow.stage NOT IN ('complete', 'remediation_required')
                            AND (
                                candidate.job_type = 'rebuild_selected_transcript'
                                OR (
                                    workflow.stage = 'sources'
                                    AND candidate.job_type IN (
                                        'extract_memory_candidates',
                                        'project_contract'
                                    )
                                )
                                OR (
                                    workflow.stage = 'aggregates'
                                    AND candidate.job_type IN (
                                        'revise_beliefs',
                                        'sync_graph',
                                        'compact_summaries'
                                    )
                                )
                                OR (
                                    workflow.stage = 'finalizing'
                                    AND candidate.job_type =
                                        'refresh_initial_context_package'
                                )
                            )
                      )
                  )
                ORDER BY candidate.queued_at ASC, candidate.job_id ASC
                LIMIT ?
            )
            RETURNING *
            """,
            (
                JobRunStatus.AWAITING_CLAIM.value,
                visibility_deadline,
                target_backend,
                JobRunStatus.QUEUED.value,
                JobRunStatus.DEFERRED.value,
                JobRunStatus.RETRYING.value,
                timestamp,
                JobRunStatus.AWAITING_CLAIM.value,
                timestamp,
                limit,
            ),
        )
        claimed = [dict(row) for row in await cursor.fetchall()]
        await self._connection.commit()
        for row in claimed:
            for key, value in tuple(row.items()):
                if key.endswith("_json") and isinstance(value, str):
                    row[key] = json_utils.loads(value)
        return claimed

    async def claim_notification(
        self,
        notification_message_id: str,
        notification: DurableJobNotification,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> ClaimedJob | None:
        """Issue the execution fence for one current durable notification."""

        started_transaction = not self._connection.in_transaction
        if started_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            now = self._clock.now()
            timestamp = now.isoformat()
            lease_expires_at = (
                now + timedelta(seconds=max(0.1, lease_seconds))
            ).isoformat()
            cursor = await self._connection.execute(
                """
            UPDATE worker_job_runs
            SET status = ?,
                attempt_count = attempt_count + 1,
                started_at = COALESCE(started_at, ?),
                last_heartbeat_at = ?,
                execution_owner = ?,
                execution_fence = execution_fence + 1,
                execution_lease_expires_at = ?
            WHERE job_id = ?
              AND status = ?
              AND dispatch_token = ?
              AND dispatch_visibility_deadline > ?
              AND lifecycle_epoch = ?
              AND lifecycle_cleanup_key = ?
              AND (
                  (
                      maintenance_operation_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS active_operation
                          WHERE (
                              active_operation.status = 'remediation_required'
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
                                OR active_operation.user_id = worker_job_runs.user_id
                            )
                      )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM admin_maintenance_operations AS owned_operation
                      WHERE owned_operation.id =
                          worker_job_runs.maintenance_operation_id
                        AND owned_operation.status = 'active'
                        AND julianday(owned_operation.lease_expires_at) >
                            julianday('now')
                        AND owned_operation.scope_kind = 'user'
                        AND owned_operation.user_id = worker_job_runs.user_id
                        AND owned_operation.lifecycle_epoch =
                            worker_job_runs.lifecycle_epoch
                        AND owned_operation.derivation_revision =
                            worker_job_runs.derivation_revision
                  )
              )
              AND (
                  NOT EXISTS (
                      SELECT 1
                      FROM conversation_transcript_selections AS selection
                      WHERE selection.user_id = worker_job_runs.user_id
                        AND selection.state IN (
                            'rebuilding',
                            'remediation_required'
                        )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM transcript_rebuild_workflows AS workflow
                      JOIN conversation_transcript_selections AS selection
                        ON selection.current_workflow_id = workflow.id
                      WHERE workflow.id = worker_job_runs.transcript_rebuild_id
                        AND workflow.user_id = worker_job_runs.user_id
                        AND selection.user_id = worker_job_runs.user_id
                        AND selection.state = 'rebuilding'
                        AND workflow.stage NOT IN ('complete', 'remediation_required')
                        AND (
                            worker_job_runs.job_type = 'rebuild_selected_transcript'
                            OR (
                                workflow.stage = 'sources'
                                AND worker_job_runs.job_type IN (
                                    'extract_memory_candidates',
                                    'project_contract'
                                )
                            )
                            OR (
                                workflow.stage = 'aggregates'
                                AND worker_job_runs.job_type IN (
                                    'revise_beliefs',
                                    'sync_graph',
                                    'compact_summaries'
                                )
                            )
                            OR (
                                workflow.stage = 'finalizing'
                                AND worker_job_runs.job_type =
                                    'refresh_initial_context_package'
                            )
                        )
                  )
              )
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision =
                        worker_job_runs.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            RETURNING
                recovery_envelope_json,
                attempt_count,
                execution_fence,
                lifecycle_epoch,
                lifecycle_cleanup_key,
                derivation_revision
            """,
                (
                    JobRunStatus.RUNNING.value,
                    timestamp,
                    timestamp,
                    owner_id,
                    lease_expires_at,
                    notification.job_id,
                    JobRunStatus.AWAITING_CLAIM.value,
                    notification.dispatch_token,
                    timestamp,
                    notification.lifecycle_epoch,
                    notification.lifecycle_cleanup_key,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                await self.cancel_jobs_with_stale_derivation(
                    job_id=notification.job_id,
                    dispatch_token=notification.dispatch_token,
                    commit=False,
                )
            await self._connection.commit()
            if row is None:
                return None
            envelope_payload = json_utils.loads(str(row["recovery_envelope_json"]))
            return ClaimedJob(
                notification_message_id=notification_message_id,
                envelope=JobEnvelope.model_validate(envelope_payload),
                owner_id=owner_id,
                attempt_count=int(row["attempt_count"]),
                execution_fence=int(row["execution_fence"]),
                lifecycle_epoch=str(row["lifecycle_epoch"]),
                lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
                derivation_revision=int(row["derivation_revision"]),
            )
        except BaseException:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def heartbeat_claim(
        self,
        claim: ClaimedJob,
        *,
        lease_seconds: float,
        commit: bool = True,
    ) -> bool:
        started_transaction = not self._connection.in_transaction
        if started_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            now = self._clock.now()
            cursor = await self._connection.execute(
                """
            UPDATE worker_job_runs
            SET last_heartbeat_at = ?,
                execution_lease_expires_at = ?
            WHERE job_id = ?
              AND status = ?
              AND execution_owner = ?
              AND execution_fence = ?
              AND lifecycle_epoch = ?
              AND derivation_revision = ?
              AND execution_lease_expires_at > ?
              AND (
                  (
                      maintenance_operation_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS active_operation
                          WHERE (
                              active_operation.status = 'remediation_required'
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
                                OR active_operation.user_id = worker_job_runs.user_id
                            )
                      )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM admin_maintenance_operations AS owned_operation
                      WHERE owned_operation.id =
                          worker_job_runs.maintenance_operation_id
                        AND owned_operation.status = 'active'
                        AND julianday(owned_operation.lease_expires_at) >
                            julianday('now')
                        AND owned_operation.scope_kind = 'user'
                        AND owned_operation.user_id = worker_job_runs.user_id
                        AND owned_operation.lifecycle_epoch =
                            worker_job_runs.lifecycle_epoch
                        AND owned_operation.derivation_revision =
                            worker_job_runs.derivation_revision
                  )
              )
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision = worker_job_runs.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            """,
                (
                    now.isoformat(),
                    (now + timedelta(seconds=max(0.1, lease_seconds))).isoformat(),
                    claim.envelope.job_id,
                    JobRunStatus.RUNNING.value,
                    claim.owner_id,
                    claim.execution_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    now.isoformat(),
                ),
            )
            if commit:
                await self._connection.commit()
            return int(cursor.rowcount or 0) == 1
        except BaseException:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def claim_is_current(self, claim: ClaimedJob) -> bool:
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM worker_job_runs
            JOIN user_lifecycles AS lifecycle
              ON lifecycle.user_id = worker_job_runs.user_id
             AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
            LEFT JOIN users ON users.id = worker_job_runs.user_id
            WHERE worker_job_runs.job_id = ?
              AND worker_job_runs.status = ?
              AND worker_job_runs.execution_owner = ?
              AND worker_job_runs.execution_fence = ?
              AND worker_job_runs.lifecycle_epoch = ?
              AND worker_job_runs.derivation_revision = ?
              AND worker_job_runs.execution_lease_expires_at > ?
              AND (
                  (
                      worker_job_runs.maintenance_operation_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS active_operation
                          WHERE (
                              active_operation.status = 'remediation_required'
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
                                OR active_operation.user_id = worker_job_runs.user_id
                            )
                      )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM admin_maintenance_operations AS owned_operation
                      WHERE owned_operation.id =
                          worker_job_runs.maintenance_operation_id
                        AND owned_operation.status = 'active'
                        AND julianday(owned_operation.lease_expires_at) >
                            julianday('now')
                        AND owned_operation.scope_kind = 'user'
                        AND owned_operation.user_id = worker_job_runs.user_id
                        AND owned_operation.lifecycle_epoch =
                            worker_job_runs.lifecycle_epoch
                        AND owned_operation.derivation_revision =
                            worker_job_runs.derivation_revision
                  )
              )
              AND lifecycle.derivation_revision = worker_job_runs.derivation_revision
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND (
                  worker_job_runs.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
            LIMIT 1
            """,
            (
                claim.envelope.job_id,
                JobRunStatus.RUNNING.value,
                claim.owner_id,
                claim.execution_fence,
                claim.lifecycle_epoch,
                claim.derivation_revision,
                self._timestamp(),
            ),
        )
        return await cursor.fetchone() is not None

    async def finish_claim(
        self,
        claim: ClaimedJob,
        *,
        status: JobRunStatus,
        metadata: dict[str, Any] | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
        commit: bool = True,
    ) -> bool:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError("finish_claim requires a terminal status")
        del error_message  # Provider text is not safe terminal diagnostic data.
        started_transaction = not self._connection.in_transaction
        if started_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            timestamp = self._timestamp()
            safe_error_class = _truncate_error(error_class, limit=128)
            metadata_json = None
            if metadata:
                existing = await self.get_job(claim.envelope.job_id)
                existing_metadata = (
                    existing.get("metadata_json")
                    if existing and isinstance(existing.get("metadata_json"), dict)
                    else {}
                )
                metadata_json = json_utils.dumps(
                    {**existing_metadata, **metadata},
                    sort_keys=True,
                )
            cursor = await self._connection.execute(
                """
            UPDATE worker_job_runs
            SET status = ?,
                finished_at = ?,
                last_heartbeat_at = ?,
                duration_ms = CASE
                    WHEN started_at IS NOT NULL
                    THEN MAX(0.0, (julianday(?) - julianday(started_at)) * 86400000.0)
                    ELSE duration_ms
                END,
                error_class = ?,
                error_message = ?,
                metadata_json = COALESCE(?, metadata_json),
                terminal_diagnostics_json = ?,
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL
            WHERE job_id = ?
              AND status = ?
              AND execution_owner = ?
              AND execution_fence = ?
              AND lifecycle_epoch = ?
              AND derivation_revision = ?
              AND execution_lease_expires_at > ?
              AND (
                  (
                      maintenance_operation_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM admin_maintenance_operations AS active_operation
                          WHERE (
                              active_operation.status = 'remediation_required'
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
                                OR active_operation.user_id = worker_job_runs.user_id
                            )
                      )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM admin_maintenance_operations AS owned_operation
                      WHERE owned_operation.id =
                          worker_job_runs.maintenance_operation_id
                        AND owned_operation.status = 'active'
                        AND julianday(owned_operation.lease_expires_at) >
                            julianday('now')
                        AND owned_operation.scope_kind = 'user'
                        AND owned_operation.user_id = worker_job_runs.user_id
                        AND owned_operation.lifecycle_epoch =
                            worker_job_runs.lifecycle_epoch
                        AND owned_operation.derivation_revision =
                            worker_job_runs.derivation_revision
                  )
              )
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision = worker_job_runs.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            """,
                (
                    status.value,
                    timestamp,
                    timestamp,
                    timestamp,
                    safe_error_class,
                    None,
                    metadata_json,
                    json_utils.dumps(
                        {
                            "error_class": safe_error_class,
                            "fence": claim.execution_fence,
                            "owner": claim.owner_id,
                            "derivation_revision": claim.derivation_revision,
                        },
                        sort_keys=True,
                    ),
                    claim.envelope.job_id,
                    JobRunStatus.RUNNING.value,
                    claim.owner_id,
                    claim.execution_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    timestamp,
                ),
            )
            if commit:
                await self._connection.commit()
            return int(cursor.rowcount or 0) == 1
        except BaseException:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def finish_transcript_rebuild_after_revision_bump(
        self,
        claim: ClaimedJob,
        *,
        workflow_id: str,
        completion_revision: int,
        commit: bool = True,
    ) -> bool:
        """Atomically terminalize the coordinator after its final source bump.

        The selected-transcript transaction must keep the coordinator running
        while domain-table effect-fence triggers execute. Once the derivation
        bump (the last domain write) succeeds, this worker-table-only transition
        rebases and terminalizes that exact fenced claim in the same transaction.
        """

        if completion_revision != claim.derivation_revision + 1:
            raise ValueError("completion revision must immediately follow the claim")
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = 'succeeded',
                derivation_revision = ?,
                finished_at = ?,
                last_heartbeat_at = ?,
                duration_ms = CASE
                    WHEN started_at IS NOT NULL
                    THEN MAX(
                        0.0,
                        (julianday(?) - julianday(started_at)) * 86400000.0
                    )
                    ELSE duration_ms
                END,
                error_class = NULL,
                error_message = NULL,
                terminal_diagnostics_json = ?,
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL
            WHERE job_id = ?
              AND job_type = 'rebuild_selected_transcript'
              AND transcript_rebuild_id = ?
              AND status = 'running'
              AND execution_owner = ?
              AND execution_fence = ?
              AND lifecycle_epoch = ?
              AND derivation_revision = ?
              AND execution_lease_expires_at > ?
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  JOIN transcript_rebuild_workflows AS workflow
                    ON workflow.id = ?
                   AND workflow.user_id = lifecycle.user_id
                  JOIN conversation_transcript_selections AS selection
                    ON selection.current_workflow_id = workflow.id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision = ?
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND workflow.stage = 'complete'
                    AND workflow.completion_derivation_revision = ?
                    AND selection.state = 'complete'
              )
            """,
            (
                completion_revision,
                timestamp,
                timestamp,
                timestamp,
                json_utils.dumps(
                    {
                        "fence": claim.execution_fence,
                        "owner": claim.owner_id,
                        "derivation_revision": completion_revision,
                        "stage": "complete",
                    },
                    sort_keys=True,
                ),
                claim.envelope.job_id,
                workflow_id,
                claim.owner_id,
                claim.execution_fence,
                claim.lifecycle_epoch,
                claim.derivation_revision,
                timestamp,
                workflow_id,
                completion_revision,
                completion_revision,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0) == 1

    async def release_claim_for_retry(
        self,
        claim: ClaimedJob,
        *,
        error_class: str,
        error_message: str,
        deferred_until: str | None = None,
        is_transient_defer: bool | None = None,
        commit: bool = True,
    ) -> bool:
        del error_message  # Provider text is not safe durable diagnostic data.
        started_transaction = not self._connection.in_transaction
        if started_transaction:
            await self._connection.execute("BEGIN IMMEDIATE")
        try:
            timestamp = self._timestamp()
            resolved_transient_defer = (
                deferred_until is not None
                if is_transient_defer is None
                else is_transient_defer
            )
            next_status = (
                JobRunStatus.DEFERRED
                if resolved_transient_defer
                else JobRunStatus.RETRYING
            )
            cursor = await self._connection.execute(
                """
            UPDATE worker_job_runs
            SET status = ?,
                last_heartbeat_at = ?,
                error_class = ?,
                error_message = ?,
                deferred_until = ?,
                transient_defer_count = transient_defer_count + ?,
                first_deferred_at = CASE
                    WHEN ? = 0 THEN first_deferred_at
                    ELSE COALESCE(first_deferred_at, ?)
                END,
                last_deferred_at = CASE WHEN ? = 0 THEN last_deferred_at ELSE ? END,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL
            WHERE job_id = ?
              AND status = ?
              AND execution_owner = ?
              AND execution_fence = ?
              AND lifecycle_epoch = ?
              AND derivation_revision = ?
              AND execution_lease_expires_at > ?
              AND EXISTS (
                  SELECT 1
                  FROM user_lifecycles AS lifecycle
                  LEFT JOIN users ON users.id = lifecycle.user_id
                  WHERE lifecycle.user_id = worker_job_runs.user_id
                    AND lifecycle.lifecycle_epoch = worker_job_runs.lifecycle_epoch
                    AND lifecycle.derivation_revision = worker_job_runs.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND (
                        lifecycle.user_id = 'atagia_system'
                        OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
                    )
              )
            """,
                (
                    next_status.value,
                    timestamp,
                    error_class,
                    None,
                    deferred_until,
                    1 if resolved_transient_defer else 0,
                    1 if resolved_transient_defer else 0,
                    timestamp,
                    1 if resolved_transient_defer else 0,
                    timestamp,
                    claim.envelope.job_id,
                    JobRunStatus.RUNNING.value,
                    claim.owner_id,
                    claim.execution_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    timestamp,
                ),
            )
            if commit:
                await self._connection.commit()
            return int(cursor.rowcount or 0) == 1
        except BaseException:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def oldest_nonterminal_queued_at(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> str | None:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            """
            SELECT MIN(queued_at) AS queued_at
            FROM worker_job_runs
            WHERE {where_clause}
              AND status IN ({placeholders})
            """.format(where_clause=where_clause, placeholders=placeholders),
            (*parameters, *(status.value for status in NONTERMINAL_JOB_STATUSES)),
        )
        row = await cursor.fetchone()
        return (
            None if row is None or row["queued_at"] is None else str(row["queued_at"])
        )

    async def status_counts(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
        window_start: str | None = None,
        nonterminal_only: bool = False,
    ) -> list[dict[str, Any]]:
        where_clause, parameters = self._optional_scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        clauses = [where_clause] if where_clause else []
        if window_start is not None:
            clauses.append("queued_at >= ?")
            parameters.append(window_start)
        if nonterminal_only:
            placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in NONTERMINAL_JOB_STATUSES)
        final_where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return await self._fetch_all(
            """
            SELECT status, job_type, COUNT(*) AS count
            FROM worker_job_runs
            {where_clause}
            GROUP BY status, job_type
            ORDER BY status ASC, job_type ASC
            """.format(where_clause=final_where),
            tuple(parameters),
        )

    async def nonterminal_count(self) -> int:
        """Return the global durable-work count used by process-wide drains."""

        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM worker_job_runs
            WHERE status IN ({placeholders})
            """,
            tuple(status.value for status in NONTERMINAL_JOB_STATUSES),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["count"] or 0)

    async def assert_target_backend_compatible(self, target_backend: str) -> None:
        """Reject a backend switch that would strand durable nonterminal work."""

        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            f"""
            SELECT target_backend, COUNT(*) AS count
            FROM worker_job_runs
            WHERE status IN ({placeholders})
              AND target_backend != ?
            GROUP BY target_backend
            ORDER BY target_backend ASC
            """,
            (
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
                target_backend,
            ),
        )
        incompatible = [dict(row) for row in await cursor.fetchall()]
        await cursor.close()
        if not incompatible:
            return
        summary = ", ".join(
            f"{row['target_backend']}={int(row['count'])}" for row in incompatible
        )
        raise RuntimeError(
            "Cannot switch durable job backend while nonterminal jobs target "
            f"another backend ({summary}); restore that backend and drain or "
            "cancel the jobs before switching"
        )

    async def nonterminal_jobs(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> list[dict[str, Any]]:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        return await self._fetch_all(
            """
            SELECT *
            FROM worker_job_runs
            WHERE {where_clause}
              AND status IN ({placeholders})
            ORDER BY queued_at ASC, job_id ASC
            """.format(where_clause=where_clause, placeholders=placeholders),
            (*parameters, *(status.value for status in NONTERMINAL_JOB_STATUSES)),
        )

    async def source_message_job_exists(
        self,
        *,
        user_id: str,
        source_message_id: str,
        job_type: JobType | str,
        statuses: Iterable[JobRunStatus] | None = None,
    ) -> bool:
        """Return whether a tracked source-message job already exists."""
        resolved_statuses = tuple(
            statuses
            if statuses is not None
            else (
                JobRunStatus.QUEUED,
                JobRunStatus.AWAITING_CLAIM,
                JobRunStatus.RUNNING,
                JobRunStatus.RETRYING,
                JobRunStatus.DEFERRED,
                JobRunStatus.SUCCEEDED,
                JobRunStatus.SKIPPED,
            )
        )
        if not resolved_statuses:
            return False
        status_placeholders = ", ".join("?" for _ in resolved_statuses)
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM worker_job_runs AS wjr,
                 json_each(wjr.source_message_ids_json) AS source_message
            WHERE wjr.user_id = ?
              AND wjr.job_type = ?
              AND CAST(source_message.value AS TEXT) = ?
              AND wjr.status IN ({status_placeholders})
            LIMIT 1
            """.format(status_placeholders=status_placeholders),
            (
                user_id,
                job_type.value if isinstance(job_type, JobType) else str(job_type),
                source_message_id,
                *(status.value for status in resolved_statuses),
            ),
        )
        return await cursor.fetchone() is not None

    async def source_message_progress(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        namespace_filter: JobNamespaceFilter | None = None,
        window_start: str | None,
    ) -> dict[str, int]:
        if window_start is None:
            return {
                "tracked_source_messages": 0,
                "processed_source_messages": 0,
                "pending_source_messages": 0,
            }
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        root_placeholders = ", ".join("?" for _ in ROOT_JOB_TYPES)
        nonterminal_placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            """
            WITH root_jobs AS (
                SELECT job_id, status, source_message_ids_json
                FROM worker_job_runs
                WHERE {where_clause}
                  AND queued_at >= ?
                  AND job_type IN ({root_placeholders})
            ),
            source_jobs AS (
                SELECT
                    CAST(json_each.value AS TEXT) AS source_message_id,
                    root_jobs.status AS status
                FROM root_jobs, json_each(root_jobs.source_message_ids_json)
            ),
            source_rollup AS (
                SELECT
                    source_message_id,
                    SUM(CASE WHEN status IN ({nonterminal_placeholders}) THEN 1 ELSE 0 END) AS nonterminal_jobs
                FROM source_jobs
                WHERE source_message_id IS NOT NULL
                  AND source_message_id != ''
                GROUP BY source_message_id
            )
            SELECT
                COUNT(*) AS tracked_source_messages,
                COALESCE(SUM(CASE WHEN nonterminal_jobs = 0 THEN 1 ELSE 0 END), 0) AS processed_source_messages,
                COALESCE(SUM(CASE WHEN nonterminal_jobs > 0 THEN 1 ELSE 0 END), 0) AS pending_source_messages
            FROM source_rollup
            """.format(
                where_clause=where_clause,
                root_placeholders=root_placeholders,
                nonterminal_placeholders=nonterminal_placeholders,
            ),
            (
                *parameters,
                window_start,
                *(job_type.value for job_type in ROOT_JOB_TYPES),
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
            ),
        )
        row = await cursor.fetchone()
        return {
            "tracked_source_messages": int(row["tracked_source_messages"] or 0),
            "processed_source_messages": int(row["processed_source_messages"] or 0),
            "pending_source_messages": int(row["pending_source_messages"] or 0),
        }

    async def recent_completed_durations(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT job_type, size_bucket, duration_ms
            FROM worker_job_runs
            WHERE status = ?
              AND duration_ms IS NOT NULL
            ORDER BY finished_at DESC, job_id DESC
            LIMIT ?
            """,
            (JobRunStatus.SUCCEEDED.value, limit),
        )

    async def newest_job_queued_at(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> str | None:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        cursor = await self._connection.execute(
            """
            SELECT MAX(queued_at) AS queued_at
            FROM worker_job_runs
            WHERE {where_clause}
            """.format(where_clause=where_clause),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return (
            None if row is None or row["queued_at"] is None else str(row["queued_at"])
        )

    async def purge_for_user(self, user_id: str, *, commit: bool = True) -> int:
        cursor = await self._connection.execute(
            "DELETE FROM worker_job_runs WHERE user_id = ?",
            (user_id,),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def purge_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM worker_job_runs
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    def _scope_where_clause(
        *,
        user_id: str,
        conversation_id: str | None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> tuple[str, list[Any]]:
        where_clause, parameters = JobRunRepository._optional_scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        if not where_clause:
            raise ValueError("user_id is required")
        return where_clause, parameters

    @staticmethod
    def _optional_scope_where_clause(
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(conversation_id)
        if namespace_filter is not None:
            clauses.append("user_persona_id IS ?")
            parameters.append(namespace_filter.user_persona_id)
            platform_id = namespace_filter.platform_id or "default"
            if namespace_filter.incognito or not namespace_filter.remember_across_chats:
                clauses.append("conversation_id IS NOT NULL")
                clauses.append("incognito_snapshot = ?")
                parameters.append(1 if namespace_filter.incognito else 0)
                if not namespace_filter.remember_across_chats:
                    clauses.append("remember_across_chats_snapshot = 0")
            else:
                clauses.append("incognito_snapshot = 0")
                clauses.append("remember_across_chats_snapshot = 1")
                clauses.append("character_id IS ?")
                parameters.append(namespace_filter.character_id)
            if namespace_filter.remember_across_devices:
                clauses.append(
                    "(remember_across_devices_snapshot = 1 OR platform_id = ?)"
                )
                parameters.append(platform_id)
            else:
                clauses.append("platform_id = ?")
                parameters.append(platform_id)
        return " AND ".join(clauses), parameters


def job_status_values(statuses: Iterable[JobRunStatus]) -> tuple[str, ...]:
    """Return enum values as a tuple for callers that need raw SQL parameters."""
    return tuple(status.value for status in statuses)


def _truncate_error(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalized[:limit]


def _optional_identifier(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _stored_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        return normalized in {"1", "true", "yes", "on"}
    return bool(value)


def _recent_message_content(row: dict[str, Any]) -> str:
    if not _stored_bool(row.get("skip_by_default")) or _stored_bool(
        row.get("include_raw"), default=True
    ):
        return str(row["text"])
    existing = " ".join(str(row.get("context_placeholder") or "").split())[:300].strip()
    if existing:
        return existing
    message_id = str(row.get("id") or f"msg_{row.get('seq', '?')}")
    seq = row.get("seq")
    seq_value = str(seq) if seq is not None else "?"
    role = str(row.get("role") or "user")
    content_kind = " ".join(str(row.get("content_kind") or "text").split()).lower()[:64]
    policy_reason = " ".join(str(row.get("policy_reason") or "").split())[:128].strip()
    if not policy_reason:
        if _stored_bool(row.get("artifact_backed")):
            policy_reason = "artifact_backed"
        elif _stored_bool(row.get("verbatim_required")):
            policy_reason = "verbatim_required"
        elif _stored_bool(row.get("heavy_content")):
            policy_reason = "heavy_content"
        else:
            policy_reason = "skip_by_default"
    return (
        f"[Skipped message | id={message_id} seq={seq_value} role={role} "
        f"kind={content_kind or 'text'} policy={policy_reason} ref={message_id}]"
    )
