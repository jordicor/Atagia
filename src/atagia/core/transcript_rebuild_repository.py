"""Durable selected-transcript rebuild state and stage gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from atagia.core.repositories import BaseRepository
from atagia.models.schemas_jobs import ClaimedJob, JobRunStatus, JobType
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
    UserDeletedError,
)


_FAILED_JOB_STATUSES = {
    JobRunStatus.FAILED.value,
    JobRunStatus.DEAD_LETTERED.value,
    JobRunStatus.CANCELLED.value,
}
_TERMINAL_JOB_STATUSES = {
    JobRunStatus.SUCCEEDED.value,
    JobRunStatus.SKIPPED.value,
    *_FAILED_JOB_STATUSES,
}


@dataclass(frozen=True, slots=True)
class UserAvailabilitySnapshot:
    """Exact canonical source identity observed while memory access was safe."""

    lifecycle_epoch: str
    derivation_revision: int


class TranscriptRebuildRepository(BaseRepository):
    """Persist workflow state without treating transient queues as authority."""

    async def get_current_selection(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversation_transcript_selections
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )

    async def get_blocking_selection(
        self,
        user_id: str,
        *,
        allowed_workflow_id: str | None = None,
    ) -> dict[str, Any] | None:
        exclusion = (
            "AND current_workflow_id != ?" if allowed_workflow_id is not None else ""
        )
        parameters: tuple[Any, ...] = (
            (user_id, allowed_workflow_id)
            if allowed_workflow_id is not None
            else (user_id,)
        )
        return await self._fetch_one(
            f"""
            SELECT *
            FROM conversation_transcript_selections
            WHERE user_id = ?
              AND state IN ('rebuilding', 'remediation_required')
              {exclusion}
            ORDER BY updated_at ASC, conversation_id ASC
            LIMIT 1
            """,
            parameters,
        )

    async def require_user_available(
        self,
        user_id: str,
        *,
        allowed_workflow_id: str | None = None,
        allowed_maintenance_operation_id: str | None = None,
    ) -> None:
        blocking = await self.get_blocking_selection(
            user_id,
            allowed_workflow_id=allowed_workflow_id,
        )
        if blocking is None:
            maintenance = await self.get_active_maintenance_operation(
                user_id,
                allowed_operation_id=allowed_maintenance_operation_id,
            )
            if maintenance is None:
                return
            raise TranscriptRebuildInProgressError(
                "An admin maintenance operation is still in progress"
            )
        if str(blocking["state"]) == "remediation_required":
            raise TranscriptRebuildRemediationRequiredError(
                "Selected transcript rebuild requires remediation before memory access"
            )
        raise TranscriptRebuildInProgressError(
            "Selected transcript rebuild is still in progress"
        )

    async def get_active_maintenance_operation(
        self,
        user_id: str,
        *,
        allowed_operation_id: str | None = None,
    ) -> dict[str, Any] | None:
        exclusion = "AND id != ?" if allowed_operation_id is not None else ""
        parameters: tuple[Any, ...] = (
            (user_id, allowed_operation_id)
            if allowed_operation_id is not None
            else (user_id,)
        )
        return await self._fetch_one(
            f"""
            SELECT *
            FROM admin_maintenance_operations
            WHERE (
                  status = 'remediation_required'
                  OR (
                      status = 'active'
                      AND (
                          phase = 'dirty'
                          OR julianday(lease_expires_at) > julianday('now')
                      )
                  )
              )
              AND (scope_kind = 'global' OR user_id = ?)
              {exclusion}
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            parameters,
        )

    async def capture_user_availability_snapshot(
        self,
        user_id: str,
        *,
        allowed_workflow_id: str | None = None,
        allowed_maintenance_operation_id: str | None = None,
    ) -> UserAvailabilitySnapshot:
        """Capture one active source revision with no rebuild blocker.

        Callers that perform work outside a SQLite transaction must validate the
        returned snapshot immediately before publishing or returning the result.
        """

        exclusion = (
            "AND selection.current_workflow_id != ?"
            if allowed_workflow_id is not None
            else ""
        )
        maintenance_exclusion = (
            "AND operation.id != ?"
            if allowed_maintenance_operation_id is not None
            else ""
        )
        parameters_list: list[Any] = [user_id]
        if allowed_workflow_id is not None:
            parameters_list.append(allowed_workflow_id)
        if allowed_maintenance_operation_id is not None:
            parameters_list.append(allowed_maintenance_operation_id)
        row = await self._fetch_one(
            f"""
            SELECT lifecycle.lifecycle_epoch, lifecycle.derivation_revision
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM conversation_transcript_selections AS selection
                  WHERE selection.user_id = lifecycle.user_id
                    AND selection.state IN ('rebuilding', 'remediation_required')
                    {exclusion}
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM admin_maintenance_operations AS operation
                  WHERE (
                      operation.status = 'remediation_required'
                      OR (
                          operation.status = 'active'
                          AND (
                              operation.phase = 'dirty'
                              OR julianday(operation.lease_expires_at) >
                                  julianday('now')
                          )
                      )
                  )
                    AND (
                        operation.scope_kind = 'global'
                        OR operation.user_id = lifecycle.user_id
                    )
                    {maintenance_exclusion}
              )
            """,
            tuple(parameters_list),
        )
        if row is not None:
            return UserAvailabilitySnapshot(
                lifecycle_epoch=str(row["lifecycle_epoch"]),
                derivation_revision=int(row["derivation_revision"]),
            )
        await self.require_user_available(
            user_id,
            allowed_workflow_id=allowed_workflow_id,
            allowed_maintenance_operation_id=allowed_maintenance_operation_id,
        )
        raise UserDeletedError("User has been erased or does not exist")

    async def require_user_availability_snapshot(
        self,
        user_id: str,
        snapshot: UserAvailabilitySnapshot,
        *,
        allowed_workflow_id: str | None = None,
        allowed_maintenance_operation_id: str | None = None,
    ) -> None:
        """Require the exact captured source identity and no rebuild blocker."""

        exclusion = (
            "AND selection.current_workflow_id != ?"
            if allowed_workflow_id is not None
            else ""
        )
        maintenance_exclusion = (
            "AND operation.id != ?"
            if allowed_maintenance_operation_id is not None
            else ""
        )
        parameters_list: list[Any] = [
            user_id,
            snapshot.lifecycle_epoch,
            snapshot.derivation_revision,
        ]
        if allowed_workflow_id is not None:
            parameters_list.append(allowed_workflow_id)
        if allowed_maintenance_operation_id is not None:
            parameters_list.append(allowed_maintenance_operation_id)
        current = await self._fetch_one(
            f"""
            SELECT 1 AS current
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.lifecycle_epoch = ?
              AND lifecycle.derivation_revision = ?
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM conversation_transcript_selections AS selection
                  WHERE selection.user_id = lifecycle.user_id
                    AND selection.state IN ('rebuilding', 'remediation_required')
                    {exclusion}
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM admin_maintenance_operations AS operation
                  WHERE (
                      operation.status = 'remediation_required'
                      OR (
                          operation.status = 'active'
                          AND (
                              operation.phase = 'dirty'
                              OR julianday(operation.lease_expires_at) >
                                  julianday('now')
                          )
                      )
                  )
                    AND (
                        operation.scope_kind = 'global'
                        OR operation.user_id = lifecycle.user_id
                    )
                    {maintenance_exclusion}
              )
            """,
            tuple(parameters_list),
        )
        if current is not None:
            return
        await self.require_user_available(
            user_id,
            allowed_workflow_id=allowed_workflow_id,
            allowed_maintenance_operation_id=allowed_maintenance_operation_id,
        )
        active = await self._fetch_one(
            """
            SELECT 1 AS active
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
            """,
            (user_id,),
        )
        if active is None:
            raise UserDeletedError("User has been erased or does not exist")
        raise TranscriptRebuildInProgressError(
            "Memory sources changed while the request was in progress; retry"
        )

    async def require_scope_available(
        self,
        user_id: str | None = None,
        *,
        allowed_maintenance_operation_id: str | None = None,
    ) -> None:
        """Block a user scope, or a global operation if any user is rebuilding."""

        if user_id is not None:
            await self.require_user_available(
                user_id,
                allowed_maintenance_operation_id=allowed_maintenance_operation_id,
            )
            return
        blocking = await self._fetch_one(
            """
            SELECT state
            FROM conversation_transcript_selections
            WHERE state IN ('rebuilding', 'remediation_required')
            ORDER BY updated_at ASC, conversation_id ASC
            LIMIT 1
            """,
            (),
        )
        if blocking is None:
            exclusion = (
                "AND id != ?" if allowed_maintenance_operation_id is not None else ""
            )
            parameters: tuple[Any, ...] = (
                (allowed_maintenance_operation_id,)
                if allowed_maintenance_operation_id is not None
                else ()
            )
            maintenance = await self._fetch_one(
                f"""
                SELECT id
                FROM admin_maintenance_operations
                WHERE (
                      status = 'remediation_required'
                      OR (
                          status = 'active'
                          AND (
                              phase = 'dirty'
                              OR julianday(lease_expires_at) > julianday('now')
                          )
                      )
                  )
                  {exclusion}
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                parameters,
            )
            if maintenance is None:
                return
            raise TranscriptRebuildInProgressError(
                "An admin maintenance operation is still in progress"
            )
        if str(blocking["state"]) == "remediation_required":
            raise TranscriptRebuildRemediationRequiredError(
                "A selected transcript rebuild requires remediation before global memory access"
            )
        raise TranscriptRebuildInProgressError(
            "A selected transcript rebuild is still in progress"
        )

    async def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            "SELECT * FROM transcript_rebuild_workflows WHERE id = ?",
            (workflow_id,),
        )

    async def get_workflow_for_operation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        operation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM transcript_rebuild_workflows
            WHERE user_id = ?
              AND conversation_id = ?
              AND operation_id = ?
            """,
            (user_id, conversation_id, operation_id),
        )

    async def list_targets(self, workflow_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM transcript_rebuild_targets
            WHERE workflow_id = ?
            ORDER BY conversation_id ASC, message_id ASC
            """,
            (workflow_id,),
        )

    async def list_jobs(
        self,
        workflow_id: str,
        *,
        job_types: Iterable[JobType] | None = None,
        exclude_job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["transcript_rebuild_id = ?"]
        parameters: list[Any] = [workflow_id]
        normalized_types = tuple(job_type.value for job_type in (job_types or ()))
        if normalized_types:
            placeholders = ", ".join("?" for _ in normalized_types)
            clauses.append(f"job_type IN ({placeholders})")
            parameters.extend(normalized_types)
        if exclude_job_id is not None:
            clauses.append("job_id != ?")
            parameters.append(exclude_job_id)
        return await self._fetch_all(
            f"""
            SELECT *
            FROM worker_job_runs
            WHERE {" AND ".join(clauses)}
            ORDER BY queued_at ASC, job_id ASC
            """,
            tuple(parameters),
        )

    @staticmethod
    def jobs_complete(jobs: Iterable[dict[str, Any]]) -> bool:
        return all(str(job["status"]) in _TERMINAL_JOB_STATUSES for job in jobs)

    @staticmethod
    def failed_jobs(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [job for job in jobs if str(job["status"]) in _FAILED_JOB_STATUSES]

    async def transition_stage(
        self,
        claim: ClaimedJob,
        *,
        expected_stage: str,
        next_stage: str,
    ) -> bool:
        workflow_id = claim.envelope.transcript_rebuild_id
        if workflow_id is None:
            return False
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            timestamp = self._timestamp()
            cursor = await self._connection.execute(
                """
                UPDATE transcript_rebuild_workflows
                SET stage = ?,
                    resume_stage = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND conversation_id = ?
                  AND orchestrator_job_id = ?
                  AND stage = ?
                  AND EXISTS (
                      SELECT 1
                      FROM worker_job_runs AS job
                      JOIN user_lifecycles AS lifecycle
                        ON lifecycle.user_id = job.user_id
                      WHERE job.job_id = ?
                        AND job.job_type = 'rebuild_selected_transcript'
                        AND job.user_id = transcript_rebuild_workflows.user_id
                        AND job.conversation_id =
                            transcript_rebuild_workflows.conversation_id
                        AND job.transcript_rebuild_id =
                            transcript_rebuild_workflows.id
                        AND job.status = 'running'
                        AND job.execution_owner = ?
                        AND job.execution_fence = ?
                        AND julianday(job.execution_lease_expires_at) >
                            julianday(?)
                        AND job.lifecycle_epoch = ?
                        AND job.derivation_revision = ?
                        AND lifecycle.lifecycle_epoch = job.lifecycle_epoch
                        AND lifecycle.derivation_revision = job.derivation_revision
                        AND lifecycle.state = 'active'
                        AND lifecycle.erasure_cleanup_id IS NULL
                  )
                """,
                (
                    next_stage,
                    next_stage,
                    timestamp,
                    workflow_id,
                    claim.envelope.user_id,
                    claim.envelope.conversation_id,
                    claim.envelope.job_id,
                    expected_stage,
                    claim.envelope.job_id,
                    claim.owner_id,
                    claim.execution_fence,
                    timestamp,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                await self._connection.rollback()
                return False
            await self._connection.commit()
            return True
        except BaseException:
            if self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def mark_remediation_required(
        self,
        claim: ClaimedJob,
        *,
        expected_stage: str,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Fence a remediation transition to the current orchestrator claim."""

        workflow_id = claim.envelope.transcript_rebuild_id
        if workflow_id is None:
            return False
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            timestamp = self._timestamp()
            workflow_cursor = await self._connection.execute(
                """
                UPDATE transcript_rebuild_workflows
                SET stage = 'remediation_required',
                    resume_stage = stage,
                    error_code = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND conversation_id = ?
                  AND orchestrator_job_id = ?
                  AND stage = ?
                  AND EXISTS (
                      SELECT 1
                      FROM worker_job_runs AS job
                      JOIN user_lifecycles AS lifecycle
                        ON lifecycle.user_id = job.user_id
                      WHERE job.job_id = ?
                        AND job.job_type = 'rebuild_selected_transcript'
                        AND job.user_id = transcript_rebuild_workflows.user_id
                        AND job.conversation_id =
                            transcript_rebuild_workflows.conversation_id
                        AND job.transcript_rebuild_id =
                            transcript_rebuild_workflows.id
                        AND job.status = 'running'
                        AND job.execution_owner = ?
                        AND job.execution_fence = ?
                        AND julianday(job.execution_lease_expires_at) >
                            julianday(?)
                        AND job.lifecycle_epoch = ?
                        AND job.derivation_revision = ?
                        AND lifecycle.lifecycle_epoch = job.lifecycle_epoch
                        AND lifecycle.derivation_revision =
                            job.derivation_revision
                        AND lifecycle.state = 'active'
                        AND lifecycle.erasure_cleanup_id IS NULL
                  )
                """,
                (
                    error_code,
                    error_message[:1000],
                    timestamp,
                    workflow_id,
                    claim.envelope.user_id,
                    claim.envelope.conversation_id,
                    claim.envelope.job_id,
                    expected_stage,
                    claim.envelope.job_id,
                    claim.owner_id,
                    claim.execution_fence,
                    timestamp,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                ),
            )
            if int(workflow_cursor.rowcount or 0) != 1:
                await self._connection.rollback()
                return False
            selection_cursor = await self._connection.execute(
                """
                UPDATE conversation_transcript_selections
                SET state = 'remediation_required',
                    updated_at = ?
                WHERE user_id = ?
                  AND conversation_id = ?
                  AND current_workflow_id = ?
                  AND state = 'rebuilding'
                """,
                (
                    timestamp,
                    claim.envelope.user_id,
                    claim.envelope.conversation_id,
                    workflow_id,
                ),
            )
            if int(selection_cursor.rowcount or 0) != 1:
                await self._connection.rollback()
                return False
            await self._connection.commit()
            return True
        except BaseException:
            if self._connection.in_transaction:
                await self._connection.rollback()
            raise
