"""Durable stage coordinator for selected-transcript rebuilds."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.storage_backend import StorageBackend
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.repositories import summary_mirror_id
from atagia.models.schemas_jobs import (
    ClaimedJob,
    JobType,
    StreamMessage,
    TRANSCRIPT_REBUILD_STREAM_NAME,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.worker_control_service import (
    WorkerControlService,
    wait_if_worker_claims_paused,
)
from atagia.services.worker_effect_fence import WorkerEffectFence
from atagia.services.worker_job_lease import JobLeaseLostError, WorkerJobLease

logger = logging.getLogger(__name__)
WORKFLOW_POLL_SECONDS = 1.0
WORKER_ERROR_RETRY_SECONDS = 1.0

_SOURCE_JOB_TYPES = (
    JobType.EXTRACT_MEMORY_CANDIDATES,
    JobType.PROJECT_CONTRACT,
)
_AGGREGATE_JOB_TYPES = (
    JobType.REVISE_BELIEFS,
    JobType.SYNC_GRAPH,
    JobType.COMPACT_SUMMARIES,
)
_FINAL_JOB_TYPES = (JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,)


class _WorkflowPending(RuntimeError):
    """Internal nonfailure signal used to release the coordinator lease."""


class TranscriptRebuildWorker:
    """Advance a rebuild only after each durable child stage is terminal."""

    def __init__(
        self,
        *,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        clock: Clock,
        settings: Settings,
        embedding_index: EmbeddingIndex,
        job_connection: aiosqlite.Connection | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._connection = connection
        self._clock = clock
        self._settings = settings
        self._embedding_index = embedding_index
        self._worker_control = WorkerControlService(connection, clock)
        self._effect_fence = WorkerEffectFence(connection, clock)
        self._job_tracking = JobTrackingService(
            job_connection or connection,
            clock,
            workers_enabled=settings.workers_enabled,
            settings=settings,
            child_job_connection=connection,
        )
        self._repository = TranscriptRebuildRepository(connection, clock)
        self._stream_reclaim_idle_ms = int(
            settings.worker_stream_reclaim_idle_seconds * 1000
        )

    async def run(self, consumer_name: str = "transcript-rebuild-1") -> None:
        await self._storage_backend.stream_ensure_group(
            TRANSCRIPT_REBUILD_STREAM_NAME,
            WORKER_GROUP_NAME,
        )
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected selected-transcript worker error")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "transcript-rebuild-1",
        block_ms: int | None = 0,
    ) -> WorkerIterationResult:
        if await wait_if_worker_claims_paused(self._worker_control, block_ms=block_ms):
            return WorkerIterationResult()
        messages = await self._next_messages(
            consumer_name=consumer_name,
            block_ms=block_ms,
        )
        if not messages:
            return WorkerIterationResult()
        acked = 0
        failed = 0
        deferred = 0
        for message in messages:
            claim = await self._job_tracking.claim_notification(
                message,
                owner_id=consumer_name,
            )
            if claim is None:
                await self._ack(message)
                acked += 1
                continue
            lease = WorkerJobLease(
                self._job_tracking,
                claim,
                effect_fence=self._effect_fence,
            )
            try:
                finalized = False
                async with lease:
                    outcome = await self.process_claim(claim)
                    if outcome == "remediation_required":
                        await lease.skip(
                            reason="transcript_rebuild_remediation_required"
                        )
                    elif outcome == "ready_to_finalize":
                        finalized = await self._finalize_ready_workflow(claim)
                        if not finalized:
                            await lease.skip(
                                reason="transcript_rebuild_completion_validation_failed"
                            )
                    else:
                        pending = _WorkflowPending(outcome)
                        await lease.defer(
                            pending,
                            deferred_until=(
                                self._clock.now()
                                + timedelta(seconds=WORKFLOW_POLL_SECONDS)
                            ),
                        )
                await self._ack(message)
                acked += 1
                if not finalized:
                    deferred += 1
            except JobLeaseLostError:
                await self._ack(message)
                acked += 1
            except Exception as exc:
                failed += 1
                logger.exception(
                    "Selected-transcript workflow job failed job_id=%s",
                    claim.envelope.job_id,
                )
                try:
                    if self._connection.in_transaction:
                        await self._connection.rollback()
                    workflow_id = claim.envelope.transcript_rebuild_id
                    workflow = (
                        await self._repository.get_workflow(workflow_id)
                        if workflow_id is not None
                        else None
                    )
                    marked = workflow is not None and (
                        await self._repository.mark_remediation_required(
                            claim,
                            expected_stage=str(workflow["stage"]),
                            error_code="orchestrator_failure",
                            error_message=str(exc),
                        )
                    )
                    if not marked:
                        raise JobLeaseLostError(
                            "Selected-transcript remediation transition lost its fence"
                        )
                    await lease.fail(exc)
                    await self._ack(message)
                    acked += 1
                except JobLeaseLostError:
                    await self._ack(message)
                    acked += 1
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            deferred=deferred,
        )

    async def process_claim(self, claim: ClaimedJob) -> str:
        envelope = claim.envelope
        if envelope.job_type is not JobType.REBUILD_SELECTED_TRANSCRIPT:
            raise ValueError(
                f"Unsupported selected-transcript job type: {envelope.job_type}"
            )
        workflow_id = envelope.transcript_rebuild_id
        if workflow_id is None or envelope.payload.get("workflow_id") != workflow_id:
            raise ValueError("Selected-transcript workflow identity is missing")
        workflow = await self._repository.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError("Selected-transcript workflow does not exist")
        stage = str(workflow["stage"])
        if stage == "remediation_required":
            return "remediation_required"
        if stage == "ready_to_finalize":
            return "ready_to_finalize"
        if stage == "complete":
            return "ready_to_finalize"
        if stage == "preparing":
            return await self._complete_preparation(claim, workflow)
        if stage == "sources":
            return await self._advance_if_complete(
                claim,
                workflow_id=workflow_id,
                expected_stage="sources",
                job_types=_SOURCE_JOB_TYPES,
                next_stage="aggregates",
            )
        if stage == "aggregates":
            return await self._advance_if_complete(
                claim,
                workflow_id=workflow_id,
                expected_stage="aggregates",
                job_types=_AGGREGATE_JOB_TYPES,
                next_stage="finalizing",
            )
        if stage == "finalizing":
            return await self._advance_if_complete(
                claim,
                workflow_id=workflow_id,
                expected_stage="finalizing",
                job_types=_FINAL_JOB_TYPES,
                next_stage="ready_to_finalize",
            )
        raise ValueError(f"Unknown selected-transcript workflow stage: {stage}")

    async def _complete_preparation(
        self,
        claim: ClaimedJob,
        workflow: dict[str, Any],
    ) -> str:
        workflow_id = str(workflow["id"])
        if workflow.get("embedding_cleanup_completed_at") is None:
            memory_ids = [
                str(item) for item in (workflow.get("affected_memory_ids_json") or [])
            ]
            summary_ids = [
                summary_mirror_id(str(item))
                for item in (workflow.get("affected_summary_ids_json") or [])
            ]
            for memory_id in dict.fromkeys([*memory_ids, *summary_ids]):
                await self._embedding_index.delete(memory_id)
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._clock.now().isoformat()
                cursor = await self._connection.execute(
                    """
                UPDATE transcript_rebuild_workflows
                SET embedding_cleanup_completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND conversation_id = ?
                  AND orchestrator_job_id = ?
                  AND stage = 'preparing'
                  AND embedding_cleanup_completed_at IS NULL
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
                        timestamp,
                        timestamp,
                        workflow_id,
                        claim.envelope.user_id,
                        claim.envelope.conversation_id,
                        claim.envelope.job_id,
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
                    raise JobLeaseLostError(
                        "Selected-transcript preparation checkpoint lost its fence"
                    )
                await self._connection.commit()
            except BaseException:
                if self._connection.in_transaction:
                    await self._connection.rollback()
                raise
        if not await self._repository.transition_stage(
            claim,
            expected_stage="preparing",
            next_stage="sources",
        ):
            raise JobLeaseLostError(
                "Selected-transcript preparation transition lost its fence"
            )
        await self._job_tracking.dispatch_pending_jobs(self._storage_backend)
        return "entered_sources"

    async def _advance_if_complete(
        self,
        claim: ClaimedJob,
        *,
        workflow_id: str,
        expected_stage: str,
        job_types: tuple[JobType, ...],
        next_stage: str,
    ) -> str:
        jobs = await self._repository.list_jobs(
            workflow_id,
            job_types=job_types,
            exclude_job_id=claim.envelope.job_id,
        )
        failed = self._repository.failed_jobs(jobs)
        if failed:
            if not await self._repository.mark_remediation_required(
                claim,
                expected_stage=expected_stage,
                error_code="child_job_failed",
                error_message=(
                    "Selected-transcript child jobs failed: "
                    + ",".join(str(job["job_id"]) for job in failed[:20])
                ),
            ):
                raise JobLeaseLostError(
                    "Selected-transcript child failure lost its remediation fence"
                )
            return "remediation_required"
        if not self._repository.jobs_complete(jobs):
            return f"waiting_for_{next_stage}"
        if not await self._repository.transition_stage(
            claim,
            expected_stage=expected_stage,
            next_stage=next_stage,
        ):
            raise JobLeaseLostError(
                "Selected-transcript stage transition lost its fence"
            )
        await self._job_tracking.dispatch_pending_jobs(self._storage_backend)
        return (
            next_stage if next_stage == "ready_to_finalize" else f"entered_{next_stage}"
        )

    async def _finalize_ready_workflow(self, claim: ClaimedJob) -> bool:
        workflow_id = claim.envelope.transcript_rebuild_id
        if workflow_id is None:
            return False
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            workflow = await self._repository.get_workflow(workflow_id)
            if workflow is None or str(workflow["stage"]) == "complete":
                await self._connection.rollback()
                return False
            if str(workflow["stage"]) != "ready_to_finalize":
                raise RuntimeError("Selected-transcript workflow is not finalizable")
            await self._validate_terminal_closure(workflow)
            lifecycle = await UserLifecycleRepository(
                self._connection,
                self._clock,
            ).get_active_identity(str(workflow["user_id"]))
            if lifecycle is None:
                raise RuntimeError("Selected-transcript user lifecycle is not active")
            if lifecycle.derivation_revision != int(
                workflow["start_derivation_revision"]
            ):
                raise RuntimeError(
                    "Selected-transcript derivation revision changed during rebuild"
                )
            completion_revision = lifecycle.derivation_revision + 1
            cache_revision = await UserLifecycleRepository(
                self._connection,
                self._clock,
            ).bump_cache_revision(
                str(workflow["user_id"]),
                expected_lifecycle_epoch=lifecycle.lifecycle_epoch,
                commit=False,
            )
            if cache_revision is None:
                raise RuntimeError("Selected-transcript cache revision bump failed")
            timestamp = self._clock.now().isoformat()
            cursor = await self._connection.execute(
                """
                UPDATE transcript_rebuild_workflows
                SET stage = 'complete',
                    completion_derivation_revision = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?
                  AND stage = 'ready_to_finalize'
                """,
                (completion_revision, timestamp, timestamp, workflow_id),
            )
            if int(cursor.rowcount or 0) != 1:
                raise RuntimeError("Selected-transcript completion CAS failed")
            await self._connection.execute(
                """
                UPDATE conversation_transcript_selections
                SET state = 'complete', updated_at = ?
                WHERE current_workflow_id = ? AND state = 'rebuilding'
                """,
                (timestamp, workflow_id),
            )
            bumped_revision = await UserLifecycleRepository(
                self._connection,
                self._clock,
            ).bump_derivation_revision(
                str(workflow["user_id"]),
                expected_lifecycle_epoch=lifecycle.lifecycle_epoch,
                commit=False,
            )
            if bumped_revision != completion_revision:
                raise RuntimeError("Selected-transcript final derivation bump failed")
            job_repository = JobRunRepository(self._connection, self._clock)
            orchestrator_finished = (
                await job_repository.finish_transcript_rebuild_after_revision_bump(
                    claim,
                    workflow_id=workflow_id,
                    completion_revision=completion_revision,
                    commit=False,
                )
            )
            if not orchestrator_finished:
                raise JobLeaseLostError(
                    "Selected-transcript completion lost its orchestrator fence"
                )
            await job_repository.reconcile_stale_root_jobs_after_revision_bump(
                user_id=str(workflow["user_id"]),
                previous_revision=lifecycle.derivation_revision,
                new_revision=completion_revision,
                excluded_source_message_ids=(
                    workflow.get("abandoned_message_ids_json") or []
                ),
                allow_validated_requeue=True,
            )
            await self._connection.commit()
        except JobLeaseLostError:
            await self._connection.rollback()
            raise
        except Exception as exc:
            await self._connection.rollback()
            if not await self._repository.mark_remediation_required(
                claim,
                expected_stage="ready_to_finalize",
                error_code="completion_validation_failed",
                error_message=str(exc),
            ):
                raise JobLeaseLostError(
                    "Selected-transcript completion failure lost its remediation fence"
                ) from exc
            return False
        try:
            await self._storage_backend.delete_context_views_for_user(
                str(workflow["user_id"])
            )
            await self._storage_backend.delete_recent_windows_for_user(
                str(workflow["user_id"])
            )
        except Exception:
            logger.warning(
                "Selected-transcript cache cleanup deferred user_id=%s",
                workflow["user_id"],
                exc_info=True,
            )
        await self._job_tracking.dispatch_pending_jobs(self._storage_backend)
        return True

    async def _validate_terminal_closure(self, workflow: dict[str, Any]) -> None:
        workflow_id = str(workflow["id"])
        jobs = await self._repository.list_jobs(
            workflow_id,
            exclude_job_id=str(workflow["orchestrator_job_id"]),
        )
        failed = self._repository.failed_jobs(jobs)
        if failed or not self._repository.jobs_complete(jobs):
            raise RuntimeError(
                "Selected-transcript workflow jobs are not all successful"
            )
        targets = await self._repository.list_targets(workflow_id)
        for target in targets:
            message_id = str(target["message_id"])
            extract_ok = any(
                str(job["job_type"]) == JobType.EXTRACT_MEMORY_CANDIDATES.value
                and message_id in (job.get("source_message_ids_json") or [])
                and str(job["status"]) in {"succeeded", "skipped"}
                for job in jobs
            )
            contract_ok = not bool(target["require_contract"]) or any(
                str(job["job_type"]) == JobType.PROJECT_CONTRACT.value
                and message_id in (job.get("source_message_ids_json") or [])
                and str(job["status"]) in {"succeeded", "skipped"}
                for job in jobs
            )
            if not extract_ok or not contract_ok:
                raise RuntimeError(
                    f"Selected-transcript target closure is incomplete: {message_id}"
                )
        forced_compaction_ok = any(
            str(job["job_type"]) == JobType.COMPACT_SUMMARIES.value
            and str(job["status"]) == "succeeded"
            and bool((job.get("metadata_json") or {}).get("force_rebuild"))
            for job in jobs
        )
        if not forced_compaction_ok:
            raise RuntimeError(
                "Selected-transcript forced conversation compaction is incomplete"
            )
        icp_jobs = [
            job
            for job in jobs
            if str(job["job_type"]) == JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value
            and str(job["status"]) == "succeeded"
        ]
        if self._settings.initial_context_package_refresh_enabled:
            icp_jobs = [
                job
                for job in icp_jobs
                if (job.get("metadata_json") or {}).get("reason") == "source_changed"
                and (job.get("metadata_json") or {}).get("status") == "refreshed"
            ]
        else:
            icp_jobs = [
                job
                for job in icp_jobs
                if (job.get("metadata_json") or {}).get("reason") == "refresh_disabled"
                and (job.get("metadata_json") or {}).get("status") == "skipped"
            ]
        if not icp_jobs:
            raise RuntimeError(
                "Selected-transcript initial context rebuild is incomplete"
            )
        await self._validate_initial_context_package_closure(workflow, icp_jobs)
        await self._validate_abandoned_sources_absent(workflow)
        await self._validate_verbatim_pin_closure(workflow)
        await self._validate_activity_consistency(workflow)
        await self._validate_conversation_summary_consistency(workflow)
        await self._validate_communication_profile_closure(workflow)

    async def _validate_initial_context_package_closure(
        self,
        workflow: dict[str, Any],
        icp_jobs: list[dict[str, Any]],
    ) -> None:
        if not self._settings.initial_context_package_refresh_enabled:
            return
        refresh_generation = max(
            int((job.get("metadata_json") or {}).get("refresh_generation") or 0)
            for job in icp_jobs
        )
        if refresh_generation <= 0:
            raise RuntimeError(
                "Selected-transcript initial context refresh generation is missing"
            )
        cursor = await self._connection.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM initial_context_packages AS package
                JOIN user_lifecycles AS user_lifecycle
                  ON user_lifecycle.user_id = package.user_id
                JOIN conversation_lifecycles AS conversation_lifecycle
                  ON conversation_lifecycle.user_id = package.user_id
                 AND conversation_lifecycle.conversation_id = package.conversation_id
                WHERE package.user_id = ?
                  AND package.conversation_id = ?
                  AND package.package_kind = 'conversation'
                  AND package.build_status = 'active'
                  AND package.refresh_generation >= ?
                  AND package.source_user_lifecycle_epoch =
                      user_lifecycle.lifecycle_epoch
                  AND package.source_user_revision = user_lifecycle.source_revision
                  AND package.source_conversation_lifecycle_epoch =
                      conversation_lifecycle.lifecycle_epoch
                  AND package.source_conversation_revision =
                      conversation_lifecycle.source_revision
            ) AS package_current
            """,
            (
                workflow["user_id"],
                workflow["conversation_id"],
                refresh_generation,
            ),
        )
        row = await cursor.fetchone()
        if row is None or not bool(row["package_current"]):
            raise RuntimeError(
                "Selected-transcript initial context package is missing or stale"
            )

    async def _validate_abandoned_sources_absent(
        self,
        workflow: dict[str, Any],
    ) -> None:
        abandoned = workflow.get("abandoned_message_ids_json") or []
        if not abandoned:
            return
        encoded = json_utils.dumps(
            abandoned,
            sort_keys=True,
        )
        cursor = await self._connection.execute(
            """
            SELECT
                EXISTS(
                    SELECT 1 FROM messages
                    JOIN json_each(?) AS old ON CAST(old.value AS TEXT) = messages.id
                )
                OR EXISTS(
                    SELECT 1 FROM memory_objects AS memory
                    JOIN json_each(json_extract(memory.payload_json, '$.source_message_ids')) AS source
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = CAST(source.value AS TEXT)
                    WHERE memory.user_id = ? AND memory.status IN ('active', 'review_required')
                )
                OR EXISTS(
                    SELECT 1 FROM memory_evidence_spans AS span
                    JOIN json_each(?) AS old ON CAST(old.value AS TEXT) = span.message_id
                    WHERE span.user_id = ?
                )
                OR EXISTS(
                    SELECT 1 FROM memory_fact_facets AS facet
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = facet.source_message_id
                    WHERE facet.user_id = ? AND facet.current_state = 1
                )
                OR EXISTS(
                    SELECT 1 FROM graph_entity_mentions AS mention
                    JOIN json_each(?) AS old ON CAST(old.value AS TEXT) = mention.message_id
                    WHERE mention.user_id = ?
                )
                OR EXISTS(
                    SELECT 1 FROM graph_relationship_sources AS source
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = source.message_id
                      OR (
                          source.source_kind = 'message'
                          AND CAST(old.value AS TEXT) = source.source_id
                      )
                    WHERE source.user_id = ?
                )
                OR EXISTS(
                    SELECT 1 FROM conversation_topic_sources AS source
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = source.source_id
                    WHERE source.user_id = ?
                      AND source.source_kind = 'message'
                )
                OR EXISTS(
                    SELECT 1 FROM artifacts AS artifact
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = artifact.message_id
                    WHERE artifact.user_id = ?
                      AND artifact.status NOT IN ('deleted', 'purged')
                )
                OR EXISTS(
                    SELECT 1 FROM artifact_links AS link
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = link.message_id
                    WHERE link.user_id = ?
                )
                OR EXISTS(
                    SELECT 1 FROM proxy_turn_runs AS turn
                    JOIN json_each(?) AS old
                      ON CAST(old.value AS TEXT) = turn.request_message_id
                      OR CAST(old.value AS TEXT) = turn.response_message_id
                    WHERE turn.user_id = ?
                )
                OR EXISTS(
                    SELECT 1
                    FROM graph_entities AS entity
                    WHERE entity.user_id = ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM graph_entity_mentions AS mention
                          WHERE mention.user_id = entity.user_id
                            AND mention.entity_id = entity.id
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM graph_relationships AS relationship
                          WHERE relationship.user_id = entity.user_id
                            AND (
                                relationship.source_entity_id = entity.id
                                OR relationship.target_entity_id = entity.id
                            )
                      )
                ) AS abandoned_reachable
            """,
            (
                encoded,
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                encoded,
                workflow["user_id"],
                workflow["user_id"],
            ),
        )
        row = await cursor.fetchone()
        if row is not None and bool(row["abandoned_reachable"]):
            raise RuntimeError("Abandoned selected-transcript source remains reachable")

    async def _validate_verbatim_pin_closure(
        self,
        workflow: dict[str, Any],
    ) -> None:
        abandoned = [
            str(item) for item in (workflow.get("abandoned_message_ids_json") or [])
        ]
        affected_memories = [
            str(item) for item in (workflow.get("affected_memory_ids_json") or [])
        ]
        affected_memories.extend(
            summary_mirror_id(str(item))
            for item in (workflow.get("affected_summary_ids_json") or [])
        )
        abandoned_json = json_utils.dumps(abandoned, sort_keys=True)
        affected_memory_json = json_utils.dumps(
            list(dict.fromkeys(affected_memories)),
            sort_keys=True,
        )
        cursor = await self._connection.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM verbatim_pins AS pin
                WHERE pin.user_id = ?
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM json_each(?) AS abandoned
                          WHERE CAST(abandoned.value AS TEXT) IN (
                              pin.target_id,
                              json_extract(pin.payload_json, '$.source_target_id'),
                              json_extract(
                                  pin.payload_json,
                                  '$.source_snapshot.message_id'
                              )
                          )
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM json_each(
                              CASE
                                  WHEN json_valid(pin.payload_json) = 1
                                  THEN json_extract(
                                      pin.payload_json,
                                      '$.source_message_ids'
                                  )
                                  ELSE '[]'
                              END
                          ) AS source
                          JOIN json_each(?) AS abandoned
                            ON CAST(abandoned.value AS TEXT) =
                               CAST(source.value AS TEXT)
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM json_each(?) AS affected
                          WHERE CAST(affected.value AS TEXT) IN (
                              pin.target_id,
                              json_extract(pin.payload_json, '$.source_target_id'),
                              json_extract(
                                  pin.payload_json,
                                  '$.source_snapshot.memory_id'
                              )
                          )
                      )
                  )
            ) AS stale_pin_reachable
            """,
            (
                workflow["user_id"],
                abandoned_json,
                abandoned_json,
                affected_memory_json,
            ),
        )
        row = await cursor.fetchone()
        if row is not None and bool(row["stale_pin_reachable"]):
            raise RuntimeError(
                "Selected-transcript verbatim pin retains abandoned evidence"
            )

    async def _validate_activity_consistency(self, workflow: dict[str, Any]) -> None:
        cursor = await self._connection.execute(
            """
            SELECT
                (
                    activity.message_count = (
                        SELECT COUNT(*)
                        FROM messages AS message
                        WHERE message.conversation_id = conversation.id
                    )
                    AND activity.user_message_count = (
                        SELECT COUNT(*)
                        FROM messages AS message
                        WHERE message.conversation_id = conversation.id
                          AND message.role = 'user'
                    )
                    AND activity.assistant_message_count = (
                        SELECT COUNT(*)
                        FROM messages AS message
                        WHERE message.conversation_id = conversation.id
                          AND message.role = 'assistant'
                    )
                    AND activity.retrieval_count = (
                        SELECT COUNT(*)
                        FROM retrieval_events AS event
                        WHERE event.user_id = conversation.user_id
                          AND event.conversation_id = conversation.id
                    )
                    AND datetime(conversation.last_activity_at) IS datetime(
                        COALESCE(
                            (
                                SELECT COALESCE(message.occurred_at, message.created_at)
                                FROM messages AS message
                                WHERE message.conversation_id = conversation.id
                                ORDER BY
                                    datetime(COALESCE(message.occurred_at, message.created_at)) DESC,
                                    message.seq DESC,
                                    message.id DESC
                                LIMIT 1
                            ),
                            conversation.created_at
                        )
                    )
                    AND datetime(activity.last_message_at) IS datetime(
                        (
                            SELECT COALESCE(message.occurred_at, message.created_at)
                            FROM messages AS message
                            WHERE message.conversation_id = conversation.id
                            ORDER BY
                                datetime(COALESCE(message.occurred_at, message.created_at)) DESC,
                                message.seq DESC,
                                message.id DESC
                            LIMIT 1
                        )
                    )
                ) AS activity_consistent
            FROM conversations AS conversation
            JOIN conversation_activity_stats AS activity
              ON activity.user_id = conversation.user_id
             AND activity.conversation_id = conversation.id
            WHERE conversation.user_id = ?
              AND conversation.id = ?
            """,
            (workflow["user_id"], workflow["conversation_id"]),
        )
        row = await cursor.fetchone()
        if row is None or not bool(row["activity_consistent"]):
            raise RuntimeError(
                "Selected-transcript conversation activity is not canonical"
            )

    async def _validate_communication_profile_closure(
        self,
        workflow: dict[str, Any],
    ) -> None:
        abandoned = [
            str(item) for item in (workflow.get("abandoned_message_ids_json") or [])
        ]
        affected_memories = [
            str(item) for item in (workflow.get("affected_memory_ids_json") or [])
        ]
        affected_memories.extend(
            summary_mirror_id(str(item))
            for item in (workflow.get("affected_summary_ids_json") or [])
        )
        abandoned_json = json_utils.dumps(abandoned, sort_keys=True)
        affected_memory_json = json_utils.dumps(
            list(dict.fromkeys(affected_memories)),
            sort_keys=True,
        )
        cursor = await self._connection.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM user_communication_profiles AS profile
                JOIN json_each(
                    CASE
                        WHEN json_valid(profile.source_refs_json) = 1
                        THEN profile.source_refs_json
                        ELSE '[]'
                    END
                ) AS source_ref
                WHERE profile.user_id = ?
                  AND profile.status = 'active'
                  AND profile.stale = 0
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM json_each(?) AS abandoned
                          WHERE CAST(abandoned.value AS TEXT) =
                                json_extract(source_ref.value, '$.source_message_id')
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM json_each(?) AS affected
                          WHERE CAST(affected.value AS TEXT) =
                                json_extract(source_ref.value, '$.memory_id')
                      )
                      OR (
                          json_extract(source_ref.value, '$.source_kind') =
                              'message_window'
                          AND json_extract(source_ref.value, '$.conversation_id') = ?
                      )
                  )
            ) AS stale_profile_reachable
            """,
            (
                workflow["user_id"],
                abandoned_json,
                affected_memory_json,
                workflow["conversation_id"],
            ),
        )
        row = await cursor.fetchone()
        if row is not None and bool(row["stale_profile_reachable"]):
            raise RuntimeError(
                "Selected-transcript communication profile retains abandoned evidence"
            )

    async def _validate_conversation_summary_consistency(
        self,
        workflow: dict[str, Any],
    ) -> None:
        cursor = await self._connection.execute(
            """
            SELECT
                conversation.temporary,
                conversation.purge_on_close,
                user.remember_across_devices,
                COUNT(message.id) AS message_count,
                MIN(message.seq) AS first_message_seq,
                MAX(message.seq) AS last_message_seq
            FROM conversations AS conversation
            JOIN users AS user ON user.id = conversation.user_id
            LEFT JOIN messages AS message
              ON message.conversation_id = conversation.id
            WHERE conversation.user_id = ? AND conversation.id = ?
            GROUP BY conversation.id
            """,
            (workflow["user_id"], workflow["conversation_id"]),
        )
        source = await cursor.fetchone()
        if source is None:
            raise RuntimeError("Selected-transcript conversation disappeared")
        cursor = await self._connection.execute(
            """
            SELECT
                COUNT(*) AS chunk_count,
                MIN(source_message_start_seq) AS first_chunk_seq,
                MAX(source_message_end_seq) AS last_chunk_seq,
                EXISTS(
                    SELECT 1
                    FROM messages AS message
                    WHERE message.conversation_id = ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM summary_views AS covering_chunk
                          WHERE covering_chunk.user_id = ?
                            AND covering_chunk.conversation_id = ?
                            AND covering_chunk.summary_kind = 'conversation_chunk'
                            AND message.seq BETWEEN
                                covering_chunk.source_message_start_seq
                                AND covering_chunk.source_message_end_seq
                      )
                ) AS has_uncovered_message,
                EXISTS(
                    SELECT 1
                    FROM summary_views AS invalid_chunk
                    WHERE invalid_chunk.user_id = ?
                      AND invalid_chunk.conversation_id = ?
                      AND invalid_chunk.summary_kind = 'conversation_chunk'
                      AND (
                          invalid_chunk.source_message_start_seq IS NULL
                          OR invalid_chunk.source_message_end_seq IS NULL
                          OR invalid_chunk.source_message_start_seq >
                              invalid_chunk.source_message_end_seq
                      )
                ) AS has_invalid_chunk
            FROM summary_views AS chunk
            WHERE chunk.user_id = ?
              AND chunk.conversation_id = ?
              AND chunk.summary_kind = 'conversation_chunk'
            """,
            (
                workflow["conversation_id"],
                workflow["user_id"],
                workflow["conversation_id"],
                workflow["user_id"],
                workflow["conversation_id"],
                workflow["user_id"],
                workflow["conversation_id"],
            ),
        )
        chunks = await cursor.fetchone()
        assert chunks is not None
        message_count = int(source["message_count"])
        chunk_count = int(chunks["chunk_count"])
        chunking_enabled = (
            not bool(source["temporary"])
            and not bool(source["purge_on_close"])
            and bool(source["remember_across_devices"])
        )
        if message_count == 0 or not chunking_enabled:
            if chunk_count != 0:
                raise RuntimeError(
                    "Selected-transcript stale conversation chunks remain"
                )
            return
        if (
            chunk_count == 0
            or bool(chunks["has_uncovered_message"])
            or bool(chunks["has_invalid_chunk"])
            or int(chunks["first_chunk_seq"]) != int(source["first_message_seq"])
            or int(chunks["last_chunk_seq"]) != int(source["last_message_seq"])
        ):
            raise RuntimeError(
                "Selected-transcript conversation chunks do not cover canonical messages"
            )

    async def _ack(self, message: StreamMessage) -> None:
        await self._storage_backend.stream_ack(
            TRANSCRIPT_REBUILD_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            TRANSCRIPT_REBUILD_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=self._stream_reclaim_idle_ms,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            TRANSCRIPT_REBUILD_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            count=1,
            block_ms=block_ms,
        )
