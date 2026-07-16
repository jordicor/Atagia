"""Atomic selected-transcript replacement and durable targeted rebuild setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.canonical import canonical_json_hash
from atagia.core.ids import generate_prefixed_id, new_job_id
from atagia.core.job_run_repository import JobRunRepository, NONTERMINAL_JOB_STATUSES
from atagia.core.presence_repository import PresenceRepository, presence_snapshot
from atagia.core.repositories import (
    MessageRepository,
    UserRepository,
    summary_mirror_id,
)
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_api import (
    ReplaceSelectedTranscriptRequest,
    SelectedTranscriptMessage,
    SelectedTranscriptRebuildResponse,
)
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    CompactionJobKind,
    CompactionJobPayload,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobType,
    TRANSCRIPT_REBUILD_STREAM_NAME,
)
from atagia.models.schemas_memory import (
    IngestOrigin,
    resolve_confirmation_strategy,
    resolve_memory_privacy_mode,
)
from atagia.services.chat_support import build_message_jobs
from atagia.services.conversation_activity_service import ConversationActivityService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    ConversationNotActiveError,
    ConversationNotFoundError,
    TranscriptRebuildUnavailableError,
    TranscriptSelectionConflictError,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.initial_context_package_refresh_service import (
    prepare_initial_context_package_refresh_payload,
)
from atagia.transport_ids import encode_path_id

if TYPE_CHECKING:
    from atagia.app import AppRuntime


@dataclass(frozen=True, slots=True)
class _ReplacementInventory:
    abandoned_message_ids: list[str]
    supporting_message_ids: list[str]
    interrupted_message_ids: list[str]
    affected_memory_ids: list[str]
    affected_summary_ids: list[str]
    artifact_ids: list[str]
    artifact_payload_blob_ids: list[str]
    communication_profile_ids: list[str]
    preserved_edited_memory_ids: list[str]


class SelectedTranscriptService:
    """Replace one host selection and launch a revision-fenced rebuild DAG."""

    def __init__(self, runtime: AppRuntime) -> None:
        self._runtime = runtime

    async def replace(
        self,
        *,
        conversation_id: str,
        request: ReplaceSelectedTranscriptRequest,
    ) -> SelectedTranscriptRebuildResponse:
        if not self._runtime.settings.workers_enabled:
            raise TranscriptRebuildUnavailableError(
                "Selected transcript replacement requires durable workers"
            )
        self._validate_complete_turns(request.messages)
        transcript_hash = self._transcript_hash(request)
        cache_service = ContextCacheService(self._runtime)
        async with cache_service.user_cache_guard(request.user_id):
            connection = await self._runtime.open_connection()
            try:
                response, _inventory = await self._install_replacement(
                    connection,
                    conversation_id=conversation_id,
                    request=request,
                    transcript_hash=transcript_hash,
                )
            finally:
                await connection.close()

            connection = await self._runtime.open_connection()
            try:
                await self._dispatch_pending(connection)
            finally:
                await connection.close()

            if response.idempotent_replay:
                return response
            await cache_service.invalidate_conversation_cache_by_id(
                request.user_id,
                conversation_id,
            )
            return response

    async def get_status(
        self,
        *,
        user_id: str,
        conversation_id: str,
        operation_id: str,
    ) -> SelectedTranscriptRebuildResponse:
        connection = await self._runtime.open_connection()
        try:
            repository = TranscriptRebuildRepository(connection, self._runtime.clock)
            workflow = await repository.get_workflow_for_operation(
                user_id=user_id,
                conversation_id=conversation_id,
                operation_id=operation_id,
            )
            if workflow is None:
                raise ConversationNotFoundError(
                    "Selected transcript rebuild operation was not found"
                )
            return self._response(workflow, idempotent_replay=True)
        finally:
            await connection.close()

    async def retry(
        self,
        *,
        user_id: str,
        conversation_id: str,
        operation_id: str,
    ) -> SelectedTranscriptRebuildResponse:
        """Restart a failed workflow from its canonical selected transcript."""

        if not self._runtime.settings.workers_enabled:
            raise TranscriptRebuildUnavailableError(
                "Selected transcript replacement requires durable workers"
            )
        cache_service = ContextCacheService(self._runtime)
        async with cache_service.user_cache_guard(user_id):
            connection = await self._runtime.open_connection()
            try:
                await connection.execute("BEGIN IMMEDIATE")
                repository = TranscriptRebuildRepository(
                    connection,
                    self._runtime.clock,
                )
                workflow = await repository.get_workflow_for_operation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    operation_id=operation_id,
                )
                if workflow is None:
                    raise ConversationNotFoundError(
                        "Selected transcript rebuild operation was not found"
                    )
                if str(workflow["stage"]) == "complete":
                    await connection.rollback()
                    return self._response(workflow, idempotent_replay=True)
                if str(workflow["stage"]) != "remediation_required":
                    raise TranscriptSelectionConflictError(
                        "Selected transcript rebuild is not awaiting remediation"
                    )
                current = await repository.get_current_selection(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if (
                    current is None
                    or str(current["current_workflow_id"]) != str(workflow["id"])
                    or str(current["state"]) != "remediation_required"
                ):
                    raise TranscriptSelectionConflictError(
                        "Selected transcript workflow is no longer canonical"
                    )
                await repository.require_user_available(
                    user_id,
                    allowed_workflow_id=str(workflow["id"]),
                )

                targets = await repository.list_targets(str(workflow["id"]))
                target_message_ids = [str(target["message_id"]) for target in targets]
                affected_memory_ids = await self._affected_memory_ids(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    source_message_ids=target_message_ids,
                )
                supporting_ids = await self._supporting_message_ids(
                    connection,
                    user_id=user_id,
                    memory_ids=affected_memory_ids,
                    excluded_message_ids=[
                        *target_message_ids,
                        *(workflow.get("abandoned_message_ids_json") or []),
                    ],
                )
                summary_ids = await self._summary_ids_for_rebuild(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_ids=affected_memory_ids,
                )
                (
                    profile_ids,
                    profile_supporting_ids,
                ) = await self._communication_profile_rebuild_inventory(
                    connection,
                    user_id=user_id,
                    rewritten_conversation_id=conversation_id,
                    invalidated_message_ids=target_message_ids,
                    affected_memory_ids=[
                        *affected_memory_ids,
                        *[summary_mirror_id(item) for item in summary_ids],
                    ],
                )
                supporting_ids = self._stable_ids(
                    [*supporting_ids, *profile_supporting_ids]
                )
                if supporting_ids:
                    await self._insert_targets(
                        connection,
                        workflow_id=str(workflow["id"]),
                        user_id=user_id,
                        selected_message_ids=set(),
                        supporting_message_ids=set(supporting_ids),
                        interrupted_message_ids=set(),
                        target_message_ids=supporting_ids,
                    )
                    targets = await repository.list_targets(str(workflow["id"]))

                inventory = _ReplacementInventory(
                    abandoned_message_ids=list(
                        workflow.get("abandoned_message_ids_json") or []
                    ),
                    supporting_message_ids=supporting_ids,
                    interrupted_message_ids=[],
                    affected_memory_ids=affected_memory_ids,
                    affected_summary_ids=summary_ids,
                    artifact_ids=[],
                    artifact_payload_blob_ids=[],
                    communication_profile_ids=profile_ids,
                    preserved_edited_memory_ids=[],
                )
                await self._delete_affected_state(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    inventory=inventory,
                )

                timestamp = self._runtime.clock.now().isoformat()
                lifecycle = await UserLifecycleRepository(
                    connection,
                    self._runtime.clock,
                ).get_active_identity(user_id)
                if lifecycle is None:
                    raise ConversationNotActiveError("User lifecycle is not active")
                new_revision = await UserLifecycleRepository(
                    connection,
                    self._runtime.clock,
                ).bump_derivation_revision(
                    user_id,
                    expected_lifecycle_epoch=lifecycle.lifecycle_epoch,
                    commit=False,
                )
                if new_revision is None:
                    raise ConversationNotActiveError("User lifecycle changed")
                await self._cancel_workflow_attempt_jobs(
                    connection,
                    workflow_id=str(workflow["id"]),
                    timestamp=timestamp,
                )
                await JobRunRepository(
                    connection,
                    self._runtime.clock,
                ).reconcile_stale_root_jobs_after_revision_bump(
                    user_id=user_id,
                    previous_revision=lifecycle.derivation_revision,
                    new_revision=new_revision,
                    excluded_source_message_ids=(
                        workflow.get("abandoned_message_ids_json") or []
                    ),
                    allow_validated_requeue=True,
                )
                await connection.execute(
                    """
                    UPDATE worker_job_runs
                    SET transcript_rebuild_id = NULL
                    WHERE transcript_rebuild_id = ?
                    """,
                    (workflow["id"],),
                )

                orchestrator_job_id = new_job_id()
                affected_union = self._stable_ids(
                    [
                        *(workflow.get("affected_memory_ids_json") or []),
                        *affected_memory_ids,
                    ]
                )
                summary_union = self._stable_ids(
                    [
                        *(workflow.get("affected_summary_ids_json") or []),
                        *summary_ids,
                    ]
                )
                cursor = await connection.execute(
                    """
                    UPDATE transcript_rebuild_workflows
                    SET orchestrator_job_id = ?,
                        stage = 'preparing',
                        resume_stage = 'preparing',
                        start_derivation_revision = ?,
                        affected_memory_ids_json = ?,
                        affected_summary_ids_json = ?,
                        embedding_cleanup_completed_at = NULL,
                        retry_count = retry_count + 1,
                        error_code = NULL,
                        error_message = NULL,
                        updated_at = ?
                    WHERE id = ? AND stage = 'remediation_required'
                    """,
                    (
                        orchestrator_job_id,
                        new_revision,
                        json_utils.dumps(affected_union, sort_keys=True),
                        json_utils.dumps(summary_union, sort_keys=True),
                        timestamp,
                        workflow["id"],
                    ),
                )
                if int(cursor.rowcount or 0) != 1:
                    raise TranscriptSelectionConflictError(
                        "Selected transcript retry lost its workflow fence"
                    )
                await connection.execute(
                    """
                    UPDATE conversation_transcript_selections
                    SET state = 'rebuilding', updated_at = ?
                    WHERE current_workflow_id = ?
                      AND state = 'remediation_required'
                    """,
                    (timestamp, workflow["id"]),
                )
                await self._enqueue_workflow_jobs(
                    connection,
                    workflow_id=str(workflow["id"]),
                    orchestrator_job_id=orchestrator_job_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    targets=targets,
                )
                await connection.commit()
                refreshed = await repository.get_workflow(str(workflow["id"]))
                if refreshed is None:
                    raise RuntimeError("Selected transcript retry commit was lost")
                await self._dispatch_pending(connection)
            except Exception:
                if connection.in_transaction:
                    await connection.rollback()
                raise
            finally:
                await connection.close()

            await cache_service.invalidate_conversation_cache_by_id(
                user_id,
                conversation_id,
            )
            return self._response(refreshed, idempotent_replay=False)

    async def _existing_replay(
        self,
        connection: aiosqlite.Connection,
        *,
        conversation_id: str,
        request: ReplaceSelectedTranscriptRequest,
        transcript_hash: str,
    ) -> SelectedTranscriptRebuildResponse | None:
        repository = TranscriptRebuildRepository(connection, self._runtime.clock)
        operation = await repository.get_workflow_for_operation(
            user_id=request.user_id,
            conversation_id=conversation_id,
            operation_id=request.operation_id,
        )
        if operation is not None:
            if (
                int(operation["selection_epoch"]) != request.selection_epoch
                or str(operation["transcript_hash"]) != transcript_hash
            ):
                raise TranscriptSelectionConflictError(
                    "operation_id already belongs to a different transcript selection"
                )
            return self._response(operation, idempotent_replay=True)

        current = await repository.get_current_selection(
            user_id=request.user_id,
            conversation_id=conversation_id,
        )
        if current is None:
            return None
        current_epoch = int(current["selection_epoch"])
        if request.selection_epoch < current_epoch:
            raise TranscriptSelectionConflictError(
                "Selected transcript epoch is older than canonical state"
            )
        if request.selection_epoch == current_epoch:
            if str(current["transcript_hash"]) != transcript_hash:
                raise TranscriptSelectionConflictError(
                    "Selected transcript epoch was reused with different content"
                )
            workflow = await repository.get_workflow(
                str(current["current_workflow_id"])
            )
            if workflow is None:
                raise RuntimeError("Canonical transcript selection lost its workflow")
            return self._response(workflow, idempotent_replay=True)
        if str(current["state"]) != "complete":
            raise TranscriptSelectionConflictError(
                "A newer selected transcript cannot replace an unfinished rebuild"
            )
        return None

    async def _install_replacement(
        self,
        connection: aiosqlite.Connection,
        *,
        conversation_id: str,
        request: ReplaceSelectedTranscriptRequest,
        transcript_hash: str,
    ) -> tuple[SelectedTranscriptRebuildResponse, _ReplacementInventory | None]:
        timestamp = self._runtime.clock.now().isoformat()
        workflow_id = generate_prefixed_id("trb")
        orchestrator_job_id = new_job_id()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            replay = await self._existing_replay(
                connection,
                conversation_id=conversation_id,
                request=request,
                transcript_hash=transcript_hash,
            )
            if replay is not None:
                await connection.rollback()
                return replay, None
            await TranscriptRebuildRepository(
                connection,
                self._runtime.clock,
            ).require_user_available(request.user_id)
            conversation = await self._conversation_for_update(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
                platform_id=request.platform_id,
            )
            existing_messages = await self._conversation_messages(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
            )
            await self._validate_and_backfill_retained_messages(
                connection,
                existing_messages,
                request.messages,
            )
            self._validate_retained_cutoff(
                existing_messages,
                request.messages,
                request.retained_cutoff_message_id,
            )
            await self._validate_global_message_ids(
                connection,
                conversation_id=conversation_id,
                messages=request.messages,
            )
            selected_ids = [message.message_id for message in request.messages]
            abandoned_ids = [
                str(message["id"])
                for message in existing_messages
                if str(message["id"]) not in set(selected_ids)
            ]
            affected_memory_ids = await self._affected_memory_ids(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
                source_message_ids=abandoned_ids,
            )
            preserved_edited_memory_ids = await self._preserved_edited_memory_ids(
                connection,
                user_id=request.user_id,
                source_message_ids=abandoned_ids,
            )
            supporting_ids = await self._supporting_message_ids(
                connection,
                user_id=request.user_id,
                memory_ids=affected_memory_ids,
                excluded_message_ids=abandoned_ids,
            )
            summary_ids = await self._summary_ids_for_rebuild(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
                memory_ids=affected_memory_ids,
            )
            (
                profile_ids,
                profile_supporting_ids,
            ) = await self._communication_profile_rebuild_inventory(
                connection,
                user_id=request.user_id,
                rewritten_conversation_id=conversation_id,
                invalidated_message_ids=abandoned_ids,
                affected_memory_ids=[
                    *affected_memory_ids,
                    *[summary_mirror_id(item) for item in summary_ids],
                ],
            )
            supporting_ids = self._stable_ids(
                [*supporting_ids, *profile_supporting_ids]
            )
            interrupted_ids: list[str] = []
            (
                artifact_ids,
                artifact_payload_blob_ids,
            ) = await self._artifacts_owned_only_by_messages(
                connection,
                user_id=request.user_id,
                message_ids=abandoned_ids,
            )
            inventory = _ReplacementInventory(
                abandoned_message_ids=abandoned_ids,
                supporting_message_ids=supporting_ids,
                interrupted_message_ids=interrupted_ids,
                affected_memory_ids=affected_memory_ids,
                affected_summary_ids=summary_ids,
                artifact_ids=artifact_ids,
                artifact_payload_blob_ids=artifact_payload_blob_ids,
                communication_profile_ids=profile_ids,
                preserved_edited_memory_ids=preserved_edited_memory_ids,
            )

            lifecycle = await UserLifecycleRepository(
                connection,
                self._runtime.clock,
            ).get_active_identity(request.user_id)
            if lifecycle is None:
                raise ConversationNotActiveError("User lifecycle is not active")
            new_revision = await UserLifecycleRepository(
                connection,
                self._runtime.clock,
            ).bump_derivation_revision(
                request.user_id,
                expected_lifecycle_epoch=lifecycle.lifecycle_epoch,
                commit=False,
            )
            if new_revision is None:
                raise ConversationNotActiveError("User lifecycle changed")

            await connection.execute(
                """
                INSERT INTO transcript_rebuild_workflows(
                    id, operation_id, user_id, conversation_id, selection_epoch,
                    transcript_hash, retained_cutoff_message_id, mutation_kind,
                    selected_message_ids_json, abandoned_message_ids_json,
                    supporting_message_ids_json, affected_memory_ids_json,
                    affected_summary_ids_json, orchestrator_job_id, stage,
                    start_derivation_revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?, ?)
                """,
                (
                    workflow_id,
                    request.operation_id,
                    request.user_id,
                    conversation_id,
                    request.selection_epoch,
                    transcript_hash,
                    request.retained_cutoff_message_id,
                    request.mutation_kind,
                    json_utils.dumps(selected_ids, sort_keys=True),
                    json_utils.dumps(abandoned_ids, sort_keys=True),
                    json_utils.dumps(supporting_ids, sort_keys=True),
                    json_utils.dumps(affected_memory_ids, sort_keys=True),
                    json_utils.dumps(summary_ids, sort_keys=True),
                    orchestrator_job_id,
                    new_revision,
                    timestamp,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                INSERT INTO conversation_transcript_selections(
                    user_id, conversation_id, selection_epoch, transcript_hash,
                    current_workflow_id, state, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'preparing', ?)
                ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                    selection_epoch = excluded.selection_epoch,
                    transcript_hash = excluded.transcript_hash,
                    current_workflow_id = excluded.current_workflow_id,
                    state = excluded.state,
                    updated_at = excluded.updated_at
                """,
                (
                    request.user_id,
                    conversation_id,
                    request.selection_epoch,
                    transcript_hash,
                    workflow_id,
                    timestamp,
                ),
            )

            await self._delete_affected_state(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
                inventory=inventory,
            )
            await self._detach_abandoned_provenance_from_edited_memories(
                connection,
                user_id=request.user_id,
                memory_ids=inventory.preserved_edited_memory_ids,
                abandoned_message_ids=inventory.abandoned_message_ids,
            )
            await self._replace_message_suffix(
                connection,
                conversation=conversation,
                selected_messages=request.messages,
                abandoned_message_ids=abandoned_ids,
            )
            await self._recompute_conversation_activity(
                connection,
                user_id=request.user_id,
                conversation_id=conversation_id,
            )
            target_ids = self._stable_ids(
                [*selected_ids, *supporting_ids, *interrupted_ids]
            )
            targets = await self._insert_targets(
                connection,
                workflow_id=workflow_id,
                user_id=request.user_id,
                selected_message_ids=set(selected_ids),
                supporting_message_ids=set(supporting_ids),
                interrupted_message_ids=set(interrupted_ids),
                target_message_ids=target_ids,
            )
            await self._cancel_pre_rebuild_jobs(
                connection,
                user_id=request.user_id,
                previous_revision=lifecycle.derivation_revision,
                new_revision=new_revision,
                abandoned_message_ids=abandoned_ids,
                timestamp=timestamp,
            )
            await connection.execute(
                """
                UPDATE conversation_transcript_selections
                SET state = 'rebuilding', updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (timestamp, request.user_id, conversation_id),
            )
            await self._enqueue_workflow_jobs(
                connection,
                workflow_id=workflow_id,
                orchestrator_job_id=orchestrator_job_id,
                user_id=request.user_id,
                conversation_id=conversation_id,
                targets=targets,
            )
            await connection.commit()
            workflow = await TranscriptRebuildRepository(
                connection,
                self._runtime.clock,
            ).get_workflow(workflow_id)
            if workflow is None:
                raise RuntimeError("Selected transcript workflow commit was lost")
            return self._response(workflow, idempotent_replay=False), inventory
        except Exception:
            if connection.in_transaction:
                await connection.rollback()
            raise

    async def _conversation_for_update(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        platform_id: str,
    ) -> dict[str, Any]:
        cursor = await connection.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError("Conversation not found for user")
        conversation = dict(row)
        if str(conversation["status"]) != "active":
            raise ConversationNotActiveError("Conversation is not active")
        if str(conversation.get("platform_id") or "") != platform_id:
            raise TranscriptSelectionConflictError(
                "Selected transcript platform does not match the conversation"
            )
        return conversation

    async def _conversation_messages(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        cursor = await connection.execute(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ? AND c.id = ?
            ORDER BY m.seq ASC, m.id ASC
            """,
            (user_id, conversation_id),
        )
        rows = [dict(row) for row in await cursor.fetchall()]
        for row in rows:
            if isinstance(row.get("metadata_json"), str):
                row["metadata_json"] = json_utils.loads(str(row["metadata_json"]))
        return rows

    @staticmethod
    def _validate_complete_turns(messages: list[SelectedTranscriptMessage]) -> None:
        if len(messages) % 2:
            raise TranscriptSelectionConflictError(
                "Selected transcript must contain complete user/assistant turns"
            )
        expected_roles = ["user", "assistant"]
        for index, message in enumerate(messages):
            if message.role != expected_roles[index % 2]:
                raise TranscriptSelectionConflictError(
                    "Selected transcript roles must alternate user then assistant"
                )

    async def _validate_and_backfill_retained_messages(
        self,
        connection: aiosqlite.Connection,
        existing: list[dict[str, Any]],
        selected: list[SelectedTranscriptMessage],
    ) -> None:
        existing_by_id = {str(message["id"]): message for message in existing}
        for message in selected:
            retained = existing_by_id.get(message.message_id)
            if retained is None:
                continue
            if (
                str(retained["role"]) != message.role
                or int(retained["seq"]) != message.source_seq
                or str(retained["text"]) != message.text
                or (retained.get("occurred_at") or None) != message.occurred_at
            ):
                raise TranscriptSelectionConflictError(
                    "A retained message identity was reused with different content"
                )
            expected_identity = self._selected_transcript_identity(message)
            metadata = retained.get("metadata_json")
            if metadata is None:
                metadata = {}
            elif isinstance(metadata, dict):
                metadata = dict(metadata)
            else:
                raise TranscriptSelectionConflictError(
                    "A retained message has non-object metadata that cannot be backfilled safely"
                )
            stored_identity = metadata.get("selected_transcript_identity")
            if stored_identity is None:
                metadata["selected_transcript_identity"] = expected_identity
                cursor = await connection.execute(
                    """
                    UPDATE messages
                    SET metadata_json = ?
                    WHERE id = ?
                      AND conversation_id = ?
                      AND (
                          json_type(
                              metadata_json,
                              '$.selected_transcript_identity'
                          ) IS NULL
                          OR json_type(
                              metadata_json,
                              '$.selected_transcript_identity'
                          ) = 'null'
                      )
                    """,
                    (
                        json_utils.dumps(metadata, sort_keys=True),
                        message.message_id,
                        str(retained["conversation_id"]),
                    ),
                )
                if int(cursor.rowcount or 0) != 1:
                    raise TranscriptSelectionConflictError(
                        "A retained message stable identity changed during validation"
                    )
                retained["metadata_json"] = metadata
                stored_identity = expected_identity
            if stored_identity != expected_identity:
                raise TranscriptSelectionConflictError(
                    "A retained message identity was reused with different host provenance"
                )

    @staticmethod
    def _selected_transcript_identity(
        message: SelectedTranscriptMessage,
    ) -> dict[str, str]:
        return {
            "host_message_id": message.host_message_id,
            "generation_id": message.generation_id,
            "source_namespace": message.source_namespace,
        }

    @staticmethod
    def _validate_retained_cutoff(
        existing: list[dict[str, Any]],
        selected: list[SelectedTranscriptMessage],
        retained_cutoff: str | None,
    ) -> None:
        common: str | None = None
        for old, new in zip(existing, selected, strict=False):
            if str(old["id"]) != new.message_id:
                break
            common = new.message_id
        if retained_cutoff != common:
            raise TranscriptSelectionConflictError(
                "retained_cutoff_message_id does not match the canonical common prefix"
            )

    async def _validate_global_message_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        conversation_id: str,
        messages: list[SelectedTranscriptMessage],
    ) -> None:
        if not messages:
            return
        placeholders = ", ".join("?" for _ in messages)
        cursor = await connection.execute(
            f"""
            SELECT id, conversation_id
            FROM messages
            WHERE id IN ({placeholders})
              AND conversation_id != ?
            LIMIT 1
            """,
            (*[message.message_id for message in messages], conversation_id),
        )
        if await cursor.fetchone() is not None:
            raise TranscriptSelectionConflictError(
                "A selected message identity belongs to another conversation"
            )

    async def _affected_memory_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        source_message_ids: list[str],
    ) -> list[str]:
        if not source_message_ids:
            return []
        source_json = json_utils.dumps(source_message_ids, sort_keys=True)
        cursor = await connection.execute(
            """
            SELECT DISTINCT mo.id
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_edit_history AS history
                  WHERE history.memory_id = mo.id
              )
              AND (
                  EXISTS (
                      SELECT 1
                      FROM json_each(json_extract(mo.payload_json, '$.source_message_ids')) AS src
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = CAST(src.value AS TEXT)
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_evidence_spans AS span
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = span.message_id
                      WHERE span.user_id = mo.user_id
                        AND span.memory_id = mo.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_fact_facets AS facet
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = facet.source_message_id
                      WHERE facet.user_id = mo.user_id
                        AND facet.memory_id = mo.id
                  )
                  OR (
                      mo.conversation_id = ?
                      AND mo.extraction_hash IS NOT NULL
                  )
              )
            ORDER BY mo.id ASC
            """,
            (user_id, source_json, source_json, source_json, conversation_id),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]

    async def _preserved_edited_memory_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        source_message_ids: list[str],
    ) -> list[str]:
        if not source_message_ids:
            return []
        source_json = json_utils.dumps(source_message_ids, sort_keys=True)
        cursor = await connection.execute(
            """
            SELECT DISTINCT memory.id
            FROM memory_objects AS memory
            WHERE memory.user_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM memory_edit_history AS history
                  WHERE history.memory_id = memory.id
              )
              AND (
                  EXISTS (
                      SELECT 1
                      FROM json_each(
                          json_extract(memory.payload_json, '$.source_message_ids')
                      ) AS source
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = CAST(source.value AS TEXT)
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_evidence_spans AS span
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = span.message_id
                      WHERE span.user_id = memory.user_id
                        AND span.memory_id = memory.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_fact_facets AS facet
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = facet.source_message_id
                      WHERE facet.user_id = memory.user_id
                        AND facet.memory_id = memory.id
                  )
              )
            ORDER BY memory.id ASC
            """,
            (user_id, source_json, source_json, source_json),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]

    async def _detach_abandoned_provenance_from_edited_memories(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        abandoned_message_ids: list[str],
    ) -> None:
        if not memory_ids or not abandoned_message_ids:
            return
        abandoned = set(abandoned_message_ids)
        memory_placeholders = ", ".join("?" for _ in memory_ids)
        message_placeholders = ", ".join("?" for _ in abandoned_message_ids)
        cursor = await connection.execute(
            f"""
            SELECT id, payload_json
            FROM memory_objects
            WHERE user_id = ? AND id IN ({memory_placeholders})
            """,
            (user_id, *memory_ids),
        )
        for row in await cursor.fetchall():
            payload = json_utils.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                payload = {}
            sources = payload.get("source_message_ids")
            if isinstance(sources, list):
                payload["source_message_ids"] = [
                    str(source) for source in sources if str(source) not in abandoned
                ]
                await connection.execute(
                    """
                    UPDATE memory_objects
                    SET payload_json = ?, updated_at = ?
                    WHERE user_id = ? AND id = ?
                    """,
                    (
                        json_utils.dumps(payload, sort_keys=True),
                        self._runtime.clock.now().isoformat(),
                        user_id,
                        row["id"],
                    ),
                )
        await connection.execute(
            f"""
            DELETE FROM memory_evidence_spans
            WHERE user_id = ?
              AND memory_id IN ({memory_placeholders})
              AND message_id IN ({message_placeholders})
            """,
            (user_id, *memory_ids, *abandoned_message_ids),
        )
        await connection.execute(
            f"""
            DELETE FROM memory_fact_facets
            WHERE user_id = ?
              AND memory_id IN ({memory_placeholders})
              AND source_message_id IN ({message_placeholders})
            """,
            (user_id, *memory_ids, *abandoned_message_ids),
        )

    async def _supporting_message_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        excluded_message_ids: list[str],
    ) -> list[str]:
        if not memory_ids:
            return []
        memory_json = json_utils.dumps(memory_ids, sort_keys=True)
        excluded = set(excluded_message_ids)
        cursor = await connection.execute(
            """
            WITH candidate_ids(id) AS (
                SELECT CAST(source.value AS TEXT)
                FROM memory_objects AS mo
                JOIN json_each(json_extract(mo.payload_json, '$.source_message_ids')) AS source
                JOIN json_each(?) AS affected ON CAST(affected.value AS TEXT) = mo.id
                WHERE mo.user_id = ?
                UNION
                SELECT span.message_id
                FROM memory_evidence_spans AS span
                JOIN json_each(?) AS affected ON CAST(affected.value AS TEXT) = span.memory_id
                WHERE span.user_id = ? AND span.message_id IS NOT NULL
                UNION
                SELECT facet.source_message_id
                FROM memory_fact_facets AS facet
                JOIN json_each(?) AS affected ON CAST(affected.value AS TEXT) = facet.memory_id
                WHERE facet.user_id = ?
            )
            SELECT DISTINCT candidate_ids.id
            FROM candidate_ids
            JOIN messages ON messages.id = candidate_ids.id
            JOIN conversations ON conversations.id = messages.conversation_id
            WHERE conversations.user_id = ?
              AND conversations.status = 'active'
            ORDER BY candidate_ids.id ASC
            """,
            (memory_json, user_id, memory_json, user_id, memory_json, user_id, user_id),
        )
        return [
            str(row["id"])
            for row in await cursor.fetchall()
            if str(row["id"]) not in excluded
        ]

    async def _derived_summary_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> list[str]:
        frontier = set(memory_ids)
        summaries: set[str] = set()
        while frontier:
            frontier_json = json_utils.dumps(sorted(frontier), sort_keys=True)
            cursor = await connection.execute(
                """
                SELECT DISTINCT summary.id
                FROM summary_views AS summary
                JOIN json_each(summary.source_object_ids_json) AS source
                JOIN json_each(?) AS frontier
                  ON CAST(frontier.value AS TEXT) = CAST(source.value AS TEXT)
                WHERE summary.user_id = ?
                ORDER BY summary.id ASC
                """,
                (frontier_json, user_id),
            )
            found = {str(row["id"]) for row in await cursor.fetchall()} - summaries
            summaries.update(found)
            frontier = {
                item
                for summary_id in found
                for item in (summary_id, summary_mirror_id(summary_id))
            }
        return sorted(summaries)

    async def _summary_ids_for_rebuild(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        memory_ids: list[str],
    ) -> list[str]:
        """Return transcript chunks plus every summary derived from affected state."""

        cursor = await connection.execute(
            """
            SELECT id
            FROM summary_views
            WHERE user_id = ?
              AND conversation_id = ?
              AND summary_kind = 'conversation_chunk'
            ORDER BY id ASC
            """,
            (user_id, conversation_id),
        )
        conversation_chunk_ids = [str(row["id"]) for row in await cursor.fetchall()]
        descendants = await self._derived_summary_ids(
            connection,
            user_id=user_id,
            memory_ids=[
                *memory_ids,
                *conversation_chunk_ids,
                *[summary_mirror_id(item) for item in conversation_chunk_ids],
            ],
        )
        return self._stable_ids([*conversation_chunk_ids, *descendants])

    async def _communication_profile_rebuild_inventory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        rewritten_conversation_id: str,
        invalidated_message_ids: list[str],
        affected_memory_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        invalidated = set(invalidated_message_ids)
        affected_memories = set(affected_memory_ids)
        cursor = await connection.execute(
            """
            SELECT id, source_refs_json
            FROM user_communication_profiles
            WHERE user_id = ? AND status = 'active' AND stale = 0
            ORDER BY id ASC
            """,
            (user_id,),
        )
        affected_profile_ids: list[str] = []
        supporting_message_ids: list[str] = []
        supporting_memory_ids: list[str] = []
        supporting_windows: list[tuple[str, int, int]] = []
        for row in await cursor.fetchall():
            try:
                decoded = json_utils.loads(str(row["source_refs_json"]))
            except (TypeError, ValueError):
                decoded = []
            refs = decoded if isinstance(decoded, list) else []
            affected = False
            for raw_ref in refs:
                if not isinstance(raw_ref, dict):
                    continue
                source_message_id = raw_ref.get("source_message_id")
                memory_id = raw_ref.get("memory_id")
                source_kind = str(raw_ref.get("source_kind") or "")
                conversation_id = raw_ref.get("conversation_id")
                if (
                    source_message_id is not None
                    and str(source_message_id) in invalidated
                ):
                    affected = True
                if memory_id is not None and str(memory_id) in affected_memories:
                    affected = True
                if (
                    source_kind == "message_window"
                    and str(conversation_id or "") == rewritten_conversation_id
                ):
                    affected = True
            if not affected:
                continue
            affected_profile_ids.append(str(row["id"]))
            for raw_ref in refs:
                if not isinstance(raw_ref, dict):
                    continue
                source_message_id = raw_ref.get("source_message_id")
                if source_message_id is not None:
                    supporting_message_ids.append(str(source_message_id))
                memory_id = raw_ref.get("memory_id")
                if memory_id is not None:
                    supporting_memory_ids.append(str(memory_id))
                if str(raw_ref.get("source_kind") or "") == "message_window":
                    conversation_id = raw_ref.get("conversation_id")
                    from_seq = raw_ref.get("from_seq")
                    to_seq = raw_ref.get("to_seq")
                    if (
                        conversation_id is not None
                        and isinstance(from_seq, int)
                        and isinstance(to_seq, int)
                    ):
                        supporting_windows.append(
                            (str(conversation_id), from_seq, to_seq)
                        )

        for conversation_id, from_seq, to_seq in supporting_windows:
            cursor = await connection.execute(
                """
                SELECT message.id
                FROM messages AS message
                JOIN conversations AS conversation
                  ON conversation.id = message.conversation_id
                WHERE conversation.user_id = ?
                  AND conversation.status = 'active'
                  AND message.conversation_id = ?
                  AND message.role = 'user'
                  AND message.seq BETWEEN ? AND ?
                ORDER BY message.seq ASC, message.id ASC
                """,
                (user_id, conversation_id, from_seq, to_seq),
            )
            supporting_message_ids.extend(
                str(item["id"]) for item in await cursor.fetchall()
            )
        if supporting_memory_ids:
            supporting_message_ids.extend(
                await self._supporting_message_ids(
                    connection,
                    user_id=user_id,
                    memory_ids=self._stable_ids(supporting_memory_ids),
                    excluded_message_ids=invalidated_message_ids,
                )
            )
        candidates = self._stable_ids(supporting_message_ids)
        if not candidates:
            return affected_profile_ids, []
        placeholders = ", ".join("?" for _ in candidates)
        cursor = await connection.execute(
            f"""
            SELECT message.id
            FROM messages AS message
            JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            WHERE conversation.user_id = ?
              AND conversation.status = 'active'
              AND message.role = 'user'
              AND message.id IN ({placeholders})
            ORDER BY message.occurred_at ASC, message.seq ASC, message.id ASC
            """,
            (user_id, *candidates),
        )
        return affected_profile_ids, [
            str(item["id"])
            for item in await cursor.fetchall()
            if str(item["id"]) not in invalidated
        ]

    async def _artifacts_owned_only_by_messages(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        message_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        if not message_ids:
            return [], []
        encoded = json_utils.dumps(message_ids, sort_keys=True)
        cursor = await connection.execute(
            """
            SELECT DISTINCT artifact.id, artifact.payload_blob_id
            FROM artifacts AS artifact
            WHERE artifact.user_id = ?
              AND (
                  EXISTS (
                      SELECT 1
                      FROM json_each(?) AS abandoned
                      WHERE CAST(abandoned.value AS TEXT) = artifact.message_id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM artifact_links AS link
                      JOIN json_each(?) AS abandoned
                        ON CAST(abandoned.value AS TEXT) = link.message_id
                      WHERE link.user_id = artifact.user_id
                        AND link.artifact_id = artifact.id
                  )
              )
              AND (
                  artifact.message_id IS NULL
                  OR EXISTS (
                      SELECT 1
                      FROM json_each(?) AS abandoned
                      WHERE CAST(abandoned.value AS TEXT) = artifact.message_id
                  )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM artifact_links AS retained_link
                  WHERE retained_link.user_id = artifact.user_id
                    AND retained_link.artifact_id = artifact.id
                    AND NOT EXISTS (
                        SELECT 1
                        FROM json_each(?) AS abandoned
                        WHERE CAST(abandoned.value AS TEXT) = retained_link.message_id
                    )
              )
            ORDER BY artifact.id ASC
            """,
            (user_id, encoded, encoded, encoded, encoded),
        )
        rows = await cursor.fetchall()
        return (
            [str(row["id"]) for row in rows],
            sorted(
                {
                    str(row["payload_blob_id"])
                    for row in rows
                    if row["payload_blob_id"] is not None
                }
            ),
        )

    async def _delete_affected_state(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        inventory: _ReplacementInventory,
    ) -> None:
        memory_ids = [
            *inventory.affected_memory_ids,
            *[summary_mirror_id(item) for item in inventory.affected_summary_ids],
        ]
        if inventory.affected_summary_ids:
            placeholders = ", ".join("?" for _ in inventory.affected_summary_ids)
            await connection.execute(
                f"DELETE FROM summary_views WHERE user_id = ? AND id IN ({placeholders})",
                (user_id, *inventory.affected_summary_ids),
            )
        if memory_ids:
            placeholders = ", ".join("?" for _ in memory_ids)
            for statement in (
                f"DELETE FROM graph_relationship_sources WHERE user_id = ? AND memory_id IN ({placeholders})",
                f"DELETE FROM graph_entity_mentions WHERE user_id = ? AND memory_id IN ({placeholders})",
                f"DELETE FROM contract_dimensions_current WHERE user_id = ? AND source_memory_id IN ({placeholders})",
                f"DELETE FROM retrieval_events WHERE user_id = ? AND EXISTS (SELECT 1 FROM json_each(retrieval_events.selected_memory_ids_json) AS selected WHERE CAST(selected.value AS TEXT) IN ({placeholders}))",
                f"DELETE FROM memory_objects WHERE user_id = ? AND id IN ({placeholders})",
            ):
                await connection.execute(statement, (user_id, *memory_ids))
            memory_json = json_utils.dumps(memory_ids, sort_keys=True)
            await connection.execute(
                """
                DELETE FROM graph_projection_runs
                WHERE user_id = ?
                  AND EXISTS (
                      SELECT 1
                      FROM json_each(graph_projection_runs.source_memory_ids_json) AS source
                      JOIN json_each(?) AS affected
                        ON CAST(affected.value AS TEXT) = CAST(source.value AS TEXT)
                  )
                """,
                (user_id, memory_json),
            )
        if inventory.abandoned_message_ids:
            placeholders = ", ".join("?" for _ in inventory.abandoned_message_ids)
            await connection.execute(
                f"""
                DELETE FROM retrieval_events
                WHERE user_id = ?
                  AND (
                      request_message_id IN ({placeholders})
                      OR response_message_id IN ({placeholders})
                  )
                """,
                (
                    user_id,
                    *inventory.abandoned_message_ids,
                    *inventory.abandoned_message_ids,
                ),
            )
            await connection.execute(
                f"""
                DELETE FROM graph_relationship_sources
                WHERE user_id = ?
                  AND (
                      message_id IN ({placeholders})
                      OR (
                          source_kind = 'message'
                          AND source_id IN ({placeholders})
                      )
                  )
                """,
                (
                    user_id,
                    *inventory.abandoned_message_ids,
                    *inventory.abandoned_message_ids,
                ),
            )
            await connection.execute(
                f"""
                DELETE FROM graph_projection_runs
                WHERE user_id = ?
                  AND source_message_id IN ({placeholders})
                """,
                (user_id, *inventory.abandoned_message_ids),
            )
            await connection.execute(
                f"DELETE FROM graph_entity_mentions WHERE user_id = ? AND message_id IN ({placeholders})",
                (user_id, *inventory.abandoned_message_ids),
            )
            await connection.execute(
                """
                DELETE FROM graph_relationships
                WHERE user_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM graph_relationship_sources AS source
                      WHERE source.user_id = graph_relationships.user_id
                        AND source.relationship_id = graph_relationships.id
                  )
                """,
                (user_id,),
            )
            await connection.execute(
                """
                DELETE FROM graph_entity_aliases
                WHERE user_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM graph_entities AS entity
                      WHERE entity.user_id = graph_entity_aliases.user_id
                        AND entity.id = graph_entity_aliases.entity_id
                  )
                """,
                (user_id,),
            )
            await connection.execute(
                """
                DELETE FROM graph_entities
                WHERE user_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM graph_entity_mentions AS mention
                      WHERE mention.user_id = graph_entities.user_id
                        AND mention.entity_id = graph_entities.id
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM graph_relationships AS relationship
                      WHERE relationship.user_id = graph_entities.user_id
                        AND (
                            relationship.source_entity_id = graph_entities.id
                            OR relationship.target_entity_id = graph_entities.id
                        )
                  )
                """,
                (user_id,),
            )
            await connection.execute(
                f"""
                DELETE FROM conversation_topic_sources
                WHERE user_id = ?
                  AND source_kind = 'message'
                  AND source_id IN ({placeholders})
                """,
                (user_id, *inventory.abandoned_message_ids),
            )
        if inventory.communication_profile_ids:
            placeholders = ", ".join("?" for _ in inventory.communication_profile_ids)
            await connection.execute(
                f"""
                UPDATE user_communication_profiles
                SET stale = 1,
                    stale_reason = 'selected_transcript_rebuild',
                    updated_at = ?
                WHERE user_id = ?
                  AND id IN ({placeholders})
                  AND status = 'active'
                """,
                (
                    self._runtime.clock.now().isoformat(),
                    user_id,
                    *inventory.communication_profile_ids,
                ),
            )
        await connection.execute(
            "DELETE FROM conversation_topics WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM conversation_topic_events WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM initial_context_packages WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            """
            UPDATE initial_context_package_build_attempts
            SET status = 'source_changed',
                finished_at = ?,
                diagnostics_json = ?
            WHERE user_id = ? AND status = 'building'
            """,
            (
                self._runtime.clock.now().isoformat(),
                json_utils.dumps(
                    {"reason": "selected_transcript_rebuild"},
                    sort_keys=True,
                ),
                user_id,
            ),
        )
        if inventory.artifact_ids:
            placeholders = ", ".join("?" for _ in inventory.artifact_ids)
            await connection.execute(
                f"DELETE FROM artifacts WHERE user_id = ? AND id IN ({placeholders})",
                (user_id, *inventory.artifact_ids),
            )
        if inventory.artifact_payload_blob_ids:
            placeholders = ", ".join("?" for _ in inventory.artifact_payload_blob_ids)
            await connection.execute(
                f"""
                DELETE FROM artifact_payload_blobs
                WHERE user_id = ?
                  AND id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM artifacts AS retained_artifact
                      WHERE retained_artifact.user_id = artifact_payload_blobs.user_id
                        AND retained_artifact.payload_blob_id = artifact_payload_blobs.id
                        AND retained_artifact.status NOT IN ('deleted', 'purged')
                  )
                """,
                (user_id, *inventory.artifact_payload_blob_ids),
            )
        if inventory.abandoned_message_ids:
            placeholders = ", ".join("?" for _ in inventory.abandoned_message_ids)
            abandoned_json = json_utils.dumps(
                inventory.abandoned_message_ids,
                sort_keys=True,
            )
            await connection.execute(
                f"""
                DELETE FROM verbatim_pins
                WHERE user_id = ?
                  AND conversation_id = ?
                  AND (
                      (
                          target_kind IN ('message', 'text_span')
                          AND target_id IN ({placeholders})
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM json_each(
                              json_extract(
                                  verbatim_pins.payload_json,
                                  '$.source_message_ids'
                              )
                          ) AS source
                          JOIN json_each(?) AS abandoned
                            ON CAST(abandoned.value AS TEXT) =
                               CAST(source.value AS TEXT)
                      )
                  )
                """,
                (
                    user_id,
                    conversation_id,
                    *inventory.abandoned_message_ids,
                    abandoned_json,
                ),
            )
        if memory_ids:
            placeholders = ", ".join("?" for _ in memory_ids)
            await connection.execute(
                f"""
                DELETE FROM verbatim_pins
                WHERE user_id = ?
                  AND target_kind = 'memory_object'
                  AND target_id IN ({placeholders})
                """,
                (user_id, *memory_ids),
            )

    async def _replace_message_suffix(
        self,
        connection: aiosqlite.Connection,
        *,
        conversation: dict[str, Any],
        selected_messages: list[SelectedTranscriptMessage],
        abandoned_message_ids: list[str],
    ) -> None:
        if abandoned_message_ids:
            placeholders = ", ".join("?" for _ in abandoned_message_ids)
            await connection.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders}) AND conversation_id = ?",
                (*abandoned_message_ids, conversation["id"]),
            )
        repository = MessageRepository(connection, self._runtime.clock)
        existing = {
            str(row["id"]): row
            for row in await self._conversation_messages(
                connection,
                user_id=str(conversation["user_id"]),
                conversation_id=str(conversation["id"]),
            )
        }
        user_id = str(conversation["user_id"])
        presences = PresenceRepository(connection, self._runtime.clock)
        active_presence_row = await presences.resolve_active_presence(
            owner_user_id=user_id,
            active_presence_id=conversation.get("active_presence_id"),
            character_id=(
                conversation.get("character_id") or conversation.get("workspace_id")
            ),
            commit=False,
        )
        human_presence_row = await presences.resolve_human_owner_presence(
            owner_user_id=user_id,
            commit=False,
        )
        active_presence = presence_snapshot(active_presence_row)
        human_presence = presence_snapshot(human_presence_row)
        if conversation.get("active_presence_id") != active_presence.presence_id:
            await connection.execute(
                """
                UPDATE conversations
                SET active_presence_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'active'
                """,
                (
                    active_presence.presence_id,
                    self._runtime.clock.now().isoformat(),
                    str(conversation["id"]),
                    user_id,
                ),
            )
            conversation["active_presence_id"] = active_presence.presence_id
        preferences = await UserRepository(
            connection,
            self._runtime.clock,
        ).get_memory_preferences(user_id)
        memory_privacy_mode = resolve_memory_privacy_mode(
            preferences.get("memory_privacy_mode")
        )
        for selected in selected_messages:
            retained = existing.get(selected.message_id)
            if retained is not None:
                retained_source_presence_id = (
                    human_presence.presence_id
                    if selected.role == "user"
                    else (
                        retained.get("source_presence_id")
                        or retained.get("active_presence_id")
                    )
                )
                await connection.execute(
                    """
                    UPDATE messages
                    SET source_presence_id = ?
                    WHERE id = ? AND conversation_id = ?
                    """,
                    (
                        retained_source_presence_id,
                        selected.message_id,
                        str(conversation["id"]),
                    ),
                )
                continue
            # Authoritative branch reconciliation is asynchronous host history,
            # not the user's current synchronous turn.  A newly discovered
            # message must therefore never inherit permission to interrupt the
            # user with a live confirmation prompt.
            ingest_origin = IngestOrigin.BACKFILL
            confirmation = resolve_confirmation_strategy(
                ingest_origin=ingest_origin,
                confirmation_strategy=None,
            )
            await repository.create_message(
                message_id=selected.message_id,
                conversation_id=str(conversation["id"]),
                role=selected.role,
                seq=selected.source_seq,
                text=selected.text,
                metadata={
                    "ingest_origin": ingest_origin.value,
                    "confirmation_strategy": confirmation.value,
                    "memory_privacy_mode": memory_privacy_mode.value,
                    "selected_transcript_identity": (
                        self._selected_transcript_identity(selected)
                    ),
                },
                occurred_at=selected.occurred_at,
                active_presence_id=active_presence.presence_id,
                source_presence_id=(
                    human_presence.presence_id
                    if selected.role == "user"
                    else active_presence.presence_id
                ),
                space_id=conversation.get("active_space_id"),
                active_mind_id=conversation.get("active_mind_id"),
                source_mind_id=conversation.get("active_mind_id"),
                active_embodiment_id=conversation.get("active_embodiment_id"),
                active_realm_id=conversation.get("active_realm_id"),
                commit=False,
            )

    async def _recompute_conversation_activity(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> None:
        timestamp = self._runtime.clock.now().isoformat()
        cursor = await connection.execute(
            """
            UPDATE conversations
            SET last_activity_at = COALESCE(
                    (
                        SELECT COALESCE(message.occurred_at, message.created_at)
                        FROM messages AS message
                        WHERE message.conversation_id = conversations.id
                        ORDER BY
                            datetime(COALESCE(message.occurred_at, message.created_at)) DESC,
                            message.seq DESC,
                            message.id DESC
                        LIMIT 1
                    ),
                    conversations.created_at
                ),
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (timestamp, conversation_id, user_id),
        )
        if int(cursor.rowcount or 0) != 1:
            raise RuntimeError("Authoritative transcript activity scope disappeared")
        await ConversationActivityService(
            self._runtime
        ).recompute_after_authoritative_transcript_change(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def _insert_targets(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        user_id: str,
        selected_message_ids: set[str],
        supporting_message_ids: set[str],
        interrupted_message_ids: set[str],
        target_message_ids: list[str],
    ) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        for message_id in target_message_ids:
            cursor = await connection.execute(
                """
                SELECT m.id, m.role, m.conversation_id
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.id = ? AND c.user_id = ? AND c.status = 'active'
                """,
                (message_id, user_id),
            )
            row = await cursor.fetchone()
            if row is None or str(row["role"]) not in {"user", "assistant"}:
                continue
            source_kind = (
                "selected"
                if message_id in selected_message_ids
                else "supporting"
                if message_id in supporting_message_ids
                else "interrupted_job"
            )
            target = {
                "workflow_id": workflow_id,
                "user_id": user_id,
                "conversation_id": str(row["conversation_id"]),
                "message_id": message_id,
                "source_kind": source_kind,
                "role": str(row["role"]),
                "require_contract": str(row["role"]) == "user",
            }
            await connection.execute(
                """
                INSERT OR IGNORE INTO transcript_rebuild_targets(
                    workflow_id, user_id, conversation_id, message_id,
                    source_kind, role, require_contract
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    user_id,
                    target["conversation_id"],
                    message_id,
                    source_kind,
                    target["role"],
                    1 if target["require_contract"] else 0,
                ),
            )
            targets.append(target)
        return targets

    async def _cancel_workflow_attempt_jobs(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        timestamp: str,
    ) -> None:
        statuses = tuple(status.value for status in NONTERMINAL_JOB_STATUSES)
        placeholders = ", ".join("?" for _ in statuses)
        await connection.execute(
            f"""
            UPDATE worker_job_runs
            SET status = 'cancelled',
                finished_at = ?,
                error_class = 'TranscriptRebuildRetry',
                error_message = NULL,
                recovery_envelope_json = NULL,
                envelope_schema_version = NULL,
                dispatch_token = NULL,
                dispatch_visibility_deadline = NULL,
                execution_owner = NULL,
                execution_lease_expires_at = NULL,
                deferred_until = NULL,
                execution_fence = execution_fence + 1
            WHERE transcript_rebuild_id = ?
              AND status IN ({placeholders})
            """,
            (timestamp, workflow_id, *statuses),
        )

    async def _cancel_pre_rebuild_jobs(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        previous_revision: int,
        new_revision: int,
        abandoned_message_ids: list[str],
        timestamp: str,
    ) -> None:
        del timestamp
        await JobRunRepository(
            connection,
            self._runtime.clock,
        ).reconcile_stale_root_jobs_after_revision_bump(
            user_id=user_id,
            previous_revision=previous_revision,
            new_revision=new_revision,
            excluded_source_message_ids=abandoned_message_ids,
            allow_validated_requeue=True,
        )

    async def _enqueue_workflow_jobs(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        orchestrator_job_id: str,
        user_id: str,
        conversation_id: str,
        targets: list[dict[str, Any]],
    ) -> None:
        tracking = JobTrackingService(
            connection,
            self._runtime.clock,
            workers_enabled=self._runtime.settings.workers_enabled,
            settings=self._runtime.settings,
        )
        orchestrator = JobEnvelope(
            job_id=orchestrator_job_id,
            job_type=JobType.REBUILD_SELECTED_TRANSCRIPT,
            user_id=user_id,
            conversation_id=conversation_id,
            transcript_rebuild_id=workflow_id,
            payload={"workflow_id": workflow_id},
            created_at=self._runtime.clock.now(),
        )
        await tracking.enqueue_job(
            self._runtime.storage_backend,
            TRANSCRIPT_REBUILD_STREAM_NAME,
            orchestrator,
            commit=False,
            dispatch=False,
        )
        await tracking.enqueue_job(
            self._runtime.storage_backend,
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            await self._selected_transcript_icp_job(
                connection,
                workflow_id=workflow_id,
                user_id=user_id,
                conversation_id=conversation_id,
            ),
            commit=False,
            dispatch=False,
        )
        for target in targets:
            jobs = await self._source_jobs_for_target(
                connection,
                workflow_id=workflow_id,
                target=target,
            )
            for stream_name, job in jobs:
                await tracking.enqueue_job(
                    self._runtime.storage_backend,
                    stream_name,
                    job,
                    commit=False,
                    dispatch=False,
                )
        await tracking.enqueue_job(
            self._runtime.storage_backend,
            COMPACT_STREAM_NAME,
            await self._forced_conversation_compaction_job(
                connection,
                workflow_id=workflow_id,
                user_id=user_id,
                conversation_id=conversation_id,
            ),
            commit=False,
            dispatch=False,
        )

    async def _forced_conversation_compaction_job(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        user_id: str,
        conversation_id: str,
    ) -> JobEnvelope:
        cursor = await connection.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ? AND user_id = ? AND status = 'active'
            """,
            (conversation_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise TranscriptSelectionConflictError(
                "Selected transcript conversation disappeared before compaction"
            )
        conversation = dict(row)
        preferences = await UserRepository(
            connection,
            self._runtime.clock,
        ).get_memory_preferences(user_id)
        cursor = await connection.execute(
            """
            SELECT id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC, id ASC
            """,
            (conversation_id,),
        )
        message_ids = [str(item["id"]) for item in await cursor.fetchall()]
        return JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.COMPACT_SUMMARIES,
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=message_ids,
            transcript_rebuild_id=workflow_id,
            payload=CompactionJobPayload(
                user_id=user_id,
                workspace_id=conversation.get("workspace_id"),
                conversation_id=conversation_id,
                user_persona_id=conversation.get("user_persona_id"),
                platform_id=str(conversation.get("platform_id") or "default"),
                character_id=(
                    conversation.get("character_id") or conversation.get("workspace_id")
                ),
                mode=conversation.get("mode") or conversation.get("assistant_mode_id"),
                incognito=bool(
                    conversation.get("incognito") or conversation.get("isolated_mode")
                ),
                remember_across_chats=bool(
                    preferences.get("remember_across_chats", True)
                ),
                remember_across_devices=bool(
                    preferences.get("remember_across_devices", True)
                ),
                temporary=bool(conversation.get("temporary")),
                temporary_ttl_seconds=conversation.get("temporary_ttl_seconds"),
                purge_on_close=bool(conversation.get("purge_on_close")),
                privacy_enforcement="enforce",
                job_kind=CompactionJobKind.CONVERSATION_CHUNK,
                force_rebuild=True,
            ).model_dump(mode="json"),
            created_at=self._runtime.clock.now(),
        )

    async def _selected_transcript_icp_job(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        user_id: str,
        conversation_id: str,
    ) -> JobEnvelope:
        cursor = await connection.execute(
            """
            SELECT id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC, id ASC
            """,
            (conversation_id,),
        )
        message_ids = [str(item["id"]) for item in await cursor.fetchall()]
        payload = await prepare_initial_context_package_refresh_payload(
            connection,
            self._runtime.clock,
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind="all",
            retrieval_profile_id=None,
            reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
            source_message_ids=message_ids,
            privacy_enforcement="enforce",
        )
        return JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=message_ids,
            transcript_rebuild_id=workflow_id,
            payload=payload.model_dump(mode="json"),
            created_at=self._runtime.clock.now(),
        )

    async def _source_jobs_for_target(
        self,
        connection: aiosqlite.Connection,
        *,
        workflow_id: str,
        target: dict[str, Any],
    ) -> list[tuple[str, JobEnvelope]]:
        cursor = await connection.execute(
            """
            SELECT m.*, c.*,
                   m.id AS message_id,
                   m.role AS message_role,
                   m.text AS message_text,
                   m.occurred_at AS message_occurred_at,
                   c.id AS conversation_id,
                   c.user_id AS owner_user_id
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.id = ? AND c.user_id = ? AND c.status = 'active'
            """,
            (target["message_id"], target["user_id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise TranscriptSelectionConflictError(
                "A rebuild target disappeared before durable job creation"
            )
        combined = dict(row)
        conversation = dict(combined)
        conversation["id"] = str(combined["conversation_id"])
        conversation["user_id"] = str(combined["owner_user_id"])
        prior_cursor = await connection.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ? AND seq < ?
            ORDER BY seq DESC, id DESC
            LIMIT 6
            """,
            (combined["conversation_id"], combined["seq"]),
        )
        prior_messages = [
            dict(item) for item in reversed(await prior_cursor.fetchall())
        ]
        preferences = await UserRepository(
            connection,
            self._runtime.clock,
        ).get_memory_preferences(str(target["user_id"]))
        metadata = combined.get("metadata_json")
        if isinstance(metadata, str):
            metadata = json_utils.loads(metadata)
        metadata = metadata if isinstance(metadata, dict) else {}
        jobs = build_message_jobs(
            clock=self._runtime.clock,
            conversation=conversation,
            message_id=str(combined["message_id"]),
            prior_messages=prior_messages,
            message_text=str(combined["message_text"]),
            occurred_at=combined.get("message_occurred_at"),
            role=str(combined["message_role"]),
            include_contract_projection=bool(target["require_contract"]),
            memory_preferences=preferences,
            # Legacy rows without provenance are being processed here only
            # because a host-history reconciliation selected them.  Fail closed
            # to backfill semantics; rows created on a real live path retain
            # their explicit live_turn metadata above.
            ingest_origin=metadata.get("ingest_origin") or IngestOrigin.BACKFILL,
            confirmation_strategy=metadata.get("confirmation_strategy"),
            memory_privacy_mode=metadata.get("memory_privacy_mode"),
            active_presence_id=combined.get("active_presence_id"),
            source_presence_id=combined.get("source_presence_id"),
            active_space_id=combined.get("space_id"),
            active_mind_id=combined.get("active_mind_id"),
            source_mind_id=combined.get("source_mind_id"),
            active_embodiment_id=combined.get("active_embodiment_id"),
            active_realm_id=combined.get("active_realm_id"),
        )
        return [
            (
                stream_name,
                job.model_copy(update={"transcript_rebuild_id": workflow_id}),
            )
            for stream_name, job in jobs
        ]

    async def _dispatch_pending(self, connection: aiosqlite.Connection) -> None:
        try:
            await JobTrackingService(
                connection,
                self._runtime.clock,
                workers_enabled=self._runtime.settings.workers_enabled,
                settings=self._runtime.settings,
            ).dispatch_pending_jobs(self._runtime.storage_backend)
        except Exception:
            # SQLite is authoritative; the continuously running dispatcher will
            # resume publication without reopening the destructive transaction.
            pass

    @staticmethod
    def _stable_ids(values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values if str(value)))

    @staticmethod
    def _transcript_hash(request: ReplaceSelectedTranscriptRequest) -> str:
        return canonical_json_hash(
            {
                "contract_version": request.contract_version,
                "selection_epoch": request.selection_epoch,
                "mutation_kind": request.mutation_kind,
                "retained_cutoff_message_id": request.retained_cutoff_message_id,
                "messages": [
                    message.model_dump(mode="json") for message in request.messages
                ],
            }
        )

    @staticmethod
    def _response(
        workflow: dict[str, Any],
        *,
        idempotent_replay: bool,
    ) -> SelectedTranscriptRebuildResponse:
        stage = str(workflow["stage"])
        status = (
            "complete"
            if stage == "complete"
            else "remediation_required"
            if stage == "remediation_required"
            else "rebuilding"
        )
        selected = workflow.get("selected_message_ids_json") or []
        abandoned = workflow.get("abandoned_message_ids_json") or []
        return SelectedTranscriptRebuildResponse(
            operation_id=str(workflow["operation_id"]),
            workflow_id=str(workflow["id"]),
            user_id=str(workflow["user_id"]),
            conversation_id=str(workflow["conversation_id"]),
            selection_epoch=int(workflow["selection_epoch"]),
            transcript_hash=str(workflow["transcript_hash"]),
            status=status,
            stage=stage,  # type: ignore[arg-type]
            selected_message_count=len(selected),
            abandoned_message_count=len(abandoned),
            poll_path=(
                "/v1/conversations/"
                f"{encode_path_id(str(workflow['conversation_id']))}"
                "/selected-transcript/"
                f"{encode_path_id(str(workflow['operation_id']))}"
            ),
            idempotent_replay=idempotent_replay,
            error_code=(
                str(workflow["error_code"])
                if workflow.get("error_code") is not None
                else None
            ),
        )
