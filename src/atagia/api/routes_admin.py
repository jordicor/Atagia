"""Admin routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from atagia.app import AppRuntime
from fastapi import APIRouter, Depends, HTTPException, Query, status

from atagia.api.dependencies import (
    AuthContext,
    get_admin_auth_context,
    get_clock,
    get_connection,
    get_embedding_index,
    get_llm_client,
    get_manifest_loader,
    get_runtime,
    get_settings,
    get_storage_backend,
)
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.ids import new_job_id
from atagia.core import json_utils
from atagia.core.metrics_repository import MetricsRepository
from atagia.core.repositories import UserRepository
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository
from atagia.memory.compactor import Compactor
from atagia.memory.grounding_analyzer import GroundingAnalyzer
from atagia.memory.inspector import MemoryInspector
from atagia.memory.lifecycle import LifecycleCycleResult
from atagia.memory.lifecycle_runner import LifecycleLockError, run_lifecycle_direct
from atagia.memory.metrics_computer import MetricsComputer, normalize_time_bucket
from atagia.memory.policy_manifest import ManifestLoader
from atagia.models.schemas_api import AdminMetricsComputeRequest
from atagia.models.schemas_api import (
    AdminEmbeddingBackfillRequest,
    AdminMemoryCoordinateCorrectionRequest,
    AdminReviewActionResponse,
    AdminReviewMemoryListResponse,
)
from atagia.models.schemas_api import WorkerControlRequest, WorkerControlResponse
from atagia.models.schemas_memory import MemoryCategory, MemoryStatus
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_evaluation import MetricName, RetrievalSummaryStats
from atagia.models.schemas_jobs import EVALUATION_STREAM_NAME, EvaluationJobPayload, JobEnvelope, JobType
from atagia.models.schemas_replay import (
    ConversationExport,
    ConversationExportRequest,
    GroundingReport,
    GroundingRequest,
    ReplayConversationRequest,
    ReplayEventRequest,
    ReplayResult,
)
from atagia.services.llm_client import LLMClient
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.embedding_backfill_service import EmbeddingBackfillResult, EmbeddingBackfillService
from atagia.services.admin_rebuild_service import AdminRebuildService, RebuildResult
from atagia.services.errors import DeletionConfirmationError, MemoryNotFoundError
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.lifecycle_service import (
    HARD_DELETE_MEMORY_CONFIRMATION,
    ConversationLifecycleService,
)
from atagia.services.worker_control_service import WorkerControlService
from atagia.services.dataset_exporter import (
    AnonymizedExportDisabledError,
    ConversationExportNotFoundError,
    DatasetExporter,
    UnsafeConversationExportRequestError,
)
from atagia.services.replay_service import ReplayService
from atagia.services.retrieval_pipeline import RetrievalPipeline
from atagia.core.storage_backend import StorageBackend

router = APIRouter(prefix="/v1/admin", tags=["admin"])


async def _worker_control_response(
    service: WorkerControlService,
    *,
    drain_completed: bool | None = None,
) -> WorkerControlResponse:
    state = await service.get_state()
    return WorkerControlResponse(
        mode=state.mode,
        reason=state.reason,
        updated_at=state.updated_at,
        updated_by=state.updated_by,
        new_source_jobs_allowed=await service.allows_new_source_jobs(),
        worker_claims_allowed=await service.allows_worker_claims(),
        periodic_work_allowed=await service.allows_periodic_work(),
        drain_completed=drain_completed,
    )


async def _audit_admin_action(
    connection: aiosqlite.Connection,
    clock: Clock,
    auth_context: AuthContext,
    *,
    action: str,
    target_type: str,
    target_id: str,
    metadata: dict[str, object] | None = None,
) -> None:
    await AdminAuditRepository(connection, clock).create_audit_entry(
        admin_user_id=auth_context.actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        metadata=metadata or {},
    )


def _decode_payload_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json_utils.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _review_memory_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = _decode_payload_json(row.get("payload_json"))
    source_message_ids = payload.get("source_message_ids")
    if not isinstance(source_message_ids, list):
        source_message_ids = []
    return {
        "memory_id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "conversation_id": row.get("conversation_id"),
        "user_persona_id": row.get("user_persona_id"),
        "platform_id": row.get("platform_id"),
        "character_id": row.get("character_id"),
        "mode": row.get("assistant_mode_id"),
        "object_type": str(row["object_type"]),
        "category": str(row["memory_category"]),
        "scope": str(row["scope"]),
        "scope_canonical": row.get("scope_canonical"),
        "sensitivity": str(row.get("sensitivity") or "unknown"),
        "privacy_level": int(row["privacy_level"]),
        "confidence": float(row["confidence"]),
        "canonical_text": str(row["canonical_text"]),
        "index_text": row.get("index_text"),
        "review_reason": payload.get("review_reason"),
        "ingest_origin": payload.get("ingest_origin"),
        "confirmation_strategy": payload.get("confirmation_strategy"),
        "memory_privacy_mode": payload.get("memory_privacy_mode"),
        "source_message_ids": [str(item) for item in source_message_ids],
        "payload": payload,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


async def _list_review_required_memories(
    connection: aiosqlite.Connection,
    *,
    user_id: str | None,
    platform_id: str | None,
    user_persona_id: str | None,
    character_id: str | None,
    category: MemoryCategory | None,
    ingest_origin: str | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    clauses = ["status = ?"]
    parameters: list[Any] = [MemoryStatus.REVIEW_REQUIRED.value]
    if user_id is not None:
        clauses.append("user_id = ?")
        parameters.append(user_id)
    if platform_id is not None:
        clauses.append("platform_id = ?")
        parameters.append(platform_id)
    if user_persona_id is not None:
        clauses.append("user_persona_id IS ?")
        parameters.append(user_persona_id)
    if character_id is not None:
        clauses.append("character_id IS ?")
        parameters.append(character_id)
    if category is not None:
        clauses.append("memory_category = ?")
        parameters.append(category.value)
    if ingest_origin is not None:
        clauses.append("json_extract(payload_json, '$.ingest_origin') = ?")
        parameters.append(ingest_origin)
    cursor = await connection.execute(
        """
        SELECT *
        FROM memory_objects
        WHERE {clauses}
        ORDER BY created_at ASC, _rowid ASC
        LIMIT ?
        OFFSET ?
        """.format(clauses=" AND ".join(clauses)),
        (*parameters, limit, offset),
    )
    rows = [dict(row) for row in await cursor.fetchall()]
    await cursor.close()
    return [_review_memory_record(row) for row in rows]


async def _get_review_required_memory(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    memory_id: str,
) -> dict[str, Any]:
    cursor = await connection.execute(
        """
        SELECT *
        FROM memory_objects
        WHERE id = ?
          AND user_id = ?
          AND status = ?
        """,
        (memory_id, user_id, MemoryStatus.REVIEW_REQUIRED.value),
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review-required memory not found",
        )
    return dict(row)


@router.get("/worker-control", response_model=WorkerControlResponse)
async def get_worker_control(
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> WorkerControlResponse:
    del auth_context
    return await _worker_control_response(WorkerControlService(connection, clock))


@router.post("/worker-control", response_model=WorkerControlResponse)
async def set_worker_control(
    payload: WorkerControlRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    runtime: "AppRuntime" = Depends(get_runtime),
) -> WorkerControlResponse:
    service = WorkerControlService(connection, clock)
    state = await service.set_mode(
        payload.mode,
        reason=payload.reason,
        updated_by=auth_context.actor_id,
    )
    drain_completed: bool | None = None
    if payload.mode is WorkerControlMode.DRAIN_AND_PAUSE:
        if not runtime.settings.workers_enabled:
            drain_completed = False
        else:
            drain_completed = await runtime.storage_backend.drain(payload.timeout_seconds)
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="set_worker_control",
        target_type="worker_control",
        target_id=state.mode.value,
        metadata={
            "mode": state.mode.value,
            "reason": state.reason,
            "drain_completed": drain_completed,
        },
    )
    return await _worker_control_response(service, drain_completed=drain_completed)


@router.get("/memory-review", response_model=AdminReviewMemoryListResponse)
async def list_review_required_memories(
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    user_id: str | None = Query(default=None),
    platform_id: str | None = Query(default=None),
    user_persona_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    category: MemoryCategory | None = Query(default=None),
    ingest_origin: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AdminReviewMemoryListResponse:
    del auth_context
    items = await _list_review_required_memories(
        connection,
        user_id=user_id,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        category=category,
        ingest_origin=ingest_origin,
        limit=limit,
        offset=offset,
    )
    return AdminReviewMemoryListResponse.model_validate({"items": items})


@router.post(
    "/memory-review/{user_id}/{memory_id}/archive",
    response_model=AdminReviewActionResponse,
)
async def archive_review_required_memory(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    runtime: "AppRuntime" = Depends(get_runtime),
) -> AdminReviewActionResponse:
    await _get_review_required_memory(
        connection,
        user_id=user_id,
        memory_id=memory_id,
    )
    await ConversationLifecycleService(runtime).delete_memory(
        connection,
        memory_id=memory_id,
        user_id=user_id,
    )
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="archive_review_required_memory",
        target_type="memory_object",
        target_id=memory_id,
        metadata={"user_id": user_id, "previous_status": "review_required"},
    )
    await connection.commit()
    return AdminReviewActionResponse(
        memory_id=memory_id,
        status=MemoryStatus.ARCHIVED.value,
    )


@router.post(
    "/memory-review/{user_id}/{memory_id}/delete",
    response_model=AdminReviewActionResponse,
)
async def delete_review_required_memory(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    runtime: "AppRuntime" = Depends(get_runtime),
) -> AdminReviewActionResponse:
    await _get_review_required_memory(
        connection,
        user_id=user_id,
        memory_id=memory_id,
    )
    try:
        await ConversationLifecycleService(runtime).delete_memory(
            connection,
            user_id=user_id,
            memory_id=memory_id,
            hard=True,
            confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
        )
    except (DeletionConfirmationError, MemoryNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="delete_review_required_memory",
        target_type="memory_object",
        target_id=memory_id,
        metadata={"user_id": user_id, "previous_status": "review_required"},
    )
    await connection.commit()
    return AdminReviewActionResponse(
        memory_id=memory_id,
        status=MemoryStatus.DELETED.value,
    )


@router.post("/embeddings/backfill")
async def backfill_embeddings(
    payload: AdminEmbeddingBackfillRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
) -> EmbeddingBackfillResult:
    try:
        result = await EmbeddingBackfillService(
            connection=connection,
            embedding_index=embedding_index,
        ).run(
            batch_size=payload.batch_size,
            delay_ms=payload.delay_ms,
            user_id=payload.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="backfill_embeddings",
        target_type="embeddings",
        target_id=payload.user_id or "all_users",
        metadata=result.model_dump(mode="json"),
    )
    return result


@router.post("/rebuild/conversation/{conversation_id}")
async def rebuild_conversation(
    conversation_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
    manifest_loader: ManifestLoader = Depends(get_manifest_loader),
    settings: Settings = Depends(get_settings),
    storage_backend: StorageBackend = Depends(get_storage_backend),
) -> RebuildResult:
    cursor = await connection.execute(
        """
        SELECT id, user_id
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    result = await AdminRebuildService(
        connection=connection,
        llm_client=llm_client,
        embedding_index=embedding_index,
        clock=clock,
        manifest_loader=manifest_loader,
        settings=settings,
        storage_backend=storage_backend,
    ).rebuild_conversation(
        user_id=str(row["user_id"]),
        conversation_id=str(row["id"]),
    )
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="rebuild_conversation",
        target_type="conversation",
        target_id=conversation_id,
        metadata=result.model_dump(mode="json"),
    )
    return result


@router.post("/rebuild/user/{user_id}")
async def rebuild_user(
    user_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
    manifest_loader: ManifestLoader = Depends(get_manifest_loader),
    settings: Settings = Depends(get_settings),
    storage_backend: StorageBackend = Depends(get_storage_backend),
) -> RebuildResult:
    user = await UserRepository(connection, clock).get_user(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    result = await AdminRebuildService(
        connection=connection,
        llm_client=llm_client,
        embedding_index=embedding_index,
        clock=clock,
        manifest_loader=manifest_loader,
        settings=settings,
        storage_backend=storage_backend,
    ).rebuild_user(user_id)
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="rebuild_user",
        target_type="user",
        target_id=user_id,
        metadata=result.model_dump(mode="json"),
    )
    return result


@router.post("/compact/conversation/{conversation_id}")
async def compact_conversation(
    conversation_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, list[str]]:
    # Admin endpoints have cross-user access by design. Auth is via admin API key.
    cursor = await connection.execute(
        """
        SELECT id, user_id
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    summary_ids = await Compactor(
        connection=connection,
        llm_client=llm_client,
        clock=clock,
        settings=settings,
    ).generate_conversation_chunks(
        user_id=str(row["user_id"]),
        conversation_id=str(row["id"]),
    )
    await AdminAuditRepository(connection, clock).create_audit_entry(
        admin_user_id=auth_context.actor_id,
        action="compact_conversation",
        target_type="conversation",
        target_id=conversation_id,
        metadata={"user_id": row["user_id"], "summary_ids": summary_ids},
    )
    return {"summary_ids": summary_ids}


@router.post("/compact/workspace/{workspace_id}")
async def compact_workspace(
    workspace_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, str | None]:
    # Admin endpoints have cross-user access by design. Auth is via admin API key.
    cursor = await connection.execute(
        """
        SELECT id, user_id
        FROM workspaces
        WHERE id = ?
        """,
        (workspace_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    summary_id = await Compactor(
        connection=connection,
        llm_client=llm_client,
        clock=clock,
        settings=settings,
    ).generate_workspace_rollup(
        user_id=str(row["user_id"]),
        workspace_id=str(row["id"]),
    )
    await AdminAuditRepository(connection, clock).create_audit_entry(
        admin_user_id=auth_context.actor_id,
        action="compact_workspace",
        target_type="workspace",
        target_id=workspace_id,
        metadata={"user_id": row["user_id"], "summary_id": summary_id},
    )
    return {"summary_id": summary_id}


@router.post("/reindex")
async def reindex(
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, str]:
    await connection.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
    await connection.execute("INSERT INTO memory_objects_fts(memory_objects_fts) VALUES ('rebuild')")
    await connection.commit()
    await AdminAuditRepository(connection, clock).create_audit_entry(
        admin_user_id=auth_context.actor_id,
        action="reindex_fts",
        target_type="fts",
        target_id="all",
        metadata={},
    )
    return {"status": "ok"}


@router.get("/retrieval-events/{event_id}")
async def inspect_retrieval_event(
    event_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, object]:
    inspector = MemoryInspector(connection, clock)
    event = await inspector.inspect_retrieval_event_by_id(
        event_id,
        admin_user_id=auth_context.actor_id,
    )
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval event not found",
        )
    return event


@router.get("/memory-coordinates/{user_id}/{memory_id}")
async def inspect_memory_coordinates(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, object]:
    inspector = MemoryInspector(connection, clock)
    inspection = await inspector.inspect_memory_coordinates(
        memory_id,
        user_id,
        admin_user_id=auth_context.actor_id,
    )
    if inspection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory object not found for user",
        )
    return inspection


@router.post("/memory-coordinates/{user_id}/{memory_id}/correct")
async def correct_memory_coordinates(
    user_id: str,
    memory_id: str,
    payload: AdminMemoryCoordinateCorrectionRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    runtime: Any = Depends(get_runtime),
) -> dict[str, object]:
    inspector = MemoryInspector(connection, clock)
    try:
        inspection = await inspector.correct_memory_coordinates(
            memory_id,
            user_id,
            admin_user_id=auth_context.actor_id,
            updates=payload.coordinates,
            reason=payload.reason,
            invalidate_user_cache=ContextCacheService(runtime).invalidate_user_cache,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if inspection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory object not found for user",
        )
    return inspection


@router.get("/memory-coordinates/{user_id}/{memory_id}/corrections")
async def inspect_memory_coordinate_corrections(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> list[dict[str, object]]:
    inspector = MemoryInspector(connection, clock)
    return await inspector.inspect_coordinate_correction_history(
        memory_id,
        user_id,
        admin_user_id=auth_context.actor_id,
    )


@router.get("/retrieval-events/{event_id}/memory-decisions/{user_id}/{memory_id}")
async def inspect_retrieval_memory_decision(
    event_id: str,
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, object]:
    inspector = MemoryInspector(connection, clock)
    decision = await inspector.inspect_retrieval_memory_decision(
        event_id,
        memory_id,
        user_id,
        admin_user_id=auth_context.actor_id,
    )
    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval event not found for user",
        )
    return decision


@router.get("/consequence-chains/{user_id}")
async def list_consequence_chains(
    user_id: str,
    workspace_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> list[dict[str, object]]:
    inspector = MemoryInspector(connection, clock)
    return await inspector.list_consequence_chains(
        user_id,
        admin_user_id=auth_context.actor_id,
        workspace_id=workspace_id,
        limit=limit,
    )


@router.post("/lifecycle/run")
async def run_lifecycle(
    dry_run: bool = Query(default=False),
    auth_context: AuthContext = Depends(get_admin_auth_context),
    clock: Clock = Depends(get_clock),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
    settings: Settings = Depends(get_settings),
    storage_backend: StorageBackend = Depends(get_storage_backend),
    runtime: AppRuntime = Depends(get_runtime),
) -> LifecycleCycleResult:
    try:
        result = await run_lifecycle_direct(
            database_path=runtime.database_path,
            clock=clock,
            settings=settings,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            artifact_blob_store=runtime.artifact_blob_store,
            llm_client=runtime.llm_client,
            lifecycle_runtime=runtime,
            dry_run=dry_run,
        )
    except LifecycleLockError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    connection = await runtime.open_connection()
    try:
        await AdminAuditRepository(connection, clock).create_audit_entry(
            admin_user_id=auth_context.actor_id,
            action="run_lifecycle_cycle",
            target_type="lifecycle",
            target_id="all",
            metadata={"dry_run": dry_run, **result.model_dump()},
        )
        await connection.commit()
    finally:
        await connection.close()
    return result


@router.get("/metrics/latest")
async def get_latest_metrics(
    user_id: str | None = Query(default=None),
    assistant_mode_id: str | None = Query(default=None),
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, dict[str, object]]:
    metrics = await MetricsRepository(connection, clock).get_latest_metrics(
        user_id=user_id,
        assistant_mode_id=assistant_mode_id,
    )
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="metrics_latest",
        target_type="metrics",
        target_id="latest",
        metadata={
            "user_id": user_id,
            "assistant_mode_id": assistant_mode_id,
            "metric_names": sorted(metrics.keys()),
        },
    )
    return metrics


@router.get("/metrics/{metric_name}/history")
async def get_metric_history(
    metric_name: MetricName,
    user_id: str | None = Query(default=None),
    assistant_mode_id: str | None = Query(default=None),
    from_date: str | None = Query(default=None),
    to_date: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=365),
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> list[dict[str, object]]:
    normalized_from = None if from_date is None else normalize_time_bucket(from_date)
    normalized_to = None if to_date is None else normalize_time_bucket(to_date)
    if normalized_from is not None and normalized_to is not None and normalized_from > normalized_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from_date must be on or before to_date",
        )
    rows = await MetricsRepository(connection, clock).list_metrics(
        metric_name=metric_name,
        user_id=user_id,
        assistant_mode_id=assistant_mode_id,
        from_bucket=normalized_from,
        to_bucket=normalized_to,
        limit=limit,
    )
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="metrics_history",
        target_type="metrics",
        target_id=metric_name.value,
        metadata={
            "user_id": user_id,
            "assistant_mode_id": assistant_mode_id,
            "from_date": normalized_from,
            "to_date": normalized_to,
            "limit": limit,
            "result_count": len(rows),
        },
    )
    return rows


@router.post("/metrics/compute")
async def compute_metrics(
    payload: AdminMetricsComputeRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    settings: Settings = Depends(get_settings),
    storage_backend: StorageBackend = Depends(get_storage_backend),
) -> dict[str, object]:
    normalized_bucket = normalize_time_bucket(payload.time_bucket)
    metrics_computer = MetricsComputer(connection, clock, settings=settings)
    metrics_repository = MetricsRepository(connection, clock)
    computed: dict[str, dict[str, float | int]] = {}
    queued_metrics: list[str] = []
    skipped_metrics: list[str] = []

    for metric_name in payload.metrics:
        if metric_name is MetricName.CCR:
            evaluation_job = JobEnvelope(
                job_id=new_job_id(),
                job_type=JobType.RUN_EVALUATION,
                user_id=payload.user_id or "admin_system",
                payload=EvaluationJobPayload(
                    time_bucket=normalized_bucket,
                    user_id=payload.user_id,
                    assistant_mode_id=payload.assistant_mode_id,
                    metrics=["ccr"],
                ).model_dump(mode="json"),
                created_at=clock.now(),
            )
            job_tracking = JobTrackingService(
                connection,
                clock,
                workers_enabled=settings.workers_enabled,
            )
            await job_tracking.create_queued_job(EVALUATION_STREAM_NAME, evaluation_job)
            try:
                await storage_backend.stream_add(
                    EVALUATION_STREAM_NAME,
                    evaluation_job.model_dump(mode="json"),
                )
            except Exception as exc:
                await job_tracking.mark_enqueue_failed(evaluation_job, exc)
                raise
            queued_metrics.append(MetricName.CCR.value)
            continue

        is_system_metric = metric_name is MetricName.SYSTEM
        results = await metrics_computer.compute_named_metric(
            metric_name=metric_name,
            user_id=payload.user_id,
            assistant_mode_id=payload.assistant_mode_id,
            time_bucket=normalized_bucket,
        )
        if not results:
            skipped_metrics.append(metric_name)
            continue
        for stored_metric_name, result in results.items():
            await metrics_repository.store_metric(
                metric_name=stored_metric_name,
                value=result.value,
                sample_count=result.sample_count,
                time_bucket=normalized_bucket,
                computed_at=clock.now().isoformat(),
                user_id=None if is_system_metric else payload.user_id,
                assistant_mode_id=None if is_system_metric else payload.assistant_mode_id,
            )
            computed[stored_metric_name] = {
                "value": result.value,
                "sample_count": result.sample_count,
            }

    response = {
        "computed": computed,
        "queued_metrics": queued_metrics,
        "skipped_metrics": skipped_metrics,
    }
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="metrics_compute",
        target_type="metrics",
        target_id=normalized_bucket,
        metadata={
            "user_id": payload.user_id,
            "assistant_mode_id": payload.assistant_mode_id,
            "requested_metrics": [metric.value for metric in payload.metrics],
            **response,
        },
    )
    return response


@router.get("/metrics/retrieval-summary")
async def get_retrieval_summary(
    from_date: str = Query(...),
    to_date: str = Query(...),
    user_id: str | None = Query(default=None),
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    settings: Settings = Depends(get_settings),
) -> RetrievalSummaryStats:
    normalized_from = normalize_time_bucket(from_date)
    normalized_to = normalize_time_bucket(to_date)
    if normalized_from > normalized_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from_date must be on or before to_date",
        )
    summary = await MetricsComputer(connection, clock, settings=settings).summarize_retrieval_events(
        from_date=normalized_from,
        to_date=normalized_to,
        user_id=user_id,
        assistant_mode_id=None,
    )
    await _audit_admin_action(
        connection,
        clock,
        auth_context,
        action="metrics_retrieval_summary",
        target_type="metrics",
        target_id=f"{normalized_from}:{normalized_to}",
        metadata={
            "user_id": user_id,
            **summary.model_dump(mode="json"),
        },
    )
    return summary


@router.post("/replay/event/{retrieval_event_id}")
async def replay_retrieval_event(
    retrieval_event_id: str,
    payload: ReplayEventRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
    settings: Settings = Depends(get_settings),
) -> ReplayResult:
    replay_service = ReplayService(
        connection=connection,
        retrieval_pipeline=RetrievalPipeline(
            connection=connection,
            llm_client=llm_client,
            embedding_index=embedding_index,
            clock=clock,
            settings=settings,
        ),
        clock=clock,
        settings=settings,
    )
    try:
        result = await replay_service.replay_retrieval_event(
            retrieval_event_id,
            payload.user_id,
            ablation=payload.ablation,
        )
        await AdminAuditRepository(connection, clock).create_audit_entry(
            admin_user_id=auth_context.actor_id,
            action="replay_event",
            target_type="retrieval_event",
            target_id=retrieval_event_id,
            metadata={"user_id": payload.user_id},
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/replay/conversation/{conversation_id}")
async def replay_conversation(
    conversation_id: str,
    payload: ReplayConversationRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    embedding_index: EmbeddingIndex = Depends(get_embedding_index),
    settings: Settings = Depends(get_settings),
) -> list[ReplayResult]:
    replay_service = ReplayService(
        connection=connection,
        retrieval_pipeline=RetrievalPipeline(
            connection=connection,
            llm_client=llm_client,
            embedding_index=embedding_index,
            clock=clock,
            settings=settings,
        ),
        clock=clock,
        settings=settings,
    )
    try:
        result = await replay_service.replay_conversation(
            conversation_id,
            payload.user_id,
            ablation=payload.ablation,
            message_limit=payload.message_limit,
        )
        await AdminAuditRepository(connection, clock).create_audit_entry(
            admin_user_id=auth_context.actor_id,
            action="replay_conversation",
            target_type="conversation",
            target_id=conversation_id,
            metadata={"user_id": payload.user_id},
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/grounding/{retrieval_event_id}")
async def analyze_grounding(
    retrieval_event_id: str,
    payload: GroundingRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> GroundingReport:
    event = await RetrievalEventRepository(connection, clock).get_event(retrieval_event_id, payload.user_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Retrieval event not found for user")
    result = await GroundingAnalyzer(connection).analyze(
        dict(event.get("context_view_json") or {}),
        payload.user_id,
    )
    await AdminAuditRepository(connection, clock).create_audit_entry(
        admin_user_id=auth_context.actor_id,
        action="grounding_analysis",
        target_type="retrieval_event",
        target_id=retrieval_event_id,
        metadata={"user_id": payload.user_id},
    )
    return result


@router.post("/export/conversation/{conversation_id}")
async def export_conversation(
    conversation_id: str,
    payload: ConversationExportRequest,
    auth_context: AuthContext = Depends(get_admin_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[object] = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> ConversationExport:
    try:
        result = await DatasetExporter(
            connection,
            clock,
            llm_client=llm_client,
            settings=settings,
        ).export_conversation(
            conversation_id,
            payload.user_id,
            include_retrieval_traces=payload.include_retrieval_traces,
            include_intimacy_context=payload.include_intimacy_context,
            anonymization_mode=payload.anonymization_mode,
        )
        await AdminAuditRepository(connection, clock).create_audit_entry(
            admin_user_id=auth_context.actor_id,
            action="export_conversation",
            target_type="conversation",
            target_id=conversation_id,
            metadata={
                "user_id": payload.user_id,
                "anonymization_mode": payload.anonymization_mode.value,
                "include_intimacy_context": payload.include_intimacy_context,
            },
        )
        return result
    except ConversationExportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AnonymizedExportDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except UnsafeConversationExportRequestError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
