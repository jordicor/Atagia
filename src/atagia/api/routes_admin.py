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
from atagia.models.schemas_api import AdminEmbeddingBackfillRequest
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            )
            await storage_backend.stream_add(
                EVALUATION_STREAM_NAME,
                evaluation_job.model_dump(mode="json"),
            )
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
            },
        )
        return result
    except ConversationExportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AnonymizedExportDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except UnsafeConversationExportRequestError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
