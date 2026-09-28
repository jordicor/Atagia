"""FastAPI application factory for Atagia."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
from typing import Any
from uuid import uuid4

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from atagia.api.request_body_limit import (
    RequestBodyLimitExceeded,
    RequestBodyLimitMiddleware,
    request_body_limit_error_response,
)

from atagia.api.routes_admin import (
    audit_memory_review_validation_failure,
    router as admin_router,
)
from atagia.api.routes_activity import router as activity_router
from atagia.api.routes_chat import router as chat_router
from atagia.api.routes_memory import router as memory_router
from atagia.api.routes_openai_proxy import router as openai_proxy_router
from atagia.api.routes_openai_proxy import openai_proxy_validation_error_response
from atagia.api.routes_verbatim_pins import router as verbatim_pins_router
from atagia.core.clock import Clock, SystemClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import (
    SQLITE_BUSY_TIMEOUT_MS,
    close_connection,
    initialize_database,
    open_connection,
    resolve_runtime_database_path,
)
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.initial_context_package_revision_repository import (
    InitialContextPackageRevisionRepository,
)
from atagia.core.redis_client import RedisBackend
from atagia.core.storage_backend import InProcessBackend, StorageBackend
from atagia.memory.operational_profile import OperationalProfileLoader
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    load_and_sync_assistant_modes,
)
from atagia.memory.token_document_frequency import TokenDocumentFrequencyCache
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    REVISE_STREAM_NAME,
    TRANSCRIPT_REBUILD_STREAM_NAME,
    WORKER_GROUP_NAME,
)
from atagia.services.artifact_blob_migration import assert_artifact_blob_runtime_ready
from atagia.services.initial_context_package_sources import (
    assert_initial_context_package_revision_coverage,
)
from atagia.services.embeddings import EmbeddingIndex, create_embedding_index
from atagia.services.durable_job_dispatcher import DurableJobDispatcher
from atagia.services.llm_client import ConfigurationError, LLMClient
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.services.model_resolution import log_resolution
from atagia.services.inference_runtime import (
    PreparedInferenceAccess,
    format_inference_access_diagnostics,
    prepare_inference_access,
)
from atagia.services.inference_routes import InferenceAccessMode
from atagia.services.providers import build_llm_client
from atagia.services.user_erasure_cleanup_service import (
    recover_pending_user_erasures,
)
from atagia.workers.compaction_worker import CompactionWorker
from atagia.workers.contract_worker import ContractWorker
from atagia.workers.evaluation_worker import EvaluationWorker
from atagia.workers.graph_sync_worker import GraphSyncWorker
from atagia.workers.initial_context_package_worker import InitialContextPackageWorker
from atagia.workers.ingest_worker import IngestWorker
from atagia.workers.lifecycle_worker import LifecycleWorker
from atagia.workers.revision_worker import RevisionWorker
from atagia.workers.transcript_rebuild_worker import TranscriptRebuildWorker


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppRuntime:
    """Shared runtime dependencies stored in app.state."""

    settings: Settings
    clock: Clock
    database_path: str
    manifest_loader: ManifestLoader
    manifests: dict[str, Any]
    operational_profile_loader: OperationalProfileLoader
    operational_profiles: dict[str, Any]
    policy_resolver: PolicyResolver
    llm_client: LLMClient[Any]
    inference_access: PreparedInferenceAccess
    embedding_index: EmbeddingIndex
    storage_backend: StorageBackend
    ingest_worker: IngestWorker | None
    contract_worker: ContractWorker | None
    graph_worker: GraphSyncWorker | None
    revision_worker: RevisionWorker | None
    compaction_worker: CompactionWorker | None
    evaluation_worker: EvaluationWorker | None
    initial_context_package_worker: InitialContextPackageWorker | None
    transcript_rebuild_worker: TranscriptRebuildWorker | None
    lifecycle_worker: LifecycleWorker | None
    durable_job_dispatcher: DurableJobDispatcher | None
    worker_tasks: list[asyncio.Task[None]]
    bootstrap_connection: aiosqlite.Connection
    embedding_connection: aiosqlite.Connection | None
    worker_connections: list[aiosqlite.Connection]
    # Corpus statistics outlive the per-request pipeline that reads them, so
    # they are owned here: one cache per database, for the process lifetime.
    token_document_frequency_cache: TokenDocumentFrequencyCache = field(
        default_factory=TokenDocumentFrequencyCache
    )
    _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    closed: bool = False

    async def open_connection(
        self,
        *,
        busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
    ) -> aiosqlite.Connection:
        """Open a short-lived SQLite connection for one unit of work."""
        return await open_connection(
            self.database_path,
            busy_timeout_ms=busy_timeout_ms,
        )

    def spawn_background_task(
        self, coro: Coroutine[Any, Any, None], *, name: str
    ) -> None:
        """Create a tracked background task that is cancelled on shutdown."""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def close(self) -> None:
        """Close worker tasks, transient backends, and SQLite resources."""
        if self.closed:
            return
        self.closed = True
        for task in self.worker_tasks:
            task.cancel()
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        bg_tasks = list(self._background_tasks)
        for task in bg_tasks:
            task.cancel()
        if bg_tasks:
            await asyncio.gather(*bg_tasks, return_exceptions=True)
        await self.llm_client.aclose()
        await self.storage_backend.close()
        for worker_connection in self.worker_connections:
            await close_connection(worker_connection)
        if self.embedding_connection is not None:
            await close_connection(self.embedding_connection)
        await close_connection(self.bootstrap_connection)


def _build_storage_backend(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "redis":
        return RedisBackend(settings.redis_url)
    return InProcessBackend()


def _validate_settings(settings: Settings) -> None:
    if settings.service_mode:
        if settings.service_api_key is None:
            raise ConfigurationError(
                "ATAGIA_SERVICE_API_KEY is required when ATAGIA_SERVICE_MODE=true"
            )
        if settings.admin_api_key is None:
            raise ConfigurationError(
                "ATAGIA_ADMIN_API_KEY is required when ATAGIA_SERVICE_MODE=true"
            )
        if settings.service_api_key == settings.admin_api_key:
            raise ConfigurationError(
                "ATAGIA_ADMIN_API_KEY must differ from ATAGIA_SERVICE_API_KEY"
            )
        return

    if not settings.allow_insecure_http:
        raise ConfigurationError(
            "create_app() requires ATAGIA_SERVICE_MODE=true with API keys or "
            "ATAGIA_ALLOW_INSECURE_HTTP=true"
        )


async def initialize_runtime(
    settings: Settings,
    *,
    additional_inference_completion_models: dict[str, str] | None = None,
) -> AppRuntime:
    """Build the shared runtime used by both FastAPI and library mode."""
    inference_access = await prepare_inference_access(
        settings,
        additional_completion_models=additional_inference_completion_models,
    )
    if (
        inference_access.policy.restricted
        or inference_access.local_catalog is not None
    ):
        logger.info("%s", format_inference_access_diagnostics(inference_access))
    worker_tasks: list[asyncio.Task[None]] = []
    worker_connections: list[aiosqlite.Connection] = []
    embedding_connection: aiosqlite.Connection | None = None
    storage_backend: StorageBackend | None = None
    llm_client: LLMClient[Any] | None = None
    database_path = resolve_runtime_database_path(settings.sqlite_path)
    clock = SystemClock()
    bootstrap_connection = await initialize_database(
        database_path,
        settings.migrations_dir(),
    )
    try:
        await assert_artifact_blob_runtime_ready(
            bootstrap_connection,
            configured_storage_kind=settings.artifact_blob_storage_kind,
        )
        await assert_initial_context_package_revision_coverage(bootstrap_connection)
        await InitialContextPackageRevisionRepository(
            bootstrap_connection,
            clock,
        ).fail_abandoned_build_attempts()
        await JobRunRepository(
            bootstrap_connection,
            clock,
        ).assert_target_backend_compatible(settings.storage_backend)
        manifest_loader = ManifestLoader(settings.manifests_dir())
        manifests = await load_and_sync_assistant_modes(
            bootstrap_connection,
            settings.manifests_dir(),
            clock,
        )
        operational_profile_loader = OperationalProfileLoader(
            settings.operational_profiles_dir()
        )
        operational_profiles = operational_profile_loader.load_all()
        log_resolution(settings)
        if (
            inference_access.policy.mode is InferenceAccessMode.UNRESTRICTED
            and inference_access.local_catalog is None
        ):
            llm_client = build_llm_client(settings)
        else:
            llm_client = build_llm_client(
                settings,
                prepared_inference_access=inference_access,
            )
        if settings.embedding_backend != "none":
            embedding_connection = await open_connection(database_path)
        embedding_index = await create_embedding_index(
            settings,
            embedding_connection or bootstrap_connection,
            llm_client,
            clock,
        )
        storage_backend = _build_storage_backend(settings)
        ingest_worker: IngestWorker | None = None
        contract_worker: ContractWorker | None = None
        graph_worker: GraphSyncWorker | None = None
        revision_worker: RevisionWorker | None = None
        compaction_worker: CompactionWorker | None = None
        evaluation_worker: EvaluationWorker | None = None
        initial_context_package_worker: InitialContextPackageWorker | None = None
        transcript_rebuild_worker: TranscriptRebuildWorker | None = None
        durable_job_dispatcher: DurableJobDispatcher | None = None
        for stream_name in (
            EXTRACT_STREAM_NAME,
            CONTRACT_STREAM_NAME,
            GRAPH_STREAM_NAME,
            REVISE_STREAM_NAME,
            COMPACT_STREAM_NAME,
            EVALUATION_STREAM_NAME,
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            TRANSCRIPT_REBUILD_STREAM_NAME,
        ):
            await storage_backend.stream_ensure_group(stream_name, WORKER_GROUP_NAME)
        await recover_pending_user_erasures(
            bootstrap_connection,
            clock,
            storage_backend,
            storage_backend_name=settings.storage_backend,
        )
        lifecycle_worker: LifecycleWorker | None = None
        if settings.workers_enabled:
            dispatcher_connection = await open_connection(database_path)
            ingest_connection = await open_connection(database_path)
            contract_connection = await open_connection(database_path)
            graph_connection = await open_connection(database_path)
            revision_connection = await open_connection(database_path)
            compaction_connection = await open_connection(database_path)
            evaluation_connection = await open_connection(database_path)
            initial_context_package_connection = await open_connection(database_path)
            transcript_rebuild_connection = await open_connection(database_path)
            ingest_job_connection = await open_connection(database_path)
            contract_job_connection = await open_connection(database_path)
            graph_job_connection = await open_connection(database_path)
            revision_job_connection = await open_connection(database_path)
            compaction_job_connection = await open_connection(database_path)
            evaluation_job_connection = await open_connection(database_path)
            initial_context_package_job_connection = await open_connection(
                database_path
            )
            transcript_rebuild_job_connection = await open_connection(database_path)
            worker_connections.extend(
                [
                    dispatcher_connection,
                    ingest_connection,
                    contract_connection,
                    graph_connection,
                    revision_connection,
                    compaction_connection,
                    evaluation_connection,
                    initial_context_package_connection,
                    transcript_rebuild_connection,
                    ingest_job_connection,
                    contract_job_connection,
                    graph_job_connection,
                    revision_job_connection,
                    compaction_job_connection,
                    evaluation_job_connection,
                    initial_context_package_job_connection,
                    transcript_rebuild_job_connection,
                ]
            )
            durable_job_dispatcher = DurableJobDispatcher(
                dispatcher_connection,
                clock,
                storage_backend=storage_backend,
                target_backend=settings.storage_backend,
                visibility_seconds=settings.worker_dispatch_visibility_seconds,
                sweep_interval_seconds=settings.worker_dispatch_sweep_interval_seconds,
                batch_size=settings.worker_dispatch_batch_size,
            )
            ingest_worker = IngestWorker(
                storage_backend=storage_backend,
                connection=ingest_connection,
                llm_client=llm_client,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
                embedding_index=embedding_index,
                job_connection=ingest_job_connection,
            )
            contract_worker = ContractWorker(
                storage_backend=storage_backend,
                connection=contract_connection,
                llm_client=llm_client,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
                job_connection=contract_job_connection,
            )
            graph_worker = GraphSyncWorker(
                storage_backend=storage_backend,
                connection=graph_connection,
                llm_client=llm_client,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
                job_connection=graph_job_connection,
            )
            revision_worker = RevisionWorker(
                storage_backend=storage_backend,
                connection=revision_connection,
                llm_client=llm_client,
                clock=clock,
                embedding_index=embedding_index,
                settings=settings,
                job_connection=revision_job_connection,
            )
            compaction_worker = CompactionWorker(
                storage_backend=storage_backend,
                connection=compaction_connection,
                llm_client=llm_client,
                clock=clock,
                embedding_index=embedding_index,
                settings=settings,
                job_connection=compaction_job_connection,
            )
            evaluation_worker = EvaluationWorker(
                storage_backend=storage_backend,
                connection=evaluation_connection,
                llm_client=llm_client,
                clock=clock,
                settings=settings,
                job_connection=evaluation_job_connection,
            )
            initial_context_package_worker = InitialContextPackageWorker(
                storage_backend=storage_backend,
                connection=initial_context_package_connection,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
                operational_profile_loader=operational_profile_loader,
                llm_client=llm_client,
                job_connection=initial_context_package_job_connection,
            )
            transcript_rebuild_worker = TranscriptRebuildWorker(
                storage_backend=storage_backend,
                connection=transcript_rebuild_connection,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                job_connection=transcript_rebuild_job_connection,
            )
            worker_tasks = [
                asyncio.create_task(
                    durable_job_dispatcher.run(),
                    name="atagia-durable-job-dispatcher",
                ),
                # Each worker owns its own SQLite connection so their transactions
                # cannot bleed across requests or each other.
                asyncio.create_task(
                    ingest_worker.run(consumer_name=f"ingest-{uuid4().hex}"),
                    name="atagia-ingest-worker",
                ),
                asyncio.create_task(
                    contract_worker.run(consumer_name=f"contract-{uuid4().hex}"),
                    name="atagia-contract-worker",
                ),
                asyncio.create_task(
                    graph_worker.run(consumer_name=f"graph-{uuid4().hex}"),
                    name="atagia-graph-worker",
                ),
                asyncio.create_task(
                    revision_worker.run(consumer_name=f"revise-{uuid4().hex}"),
                    name="atagia-revision-worker",
                ),
                asyncio.create_task(
                    compaction_worker.run(consumer_name=f"compact-{uuid4().hex}"),
                    name="atagia-compaction-worker",
                ),
                asyncio.create_task(
                    evaluation_worker.run(consumer_name=f"evaluate-{uuid4().hex}"),
                    name="atagia-evaluation-worker",
                ),
                asyncio.create_task(
                    initial_context_package_worker.run(
                        consumer_name=f"initial-context-package-{uuid4().hex}"
                    ),
                    name="atagia-initial-context-package-worker",
                ),
                asyncio.create_task(
                    transcript_rebuild_worker.run(
                        consumer_name=f"transcript-rebuild-{uuid4().hex}"
                    ),
                    name="atagia-transcript-rebuild-worker",
                ),
            ]
        if settings.lifecycle_worker_enabled:
            lifecycle_worker = LifecycleWorker(
                database_path=database_path,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                storage_backend=storage_backend,
                llm_client=llm_client,
            )
            worker_tasks.append(
                asyncio.create_task(
                    lifecycle_worker.run(), name="atagia-lifecycle-worker"
                )
            )
        return AppRuntime(
            settings=settings,
            clock=clock,
            database_path=database_path,
            manifest_loader=manifest_loader,
            manifests=manifests,
            operational_profile_loader=operational_profile_loader,
            operational_profiles=operational_profiles,
            policy_resolver=PolicyResolver(),
            llm_client=llm_client,
            inference_access=inference_access,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            ingest_worker=ingest_worker,
            contract_worker=contract_worker,
            graph_worker=graph_worker,
            revision_worker=revision_worker,
            compaction_worker=compaction_worker,
            evaluation_worker=evaluation_worker,
            initial_context_package_worker=initial_context_package_worker,
            transcript_rebuild_worker=transcript_rebuild_worker,
            lifecycle_worker=lifecycle_worker,
            durable_job_dispatcher=durable_job_dispatcher,
            worker_tasks=worker_tasks,
            bootstrap_connection=bootstrap_connection,
            embedding_connection=embedding_connection,
            worker_connections=worker_connections,
        )
    except Exception:
        for task in worker_tasks:
            task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        if storage_backend is not None:
            await storage_backend.close()
        if llm_client is not None:
            await llm_client.aclose()
        for worker_connection in worker_connections:
            await close_connection(worker_connection)
        if embedding_connection is not None:
            await close_connection(embedding_connection)
        await close_connection(bootstrap_connection)
        raise


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application and wire runtime dependencies."""
    resolved_settings = settings or Settings.from_env()
    _validate_settings(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = await initialize_runtime(resolved_settings)
        try:
            yield
        finally:
            await app.state.runtime.close()

    app = FastAPI(
        title="Atagia",
        version="0.1.0",
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        if request.url.path in {"/v1/chat/completions", "/v1/models"}:
            return openai_proxy_validation_error_response(exc)
        if request.url.path == "/v1/admin/memory-review":
            await audit_memory_review_validation_failure(request, exc)
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(RequestBodyLimitExceeded)
    async def body_limit_exception_handler(
        request: Request,
        exc: RequestBodyLimitExceeded,
    ) -> JSONResponse:
        return request_body_limit_error_response(request.url.path, exc.limit)

    @app.exception_handler(TranscriptRebuildInProgressError)
    async def transcript_rebuild_in_progress_handler(
        request: Request,
        exc: TranscriptRebuildInProgressError,
    ) -> JSONResponse:
        detail = str(exc)
        if request.url.path == "/v1/chat/completions":
            return JSONResponse(
                status_code=409,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "message": detail,
                        "type": "conflict_error",
                        "code": "selected_transcript_rebuild_in_progress",
                    }
                },
            )
        return JSONResponse(
            status_code=409,
            headers={"Retry-After": "1"},
            content={
                "detail": detail,
                "code": "selected_transcript_rebuild_in_progress",
            },
        )

    @app.exception_handler(TranscriptRebuildRemediationRequiredError)
    async def transcript_rebuild_remediation_handler(
        request: Request,
        exc: TranscriptRebuildRemediationRequiredError,
    ) -> JSONResponse:
        detail = str(exc)
        if request.url.path == "/v1/chat/completions":
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": detail,
                        "type": "service_unavailable_error",
                        "code": "selected_transcript_remediation_required",
                    }
                },
            )
        return JSONResponse(
            status_code=503,
            content={
                "detail": detail,
                "code": "selected_transcript_remediation_required",
            },
        )

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=resolved_settings.request_max_body_bytes,
    )

    if resolved_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(chat_router)
    app.include_router(openai_proxy_router)
    app.include_router(activity_router)
    app.include_router(memory_router)
    app.include_router(verbatim_pins_router)
    app.include_router(admin_router)
    return app
