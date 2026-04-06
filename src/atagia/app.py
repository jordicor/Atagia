"""FastAPI application factory for Atagia."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import aiosqlite
from fastapi import FastAPI

from atagia.api.routes_admin import router as admin_router
from atagia.api.routes_chat import router as chat_router
from atagia.api.routes_memory import router as memory_router
from atagia.core.clock import Clock, SystemClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database, open_connection, resolve_runtime_database_path
from atagia.core.redis_client import RedisBackend
from atagia.core.storage_backend import InProcessBackend, StorageBackend
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, load_and_sync_assistant_modes
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    REVISE_STREAM_NAME,
    WORKER_GROUP_NAME,
)
from atagia.services.embeddings import EmbeddingIndex, create_embedding_index
from atagia.services.llm_client import ConfigurationError, LLMClient
from atagia.services.providers import build_llm_client
from atagia.workers.compaction_worker import CompactionWorker
from atagia.workers.contract_worker import ContractWorker
from atagia.workers.evaluation_worker import EvaluationWorker
from atagia.workers.ingest_worker import IngestWorker
from atagia.workers.lifecycle_worker import LifecycleWorker
from atagia.workers.revision_worker import RevisionWorker


@dataclass(slots=True)
class AppRuntime:
    """Shared runtime dependencies stored in app.state."""

    settings: Settings
    clock: Clock
    database_path: str
    manifest_loader: ManifestLoader
    manifests: dict[str, Any]
    policy_resolver: PolicyResolver
    llm_client: LLMClient[Any]
    embedding_index: EmbeddingIndex
    storage_backend: StorageBackend
    ingest_worker: IngestWorker | None
    contract_worker: ContractWorker | None
    revision_worker: RevisionWorker | None
    compaction_worker: CompactionWorker | None
    evaluation_worker: EvaluationWorker | None
    lifecycle_worker: LifecycleWorker | None
    worker_tasks: list[asyncio.Task[None]]
    bootstrap_connection: aiosqlite.Connection
    embedding_connection: aiosqlite.Connection | None
    worker_connections: list[aiosqlite.Connection]
    _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    closed: bool = False

    async def open_connection(self) -> aiosqlite.Connection:
        """Open a short-lived SQLite connection for one unit of work."""
        return await open_connection(self.database_path)

    def spawn_background_task(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
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
        await self.storage_backend.close()
        for worker_connection in self.worker_connections:
            await worker_connection.close()
        if self.embedding_connection is not None:
            await self.embedding_connection.close()
        await self.bootstrap_connection.close()


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


async def initialize_runtime(settings: Settings) -> AppRuntime:
    """Build the shared runtime used by both FastAPI and library mode."""
    worker_tasks: list[asyncio.Task[None]] = []
    worker_connections: list[aiosqlite.Connection] = []
    embedding_connection: aiosqlite.Connection | None = None
    storage_backend: StorageBackend | None = None
    database_path = resolve_runtime_database_path(settings.sqlite_path)
    clock = SystemClock()
    bootstrap_connection = await initialize_database(
        database_path,
        settings.migrations_dir(),
    )
    try:
        manifest_loader = ManifestLoader(settings.manifests_dir())
        manifests = await load_and_sync_assistant_modes(
            bootstrap_connection,
            settings.manifests_dir(),
            clock,
        )
        llm_client = build_llm_client(settings)
        if settings.embedding_backend != "none":
            embedding_connection = await open_connection(database_path)
        embedding_index = await create_embedding_index(
            settings,
            embedding_connection or bootstrap_connection,
            llm_client,
        )
        storage_backend = _build_storage_backend(settings)
        ingest_worker: IngestWorker | None = None
        contract_worker: ContractWorker | None = None
        revision_worker: RevisionWorker | None = None
        compaction_worker: CompactionWorker | None = None
        evaluation_worker: EvaluationWorker | None = None
        for stream_name in (
            EXTRACT_STREAM_NAME,
            CONTRACT_STREAM_NAME,
            REVISE_STREAM_NAME,
            COMPACT_STREAM_NAME,
            EVALUATION_STREAM_NAME,
        ):
            await storage_backend.stream_ensure_group(stream_name, WORKER_GROUP_NAME)
        lifecycle_worker: LifecycleWorker | None = None
        if settings.workers_enabled:
            ingest_connection = await open_connection(database_path)
            contract_connection = await open_connection(database_path)
            revision_connection = await open_connection(database_path)
            compaction_connection = await open_connection(database_path)
            evaluation_connection = await open_connection(database_path)
            worker_connections.extend(
                [
                    ingest_connection,
                    contract_connection,
                    revision_connection,
                    compaction_connection,
                    evaluation_connection,
                ]
            )
            ingest_worker = IngestWorker(
                storage_backend=storage_backend,
                connection=ingest_connection,
                llm_client=llm_client,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
                embedding_index=embedding_index,
            )
            contract_worker = ContractWorker(
                storage_backend=storage_backend,
                connection=contract_connection,
                llm_client=llm_client,
                clock=clock,
                manifest_loader=manifest_loader,
                settings=settings,
            )
            revision_worker = RevisionWorker(
                storage_backend=storage_backend,
                connection=revision_connection,
                llm_client=llm_client,
                clock=clock,
                settings=settings,
            )
            compaction_worker = CompactionWorker(
                storage_backend=storage_backend,
                connection=compaction_connection,
                llm_client=llm_client,
                clock=clock,
                embedding_index=embedding_index,
                settings=settings,
            )
            evaluation_worker = EvaluationWorker(
                storage_backend=storage_backend,
                connection=evaluation_connection,
                llm_client=llm_client,
                clock=clock,
                settings=settings,
            )
            worker_tasks = [
                # Each worker owns its own SQLite connection so their transactions
                # cannot bleed across requests or each other.
                asyncio.create_task(ingest_worker.run(), name="atagia-ingest-worker"),
                asyncio.create_task(contract_worker.run(), name="atagia-contract-worker"),
                asyncio.create_task(revision_worker.run(), name="atagia-revision-worker"),
                asyncio.create_task(compaction_worker.run(), name="atagia-compaction-worker"),
                asyncio.create_task(evaluation_worker.run(), name="atagia-evaluation-worker"),
            ]
        if settings.lifecycle_worker_enabled:
            lifecycle_worker = LifecycleWorker(
                database_path=database_path,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                storage_backend=storage_backend,
            )
            worker_tasks.append(
                asyncio.create_task(lifecycle_worker.run(), name="atagia-lifecycle-worker")
            )
        return AppRuntime(
            settings=settings,
            clock=clock,
            database_path=database_path,
            manifest_loader=manifest_loader,
            manifests=manifests,
            policy_resolver=PolicyResolver(),
            llm_client=llm_client,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            ingest_worker=ingest_worker,
            contract_worker=contract_worker,
            revision_worker=revision_worker,
            compaction_worker=compaction_worker,
            evaluation_worker=evaluation_worker,
            lifecycle_worker=lifecycle_worker,
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
        for worker_connection in worker_connections:
            await worker_connection.close()
        if embedding_connection is not None:
            await embedding_connection.close()
        await bootstrap_connection.close()
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
    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(admin_router)
    return app
