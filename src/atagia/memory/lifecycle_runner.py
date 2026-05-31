"""Lifecycle automation runner with cooldown and mutex coordination."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, open_connection
from atagia.core.storage_backend import StorageBackend
from atagia.memory.lifecycle import LifecycleCycleResult, MemoryLifecycleManager
from atagia.services.embeddings import EmbeddingIndex

if TYPE_CHECKING:
    from atagia.app import AppRuntime

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS = 300  # 5 min safety net for one cycle


class LifecycleLockError(Exception):
    """Raised when a lifecycle cycle cannot acquire the execution lock."""


@dataclass(slots=True)
class _LifecycleRuntimeView:
    """Small runtime facade for lifecycle services used by detached workers."""

    database_path: str
    clock: Clock
    settings: Settings
    embedding_index: EmbeddingIndex
    storage_backend: StorageBackend
    artifact_blob_store: Any | None = None
    llm_client: Any | None = None


async def _open_tracked_connection(
    database_path: str,
    connections: list[aiosqlite.Connection],
) -> aiosqlite.Connection:
    task = asyncio.create_task(open_connection(database_path))
    try:
        connection = await task
    except asyncio.CancelledError:
        if task.done() and not task.cancelled():
            connection = task.result()
            await close_connection(connection)
        else:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        raise
    connections.append(connection)
    return connection


def _runtime_prefix(database_path: str) -> str:
    """Short hash prefix to namespace lifecycle keys by database identity."""
    return hashlib.sha256(database_path.encode()).hexdigest()[:12]


def cache_generation_key(database_path: str, user_id: str) -> str:
    """Build a namespaced cache generation key for a user."""
    return f"{_runtime_prefix(database_path)}:{user_id}"


def _is_sqlite_busy_error(exc: BaseException) -> bool:
    """Return True for SQLite writer contention that lazy lifecycle can skip."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    error_name = getattr(exc, "sqlite_errorname", None)
    if isinstance(error_name, str) and error_name.startswith("SQLITE_BUSY"):
        return True
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        return (error_code & 0xFF) == sqlite3.SQLITE_BUSY
    return "database is locked" in str(exc).lower()


async def _has_any_dedupe(storage_backend: StorageBackend, keys: tuple[str, ...]) -> bool:
    for key in keys:
        if await storage_backend.has_dedupe(key):
            return True
    return False


async def try_run_lifecycle(
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
    artifact_blob_store: Any | None = None,
    llm_client: Any | None = None,
    lifecycle_runtime: AppRuntime | None = None,
) -> bool:
    """Attempt a lifecycle cycle if the cooldown allows and no cycle is running.

    Returns True if a cycle ran, False if throttled or locked.
    Raises on failure (callers decide how to handle).
    """
    prefix = _runtime_prefix(database_path)
    cooldown_key = f"lifecycle:cooldown:{prefix}"
    busy_backoff_key = f"lifecycle:busy_backoff:{prefix}"
    failure_backoff_key = f"lifecycle:failure_backoff:{prefix}"
    lock_key = f"lifecycle:lock:{prefix}"
    skip_keys = (cooldown_key, busy_backoff_key, failure_backoff_key)

    if await _has_any_dedupe(storage_backend, skip_keys):
        return False

    token = await storage_backend.acquire_lock(lock_key, LOCK_TTL_SECONDS)
    if token is None:
        return False

    connections: list[aiosqlite.Connection] = []
    try:
        if await _has_any_dedupe(storage_backend, skip_keys):
            return False

        try:
            connection = await _open_tracked_connection(database_path, connections)
            await connection.execute(
                f"PRAGMA busy_timeout = {settings.lifecycle_busy_timeout_ms};"
            )
            manager = MemoryLifecycleManager(
                connection, clock,
                settings=settings,
                embedding_index=embedding_index,
            )
            result = await manager.run_cycle(dry_run=False)
            result.expired_temporary_conversations_count = await _expire_idle_temporary_conversations(
                connection,
                database_path=database_path,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                storage_backend=storage_backend,
                artifact_blob_store=artifact_blob_store,
                llm_client=llm_client,
                lifecycle_runtime=lifecycle_runtime,
                dry_run=False,
            )
            result.purged_pending_conversations_count = await _purge_pending_deleted_conversations(
                connection,
                database_path=database_path,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                storage_backend=storage_backend,
                artifact_blob_store=artifact_blob_store,
                llm_client=llm_client,
                lifecycle_runtime=lifecycle_runtime,
                dry_run=False,
            )
            result.processed_pending_file_deletions_count = await _process_pending_file_deletions(
                connection,
                database_path=database_path,
                clock=clock,
                settings=settings,
                embedding_index=embedding_index,
                storage_backend=storage_backend,
                artifact_blob_store=artifact_blob_store,
                llm_client=llm_client,
                lifecycle_runtime=lifecycle_runtime,
                dry_run=False,
            )
        except Exception as exc:
            if _is_sqlite_busy_error(exc):
                await storage_backend.force_dedupe(
                    busy_backoff_key,
                    settings.lifecycle_busy_backoff_seconds,
                )
                logger.debug("Lifecycle skipped because SQLite is busy")
                return False
            await storage_backend.force_dedupe(
                failure_backoff_key,
                settings.lifecycle_failure_backoff_seconds,
            )
            raise

        for user_id in manager.affected_user_ids:
            await storage_backend.delete_context_views_for_user(user_id)
            await storage_backend.increment_cache_generation(
                cache_generation_key(database_path, user_id)
            )
        await storage_backend.force_dedupe(
            cooldown_key,
            settings.lifecycle_min_interval_seconds,
        )
        logger.info(
            "Lifecycle cycle completed: decayed=%d archived=%d deleted=%d",
            result.decayed_count,
            result.archived_count,
            result.deleted_count,
        )
        return True
    finally:
        for connection in connections:
            await close_connection(connection)
        await storage_backend.release_lock(lock_key, token)


async def run_lifecycle_direct(
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
    artifact_blob_store: Any | None = None,
    llm_client: Any | None = None,
    lifecycle_runtime: AppRuntime | None = None,
    dry_run: bool = False,
) -> LifecycleCycleResult:
    """Run lifecycle directly for admin use. Acquires the same mutex lock.

    Bypasses the cooldown gate (admin decides when to run), but still
    acquires the lock to prevent overlap with automated runs.
    On non-dry-run, also sets the cooldown to suppress automated runs
    for the next interval.
    """
    prefix = _runtime_prefix(database_path)
    lock_key = f"lifecycle:lock:{prefix}"
    cooldown_key = f"lifecycle:cooldown:{prefix}"

    token = await storage_backend.acquire_lock(lock_key, LOCK_TTL_SECONDS)
    if token is None:
        raise LifecycleLockError("A lifecycle cycle is already running")

    connections: list[aiosqlite.Connection] = []
    try:
        connection = await _open_tracked_connection(database_path, connections)
        manager = MemoryLifecycleManager(
            connection, clock,
            settings=settings,
            embedding_index=embedding_index,
        )
        result = await manager.run_cycle(dry_run=dry_run)
        result.expired_temporary_conversations_count = await _expire_idle_temporary_conversations(
            connection,
            database_path=database_path,
            clock=clock,
            settings=settings,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            artifact_blob_store=artifact_blob_store,
            llm_client=llm_client,
            lifecycle_runtime=lifecycle_runtime,
            dry_run=dry_run,
        )
        result.purged_pending_conversations_count = await _purge_pending_deleted_conversations(
            connection,
            database_path=database_path,
            clock=clock,
            settings=settings,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            artifact_blob_store=artifact_blob_store,
            llm_client=llm_client,
            lifecycle_runtime=lifecycle_runtime,
            dry_run=dry_run,
        )
        result.processed_pending_file_deletions_count = await _process_pending_file_deletions(
            connection,
            database_path=database_path,
            clock=clock,
            settings=settings,
            embedding_index=embedding_index,
            storage_backend=storage_backend,
            artifact_blob_store=artifact_blob_store,
            llm_client=llm_client,
            lifecycle_runtime=lifecycle_runtime,
            dry_run=dry_run,
        )
        if not dry_run:
            await storage_backend.force_dedupe(
                cooldown_key, settings.lifecycle_min_interval_seconds,
            )
            for user_id in manager.affected_user_ids:
                await storage_backend.delete_context_views_for_user(user_id)
                await storage_backend.increment_cache_generation(
                    cache_generation_key(database_path, user_id)
                )
        return result
    finally:
        for connection in connections:
            await close_connection(connection)
        await storage_backend.release_lock(lock_key, token)


async def _expire_idle_temporary_conversations(
    connection: aiosqlite.Connection,
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
    artifact_blob_store: Any | None,
    llm_client: Any | None,
    lifecycle_runtime: AppRuntime | None,
    dry_run: bool,
) -> int:
    from atagia.services.lifecycle_service import ConversationLifecycleService

    runtime = lifecycle_runtime or _LifecycleRuntimeView(
        database_path=database_path,
        clock=clock,
        settings=settings,
        embedding_index=embedding_index,
        storage_backend=storage_backend,
        artifact_blob_store=artifact_blob_store,
        llm_client=llm_client,
    )
    return await ConversationLifecycleService(runtime).expire_idle_temporary_conversations(
        connection,
        dry_run=dry_run,
    )


async def _process_pending_file_deletions(
    connection: aiosqlite.Connection,
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
    artifact_blob_store: Any | None,
    llm_client: Any | None,
    lifecycle_runtime: AppRuntime | None,
    dry_run: bool,
) -> int:
    from atagia.services.lifecycle_service import ConversationLifecycleService

    runtime = lifecycle_runtime or _LifecycleRuntimeView(
        database_path=database_path,
        clock=clock,
        settings=settings,
        embedding_index=embedding_index,
        storage_backend=storage_backend,
        artifact_blob_store=artifact_blob_store,
        llm_client=llm_client,
    )
    return await ConversationLifecycleService(runtime).process_pending_file_deletions(
        connection,
        dry_run=dry_run,
    )


async def _purge_pending_deleted_conversations(
    connection: aiosqlite.Connection,
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
    artifact_blob_store: Any | None,
    llm_client: Any | None,
    lifecycle_runtime: AppRuntime | None,
    dry_run: bool,
) -> int:
    from atagia.services.lifecycle_service import ConversationLifecycleService

    runtime = lifecycle_runtime or _LifecycleRuntimeView(
        database_path=database_path,
        clock=clock,
        settings=settings,
        embedding_index=embedding_index,
        storage_backend=storage_backend,
        artifact_blob_store=artifact_blob_store,
        llm_client=llm_client,
    )
    return await ConversationLifecycleService(runtime).purge_pending_deleted_conversations(
        connection,
        dry_run=dry_run,
    )


async def piggyback_lifecycle(runtime: AppRuntime, *, reason: str | None = None) -> None:
    """Fire-and-forget lifecycle attempt after retrieval."""
    del reason
    try:
        await try_run_lifecycle(
            database_path=runtime.database_path,
            clock=runtime.clock,
            settings=runtime.settings,
            embedding_index=runtime.embedding_index,
            storage_backend=runtime.storage_backend,
            artifact_blob_store=runtime.artifact_blob_store,
            llm_client=runtime.llm_client,
            lifecycle_runtime=runtime,
        )
    except Exception:
        logger.exception("Piggyback lifecycle failed")


def request_lifecycle_piggyback(runtime: AppRuntime, *, reason: str) -> bool:
    """Schedule a lazy lifecycle attempt when the runtime can accept background work."""
    if getattr(runtime, "closed", False):
        return False
    runtime.spawn_background_task(
        piggyback_lifecycle(runtime, reason=reason),
        name="atagia-lifecycle-piggyback",
    )
    return True
