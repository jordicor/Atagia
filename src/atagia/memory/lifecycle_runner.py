"""Lifecycle automation runner with cooldown and mutex coordination."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

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


async def try_run_lifecycle(
    *,
    database_path: str,
    clock: Clock,
    settings: Settings,
    embedding_index: EmbeddingIndex,
    storage_backend: StorageBackend,
) -> bool:
    """Attempt a lifecycle cycle if the cooldown allows and no cycle is running.

    Returns True if a cycle ran, False if throttled or locked.
    Raises on failure (callers decide how to handle).
    """
    prefix = _runtime_prefix(database_path)
    cooldown_key = f"lifecycle:cooldown:{prefix}"
    lock_key = f"lifecycle:lock:{prefix}"

    token = await storage_backend.acquire_lock(lock_key, LOCK_TTL_SECONDS)
    if token is None:
        return False

    connections: list[aiosqlite.Connection] = []
    try:
        should_run = await storage_backend.remember_dedupe(
            cooldown_key, settings.lifecycle_min_interval_seconds,
        )
        if not should_run:
            return False

        connection = await _open_tracked_connection(database_path, connections)
        manager = MemoryLifecycleManager(
            connection, clock,
            settings=settings,
            embedding_index=embedding_index,
        )
        result = await manager.run_cycle(dry_run=False)
        for user_id in manager.affected_user_ids:
            await storage_backend.delete_context_views_for_user(user_id)
            await storage_backend.increment_cache_generation(
                cache_generation_key(database_path, user_id)
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


async def piggyback_lifecycle(runtime: AppRuntime) -> None:
    """Fire-and-forget lifecycle attempt after retrieval."""
    try:
        await try_run_lifecycle(
            database_path=runtime.database_path,
            clock=runtime.clock,
            settings=runtime.settings,
            embedding_index=runtime.embedding_index,
            storage_backend=runtime.storage_backend,
        )
    except Exception:
        logger.exception("Piggyback lifecycle failed")
