"""Runtime safety helpers for interactive SQLite flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atagia.core.db_sqlite import is_in_memory_database

if TYPE_CHECKING:
    from atagia.app import AppRuntime


INTERACTIVE_SQLITE_DRAIN_TIMEOUT_SECONDS = 30.0


async def wait_for_in_memory_worker_quiescence(
    runtime: AppRuntime,
    *,
    timeout_seconds: float = INTERACTIVE_SQLITE_DRAIN_TIMEOUT_SECONDS,
) -> None:
    """Drain background jobs before interactive writes on shared in-memory SQLite."""
    if not runtime.settings.workers_enabled:
        return
    if not is_in_memory_database(runtime.database_path):
        return
    drained = await runtime.storage_backend.drain(timeout_seconds)
    if not drained:
        raise RuntimeError(
            "Background workers did not become idle before the interactive SQLite operation"
        )
