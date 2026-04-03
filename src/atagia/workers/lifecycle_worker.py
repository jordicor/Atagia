"""Periodic lifecycle worker -- opt-in alternative to lazy piggyback."""

from __future__ import annotations

import asyncio
import logging

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.storage_backend import StorageBackend
from atagia.memory.lifecycle_runner import try_run_lifecycle
from atagia.services.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)


class LifecycleWorker:
    """Runs lifecycle cycles on a fixed interval.

    Unlike stream-based workers, this uses a simple sleep loop.
    Shares cooldown and lock with lazy piggyback to prevent duplicate runs.
    The worker_interval_seconds controls how often the worker wakes up to
    check; the actual lifecycle cooldown is lifecycle_min_interval_seconds.
    """

    def __init__(
        self,
        *,
        database_path: str,
        clock: Clock,
        settings: Settings,
        embedding_index: EmbeddingIndex,
        storage_backend: StorageBackend,
    ) -> None:
        self._database_path = database_path
        self._clock = clock
        self._settings = settings
        self._embedding_index = embedding_index
        self._storage_backend = storage_backend

    async def run(self) -> None:
        """Loop forever, running lifecycle at the configured interval."""
        interval = self._settings.lifecycle_worker_interval_seconds
        logger.info("Lifecycle worker started (poll_interval=%ds)", interval)
        while True:
            try:
                await try_run_lifecycle(
                    database_path=self._database_path,
                    clock=self._clock,
                    settings=self._settings,
                    embedding_index=self._embedding_index,
                    storage_backend=self._storage_backend,
                )
            except asyncio.CancelledError:
                logger.info("Lifecycle worker cancelled")
                raise
            except Exception:
                logger.exception("Lifecycle worker iteration failed")
            await asyncio.sleep(interval)
