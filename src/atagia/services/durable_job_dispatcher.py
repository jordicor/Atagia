"""Continuous durable-job dispatcher and lifecycle-mirror reconciler."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from uuid import uuid4

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.storage_backend import StorageBackend
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_jobs import DurableJobNotification

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DurableDispatchResult:
    """One bounded dispatcher sweep."""

    claimed: int = 0
    published: int = 0
    delivery_failed: int = 0
    recovered_leases: int = 0
    cancelled_lifecycles: int = 0
    cancelled_derivations: int = 0


class DurableJobDispatcher:
    """Publish content-free wake-ups only after a committed SQLite claim."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        storage_backend: StorageBackend,
        target_backend: str,
        visibility_seconds: float,
        sweep_interval_seconds: float,
        batch_size: int,
    ) -> None:
        if target_backend not in {"inprocess", "redis"}:
            raise ValueError("target_backend must be inprocess or redis")
        if visibility_seconds <= 0:
            raise ValueError("visibility_seconds must be positive")
        if sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._clock = clock
        self._storage_backend = storage_backend
        self._target_backend = target_backend
        self._visibility_seconds = visibility_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._batch_size = batch_size
        self._jobs = JobRunRepository(connection, clock)
        self._lifecycles = UserLifecycleRepository(connection, clock)

    async def run(self) -> None:
        """Dispatch continuously; transient failures never terminalize durable work."""

        while True:
            try:
                await self.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("durable_job_dispatch_sweep_failed")
            await asyncio.sleep(self._sweep_interval_seconds)

    async def dispatch_once(
        self,
        *,
        perform_maintenance: bool = True,
    ) -> DurableDispatchResult:
        cancelled = (
            await self._jobs.cancel_jobs_with_inactive_lifecycle()
            if perform_maintenance
            else 0
        )
        cancelled_derivations = (
            await self._jobs.cancel_jobs_with_stale_derivation()
            if perform_maintenance
            else 0
        )
        recovered = (
            await self._jobs.recover_expired_execution_leases()
            if perform_maintenance
            else 0
        )
        rows = await self._jobs.claim_dispatchable_jobs(
            target_backend=self._target_backend,
            limit=self._batch_size,
            visibility_seconds=self._visibility_seconds,
        )
        published = 0
        failed = 0
        for row in rows:
            try:
                if not await self._ensure_active_mirror(row):
                    failed += 1
                    continue
                notification = DurableJobNotification(
                    job_id=str(row["job_id"]),
                    dispatch_token=str(row["dispatch_token"]),
                    lifecycle_epoch=str(row["lifecycle_epoch"]),
                    lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
                )
                delivery_id = await self._storage_backend.publish_job_notification(
                    str(row["stream_name"]),
                    notification.model_dump(mode="json"),
                    lifecycle_cleanup_key=notification.lifecycle_cleanup_key,
                    lifecycle_epoch=notification.lifecycle_epoch,
                )
                if delivery_id is None:
                    failed += 1
                else:
                    published += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                failed += 1
                logger.warning(
                    "durable_job_notification_publish_failed",
                    extra={"job_id": str(row.get("job_id") or "")},
                    exc_info=True,
                )
        return DurableDispatchResult(
            claimed=len(rows),
            published=published,
            delivery_failed=failed,
            recovered_leases=recovered,
            cancelled_lifecycles=cancelled,
            cancelled_derivations=cancelled_derivations,
        )

    async def _ensure_active_mirror(self, row: dict[str, object]) -> bool:
        user_id = str(row["user_id"])
        lifecycle_epoch = str(row["lifecycle_epoch"])
        lifecycle_cleanup_key = str(row["lifecycle_cleanup_key"])
        active_value = f"active:{lifecycle_epoch}"
        nonce = uuid4().hex
        state = await self._storage_backend.prepare_lifecycle_mirror(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            nonce,
        )
        if state == active_value:
            return True
        expected_preparing = f"preparing:{lifecycle_epoch}:{nonce}"
        if state != expected_preparing:
            return False

        # Recheck SQLite after the backend entered preparing. Erasure can replace
        # the mirror with revoked at any point; the exact-nonce CAS then fails.
        identity = await self._lifecycles.get_active_identity(user_id)
        if (
            identity is None
            or identity.lifecycle_epoch != lifecycle_epoch
            or identity.lifecycle_cleanup_key != lifecycle_cleanup_key
        ):
            return False
        return await self._storage_backend.activate_lifecycle_mirror(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            nonce,
        )
