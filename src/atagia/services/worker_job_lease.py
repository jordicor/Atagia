"""Renewable execution-lease guard shared by every durable worker type."""

from __future__ import annotations

import asyncio
from contextvars import Token
import logging
from typing import Any

from atagia.core.storage_backend import StorageBackend
from atagia.models.schemas_jobs import (
    ClaimedJob,
    DurableJobNotification,
    StreamMessage,
)
from atagia.services.job_execution_context import (
    StaleParentJobFenceError,
    bind_job_claim,
    reset_job_claim,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.worker_effect_fence import (
    StaleJobEffectFenceError,
    WorkerEffectFence,
)

logger = logging.getLogger(__name__)


class JobLeaseLostError(RuntimeError):
    """The worker no longer owns the durable execution fence."""


class WorkerJobLease:
    """Heartbeat and finalize one claimed job under its immutable fence."""

    def __init__(
        self,
        tracking: JobTrackingService,
        claim: ClaimedJob,
        *,
        effect_fence: WorkerEffectFence,
    ) -> None:
        self._tracking = tracking
        self.claim = claim
        self._effect_fence = effect_fence
        self._effect_context = None
        self._claim_context_token: Token[ClaimedJob | None] | None = None
        self._lost = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "WorkerJobLease":
        self._effect_context = self._effect_fence.activate(self.claim)
        await self._effect_context.__aenter__()
        self._claim_context_token = bind_job_claim(self.claim)
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"atagia-job-heartbeat-{self.claim.envelope.job_id}",
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        token = self._claim_context_token
        self._claim_context_token = None
        if token is not None:
            reset_job_claim(token)
        context = self._effect_context
        self._effect_context = None
        if context is None:
            return
        try:
            await context.__aexit__(exc_type, exc, traceback)
        except (StaleJobEffectFenceError, StaleParentJobFenceError) as fence_error:
            self._lost.set()
            raise JobLeaseLostError(str(fence_error)) from fence_error

    async def require_current(self) -> None:
        if self._lost.is_set() or not await self._tracking.claim_is_current(self.claim):
            self._lost.set()
            raise JobLeaseLostError(
                "Execution or derivation fence lost for durable job "
                f"{self.claim.envelope.job_id}"
            )

    async def succeed(self, *, metadata: dict[str, Any] | None = None) -> None:
        await self.require_current()
        if not await self._tracking.finish_claim_succeeded(
            self.claim, metadata=metadata
        ):
            self._lost.set()
            raise JobLeaseLostError(
                f"Terminal success fence lost for durable job {self.claim.envelope.job_id}"
            )

    async def skip(self, *, reason: str | None = None) -> None:
        await self.require_current()
        if not await self._tracking.finish_claim_skipped(self.claim, reason=reason):
            self._lost.set()
            raise JobLeaseLostError(
                f"Terminal skip fence lost for durable job {self.claim.envelope.job_id}"
            )

    async def retry(self, exc: Exception) -> None:
        if not await self._tracking.release_claim_for_retry(self.claim, exc):
            self._lost.set()
            raise JobLeaseLostError(
                f"Retry fence lost for durable job {self.claim.envelope.job_id}"
            )

    async def defer(self, exc: Exception, *, deferred_until) -> None:
        if not await self._tracking.release_claim_for_retry(
            self.claim,
            exc,
            deferred_until=deferred_until,
        ):
            self._lost.set()
            raise JobLeaseLostError(
                f"Defer fence lost for durable job {self.claim.envelope.job_id}"
            )

    async def fail(self, exc: Exception, *, dead_lettered: bool = False) -> None:
        if not await self._tracking.finish_claim_failed(
            self.claim,
            exc,
            dead_lettered=dead_lettered,
        ):
            self._lost.set()
            raise JobLeaseLostError(
                f"Failure fence lost for durable job {self.claim.envelope.job_id}"
            )

    async def dead_letter(
        self,
        storage_backend: StorageBackend,
        *,
        stream_name: str,
        group_name: str,
        message: StreamMessage,
        exc: Exception,
    ) -> bool:
        """Lifecycle-fence one diagnostic, terminalize its job, and ack its wake-up.

        The backend publish and lifecycle revoke/purge operations share one atomic
        mirror gate. If the mirror is unavailable while the SQLite claim remains
        current, the wake-up stays pending so recovery can retry without silently
        losing the diagnostic.
        """

        notification = DurableJobNotification.model_validate(message.payload)
        if (
            notification.job_id != self.claim.envelope.job_id
            or notification.lifecycle_epoch != self.claim.lifecycle_epoch
        ):
            self._lost.set()
            await storage_backend.stream_ack(
                stream_name,
                group_name,
                message.message_id,
            )
            return False

        async def publish_diagnostic() -> str | None:
            return await storage_backend.publish_lifecycle_diagnostic(
                f"dead_letter:{stream_name}",
                {
                    "job_id": self.claim.envelope.job_id,
                    "user_id": self.claim.envelope.user_id,
                    "conversation_id": self.claim.envelope.conversation_id,
                    "lifecycle_epoch": self.claim.lifecycle_epoch,
                    "derivation_revision": self.claim.derivation_revision,
                    "message_id": message.message_id,
                    "attempt_count": self.claim.attempt_count,
                    "error_class": exc.__class__.__name__,
                },
                lifecycle_cleanup_key=notification.lifecycle_cleanup_key,
                lifecycle_epoch=self.claim.lifecycle_epoch,
            )

        finalized = await self._tracking.finish_claim_dead_lettered_with_diagnostic(
            self.claim,
            exc,
            publish_diagnostic=publish_diagnostic,
        )
        if finalized is False:
            self._lost.set()
            await storage_backend.stream_ack(
                stream_name,
                group_name,
                message.message_id,
            )
            return False
        if finalized is None:
            logger.warning(
                "dead_letter_lifecycle_mirror_unavailable",
                extra={"job_id": self.claim.envelope.job_id},
            )
            return False
        await storage_backend.stream_ack(
            stream_name,
            group_name,
            message.message_id,
        )
        return True

    async def _heartbeat_loop(self) -> None:
        interval = self._tracking.heartbeat_interval_seconds
        try:
            while True:
                await asyncio.sleep(interval)
                if not await self._tracking.heartbeat_claim(self.claim):
                    self._lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._lost.set()
