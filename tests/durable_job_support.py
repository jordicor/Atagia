"""Test-only helpers for publishing jobs through the durable dispatcher."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import aiosqlite
from pydantic import ValidationError

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.storage_backend import InProcessBackend, StorageBackend
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_jobs import (
    ClaimedJob,
    DurableJobNotification,
    JobEnvelope,
    StreamMessage,
)
from atagia.services.durable_job_dispatcher import DurableJobDispatcher
from atagia.services.job_execution_context import bind_job_claim, reset_job_claim
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.lifecycle_mirror_reconciler import (
    reconcile_active_lifecycle_mirror,
)


@asynccontextmanager
async def bound_test_job_claim(
    connection: aiosqlite.Connection,
    backend: StorageBackend,
    clock: Clock,
    payload: JobEnvelope | dict[str, Any],
) -> AsyncIterator[ClaimedJob]:
    """Bind an exact active lifecycle for a direct worker unit invocation."""

    envelope = (
        payload
        if isinstance(payload, JobEnvelope)
        else JobEnvelope.model_validate(payload)
    )
    cursor = await connection.execute(
        """
        SELECT
            lifecycle_epoch,
            lifecycle_cleanup_key,
            derivation_revision,
            execution_fence,
            attempt_count
        FROM worker_job_runs
        WHERE job_id = ?
        """,
        (envelope.job_id,),
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        identity = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(envelope.user_id)
        if identity is None:
            raise AssertionError("Direct worker test requires an active user lifecycle")
        lifecycle_epoch = identity.lifecycle_epoch
        lifecycle_cleanup_key = identity.lifecycle_cleanup_key
        derivation_revision = identity.derivation_revision
        execution_fence = 1
        attempt_count = 1
    else:
        lifecycle_epoch = str(row["lifecycle_epoch"])
        lifecycle_cleanup_key = str(row["lifecycle_cleanup_key"])
        derivation_revision = int(row["derivation_revision"])
        # Direct process_job units do not run the durable claim transition.
        # An awaiting-claim row therefore still has fence zero; bind the first
        # valid synthetic attempt rather than exposing an impossible lock scope.
        execution_fence = max(1, int(row["execution_fence"]))
        attempt_count = max(1, int(row["attempt_count"]))
    if not await reconcile_active_lifecycle_mirror(
        connection,
        backend,
        user_id=envelope.user_id,
        lifecycle_epoch=lifecycle_epoch,
        lifecycle_cleanup_key=lifecycle_cleanup_key,
    ):
        raise AssertionError("Direct worker test lifecycle mirror is not active")
    claim = ClaimedJob(
        notification_message_id="direct-test",
        envelope=envelope,
        owner_id="direct-test",
        attempt_count=attempt_count,
        execution_fence=execution_fence,
        lifecycle_epoch=lifecycle_epoch,
        lifecycle_cleanup_key=lifecycle_cleanup_key,
        derivation_revision=derivation_revision,
    )
    token = bind_job_claim(claim)
    try:
        yield claim
    finally:
        reset_job_claim(token)


class DurableJobTestBackend(InProcessBackend):
    """Route legacy test setup calls through the production durable write path.

    Worker tests historically inserted complete envelopes directly into a stream.
    Keeping that setup syntax compact is useful, but the adapter deliberately lives
    in tests and publishes only through ``JobTrackingService``. Non-envelope payloads
    such as bounded dead-letter diagnostics retain the ordinary backend behavior.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self._connection = connection
        self._clock = clock
        self.job_tracking_service = JobTrackingService(
            connection,
            clock,
            workers_enabled=True,
            settings=settings,
        )
        self._dispatcher = DurableJobDispatcher(
            connection,
            clock,
            storage_backend=self,
            target_backend="inprocess",
            visibility_seconds=(
                settings.worker_dispatch_visibility_seconds if settings else 30.0
            ),
            sweep_interval_seconds=0.01,
            batch_size=(settings.worker_dispatch_batch_size if settings else 100),
        )

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        try:
            envelope = JobEnvelope.model_validate(payload)
        except ValidationError:
            return await super().stream_add(stream_name, payload)
        await self.job_tracking_service.enqueue_job(self, stream_name, envelope)
        return f"durable:{envelope.job_id}"

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        # Deterministically model the runtime's continuous dispatcher before a
        # manually driven worker iteration.
        await self.dispatch_durable_jobs()
        return await super().stream_read(
            stream_name,
            group_name,
            consumer_name,
            count=count,
            block_ms=block_ms,
        )

    async def dispatch_durable_jobs(self) -> None:
        await self._dispatcher.dispatch_once()

    async def advance_to_next_retry(self) -> None:
        """Advance a frozen test clock to the next durable retry deadline."""

        cursor = await self._connection.execute(
            """
            SELECT MIN(deferred_until) AS deferred_until
            FROM worker_job_runs
            WHERE status IN ('retrying', 'deferred')
              AND deferred_until IS NOT NULL
            """
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row["deferred_until"] is None:
            raise AssertionError("No durable retry deadline is pending")
        advance = getattr(self._clock, "advance", None)
        if advance is None:
            raise AssertionError("advance_to_next_retry requires a FrozenClock")
        deadline = datetime.fromisoformat(str(row["deferred_until"]))
        delay = max(0.0, (deadline - self._clock.now()).total_seconds())
        advance(seconds=delay)

    async def dequeue_durable_envelope(
        self,
        stream_name: str,
    ) -> dict[str, Any] | None:
        """Pop one opaque notification and resolve its canonical SQLite envelope."""

        await self.dispatch_durable_jobs()
        delivery = await super().dequeue_job(
            f"stream:{stream_name}",
            timeout_seconds=0,
        )
        if delivery is None:
            return None
        notification = DurableJobNotification.model_validate(delivery["payload"])
        row = await self.job_tracking_service.get_job_run(notification.job_id)
        if row is None or row.get("recovery_envelope_json") is None:
            return None
        return dict(row["recovery_envelope_json"])
