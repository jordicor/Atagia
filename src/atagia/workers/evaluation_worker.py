"""Evaluation worker for aggregated metric computation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.metrics_repository import MetricsRepository
from atagia.core.storage_backend import StorageBackend
from atagia.memory.metrics_computer import MetricsComputer
from atagia.models.schemas_jobs import (
    EVALUATION_STREAM_NAME,
    EvaluationJobPayload,
    JobEnvelope,
    JobType,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.services.llm_client import LLMClient

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3


class EvaluationWorker:
    """Consumes metric computation jobs from the configured stream backend."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._llm_client = llm_client
        self._metrics_computer = MetricsComputer(connection, clock, settings=settings)
        self._metrics_repository = MetricsRepository(connection, clock)
        self._clock = clock

    async def run(self, consumer_name: str = "evaluate-1") -> None:
        await self._storage_backend.stream_ensure_group(EVALUATION_STREAM_NAME, WORKER_GROUP_NAME)
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in evaluation worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "evaluate-1",
        block_ms: int | None = 0,
    ) -> WorkerIterationResult:
        messages = await self._next_messages(consumer_name=consumer_name, block_ms=block_ms)
        if not messages:
            return WorkerIterationResult()

        acked = 0
        failed = 0
        dead_lettered = 0
        for message in messages:
            try:
                await self.process_job(message.payload)
                await self._storage_backend.stream_ack(
                    EVALUATION_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                logger.exception("Failed to process evaluation job %s", message.message_id)
                if await self._dead_letter_if_exhausted(message, exc):
                    dead_lettered += 1
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            dead_lettered=dead_lettered,
        )

    async def process_job(self, payload: dict[str, object]) -> dict[str, Any]:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.RUN_EVALUATION:
            raise ValueError(f"Unsupported evaluation job type: {envelope.job_type}")
        job_payload = EvaluationJobPayload.model_validate(envelope.payload)

        computed: dict[str, dict[str, float | int]] = {}
        skipped_metrics: list[str] = []
        for metric_name in job_payload.metrics:
            is_system_metric = metric_name == "system"
            results = await self._metrics_computer.compute_named_metric(
                metric_name=metric_name,
                user_id=job_payload.user_id,
                assistant_mode_id=job_payload.assistant_mode_id,
                time_bucket=job_payload.time_bucket,
                llm_client=self._llm_client,
            )
            if not results:
                skipped_metrics.append(metric_name)
                continue
            for stored_metric_name, result in results.items():
                await self._metrics_repository.store_metric(
                    metric_name=stored_metric_name,
                    value=result.value,
                    sample_count=result.sample_count,
                    time_bucket=job_payload.time_bucket,
                    computed_at=self._clock.now().isoformat(),
                    user_id=None if is_system_metric else job_payload.user_id,
                    assistant_mode_id=None if is_system_metric else job_payload.assistant_mode_id,
                )
                computed[stored_metric_name] = {
                    "value": result.value,
                    "sample_count": result.sample_count,
                }
        return {"computed": computed, "skipped_metrics": skipped_metrics}

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            EVALUATION_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            EVALUATION_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            count=1,
            block_ms=block_ms,
        )

    async def _dead_letter_if_exhausted(
        self,
        message: StreamMessage,
        exc: Exception,
    ) -> bool:
        if message.delivery_count < MAX_STREAM_DELIVERIES:
            return False
        await self._storage_backend.enqueue_job(
            f"dead_letter:{EVALUATION_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
            },
        )
        await self._storage_backend.stream_ack(
            EVALUATION_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        return True
