"""Worker job tracking and memory-processing status helpers."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
import logging
import math
import sqlite3
from statistics import median
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.job_run_repository import JobNamespaceFilter, JobRunRepository
from atagia.memory.context_composer import ContextComposer
from atagia.models.schemas_api import (
    MemoryProcessingEstimate,
    MemoryProcessingStatus,
)
from atagia.models.schemas_jobs import (
    JobEnvelope,
    JobRunStatus,
    JobType,
    StreamMessage,
)
from atagia.services.worker_circuit_breaker_service import WorkerCircuitBreakerService

SMALL_BUCKET_MAX_TOKENS = 1024
MEDIUM_BUCKET_MAX_TOKENS = 4096
LARGE_BUCKET_MAX_TOKENS = 16384
ETA_DURATION_SAMPLE_LIMIT = 500
ETA_MIN_SAMPLES_FOR_MEDIUM_CONFIDENCE = 20
GLOBAL_NORMAL_JOB_LIMIT = 10
GLOBAL_BUSY_JOB_LIMIT = 100
JOB_TRACKING_BUSY_RETRIES = 5
JOB_TRACKING_BUSY_BASE_DELAY_SECONDS = 0.02

_T = TypeVar("_T")
_JOB_TRACKING_LOCKS: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    WeakKeyDictionary()
)
logger = logging.getLogger(__name__)


class JobTrackingService:
    """Record worker job runs and summarize current memory processing status."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        workers_enabled: bool,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._workers_enabled = workers_enabled
        self._repository = JobRunRepository(connection, clock)
        self._circuit_breaker = (
            WorkerCircuitBreakerService(connection, clock, settings)
            if settings is not None
            else None
        )

    async def create_queued_job(
        self,
        stream_name: str,
        envelope: JobEnvelope,
    ) -> None:
        source_token_estimate = self._source_token_estimate(envelope)
        metadata = self._safe_metadata(envelope)
        metadata["workers_enabled_at_enqueue"] = self._workers_enabled
        snapshot = self._policy_snapshot(envelope)
        await self._run_best_effort(
            "create_queued_job",
            lambda: self._repository.create_queued_job(
                job_id=envelope.job_id,
                stream_name=stream_name,
                job_type=envelope.job_type.value,
                user_id=envelope.user_id,
                conversation_id=envelope.conversation_id,
                source_message_ids=[str(item) for item in envelope.message_ids],
                source_token_estimate=source_token_estimate,
                size_bucket=self._size_bucket(source_token_estimate),
                queued_at=(
                    envelope.created_at.isoformat()
                    if envelope.created_at is not None
                    else None
                ),
                metadata=metadata,
                user_persona_id=snapshot.get("user_persona_id"),
                platform_id=snapshot.get("platform_id"),
                character_id=snapshot.get("character_id"),
                incognito_snapshot=bool(snapshot.get("incognito")),
                remember_across_chats_snapshot=bool(snapshot.get("remember_across_chats", True)),
                remember_across_devices_snapshot=bool(snapshot.get("remember_across_devices", True)),
                temporary_snapshot=bool(snapshot.get("temporary")),
                purge_on_close_snapshot=bool(snapshot.get("purge_on_close")),
                policy_snapshot=snapshot,
            )
        )

    async def mark_enqueue_failed(
        self,
        envelope: JobEnvelope,
        exc: Exception,
    ) -> None:
        await self._run_best_effort(
            "mark_enqueue_failed",
            lambda: self._repository.mark_failed(
                envelope.job_id,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
        )
        await self._maybe_auto_pause_after_failure(exc)

    async def mark_running(self, message: StreamMessage) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_running(
                envelope.job_id,
                attempt_count=message.delivery_count,
            )

        await self._run_best_effort(
            "mark_running",
            operation,
        )

    async def mark_succeeded(
        self,
        message: StreamMessage,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_succeeded(envelope.job_id, metadata=metadata)

        await self._run_best_effort(
            "mark_succeeded",
            operation,
        )

    async def mark_retrying(self, message: StreamMessage, exc: Exception) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_retrying(
                envelope.job_id,
                attempt_count=message.delivery_count,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            await self._maybe_auto_pause_after_failure(exc)

        await self._run_best_effort(
            "mark_retrying",
            operation,
        )

    async def mark_deferred(
        self,
        message: StreamMessage,
        exc: Exception,
        *,
        deferred_until: datetime,
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            envelope = JobEnvelope.model_validate(message.payload)
            row = await self._repository.mark_deferred(
                envelope.job_id,
                attempt_count=message.delivery_count,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
                deferred_until=deferred_until.isoformat(),
            )
            await self._maybe_auto_pause_after_failure(exc)
            return row

        return await self._run_repository_operation(operation)

    async def mark_skipped(self, message: StreamMessage, *, reason: str | None = None) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_skipped(envelope.job_id, reason=reason)

        await self._run_best_effort(
            "mark_skipped",
            operation,
        )

    async def mark_failed(self, message: StreamMessage, exc: Exception) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_failed(
                envelope.job_id,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            await self._maybe_auto_pause_after_failure(exc)

        await self._run_best_effort(
            "mark_failed",
            operation,
        )

    async def mark_dead_lettered(self, message: StreamMessage, exc: Exception) -> None:
        async def operation() -> None:
            envelope = JobEnvelope.model_validate(message.payload)
            await self._repository.mark_dead_lettered(
                envelope.job_id,
                attempt_count=message.delivery_count,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            await self._maybe_auto_pause_after_failure(exc)

        await self._run_best_effort(
            "mark_dead_lettered",
            operation,
        )

    async def get_status(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        admin: bool = False,
    ) -> MemoryProcessingStatus:
        namespace_filter = self._namespace_filter(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            admin=admin,
        )
        return await self._run_repository_operation(
            lambda: self._get_status_unlocked(
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_filter=namespace_filter,
                admin=admin,
            )
        )

    async def source_message_job_exists(
        self,
        *,
        user_id: str,
        source_message_id: str,
        job_type: JobType | str,
    ) -> bool:
        """Return whether a nonfailed tracked job already covers a source message."""
        return await self._run_repository_operation(
            lambda: self._repository.source_message_job_exists(
                user_id=user_id,
                source_message_id=source_message_id,
                job_type=job_type,
            )
        )

    async def _get_status_unlocked(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
        admin: bool = False,
    ) -> MemoryProcessingStatus:
        window_start = await self._repository.oldest_nonterminal_queued_at(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        counts = await self._repository.status_counts(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
            window_start=window_start,
        )
        progress = await self._repository.source_message_progress(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
            window_start=window_start,
        )
        nonterminal_jobs = await self._repository.nonterminal_jobs(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        newest_job_queued_at = await self._repository.newest_job_queued_at(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        global_counts = (
            await self._repository.status_counts(nonterminal_only=True)
            if admin
            else counts
        )

        pending_by_type = self._counts_by_type(counts, JobRunStatus.QUEUED)
        running_by_type = self._counts_by_type(counts, JobRunStatus.RUNNING)
        pending_jobs = self._count_status(counts, JobRunStatus.QUEUED)
        running_jobs = self._count_status(counts, JobRunStatus.RUNNING)
        retrying_jobs = self._count_status(counts, JobRunStatus.RETRYING)
        failed_jobs = self._count_status(counts, JobRunStatus.FAILED)
        dead_lettered_jobs = self._count_status(counts, JobRunStatus.DEAD_LETTERED)
        processing = pending_jobs + running_jobs + retrying_jobs > 0
        status = self._status_label(
            processing=processing,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            retrying_jobs=retrying_jobs,
            failed_jobs=failed_jobs,
            dead_lettered_jobs=dead_lettered_jobs,
        )
        oldest_pending_age_seconds = self._oldest_pending_age_seconds(nonterminal_jobs)
        global_pending_jobs = self._count_status(global_counts, JobRunStatus.QUEUED)
        global_running_jobs = self._count_status(global_counts, JobRunStatus.RUNNING)

        return MemoryProcessingStatus(
            workers_enabled=self._workers_enabled,
            processing=processing,
            status=status,
            pending_source_messages=progress["pending_source_messages"],
            processed_source_messages=progress["processed_source_messages"],
            tracked_source_messages=progress["tracked_source_messages"],
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            retrying_jobs=retrying_jobs,
            failed_jobs=failed_jobs,
            dead_lettered_jobs=dead_lettered_jobs,
            pending_jobs_by_type=pending_by_type,
            running_jobs_by_type=running_by_type,
            oldest_pending_age_seconds=oldest_pending_age_seconds,
            newest_job_queued_at=newest_job_queued_at,
            estimate=await self._estimate_remaining(nonterminal_jobs),
            global_queue_state=self._global_queue_state(
                global_pending_jobs + global_running_jobs
            ),
            global_pending_jobs=global_pending_jobs,
            global_running_jobs=global_running_jobs,
        )

    async def _run_repository_operation(
        self,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        for attempt in range(JOB_TRACKING_BUSY_RETRIES + 1):
            try:
                async with _job_tracking_lock():
                    return await operation()
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_busy(exc) or attempt >= JOB_TRACKING_BUSY_RETRIES:
                    raise
                await asyncio.sleep(
                    JOB_TRACKING_BUSY_BASE_DELAY_SECONDS * (attempt + 1)
                )
        raise RuntimeError("unreachable job tracking retry state")

    async def _run_best_effort(
        self,
        action: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> None:
        try:
            await self._run_repository_operation(operation)
        except Exception:
            logger.warning("worker_job_tracking_%s_failed", action, exc_info=True)

    async def _maybe_auto_pause_after_failure(self, exc: Exception) -> None:
        if self._circuit_breaker is None:
            return
        try:
            snapshot = await self._circuit_breaker.evaluate_after_worker_failure(
                error_class=exc.__class__.__name__,
            )
        except Exception:
            logger.warning("worker_circuit_breaker_evaluation_failed", exc_info=True)
            return
        if snapshot is not None:
            logger.info(
                "worker_circuit_breaker_evaluated",
                extra={
                    "failure_count": snapshot.failure_count,
                    "attempted_count": snapshot.attempted_count,
                    "failure_ratio": snapshot.failure_ratio,
                },
            )

    def _status_label(
        self,
        *,
        processing: bool,
        pending_jobs: int,
        running_jobs: int,
        retrying_jobs: int,
        failed_jobs: int,
        dead_lettered_jobs: int,
    ) -> str:
        if processing and not self._workers_enabled:
            return "blocked"
        if retrying_jobs:
            return "retrying"
        if running_jobs:
            return "running"
        if pending_jobs:
            return "queued"
        if failed_jobs or dead_lettered_jobs:
            return "degraded"
        return "idle"

    async def _estimate_remaining(
        self,
        nonterminal_jobs: list[dict[str, Any]],
    ) -> MemoryProcessingEstimate:
        if not nonterminal_jobs:
            return MemoryProcessingEstimate()
        duration_rows = await self._repository.recent_completed_durations(
            limit=ETA_DURATION_SAMPLE_LIMIT,
        )
        grouped = self._duration_groups(duration_rows)
        lower_ms = 0.0
        upper_ms = 0.0
        sample_counts: list[int] = []
        missing_samples = False

        for job in nonterminal_jobs:
            job_type = str(job["job_type"])
            size_bucket = job.get("size_bucket")
            samples = grouped.get((job_type, str(size_bucket or "")))
            if not samples:
                samples = grouped.get((job_type, ""))
            if not samples:
                missing_samples = True
                continue
            sample_counts.append(len(samples))
            p50, p90 = self._p50_p90(samples)
            elapsed_ms = self._running_elapsed_ms(job)
            lower_ms += max(0.0, p50 - elapsed_ms)
            upper_ms += max(0.0, p90 - elapsed_ms)

        if not sample_counts or lower_ms == 0.0 and upper_ms == 0.0:
            return MemoryProcessingEstimate()
        confidence = (
            "low"
            if missing_samples or min(sample_counts) < ETA_MIN_SAMPLES_FOR_MEDIUM_CONFIDENCE
            else "medium"
        )
        midpoint_seconds = ((lower_ms + upper_ms) / 2.0) / 1000.0
        return MemoryProcessingEstimate(
            estimated_remaining_seconds=round(midpoint_seconds, 3),
            estimate_range_seconds=[
                round(lower_ms / 1000.0, 3),
                round(max(lower_ms, upper_ms) / 1000.0, 3),
            ],
            confidence=confidence,
            basis="historical_jobs",
        )

    def _duration_groups(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[str, str], list[float]]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            duration = float(row.get("duration_ms") or 0.0)
            if duration <= 0:
                continue
            job_type = str(row["job_type"])
            size_bucket = str(row.get("size_bucket") or "")
            grouped[(job_type, size_bucket)].append(duration)
            grouped[(job_type, "")].append(duration)
        return dict(grouped)

    @staticmethod
    def _p50_p90(samples: list[float]) -> tuple[float, float]:
        ordered = sorted(samples)
        p50 = median(ordered)
        p90_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1)
        return float(p50), float(ordered[p90_index])

    def _oldest_pending_age_seconds(self, nonterminal_jobs: list[dict[str, Any]]) -> float | None:
        queued_values = [
            self._parse_timestamp(str(job["queued_at"]))
            for job in nonterminal_jobs
            if job.get("queued_at") is not None
        ]
        queued_times = [value for value in queued_values if value is not None]
        if not queued_times:
            return None
        oldest = min(queued_times)
        return max(0.0, (self._clock.now() - oldest).total_seconds())

    def _running_elapsed_ms(self, job: dict[str, Any]) -> float:
        if str(job.get("status")) != JobRunStatus.RUNNING.value:
            return 0.0
        started_at = self._parse_timestamp(str(job.get("started_at") or ""))
        if started_at is None:
            return 0.0
        return max(0.0, (self._clock.now() - started_at).total_seconds() * 1000.0)

    @staticmethod
    def _counts_by_type(
        counts: list[dict[str, Any]],
        status: JobRunStatus,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in counts:
            if str(row["status"]) != status.value:
                continue
            result[str(row["job_type"])] = int(row["count"])
        return result

    @staticmethod
    def _count_status(counts: list[dict[str, Any]], status: JobRunStatus) -> int:
        return sum(
            int(row["count"])
            for row in counts
            if str(row["status"]) == status.value
        )

    @staticmethod
    def _global_queue_state(total_nonterminal_jobs: int) -> str:
        if total_nonterminal_jobs <= 0:
            return "idle"
        if total_nonterminal_jobs < GLOBAL_NORMAL_JOB_LIMIT:
            return "normal"
        if total_nonterminal_jobs < GLOBAL_BUSY_JOB_LIMIT:
            return "busy"
        return "backlogged"

    @staticmethod
    def _source_token_estimate(envelope: JobEnvelope) -> int | None:
        payload = envelope.payload
        message_text = payload.get("message_text") if isinstance(payload, dict) else None
        if isinstance(message_text, str):
            return ContextComposer.estimate_tokens(message_text)
        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if isinstance(chunks, list):
            total = 0
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get("text")
                if isinstance(text, str):
                    total += ContextComposer.estimate_tokens(text)
            return total or None
        return None

    @staticmethod
    def _size_bucket(token_estimate: int | None) -> str | None:
        if token_estimate is None:
            return None
        if token_estimate <= SMALL_BUCKET_MAX_TOKENS:
            return "small"
        if token_estimate <= MEDIUM_BUCKET_MAX_TOKENS:
            return "medium"
        if token_estimate <= LARGE_BUCKET_MAX_TOKENS:
            return "large"
        return "huge"

    @staticmethod
    def _policy_snapshot(envelope: JobEnvelope) -> dict[str, Any]:
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        snapshot: dict[str, Any] = {
            "user_persona_id": payload.get("user_persona_id"),
            "platform_id": str(payload.get("platform_id") or "default"),
            "character_id": payload.get("character_id"),
            "conversation_id": envelope.conversation_id,
            "mode": payload.get("mode") or payload.get("assistant_mode_id"),
            "incognito": bool(payload.get("incognito", False)),
            "remember_across_chats": bool(payload.get("remember_across_chats", True)),
            "remember_across_devices": bool(payload.get("remember_across_devices", True)),
            "memory_privacy_mode": str(
                payload.get("memory_privacy_mode") or "balanced"
            ),
            "temporary": bool(payload.get("temporary", False)),
            "temporary_ttl_seconds": payload.get("temporary_ttl_seconds"),
            "purge_on_close": bool(payload.get("purge_on_close", False)),
            "valid_to": payload.get("valid_to"),
        }
        return snapshot

    @staticmethod
    def _namespace_filter(
        *,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        admin: bool,
    ) -> JobNamespaceFilter | None:
        if admin or conversation_id is not None:
            return None
        if platform_id is None:
            raise ValueError("conversation_id or platform_id is required for non-admin job status")
        if incognito or not remember_across_chats:
            raise ValueError("conversation_id is required for chat-local job status")
        return JobNamespaceFilter(
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )

    @staticmethod
    def _safe_metadata(envelope: JobEnvelope) -> dict[str, Any]:
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        metadata: dict[str, Any] = {
            "message_count": len(envelope.message_ids),
        }
        for key in (
            "job_kind",
            "chunked",
            "chunk_count",
            "fallback_count",
            "package_kind",
            "reason",
            "retrieval_profile_id",
            "privacy_enforcement",
        ):
            value = payload.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            metadata["chunk_count"] = len(chunks)
        if envelope.operational_profile is not None:
            metadata["operational_profile"] = envelope.operational_profile.profile_id
            metadata["operational_profile_snapshot"] = envelope.operational_profile.model_dump(
                mode="json"
            )
        return metadata

    @staticmethod
    def _parse_timestamp(value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


def render_memory_processing_status_block(
    status: MemoryProcessingStatus | None,
) -> str:
    """Render a compact prompt block describing background memory work."""
    if status is None or (not status.processing and status.status == "idle"):
        return ""
    lines = [
        "[Memory Processing Status]",
    ]
    if status.processing:
        lines.append("Some recent source messages may still be processing into durable memory.")
    else:
        lines.append("Recent memory processing encountered worker issues.")
    if status.tracked_source_messages:
        lines.append(
            "Processed source messages in current window: "
            f"{status.processed_source_messages}/{status.tracked_source_messages}"
        )
    if status.pending_source_messages:
        lines.append(f"Pending source messages: {status.pending_source_messages}")
    pending_parts = [
        f"{job_type}={count}"
        for job_type, count in sorted(status.pending_jobs_by_type.items())
        if count > 0
    ]
    running_parts = [
        f"{job_type}={count}"
        for job_type, count in sorted(status.running_jobs_by_type.items())
        if count > 0
    ]
    if pending_parts:
        lines.append(f"Pending work: {', '.join(pending_parts)}")
    if running_parts:
        lines.append(f"Running work: {', '.join(running_parts)}")
    if status.failed_jobs or status.dead_lettered_jobs:
        lines.append(
            "Recent worker issues: "
            f"failed={status.failed_jobs}, dead_lettered={status.dead_lettered_jobs}"
        )
    estimate = status.estimate
    if estimate.confidence != "none" and estimate.estimate_range_seconds is not None:
        lower, upper = estimate.estimate_range_seconds
        lines.append(
            "Rough remaining time: "
            f"about {lower:.0f}-{upper:.0f} seconds, {estimate.confidence} confidence."
        )
    if status.status == "blocked":
        lines.append("Workers are not currently enabled, so queued memory work may not drain yet.")
    lines.append(
        "Use retrieved memories and the recent transcript normally. If the user asks "
        "about details that may come from still-processing messages, say that memory "
        "processing is still running and ask for the specific detail if needed."
    )
    return "\n".join(lines)


def _is_sqlite_busy(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _job_tracking_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _JOB_TRACKING_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _JOB_TRACKING_LOCKS[loop] = lock
    return lock
