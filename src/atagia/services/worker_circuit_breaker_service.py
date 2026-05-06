"""Automatic worker pause when background processing is broadly unhealthy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.models.schemas_jobs import JobRunStatus, WorkerControlMode
from atagia.services.worker_control_service import WorkerControlService

_ATTEMPTED_STATUSES = (
    JobRunStatus.SUCCEEDED.value,
    JobRunStatus.SKIPPED.value,
    JobRunStatus.RETRYING.value,
    JobRunStatus.FAILED.value,
    JobRunStatus.DEAD_LETTERED.value,
)
_FAILURE_STATUSES = {
    JobRunStatus.RETRYING.value,
    JobRunStatus.FAILED.value,
    JobRunStatus.DEAD_LETTERED.value,
}
_AUTO_PAUSE_ACTOR = "worker_circuit_breaker"


@dataclass(frozen=True, slots=True)
class WorkerCircuitBreakerSnapshot:
    """Recent worker health counters used to decide auto-pause."""

    attempted_count: int
    failure_count: int
    failure_ratio: float
    failure_classes: dict[str, int]


class WorkerCircuitBreakerService:
    """Trip the global worker stop switch after a sustained failure storm."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        settings: Settings,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._settings = settings
        self._worker_control = WorkerControlService(connection, clock)

    async def evaluate_after_worker_failure(
        self,
        *,
        error_class: str,
    ) -> WorkerCircuitBreakerSnapshot | None:
        """Hard-pause workers when recent failures exceed configured thresholds."""
        if not self._settings.worker_circuit_breaker_enabled:
            return None

        state = await self._worker_control.get_state()
        if state.mode is not WorkerControlMode.ACTIVE:
            return None

        snapshot = await self.recent_snapshot()
        if snapshot.failure_count < self._settings.worker_circuit_breaker_failure_threshold:
            return snapshot
        if snapshot.failure_ratio < self._settings.worker_circuit_breaker_min_failure_ratio:
            return snapshot

        await self._worker_control.set_mode(
            WorkerControlMode.HARD_PAUSE,
            reason=self._pause_reason(snapshot, error_class=error_class),
            updated_by=_AUTO_PAUSE_ACTOR,
        )
        return snapshot

    async def recent_snapshot(self) -> WorkerCircuitBreakerSnapshot:
        """Return recent attempted and failed worker-job counts."""
        since = (
            self._clock.now()
            - timedelta(seconds=self._settings.worker_circuit_breaker_window_seconds)
        ).isoformat()
        placeholders = ", ".join("?" for _ in _ATTEMPTED_STATUSES)
        cursor = await self._connection.execute(
            f"""
            SELECT status, error_class, COUNT(*) AS count
            FROM worker_job_runs
            WHERE status IN ({placeholders})
              AND COALESCE(finished_at, last_heartbeat_at, started_at, queued_at) >= ?
            GROUP BY status, error_class
            """,
            (*_ATTEMPTED_STATUSES, since),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        attempted_count = 0
        failure_count = 0
        failure_classes: dict[str, int] = defaultdict(int)
        for row in rows:
            count = int(row["count"] or 0)
            status = str(row["status"])
            attempted_count += count
            if status in _FAILURE_STATUSES:
                failure_count += count
                error_class = row["error_class"]
                label = str(error_class).strip() if error_class else "UnknownError"
                failure_classes[label] += count

        failure_ratio = failure_count / attempted_count if attempted_count else 0.0
        return WorkerCircuitBreakerSnapshot(
            attempted_count=attempted_count,
            failure_count=failure_count,
            failure_ratio=failure_ratio,
            failure_classes=dict(failure_classes),
        )

    def _pause_reason(
        self,
        snapshot: WorkerCircuitBreakerSnapshot,
        *,
        error_class: str,
    ) -> str:
        top_classes = sorted(
            snapshot.failure_classes.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        class_summary = ", ".join(
            f"{label}={count}" for label, count in top_classes
        )
        if not class_summary:
            class_summary = error_class or "UnknownError"
        ratio_percent = snapshot.failure_ratio * 100.0
        return (
            "Auto hard pause: "
            f"{snapshot.failure_count}/{snapshot.attempted_count} recent worker attempts "
            f"failed in {self._settings.worker_circuit_breaker_window_seconds}s "
            f"({ratio_percent:.0f}%). "
            f"Top errors: {class_summary}. "
            "Check provider/network/configuration, then resume manually."
        )
