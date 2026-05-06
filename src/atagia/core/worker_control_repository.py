"""Durable background-processing control state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.core.repositories import BaseRepository
from atagia.models.schemas_jobs import WorkerControlMode

WORKER_CONTROL_STATE_ID = 1


@dataclass(frozen=True, slots=True)
class WorkerControlState:
    """Single persisted processing-control state."""

    mode: WorkerControlMode
    reason: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


class WorkerControlRepository(BaseRepository):
    """Persistence operations for the singleton worker-control state."""

    async def get_state(self) -> WorkerControlState:
        row = await self._fetch_one(
            """
            SELECT mode, reason, updated_at, updated_by
            FROM worker_control_state
            WHERE id = ?
            """,
            (WORKER_CONTROL_STATE_ID,),
        )
        if row is None:
            return WorkerControlState(mode=WorkerControlMode.ACTIVE)
        return _state_from_row(row)

    async def set_state(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        updated_by: str | None = None,
        commit: bool = True,
    ) -> WorkerControlState:
        resolved_mode = WorkerControlMode(mode)
        timestamp = self._timestamp()
        normalized_reason = _clean_optional(reason)
        normalized_updated_by = _clean_optional(updated_by)
        await self._connection.execute(
            """
            INSERT INTO worker_control_state(id, mode, reason, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                mode = excluded.mode,
                reason = excluded.reason,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (
                WORKER_CONTROL_STATE_ID,
                resolved_mode.value,
                normalized_reason,
                timestamp,
                normalized_updated_by,
            ),
        )
        if commit:
            await self._connection.commit()
        return WorkerControlState(
            mode=resolved_mode,
            reason=normalized_reason,
            updated_at=timestamp,
            updated_by=normalized_updated_by,
        )


def _state_from_row(row: dict[str, Any]) -> WorkerControlState:
    return WorkerControlState(
        mode=WorkerControlMode(str(row["mode"])),
        reason=row.get("reason") if isinstance(row.get("reason"), str) else None,
        updated_at=row.get("updated_at") if isinstance(row.get("updated_at"), str) else None,
        updated_by=row.get("updated_by") if isinstance(row.get("updated_by"), str) else None,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
