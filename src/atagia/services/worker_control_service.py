"""Background-processing stop-switch semantics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.worker_control_repository import (
    WorkerControlRepository,
    WorkerControlState,
)
from atagia.models.schemas_jobs import WorkerControlMode

WORKER_CONTROL_POLL_SECONDS = 1.0


@dataclass(slots=True)
class WorkerControlService:
    """Interpret durable processing-control mode for enqueuers and workers."""

    connection: aiosqlite.Connection
    clock: Clock

    async def get_state(self) -> WorkerControlState:
        return await WorkerControlRepository(self.connection, self.clock).get_state()

    async def set_mode(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        updated_by: str | None = None,
    ) -> WorkerControlState:
        return await WorkerControlRepository(self.connection, self.clock).set_state(
            mode,
            reason=reason,
            updated_by=updated_by,
        )

    async def allows_new_source_jobs(self) -> bool:
        state = await self.get_state()
        return state.mode is WorkerControlMode.ACTIVE

    async def allows_worker_claims(self) -> bool:
        state = await self.get_state()
        return state.mode is not WorkerControlMode.HARD_PAUSE

    async def allows_periodic_work(self) -> bool:
        state = await self.get_state()
        return state.mode is WorkerControlMode.ACTIVE


async def wait_if_worker_claims_paused(
    control_service: WorkerControlService,
    *,
    block_ms: int | None,
) -> bool:
    """Return True after waiting briefly when hard pause blocks stream claims."""
    if await control_service.allows_worker_claims():
        return False
    sleep_seconds = WORKER_CONTROL_POLL_SECONDS
    if block_ms is not None:
        sleep_seconds = min(sleep_seconds, max(0.0, block_ms / 1000))
    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)
    return True
