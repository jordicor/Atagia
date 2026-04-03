"""Shared locking helpers for belief mutations."""

from __future__ import annotations

import asyncio

from atagia.core.storage_backend import StorageBackend


def belief_lock_key(belief_id: str) -> str:
    """Return the transient lock key used for a belief mutation."""
    return f"belief:{belief_id}"


async def acquire_belief_lock(
    storage_backend: StorageBackend,
    belief_id: str,
    *,
    ttl_seconds: int = 30,
    attempts: int = 3,
    base_delay_seconds: float = 0.05,
) -> str | None:
    """Acquire a Redis-style belief lock token with bounded retries."""
    delay = base_delay_seconds
    lock_key = belief_lock_key(belief_id)
    for attempt in range(1, attempts + 1):
        token = await storage_backend.acquire_lock(lock_key, ttl_seconds=ttl_seconds)
        if token is not None:
            return token
        if attempt == attempts:
            return None
        await asyncio.sleep(delay)
        delay *= 2
    return None
