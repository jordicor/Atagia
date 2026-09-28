"""Unfenced recent-window probe for storage-layer tests.

Production reads go through ``StorageBackend.get_recent_window_for_cache_identity``,
which returns a window only to a caller holding the exact identity it was
published under. That fence is the point of the API, and it makes the fenced
read useless for asserting what is PHYSICALLY stored: "the entry is gone" would
also be satisfied by an identity that merely stopped matching.

Ownership, cleanup and revocation tests need the stronger statement, so this
probe reads each backend's own storage directly. It is a test-only observation
of internal state and must never be mirrored into ``src/atagia``.
"""

from __future__ import annotations

import copy
from typing import Any

from atagia.core import json_utils
from atagia.core.redis_client import RedisBackend
from atagia.core.storage_backend import InProcessBackend, StorageBackend


async def stored_recent_window(
    backend: StorageBackend,
    key: str,
) -> list[dict[str, Any]] | None:
    """Return the stored recent-window payload with no identity fence."""

    if isinstance(backend, InProcessBackend):
        value = backend._recent_windows.get(key)
        return copy.deepcopy(value) if value is not None else None
    if isinstance(backend, RedisBackend):
        raw = await backend._client.get(f"recent_window:{key}")
        return None if raw is None else json_utils.loads(raw)
    raise TypeError(f"No recent-window probe for backend {type(backend).__name__}")
