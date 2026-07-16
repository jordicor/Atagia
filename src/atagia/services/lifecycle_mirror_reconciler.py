"""Two-phase reconciliation of transient lifecycle mirrors from SQLite truth."""

from __future__ import annotations

from uuid import uuid4

import aiosqlite

from atagia.core.storage_backend import StorageBackend


async def reconcile_active_lifecycle_mirror(
    connection: aiosqlite.Connection,
    storage_backend: StorageBackend,
    *,
    user_id: str,
    lifecycle_epoch: str,
    lifecycle_cleanup_key: str,
) -> bool:
    """Restore one missing mirror without allowing reset to revive erased work.

    The backend first exposes a non-publishable ``preparing`` value. SQLite is
    then rechecked for the exact active lifecycle before a backend CAS can make
    it active. Erasure may replace preparing with revoked at any point and wins
    that race.
    """

    nonce = uuid4().hex
    try:
        state = await storage_backend.prepare_lifecycle_mirror(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            nonce,
        )
    except Exception:
        return False
    active_state = f"active:{lifecycle_epoch}"
    if state == active_state:
        return True
    if state != f"preparing:{lifecycle_epoch}:{nonce}":
        return False

    cursor = await connection.execute(
        """
        SELECT 1
        FROM user_lifecycles AS lifecycle
        LEFT JOIN users ON users.id = lifecycle.user_id
        WHERE lifecycle.user_id = ?
          AND lifecycle.lifecycle_epoch = ?
          AND lifecycle.lifecycle_cleanup_key = ?
          AND lifecycle.state = 'active'
          AND lifecycle.erasure_cleanup_id IS NULL
          AND (
              lifecycle.user_id = 'atagia_system'
              OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
          )
        LIMIT 1
        """,
        (user_id, lifecycle_epoch, lifecycle_cleanup_key),
    )
    if await cursor.fetchone() is None:
        return False
    try:
        return await storage_backend.activate_lifecycle_mirror(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            nonce,
        )
    except Exception:
        return False
