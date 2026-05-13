"""SQLite repository for OjoCentauri overseer grants."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import OverseerGrantKind, OverseerGrantTargetKind


def _decode_json_columns(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in tuple(payload.items()):
        if key.endswith("_json") and isinstance(value, str):
            payload[key] = json_utils.loads(value)
    return payload


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


class OverseerGrantRepository:
    """Persistence operations for explicit overseer visibility grants."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def upsert_grant(
        self,
        *,
        owner_user_id: str,
        overseer_mind_id: str,
        target_kind: OverseerGrantTargetKind | str,
        target_id: str,
        grant_kind: OverseerGrantKind | str = OverseerGrantKind.READ,
        visibility: str = "attributed",
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_target_kind = OverseerGrantTargetKind(target_kind)
        resolved_grant_kind = OverseerGrantKind(grant_kind)
        resolved_target_id = _normalize_optional_text(target_id)
        if resolved_target_id is None:
            raise ValueError("target_id is required for overseer grants")
        if visibility not in {"attributed", "applicable"}:
            raise ValueError("visibility must be attributed or applicable")
        if not await self._is_overseer_mind(
            owner_user_id=owner_user_id,
            overseer_mind_id=overseer_mind_id,
        ):
            raise ValueError("overseer_mind_id must reference an overseer Mind")
        if not await self._target_exists(
            owner_user_id=owner_user_id,
            target_kind=resolved_target_kind,
            target_id=resolved_target_id,
        ):
            raise ValueError(
                f"overseer grant target not found: "
                f"{resolved_target_kind.value}:{resolved_target_id}"
            )

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO overseer_grants(
                owner_user_id,
                overseer_mind_id,
                target_kind,
                target_id,
                grant_kind,
                visibility,
                expires_at,
                revoked_at,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            ON CONFLICT(owner_user_id, overseer_mind_id, target_kind, target_id, grant_kind)
            DO UPDATE SET
                visibility = excluded.visibility,
                expires_at = excluded.expires_at,
                revoked_at = NULL,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                owner_user_id,
                overseer_mind_id,
                resolved_target_kind.value,
                resolved_target_id,
                resolved_grant_kind.value,
                visibility,
                expires_at,
                json_utils.dumps(metadata or {}, sort_keys=True),
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self.get_grant(
            owner_user_id=owner_user_id,
            overseer_mind_id=overseer_mind_id,
            target_kind=resolved_target_kind,
            target_id=resolved_target_id,
            grant_kind=resolved_grant_kind,
        )
        if row is None:
            raise RuntimeError("Failed to resolve overseer grant")
        return row

    async def revoke_grant(
        self,
        *,
        owner_user_id: str,
        overseer_mind_id: str,
        target_kind: OverseerGrantTargetKind | str,
        target_id: str,
        grant_kind: OverseerGrantKind | str = OverseerGrantKind.READ,
        commit: bool = True,
    ) -> None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE overseer_grants
            SET revoked_at = ?,
                updated_at = ?
            WHERE owner_user_id = ?
              AND overseer_mind_id = ?
              AND target_kind = ?
              AND target_id = ?
              AND grant_kind = ?
            """,
            (
                timestamp,
                timestamp,
                owner_user_id,
                overseer_mind_id,
                OverseerGrantTargetKind(target_kind).value,
                target_id,
                OverseerGrantKind(grant_kind).value,
            ),
        )
        if commit:
            await self._connection.commit()

    async def get_grant(
        self,
        *,
        owner_user_id: str,
        overseer_mind_id: str,
        target_kind: OverseerGrantTargetKind | str,
        target_id: str,
        grant_kind: OverseerGrantKind | str = OverseerGrantKind.READ,
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM overseer_grants
            WHERE owner_user_id = ?
              AND overseer_mind_id = ?
              AND target_kind = ?
              AND target_id = ?
              AND grant_kind = ?
            """,
            (
                owner_user_id,
                overseer_mind_id,
                OverseerGrantTargetKind(target_kind).value,
                target_id,
                OverseerGrantKind(grant_kind).value,
            ),
        )
        return _decode_json_columns(await cursor.fetchone())

    async def _is_overseer_mind(
        self,
        *,
        owner_user_id: str,
        overseer_mind_id: str,
    ) -> bool:
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM minds
            WHERE owner_user_id = ?
              AND id = ?
              AND kind = 'overseer'
            """,
            (owner_user_id, overseer_mind_id),
        )
        return await cursor.fetchone() is not None

    async def _target_exists(
        self,
        *,
        owner_user_id: str,
        target_kind: OverseerGrantTargetKind,
        target_id: str,
    ) -> bool:
        if target_kind is OverseerGrantTargetKind.MIND:
            query = """
                SELECT 1
                FROM minds
                WHERE owner_user_id = ?
                  AND id = ?
            """
        elif target_kind is OverseerGrantTargetKind.SPACE:
            query = """
                SELECT 1
                FROM spaces
                WHERE owner_user_id = ?
                  AND id = ?
            """
        elif target_kind is OverseerGrantTargetKind.REALM:
            query = """
                SELECT 1
                FROM realms
                WHERE owner_user_id = ?
                  AND id = ?
            """
        else:
            return False
        cursor = await self._connection.execute(
            query,
            (owner_user_id, target_id),
        )
        return await cursor.fetchone() is not None


__all__ = ["OverseerGrantRepository"]
