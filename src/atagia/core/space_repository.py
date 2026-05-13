"""SQLite repository for Space boundary rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import SpaceBoundaryMode


def _decode_json_columns(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in tuple(payload.items()):
        if key.endswith("_json") and isinstance(value, str):
            payload[key] = json_utils.loads(value)
    return payload


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class SpaceSnapshot:
    """Small immutable Space view carried through request/work queues."""

    space_id: str
    boundary_mode: SpaceBoundaryMode
    display_name: str | None = None


class SpaceRepository:
    """Persistence operations for Space boundaries."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def get_space(
        self,
        *,
        owner_user_id: str,
        space_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM spaces
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (owner_user_id, space_id),
        )

    async def resolve_active_space(
        self,
        *,
        owner_user_id: str,
        space_id: str | None,
        workspace_id: str | None,
        boundary_mode: SpaceBoundaryMode = SpaceBoundaryMode.FOCUS,
        display_name: str | None = None,
    ) -> dict[str, Any] | None:
        if space_id is not None:
            return await self.resolve_space(
                owner_user_id=owner_user_id,
                space_id=space_id,
                boundary_mode=boundary_mode,
                display_name=display_name or space_id,
                source_kind="explicit",
                source_id=space_id,
            )
        if workspace_id is not None:
            return await self.resolve_space(
                owner_user_id=owner_user_id,
                space_id=workspace_id,
                boundary_mode=boundary_mode,
                display_name=display_name or workspace_id,
                source_kind="workspace_id",
                source_id=workspace_id,
            )
        return None

    async def resolve_space(
        self,
        *,
        owner_user_id: str,
        space_id: str,
        boundary_mode: SpaceBoundaryMode,
        display_name: str | None,
        source_kind: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_space(
            owner_user_id=owner_user_id,
            space_id=space_id,
        )
        timestamp = self._timestamp()
        if existing is not None:
            await self._connection.execute(
                """
                UPDATE spaces
                SET boundary_mode = ?,
                    display_name = COALESCE(?, display_name),
                    updated_at = ?
                WHERE owner_user_id = ?
                  AND id = ?
                """,
                (
                    boundary_mode.value,
                    _normalize_optional_text(display_name),
                    timestamp,
                    owner_user_id,
                    space_id,
                ),
            )
            if commit:
                await self._connection.commit()
            row = await self.get_space(owner_user_id=owner_user_id, space_id=space_id)
            if row is None:
                raise RuntimeError(f"Failed to resolve space {space_id}")
            return row

        await self._connection.execute(
            """
            INSERT INTO spaces(
                id,
                owner_user_id,
                boundary_mode,
                display_name,
                source_kind,
                source_id,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, source_kind, source_id)
            DO UPDATE SET
                boundary_mode = excluded.boundary_mode,
                display_name = COALESCE(excluded.display_name, spaces.display_name),
                updated_at = excluded.updated_at
            """,
            (
                space_id,
                owner_user_id,
                boundary_mode.value,
                _normalize_optional_text(display_name),
                source_kind,
                source_id,
                json_utils.dumps(metadata or {}, sort_keys=True),
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self._fetch_one(
            """
            SELECT *
            FROM spaces
            WHERE owner_user_id = ?
              AND source_kind = ?
              AND source_id = ?
            """,
            (owner_user_id, source_kind, source_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to resolve space {space_id}")
        return row

    async def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)


def space_snapshot(row: dict[str, Any]) -> SpaceSnapshot:
    return SpaceSnapshot(
        space_id=str(row["id"]),
        boundary_mode=SpaceBoundaryMode(str(row.get("boundary_mode") or SpaceBoundaryMode.FOCUS.value)),
        display_name=_normalize_optional_text(row.get("display_name")),
    )


__all__ = [
    "SpaceRepository",
    "SpaceSnapshot",
    "space_snapshot",
]
