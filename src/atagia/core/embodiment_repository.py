"""SQLite repository for Embodiment body/device rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import EmbodimentBoundaryMode


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


@dataclass(frozen=True, slots=True)
class EmbodimentSnapshot:
    """Small immutable Embodiment view carried through request/work queues."""

    embodiment_id: str
    cross_embodiment_mode: EmbodimentBoundaryMode
    display_name: str | None = None


class EmbodimentRepository:
    """Persistence operations for physical body/device coordinates."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def get_embodiment(
        self,
        *,
        owner_user_id: str,
        embodiment_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM embodiments
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (owner_user_id, embodiment_id),
        )

    async def resolve_active_embodiment(
        self,
        *,
        owner_user_id: str,
        embodiment_id: str | None,
        cross_embodiment_mode: EmbodimentBoundaryMode | str | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any] | None:
        resolved_id = _normalize_optional_text(embodiment_id)
        if resolved_id is None:
            return None
        return await self.resolve_embodiment(
            owner_user_id=owner_user_id,
            embodiment_id=resolved_id,
            cross_embodiment_mode=EmbodimentBoundaryMode(
                cross_embodiment_mode
                or EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY.value
            ),
            display_name=display_name or resolved_id,
            source_kind="explicit",
            source_id=resolved_id,
        )

    async def resolve_embodiment(
        self,
        *,
        owner_user_id: str,
        embodiment_id: str,
        cross_embodiment_mode: EmbodimentBoundaryMode,
        display_name: str | None,
        source_kind: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_embodiment(
            owner_user_id=owner_user_id,
            embodiment_id=embodiment_id,
        )
        timestamp = self._timestamp()
        if existing is not None:
            await self._connection.execute(
                """
                UPDATE embodiments
                SET cross_embodiment_mode = ?,
                    display_name = COALESCE(?, display_name),
                    updated_at = ?
                WHERE owner_user_id = ?
                  AND id = ?
                """,
                (
                    cross_embodiment_mode.value,
                    _normalize_optional_text(display_name),
                    timestamp,
                    owner_user_id,
                    embodiment_id,
                ),
            )
            if commit:
                await self._connection.commit()
            row = await self.get_embodiment(
                owner_user_id=owner_user_id,
                embodiment_id=embodiment_id,
            )
            if row is None:
                raise RuntimeError(f"Failed to resolve embodiment {embodiment_id}")
            return row

        await self._connection.execute(
            """
            INSERT INTO embodiments(
                id,
                owner_user_id,
                display_name,
                source_kind,
                source_id,
                cross_embodiment_mode,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, source_kind, source_id)
            DO UPDATE SET
                cross_embodiment_mode = excluded.cross_embodiment_mode,
                display_name = COALESCE(excluded.display_name, embodiments.display_name),
                updated_at = excluded.updated_at
            """,
            (
                embodiment_id,
                owner_user_id,
                _normalize_optional_text(display_name),
                source_kind,
                source_id,
                cross_embodiment_mode.value,
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
            FROM embodiments
            WHERE owner_user_id = ?
              AND source_kind = ?
              AND source_id = ?
            """,
            (owner_user_id, source_kind, source_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to resolve embodiment {embodiment_id}")
        return row

    async def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)


def embodiment_snapshot(row: dict[str, Any]) -> EmbodimentSnapshot:
    return EmbodimentSnapshot(
        embodiment_id=str(row["id"]),
        cross_embodiment_mode=EmbodimentBoundaryMode(
            str(
                row.get("cross_embodiment_mode")
                or EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY.value
            )
        ),
        display_name=_normalize_optional_text(row.get("display_name")),
    )


__all__ = [
    "EmbodimentRepository",
    "EmbodimentSnapshot",
    "embodiment_snapshot",
]
