"""SQLite repository for Presence attribution rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import PresenceKind


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
class PresenceSnapshot:
    """Small immutable Presence view carried through request/work queues."""

    presence_id: str
    kind: PresenceKind
    display_name: str | None = None
    presence_cluster_id: str | None = None


class PresenceRepository:
    """Persistence operations for Presence attribution."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def get_presence(
        self,
        *,
        owner_user_id: str,
        presence_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM presences
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (owner_user_id, presence_id),
        )

    async def resolve_active_presence(
        self,
        *,
        owner_user_id: str,
        active_presence_id: str | None,
        character_id: str | None,
    ) -> dict[str, Any]:
        if active_presence_id is not None:
            return await self.resolve_presence(
                owner_user_id=owner_user_id,
                presence_id=active_presence_id,
                kind=PresenceKind.UNKNOWN,
                display_name=active_presence_id,
                source_kind="explicit",
                source_id=active_presence_id,
            )
        if character_id is not None:
            return await self.resolve_presence(
                owner_user_id=owner_user_id,
                presence_id=character_id,
                kind=PresenceKind.OWNED_FACET,
                display_name=character_id,
                source_kind="character_id",
                source_id=character_id,
            )
        return await self.resolve_presence(
            owner_user_id=owner_user_id,
            presence_id="default_assistant",
            kind=PresenceKind.OWNED_AI,
            display_name="Assistant",
            source_kind="default_ai",
            source_id="default_assistant",
        )

    async def resolve_human_owner_presence(
        self,
        *,
        owner_user_id: str,
    ) -> dict[str, Any]:
        return await self.resolve_presence(
            owner_user_id=owner_user_id,
            presence_id="human_owner",
            kind=PresenceKind.HUMAN,
            display_name="User",
            source_kind="human_owner",
            source_id="human_owner",
        )

    async def resolve_presence(
        self,
        *,
        owner_user_id: str,
        presence_id: str,
        kind: PresenceKind,
        display_name: str | None,
        source_kind: str,
        source_id: str,
        presence_cluster_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_presence(
            owner_user_id=owner_user_id,
            presence_id=presence_id,
        )
        if existing is not None:
            return existing

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO presences(
                id,
                owner_user_id,
                kind,
                display_name,
                source_kind,
                source_id,
                presence_cluster_id,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, source_kind, source_id)
            DO UPDATE SET
                display_name = COALESCE(presences.display_name, excluded.display_name),
                updated_at = excluded.updated_at
            """,
            (
                presence_id,
                owner_user_id,
                kind.value,
                _normalize_optional_text(display_name),
                source_kind,
                source_id,
                _normalize_optional_text(presence_cluster_id),
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
            FROM presences
            WHERE owner_user_id = ?
              AND source_kind = ?
              AND source_id = ?
            """,
            (owner_user_id, source_kind, source_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to resolve presence {presence_id}")
        return row

    async def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)


def presence_snapshot(row: dict[str, Any]) -> PresenceSnapshot:
    return PresenceSnapshot(
        presence_id=str(row["id"]),
        kind=PresenceKind(str(row.get("kind") or PresenceKind.UNKNOWN.value)),
        display_name=_normalize_optional_text(row.get("display_name")),
        presence_cluster_id=_normalize_optional_text(row.get("presence_cluster_id")),
    )


__all__ = [
    "PresenceRepository",
    "PresenceSnapshot",
    "presence_snapshot",
]
