"""SQLite repository for Mind perspective rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import MindKind, MindTopology, PresenceKind

DEFAULT_MIND_ID = "default_mind"
DEFAULT_OVERSEER_MIND_ID = "ojocentauri"


class MindNotFoundError(ValueError):
    """Raised when an explicit Mind reference is outside the owner boundary."""


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


def _kind_from_presence_kind(value: PresenceKind | str | None) -> MindKind:
    if isinstance(value, PresenceKind):
        value = value.value
    try:
        return MindKind(str(value or MindKind.UNKNOWN.value))
    except ValueError:
        return MindKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class MindSnapshot:
    """Small immutable Mind view carried through request/work queues."""

    mind_id: str
    kind: MindKind
    topology: MindTopology
    display_name: str | None = None


class MindRepository:
    """Persistence operations for Mind perspectives."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def get_mind(
        self,
        *,
        owner_user_id: str,
        mind_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM minds
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (owner_user_id, mind_id),
        )

    async def resolve_active_mind(
        self,
        *,
        owner_user_id: str,
        mind_id: str | None,
        active_presence_id: str | None,
        active_presence_kind: PresenceKind | str | None = None,
        active_presence_display_name: str | None = None,
        character_id: str | None = None,
        topology: MindTopology | str | None = None,
    ) -> dict[str, Any]:
        resolved_topology = MindTopology(topology or MindTopology.UNIMIND.value)
        if mind_id is not None:
            existing = await self.get_mind(
                owner_user_id=owner_user_id,
                mind_id=mind_id,
            )
            if existing is None:
                raise MindNotFoundError("Mind not found for user")
            if (
                resolved_topology is MindTopology.OJOCENTAURI
                and existing.get("kind") != MindKind.OVERSEER.value
            ):
                raise MindNotFoundError("Overseer Mind not found for user")
            return existing

        if resolved_topology is MindTopology.OJOCENTAURI:
            return await self.resolve_mind(
                owner_user_id=owner_user_id,
                mind_id=DEFAULT_OVERSEER_MIND_ID,
                kind=MindKind.OVERSEER,
                display_name="OjoCentauri",
                source_kind="ojocentauri",
                source_id=DEFAULT_OVERSEER_MIND_ID,
            )

        if resolved_topology is MindTopology.MULTI_MIND:
            if active_presence_id is not None:
                return await self.resolve_mind(
                    owner_user_id=owner_user_id,
                    mind_id=active_presence_id,
                    kind=_kind_from_presence_kind(active_presence_kind),
                    display_name=active_presence_display_name or active_presence_id,
                    source_kind="active_presence",
                    source_id=active_presence_id,
                )
            if character_id is not None:
                return await self.resolve_mind(
                    owner_user_id=owner_user_id,
                    mind_id=character_id,
                    kind=MindKind.OWNED_FACET,
                    display_name=character_id,
                    source_kind="character_id",
                    source_id=character_id,
                )

        return await self.resolve_mind(
            owner_user_id=owner_user_id,
            mind_id=DEFAULT_MIND_ID,
            kind=MindKind.OWNED_AI,
            display_name="Default Mind",
            source_kind="default_mind",
            source_id=DEFAULT_MIND_ID,
        )

    async def resolve_mind(
        self,
        *,
        owner_user_id: str,
        mind_id: str,
        kind: MindKind,
        display_name: str | None,
        source_kind: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_mind(
            owner_user_id=owner_user_id,
            mind_id=mind_id,
        )
        if existing is not None:
            return existing

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO minds(
                id,
                owner_user_id,
                kind,
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
                display_name = COALESCE(minds.display_name, excluded.display_name),
                updated_at = excluded.updated_at
            """,
            (
                mind_id,
                owner_user_id,
                kind.value,
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
            FROM minds
            WHERE owner_user_id = ?
              AND source_kind = ?
              AND source_id = ?
            """,
            (owner_user_id, source_kind, source_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to resolve mind {mind_id}")
        return row

    async def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)


def mind_snapshot(row: dict[str, Any], topology: MindTopology | str | None) -> MindSnapshot:
    return MindSnapshot(
        mind_id=str(row["id"]),
        kind=MindKind(str(row.get("kind") or MindKind.UNKNOWN.value)),
        topology=MindTopology(topology or MindTopology.UNIMIND.value),
        display_name=_normalize_optional_text(row.get("display_name")),
    )


__all__ = [
    "DEFAULT_MIND_ID",
    "DEFAULT_OVERSEER_MIND_ID",
    "MindNotFoundError",
    "MindRepository",
    "MindSnapshot",
    "mind_snapshot",
]
