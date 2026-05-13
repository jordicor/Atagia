"""SQLite repository for Realm world/domain rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.models.schemas_memory import CrossRealmMode


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
class RealmSnapshot:
    """Small immutable Realm view carried through request/work queues."""

    realm_id: str
    cross_realm_mode: CrossRealmMode
    display_name: str | None = None


class RealmRepository:
    """Persistence operations for world/reality/domain coordinates."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def get_realm(
        self,
        *,
        owner_user_id: str,
        realm_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM realms
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (owner_user_id, realm_id),
        )

    async def resolve_active_realm(
        self,
        *,
        owner_user_id: str,
        realm_id: str | None,
        cross_realm_mode: CrossRealmMode | str | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any] | None:
        resolved_id = _normalize_optional_text(realm_id)
        if resolved_id is None:
            return None
        return await self.resolve_realm(
            owner_user_id=owner_user_id,
            realm_id=resolved_id,
            cross_realm_mode=CrossRealmMode(
                cross_realm_mode or CrossRealmMode.NONE.value
            ),
            display_name=display_name or resolved_id,
            source_kind="explicit",
            source_id=resolved_id,
        )

    async def resolve_realm(
        self,
        *,
        owner_user_id: str,
        realm_id: str,
        cross_realm_mode: CrossRealmMode,
        display_name: str | None,
        source_kind: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_realm(
            owner_user_id=owner_user_id,
            realm_id=realm_id,
        )
        timestamp = self._timestamp()
        if existing is not None:
            await self._connection.execute(
                """
                UPDATE realms
                SET cross_realm_mode = ?,
                    display_name = COALESCE(?, display_name),
                    updated_at = ?
                WHERE owner_user_id = ?
                  AND id = ?
                """,
                (
                    cross_realm_mode.value,
                    _normalize_optional_text(display_name),
                    timestamp,
                    owner_user_id,
                    realm_id,
                ),
            )
            if commit:
                await self._connection.commit()
            row = await self.get_realm(
                owner_user_id=owner_user_id,
                realm_id=realm_id,
            )
            if row is None:
                raise RuntimeError(f"Failed to resolve realm {realm_id}")
            return row

        await self._connection.execute(
            """
            INSERT INTO realms(
                id,
                owner_user_id,
                display_name,
                source_kind,
                source_id,
                cross_realm_mode,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, source_kind, source_id)
            DO UPDATE SET
                cross_realm_mode = excluded.cross_realm_mode,
                display_name = COALESCE(excluded.display_name, realms.display_name),
                updated_at = excluded.updated_at
            """,
            (
                realm_id,
                owner_user_id,
                _normalize_optional_text(display_name),
                source_kind,
                source_id,
                cross_realm_mode.value,
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
            FROM realms
            WHERE owner_user_id = ?
              AND source_kind = ?
              AND source_id = ?
            """,
            (owner_user_id, source_kind, source_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to resolve realm {realm_id}")
        return row

    async def upsert_realm_bridge(
        self,
        *,
        owner_user_id: str,
        source_realm_id: str,
        target_realm_id: str,
        cross_realm_mode: CrossRealmMode | str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        mode = CrossRealmMode(cross_realm_mode)
        if mode == CrossRealmMode.NONE:
            raise ValueError("Realm bridges require attributed or applicable mode")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO realm_bridges(
                owner_user_id,
                source_realm_id,
                target_realm_id,
                cross_realm_mode,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, source_realm_id, target_realm_id)
            DO UPDATE SET
                cross_realm_mode = excluded.cross_realm_mode,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                owner_user_id,
                source_realm_id,
                target_realm_id,
                mode.value,
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
            FROM realm_bridges
            WHERE owner_user_id = ?
              AND source_realm_id = ?
              AND target_realm_id = ?
            """,
            (owner_user_id, source_realm_id, target_realm_id),
        )
        if row is None:
            raise RuntimeError("Failed to resolve realm bridge")
        return row

    async def _fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)


def realm_snapshot(row: dict[str, Any]) -> RealmSnapshot:
    return RealmSnapshot(
        realm_id=str(row["id"]),
        cross_realm_mode=CrossRealmMode(
            str(row.get("cross_realm_mode") or CrossRealmMode.NONE.value)
        ),
        display_name=_normalize_optional_text(row.get("display_name")),
    )


__all__ = [
    "RealmRepository",
    "RealmSnapshot",
    "realm_snapshot",
]
