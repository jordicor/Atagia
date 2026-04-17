"""SQLite persistence for user-controlled verbatim pins."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_memory import MemoryScope, VerbatimPinStatus, VerbatimPinTargetKind


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _normalize_required_text(value: Any) -> str:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        raise ValueError("value must be a non-empty string")
    return normalized


class VerbatimPinRepository(BaseRepository):
    """Persistence operations for verbatim pins and their search index."""

    @staticmethod
    def _strip_rowid(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        sanitized = dict(row)
        sanitized.pop("_rowid", None)
        return sanitized

    async def create_verbatim_pin(
        self,
        *,
        user_id: str,
        scope: MemoryScope,
        target_kind: VerbatimPinTargetKind,
        target_id: str,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        canonical_text: str,
        index_text: str,
        privacy_level: int,
        created_by: str,
        reason: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
        pin_id: str | None = None,
        status: VerbatimPinStatus = VerbatimPinStatus.ACTIVE,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_pin_id = pin_id or generate_prefixed_id("vbp")
        timestamp = self._timestamp()
        normalized_expires_at = normalize_optional_timestamp(expires_at)
        parameters = (
            resolved_pin_id,
            user_id,
            _normalize_optional_text(workspace_id),
            _normalize_optional_text(conversation_id),
            _normalize_optional_text(assistant_mode_id),
            scope.value,
            target_kind.value,
            _normalize_required_text(target_id),
            target_span_start,
            target_span_end,
            _normalize_required_text(canonical_text),
            _normalize_required_text(index_text),
            int(privacy_level),
            status.value,
            _normalize_optional_text(reason),
            _normalize_required_text(created_by),
            timestamp,
            timestamp,
            normalized_expires_at,
            None,
            _encode_json(payload_json),
        )
        await self._connection.execute(
            """
            INSERT INTO verbatim_pins(
                id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope,
                target_kind,
                target_id,
                target_span_start,
                target_span_end,
                canonical_text,
                index_text,
                privacy_level,
                status,
                reason,
                created_by,
                created_at,
                updated_at,
                expires_at,
                deleted_at,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            parameters,
        )
        if commit:
            await self._connection.commit()
        created = await self.get_verbatim_pin(resolved_pin_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create verbatim pin {resolved_pin_id}")
        return created

    async def get_verbatim_pin(self, pin_id: str, user_id: str) -> dict[str, Any] | None:
        return self._strip_rowid(await self._fetch_one(
            """
            SELECT *
            FROM verbatim_pins
            WHERE id = ?
              AND user_id = ?
            """,
            (pin_id, user_id),
        ))

    async def list_verbatim_pins(
        self,
        user_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        scope_filter: list[MemoryScope] | None = None,
        target_kind_filter: list[VerbatimPinTargetKind] | None = None,
        status_filter: list[VerbatimPinStatus] | None = None,
        target_id: str | None = None,
        include_deleted: bool = False,
        active_only: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]

        if active_only:
            clauses.append("status = ?")
            parameters.append(VerbatimPinStatus.ACTIVE.value)
            clauses.append("(expires_at IS NULL OR datetime(expires_at) > datetime(?))")
            parameters.append(normalize_optional_timestamp(as_of) or self._timestamp())
        elif status_filter is not None:
            if not status_filter:
                return []
            placeholders = ", ".join("?" for _ in status_filter)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in status_filter)
        elif not include_deleted:
            clauses.append("status != ?")
            parameters.append(VerbatimPinStatus.DELETED.value)

        if scope_filter is not None:
            if not scope_filter:
                return []
            placeholders = ", ".join("?" for _ in scope_filter)
            clauses.append(f"scope IN ({placeholders})")
            parameters.extend(scope.value for scope in scope_filter)

        if target_kind_filter is not None:
            if not target_kind_filter:
                return []
            placeholders = ", ".join("?" for _ in target_kind_filter)
            clauses.append(f"target_kind IN ({placeholders})")
            parameters.extend(kind.value for kind in target_kind_filter)

        if target_id is not None:
            clauses.append("target_id = ?")
            parameters.append(_normalize_required_text(target_id))

        resolved_limit = max(1, min(500, int(limit)))
        resolved_offset = max(0, int(offset))
        rows = await self._fetch_all(
            """
            SELECT *
            FROM verbatim_pins
            WHERE {where_clause}
            ORDER BY updated_at DESC, created_at DESC, id ASC
            LIMIT ?
            OFFSET ?
            """.format(where_clause=" AND ".join(clauses)),
            tuple([*parameters, resolved_limit, resolved_offset]),
        )
        return [
            sanitized
            for sanitized in (self._strip_rowid(row) for row in rows)
            if sanitized is not None
        ]

    async def search_active_verbatim_pins(
        self,
        *,
        user_id: str,
        query: str,
        privacy_ceiling: int,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        limit: int,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        normalized_query = _normalize_optional_text(query)
        if normalized_query is None:
            return []
        scope_clauses, scope_parameters = self._scope_clauses(
            scope_filter,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if not scope_clauses:
            return []
        resolved_as_of = normalize_optional_timestamp(as_of) or self._timestamp()
        cursor = await self._connection.execute(
            """
            SELECT
                vp.*,
                bm25(verbatim_pins_fts) AS rank
            FROM verbatim_pins_fts
            JOIN verbatim_pins AS vp ON vp._rowid = verbatim_pins_fts.rowid
            WHERE vp.user_id = ?
              AND ({scope_clauses})
              AND vp.status = ?
              AND vp.privacy_level <= ?
              AND (vp.expires_at IS NULL OR datetime(vp.expires_at) > datetime(?))
              AND verbatim_pins_fts MATCH ?
            ORDER BY rank ASC, vp.updated_at DESC, vp.created_at DESC, vp.id ASC
            LIMIT ?
            """.format(scope_clauses=" OR ".join(scope_clauses)),
            (
                user_id,
                *scope_parameters,
                VerbatimPinStatus.ACTIVE.value,
                privacy_ceiling,
                resolved_as_of,
                normalized_query,
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return [
            sanitized
            for sanitized in (self._strip_rowid(decoded) for decoded in (_decode_json_columns(row) for row in rows) if decoded is not None)
            if sanitized is not None
        ]

    async def update_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int | None = None,
        status: VerbatimPinStatus | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
        deleted_at: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_verbatim_pin(pin_id, user_id)
        if existing is None:
            return None

        updates: list[str] = []
        parameters: list[Any] = []
        current_payload = existing.get("payload_json")
        normalized_payload = dict(current_payload) if isinstance(current_payload, dict) else {}
        if payload_json is not None:
            normalized_payload = dict(payload_json)

        if canonical_text is not None:
            updates.append("canonical_text = ?")
            parameters.append(_normalize_required_text(canonical_text))
        if index_text is not None:
            updates.append("index_text = ?")
            parameters.append(_normalize_required_text(index_text))
        if target_span_start is not None or target_span_end is not None:
            if target_span_start is not None and target_span_end is not None and target_span_end < target_span_start:
                raise ValueError("target_span_end must be greater than or equal to target_span_start")
            if target_span_start is not None:
                updates.append("target_span_start = ?")
                parameters.append(target_span_start)
            if target_span_end is not None:
                updates.append("target_span_end = ?")
                parameters.append(target_span_end)
        if privacy_level is not None:
            updates.append("privacy_level = ?")
            parameters.append(int(privacy_level))
        if status is not None:
            updates.append("status = ?")
            parameters.append(status.value)
            if status is VerbatimPinStatus.DELETED and deleted_at is None:
                deleted_at = self._timestamp()
        if reason is not None:
            updates.append("reason = ?")
            parameters.append(_normalize_optional_text(reason))
        if expires_at is not None:
            updates.append("expires_at = ?")
            parameters.append(normalize_optional_timestamp(expires_at))
        if payload_json is not None:
            updates.append("payload_json = ?")
            parameters.append(_encode_json(normalized_payload))
        if deleted_at is not None:
            updates.append("deleted_at = ?")
            parameters.append(normalize_optional_timestamp(deleted_at))

        if updates:
            updates.append("updated_at = ?")
            parameters.append(self._timestamp())
            parameters.extend([pin_id, user_id])
            await self._connection.execute(
                """
                UPDATE verbatim_pins
                SET {updates}
                WHERE id = ?
                  AND user_id = ?
                """.format(updates=", ".join(updates)),
                tuple(parameters),
            )
            if commit:
                await self._connection.commit()
        elif commit:
            await self._connection.commit()
        return await self.get_verbatim_pin(pin_id, user_id)

    async def archive_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        return await self.update_verbatim_pin(
            pin_id,
            user_id,
            status=VerbatimPinStatus.ARCHIVED,
            commit=commit,
        )

    async def delete_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        return await self.update_verbatim_pin(
            pin_id,
            user_id,
            status=VerbatimPinStatus.DELETED,
            deleted_at=self._timestamp(),
            commit=commit,
        )

    @staticmethod
    def _scope_clauses(
        scope_filter: list[MemoryScope],
        *,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        alias: str = "vp",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scope_filter:
            if scope is MemoryScope.WORKSPACE and workspace_id is None:
                continue
            if scope is MemoryScope.ASSISTANT_MODE:
                clauses.append(f"({alias}.scope = ? AND {alias}.assistant_mode_id = ?)")
                parameters.extend([scope.value, assistant_mode_id])
                continue
            if scope is MemoryScope.WORKSPACE:
                clauses.append(
                    f"({alias}.scope = ? AND {alias}.assistant_mode_id = ? AND {alias}.workspace_id = ?)"
                )
                parameters.extend([scope.value, assistant_mode_id, workspace_id])
                continue
            if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION}:
                clauses.append(
                    f"({alias}.scope = ? AND {alias}.assistant_mode_id = ? AND {alias}.conversation_id = ?)"
                )
                parameters.extend([scope.value, assistant_mode_id, conversation_id])
                continue
            clauses.append(f"({alias}.scope = ?)")
            parameters.append(scope.value)
        return clauses, parameters
