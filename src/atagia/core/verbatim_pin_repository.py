"""SQLite persistence for user-controlled verbatim pins."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import (
    BaseRepository,
    MemoryObjectRepository,
    _decode_json_columns,
    _derive_sensitivity_from_privacy,
    _encode_json,
    conversation_visibility_clause,
)
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.memory.intimacy_boundary_policy import (
    is_blocked_intimacy_boundary,
    memory_object_intimacy_sql_clause,
    minimum_privacy_for_intimacy_boundary,
    normalize_intimacy_boundary,
)
from atagia.memory.embodiment_policy import embodiment_visibility_sql_clause_for_context
from atagia.memory.mind_policy import mind_visibility_sql_clause_for_context
from atagia.memory.realm_policy import realm_visibility_sql_clause_for_context
from atagia.memory.space_policy import space_visibility_sql_clause_for_context
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryCategory,
    MemoryScope,
    MemorySensitivity,
    MindTopology,
    SpaceBoundaryMode,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)


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


def _canonical_pin_scope(scope: MemoryScope) -> str:
    if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
        return MemoryScope.CHAT.value
    if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
        return MemoryScope.CHARACTER.value
    return MemoryScope.USER.value


def _normalize_optional_space_boundary_mode(value: SpaceBoundaryMode | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, SpaceBoundaryMode):
        return value.value
    try:
        return SpaceBoundaryMode(str(value)).value
    except ValueError as exc:
        raise ValueError(f"Unsupported space boundary mode: {value!r}") from exc


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
        intimacy_boundary: IntimacyBoundary | str = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        created_by: str,
        reason: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
        pin_id: str | None = None,
        status: VerbatimPinStatus = VerbatimPinStatus.ACTIVE,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        incognito_snapshot: bool = False,
        remember_across_chats_snapshot: bool = True,
        remember_across_devices_snapshot: bool = True,
        policy_snapshot: dict[str, Any] | None = None,
        space_id: str | None = None,
        space_boundary_mode: SpaceBoundaryMode | str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_pin_id = pin_id or generate_prefixed_id("vbp")
        resolved_intimacy_boundary = normalize_intimacy_boundary(intimacy_boundary)
        if is_blocked_intimacy_boundary(resolved_intimacy_boundary):
            raise ValueError("safety_blocked verbatim pins cannot be created")
        resolved_privacy_level = minimum_privacy_for_intimacy_boundary(
            resolved_intimacy_boundary,
            privacy_level=privacy_level,
        )
        resolved_sensitivity = sensitivity or _derive_sensitivity_from_privacy(
            resolved_privacy_level,
            resolved_intimacy_boundary,
            memory_category=MemoryCategory.UNKNOWN,
        )
        resolved_scope_canonical = scope_canonical or _canonical_pin_scope(scope)
        resolved_storage_scope = resolved_scope_canonical
        resolved_character_id = character_id if character_id is not None else workspace_id
        resolved_space_id = _normalize_optional_text(space_id)
        resolved_space_boundary_mode = (
            _normalize_optional_space_boundary_mode(space_boundary_mode)
            if resolved_space_id is not None
            else None
        )
        timestamp = self._timestamp()
        normalized_expires_at = normalize_optional_timestamp(expires_at)
        parameters = (
            resolved_pin_id,
            user_id,
            _normalize_optional_text(workspace_id),
            _normalize_optional_text(conversation_id),
            _normalize_optional_text(assistant_mode_id),
            resolved_storage_scope,
            target_kind.value,
            _normalize_required_text(target_id),
            target_span_start,
            target_span_end,
            _normalize_required_text(canonical_text),
            _normalize_required_text(index_text),
            int(resolved_privacy_level),
            resolved_intimacy_boundary.value,
            float(intimacy_boundary_confidence),
            status.value,
            _normalize_optional_text(reason),
            _normalize_required_text(created_by),
            timestamp,
            timestamp,
            normalized_expires_at,
            None,
            _encode_json(payload_json),
            user_persona_id,
            platform_id or "default",
            resolved_character_id,
            resolved_sensitivity.value,
            _encode_json(list(themes or [])),
            int(platform_locked),
            platform_id_lock,
            resolved_scope_canonical,
            int(incognito_snapshot),
            int(remember_across_chats_snapshot),
            int(remember_across_devices_snapshot),
            _encode_json(policy_snapshot or {}),
            resolved_space_id,
            resolved_space_boundary_mode,
            _normalize_optional_text(memory_owner_id),
            _normalize_optional_text(source_mind_id),
            _normalize_optional_text(embodiment_id),
            _normalize_optional_text(realm_id),
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
                intimacy_boundary,
                intimacy_boundary_confidence,
                status,
                reason,
                created_by,
                created_at,
                updated_at,
                expires_at,
                deleted_at,
                payload_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                scope_canonical,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                policy_snapshot_json,
                space_id,
                space_boundary_mode,
                memory_owner_id,
                source_mind_id,
                embodiment_id,
                realm_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            parameters,
        )
        if commit:
            await self._connection.commit()
        created = await self.get_verbatim_pin(
            resolved_pin_id,
            user_id,
            active_space_id=resolved_space_id,
            active_space_boundary_mode=resolved_space_boundary_mode,
            active_mind_id=_normalize_optional_text(memory_owner_id),
            mind_topology=MindTopology.UNIMIND,
            active_embodiment_id=_normalize_optional_text(embodiment_id),
            active_realm_id=_normalize_optional_text(realm_id),
        )
        if created is None:
            raise RuntimeError(f"Failed to create verbatim pin {resolved_pin_id}")
        return created

    async def get_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        visibility_clause, visibility_parameters = self._namespace_crud_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="vp",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="vp",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="vp",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="vp",
        )
        return self._strip_rowid(await self._fetch_one(
            f"""
            SELECT *
            FROM verbatim_pins AS vp
            WHERE id = ?
              AND user_id = ?
              {visibility_clause}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
            """,
            (
                pin_id,
                user_id,
                *visibility_parameters,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
            ),
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
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
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
            canonical_scope_filter = sorted({_canonical_pin_scope(scope) for scope in scope_filter})
            placeholders = ", ".join("?" for _ in canonical_scope_filter)
            clauses.append(f"scope IN ({placeholders})")
            parameters.extend(canonical_scope_filter)

        if target_kind_filter is not None:
            if not target_kind_filter:
                return []
            placeholders = ", ".join("?" for _ in target_kind_filter)
            clauses.append(f"target_kind IN ({placeholders})")
            parameters.extend(kind.value for kind in target_kind_filter)

        if target_id is not None:
            clauses.append("target_id = ?")
            parameters.append(_normalize_required_text(target_id))
        visibility_clause, visibility_parameters = self._namespace_crud_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        if visibility_clause:
            clauses.append(visibility_clause.removeprefix(" AND "))
            parameters.extend(visibility_parameters)
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="vp",
        )
        clauses.append(space_clause)
        parameters.extend(space_parameters)
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="vp",
        )
        clauses.append(mind_clause)
        parameters.extend(mind_parameters)
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="vp",
        )
        clauses.append(embodiment_clause)
        parameters.extend(embodiment_parameters)
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="vp",
        )
        clauses.append(realm_clause)
        parameters.extend(realm_parameters)

        resolved_limit = max(1, min(500, int(limit)))
        resolved_offset = max(0, int(offset))
        rows = await self._fetch_all(
            """
            SELECT *
            FROM verbatim_pins AS vp
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
        allow_intimacy_context: bool = False,
        as_of: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        allow_private_sensitivity: bool = False,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        normalized_query = _normalize_optional_text(query)
        if normalized_query is None:
            return []
        if platform_id is not None:
            visibility_clauses, visibility_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                scope_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id if character_id is not None else workspace_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                sensitivity_gates_enabled=allow_intimacy_context,
                allow_private_sensitivity=allow_private_sensitivity,
                table_alias="vp",
            )
            visibility_joiner = " AND "
        else:
            visibility_clauses, visibility_parameters = self._scope_clauses(
                scope_filter,
                assistant_mode_id=assistant_mode_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
            )
            visibility_joiner = " OR "
        if not visibility_clauses:
            return []
        resolved_as_of = normalize_optional_timestamp(as_of) or self._timestamp()
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="vp",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="vp",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="vp",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="vp",
        )
        cursor = await self._connection.execute(
            """
            SELECT
                vp.*,
                bm25(verbatim_pins_fts) AS rank
            FROM verbatim_pins_fts
            JOIN verbatim_pins AS vp ON vp._rowid = verbatim_pins_fts.rowid
            WHERE vp.user_id = ?
              AND ({visibility_clauses})
              AND vp.status = ?
              AND vp.privacy_level <= ?
              AND {intimacy_filter}
              AND {visibility_clause}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND (vp.expires_at IS NULL OR datetime(vp.expires_at) > datetime(?))
              AND verbatim_pins_fts MATCH ?
            ORDER BY rank ASC, vp.updated_at DESC, vp.created_at DESC, vp.id ASC
            LIMIT ?
            """.format(
                visibility_clauses=visibility_joiner.join(visibility_clauses),
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "vp",
                    allow_intimacy_context=allow_intimacy_context,
                ),
                visibility_clause=conversation_visibility_clause("vp"),
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
            ),
            (
                user_id,
                *visibility_parameters,
                VerbatimPinStatus.ACTIVE.value,
                privacy_ceiling,
                conversation_id,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
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
        intimacy_boundary: IntimacyBoundary | str | None = None,
        intimacy_boundary_confidence: float | None = None,
        status: VerbatimPinStatus | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
        deleted_at: str | None = None,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_verbatim_pin(
            pin_id,
            user_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
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
        if intimacy_boundary is not None:
            resolved_intimacy_boundary = normalize_intimacy_boundary(intimacy_boundary)
            if is_blocked_intimacy_boundary(resolved_intimacy_boundary):
                raise ValueError("safety_blocked verbatim pins cannot be activated")
            updates.append("intimacy_boundary = ?")
            parameters.append(resolved_intimacy_boundary.value)
            if privacy_level is None:
                updates.append("privacy_level = ?")
                parameters.append(
                    minimum_privacy_for_intimacy_boundary(
                        resolved_intimacy_boundary,
                        privacy_level=int(existing.get("privacy_level", 0) or 0),
                    )
                )
        if intimacy_boundary_confidence is not None:
            updates.append("intimacy_boundary_confidence = ?")
            parameters.append(float(intimacy_boundary_confidence))
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
            visibility_clause, visibility_parameters = self._namespace_crud_sql(
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                table_alias="",
            )
            space_clause, space_parameters = space_visibility_sql_clause_for_context(
                active_space_id=active_space_id,
                active_space_boundary_mode=active_space_boundary_mode,
                alias="",
            )
            mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
                alias="",
            )
            embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
                active_embodiment_id=active_embodiment_id,
                alias="",
            )
            realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
                active_realm_id=active_realm_id,
                alias="",
            )
            parameters.extend(
                [
                    pin_id,
                    user_id,
                    *visibility_parameters,
                    *space_parameters,
                    *mind_parameters,
                    *embodiment_parameters,
                    *realm_parameters,
                ]
            )
            await self._connection.execute(
                """
                UPDATE verbatim_pins
                SET {updates}
                WHERE id = ?
                  AND user_id = ?
                  {visibility_clause}
                  AND {space_clause}
                  AND {mind_clause}
                  AND {embodiment_clause}
                  AND {realm_clause}
                """.format(
                    updates=", ".join(updates),
                    visibility_clause=visibility_clause,
                    space_clause=space_clause,
                    mind_clause=mind_clause,
                    embodiment_clause=embodiment_clause,
                    realm_clause=realm_clause,
                ),
                tuple(parameters),
            )
            if commit:
                await self._connection.commit()
        elif commit:
            await self._connection.commit()
        return await self.get_verbatim_pin(
            pin_id,
            user_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )

    async def archive_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        return await self.update_verbatim_pin(
            pin_id,
            user_id,
            status=VerbatimPinStatus.ARCHIVED,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            commit=commit,
        )

    async def delete_verbatim_pin(
        self,
        pin_id: str,
        user_id: str,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        return await self.update_verbatim_pin(
            pin_id,
            user_id,
            status=VerbatimPinStatus.DELETED,
            deleted_at=self._timestamp(),
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            commit=commit,
        )

    @staticmethod
    def _namespace_crud_sql(
        *,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        table_alias: str = "vp",
    ) -> tuple[str, list[Any]]:
        if platform_id is None or conversation_id is None:
            return "", []
        clauses, parameters = MemoryObjectRepository.namespace_visibility_clauses(
            [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            incognito=incognito,
            sensitivity_gates_enabled=False,
            table_alias=table_alias,
        )
        if not clauses:
            return " AND 0", []
        return " AND " + " AND ".join(clauses), parameters

    @staticmethod
    def _scope_clauses(
        scope_filter: list[MemoryScope],
        *,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        alias: str = "vp",
    ) -> tuple[list[str], list[Any]]:
        del assistant_mode_id
        scope_expr = (
            f"CASE "
            f"WHEN {alias}.scope_canonical IS NOT NULL THEN {alias}.scope_canonical "
            f"WHEN {alias}.scope IN ('conversation', 'ephemeral_session') THEN 'chat' "
            f"WHEN {alias}.scope = 'workspace' THEN 'character' "
            f"WHEN {alias}.scope IN ('global_user', 'assistant_mode') THEN 'user' "
            f"ELSE {alias}.scope END"
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        canonical_scopes = MemoryObjectRepository.canonical_retrieval_scopes(scope_filter)
        for scope in canonical_scopes:
            if scope is MemoryScope.CHARACTER and workspace_id is not None:
                clauses.append(
                    f"({scope_expr} = 'character' "
                    f"AND ({alias}.character_id = ? OR ({alias}.character_id IS NULL AND {alias}.workspace_id = ?)))"
                )
                parameters.extend([workspace_id, workspace_id])
                continue
            if scope is MemoryScope.CHAT:
                clauses.append(
                    f"({scope_expr} = 'chat' AND {alias}.conversation_id = ?)"
                )
                parameters.append(conversation_id)
                continue
            clauses.append(f"({scope_expr} = 'user')")
        return clauses, parameters
