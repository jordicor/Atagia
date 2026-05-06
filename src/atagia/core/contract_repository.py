"""SQLite repository helpers for interaction contract projections."""

from __future__ import annotations

from typing import Any

from atagia.core import json_utils
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import (
    BaseRepository,
    MemoryObjectRepository,
    _decode_json_columns,
    _encode_json,
    conversation_visibility_clause,
)
from atagia.models.schemas_memory import (
    RetrievalProfileManifest,
    MemoryObjectType,
    MemoryScope,
    MemoryStatus,
)


class ContractDimensionRepository(BaseRepository):
    """Persistence operations for contract_dimensions_current."""

    async def count_for_context(
        self,
        user_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> int:
        if platform_id is not None and conversation_id is not None:
            source_clauses, source_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                table_alias="source",
            )
            row_clauses, row_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                table_alias="cdc",
            )
            if not source_clauses or not row_clauses:
                return 0
            cursor = await self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM contract_dimensions_current AS cdc
                JOIN memory_objects AS source
                  ON source.id = cdc.source_memory_id
                 AND source.user_id = cdc.user_id
                WHERE cdc.user_id = ?
                  AND source.status = ?
                  AND source.archived_by_conversation_id IS NULL
                  AND {visibility_clause}
                  AND {source_clauses}
                  AND {row_clauses}
                """.format(
                    source_clauses=" AND ".join(source_clauses),
                    row_clauses=" AND ".join(row_clauses),
                    visibility_clause=conversation_visibility_clause("source"),
                ),
                (
                    user_id,
                    MemoryStatus.ACTIVE.value,
                    conversation_id,
                    *source_parameters,
                    *row_parameters,
                ),
            )
            row = await cursor.fetchone()
            return int(row["count"])

        clauses, parameters = self._context_clauses(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM contract_dimensions_current
            WHERE user_id = ?
              AND ({clauses})
            """.format(clauses=" OR ".join(clauses)),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def list_for_context(
        self,
        user_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        if platform_id is not None and conversation_id is not None:
            source_clauses, source_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                table_alias="source",
            )
            row_clauses, row_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                table_alias="cdc",
            )
            if not source_clauses or not row_clauses:
                return []
            return await self._fetch_all(
                """
                SELECT cdc.*
                FROM contract_dimensions_current AS cdc
                JOIN memory_objects AS source
                  ON source.id = cdc.source_memory_id
                 AND source.user_id = cdc.user_id
                WHERE cdc.user_id = ?
                  AND source.status = ?
                  AND source.archived_by_conversation_id IS NULL
                  AND {visibility_clause}
                  AND {source_clauses}
                  AND {row_clauses}
                ORDER BY cdc.updated_at DESC, cdc.id ASC
                """.format(
                    source_clauses=" AND ".join(source_clauses),
                    row_clauses=" AND ".join(row_clauses),
                    visibility_clause=conversation_visibility_clause("source"),
                ),
                (
                    user_id,
                    MemoryStatus.ACTIVE.value,
                    conversation_id,
                    *source_parameters,
                    *row_parameters,
                ),
            )

        clauses, parameters = self._context_clauses(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return await self._fetch_all(
            """
            SELECT *
            FROM contract_dimensions_current
            WHERE user_id = ?
              AND ({clauses})
            ORDER BY updated_at DESC, id ASC
            """.format(clauses=" OR ".join(clauses)),
            tuple(parameters),
        )

    async def get_mode_contract_dimensions_priority(self, assistant_mode_id: str) -> list[str]:
        cursor = await self._connection.execute(
            """
            SELECT memory_policy_json
            FROM assistant_modes
            WHERE id = ?
            """,
            (assistant_mode_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown assistant mode: {assistant_mode_id}")
        manifest = RetrievalProfileManifest.model_validate(json_utils.loads(row["memory_policy_json"]))
        return list(manifest.contract_dimensions_priority)

    async def upsert_projection(
        self,
        *,
        user_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        scope: MemoryScope,
        dimension_name: str,
        value_json: dict[str, Any],
        confidence: float,
        source_memory_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        source_memory = await self._fetch_one(
            """
            SELECT
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                scope_canonical,
                payload_json
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (source_memory_id, user_id),
        )
        if source_memory is None:
            raise ValueError(f"Source memory {source_memory_id} does not belong to user {user_id}")
        resolved_scope_canonical = source_memory.get("scope_canonical") or self._canonical_scope(scope)
        if resolved_scope_canonical not in {
            MemoryScope.CHAT.value,
            MemoryScope.CHARACTER.value,
            MemoryScope.USER.value,
        }:
            resolved_scope_canonical = self._canonical_scope(scope)
        source_payload = source_memory.get("payload_json") or {}
        source_policy = (
            source_payload.get("source_turn_policy")
            if isinstance(source_payload, dict)
            else None
        )
        if not isinstance(source_policy, dict):
            source_policy = {}
        timestamp = self._timestamp()
        projection_id = generate_prefixed_id("ctd")
        cursor = await self._connection.execute(
            """
            INSERT INTO contract_dimensions_current(
                id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope,
                dimension_name,
                value_json,
                confidence,
                source_memory_id,
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
                temporary_snapshot,
                purge_on_close_snapshot,
                policy_snapshot_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                user_id,
                user_persona_key,
                character_key,
                conversation_key,
                scope_canonical_key,
                dimension_name
            )
            DO UPDATE SET
                value_json = excluded.value_json,
                confidence = excluded.confidence,
                source_memory_id = excluded.source_memory_id,
                user_persona_id = excluded.user_persona_id,
                platform_id = excluded.platform_id,
                character_id = excluded.character_id,
                sensitivity = CASE
                    WHEN contract_dimensions_current.sensitivity = 'secret'
                      OR excluded.sensitivity = 'secret' THEN 'secret'
                    WHEN contract_dimensions_current.sensitivity = 'private'
                      OR excluded.sensitivity = 'private' THEN 'private'
                    ELSE excluded.sensitivity
                END,
                themes_json = excluded.themes_json,
                platform_locked = MAX(contract_dimensions_current.platform_locked, excluded.platform_locked),
                platform_id_lock = COALESCE(contract_dimensions_current.platform_id_lock, excluded.platform_id_lock),
                scope_canonical = excluded.scope_canonical,
                incognito_snapshot = MAX(
                    contract_dimensions_current.incognito_snapshot,
                    excluded.incognito_snapshot
                ),
                remember_across_chats_snapshot = MIN(
                    contract_dimensions_current.remember_across_chats_snapshot,
                    excluded.remember_across_chats_snapshot
                ),
                remember_across_devices_snapshot = MIN(
                    contract_dimensions_current.remember_across_devices_snapshot,
                    excluded.remember_across_devices_snapshot
                ),
                temporary_snapshot = MAX(
                    contract_dimensions_current.temporary_snapshot,
                    excluded.temporary_snapshot
                ),
                purge_on_close_snapshot = MAX(
                    contract_dimensions_current.purge_on_close_snapshot,
                    excluded.purge_on_close_snapshot
                ),
                policy_snapshot_json = excluded.policy_snapshot_json,
                updated_at = excluded.updated_at
            WHERE excluded.confidence >= contract_dimensions_current.confidence
            RETURNING *
            """,
            (
                projection_id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                resolved_scope_canonical,
                dimension_name,
                _encode_json(value_json),
                confidence,
                source_memory_id,
                source_memory.get("user_persona_id"),
                source_memory.get("platform_id") or "default",
                source_memory.get("character_id"),
                source_memory.get("sensitivity") or "unknown",
                _encode_json(source_memory.get("themes_json") or []),
                int(source_memory.get("platform_locked") or 0),
                source_memory.get("platform_id_lock"),
                resolved_scope_canonical,
                int(bool(source_policy.get("incognito"))),
                int(source_policy.get("remember_across_chats", True) is not False),
                int(source_policy.get("remember_across_devices", True) is not False),
                int(bool(source_policy.get("temporary"))),
                int(bool(source_policy.get("purge_on_close"))),
                _encode_json(source_policy),
                timestamp,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if commit:
            await self._connection.commit()
        if row is not None:
            decoded = _decode_json_columns(row)
            if decoded is None:
                raise RuntimeError("Failed to decode contract projection row")
            return decoded

        existing = await self._fetch_exact_projection(
            user_id=user_id,
            user_persona_id=source_memory.get("user_persona_id"),
            character_id=source_memory.get("character_id"),
            conversation_id=conversation_id,
            scope_canonical=resolved_scope_canonical,
            dimension_name=dimension_name,
        )
        if existing is None:
            raise RuntimeError("Failed to upsert contract projection row")
        return existing

    async def list_projection_keys_for_sources(self, source_memory_ids: list[str]) -> list[dict[str, Any]]:
        if not source_memory_ids:
            return []
        placeholders = ", ".join("?" for _ in source_memory_ids)
        return await self._fetch_all(
            f"""
            SELECT
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope,
                user_persona_id,
                character_id,
                scope_canonical,
                dimension_name
            FROM contract_dimensions_current
            WHERE source_memory_id IN ({placeholders})
            ORDER BY updated_at DESC, id ASC
            """,
            tuple(source_memory_ids),
        )

    async def reproject_best_remaining(
        self,
        *,
        user_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        scope: MemoryScope,
        dimension_name: str,
        user_persona_id: str | None = None,
        character_id: str | None = None,
        scope_canonical: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        resolved_scope_canonical = scope_canonical or self._canonical_scope(scope)
        await self._connection.execute(
            """
            DELETE FROM contract_dimensions_current
            WHERE user_id = ?
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND user_persona_id IS ?
              AND character_id IS ?
              AND scope_canonical = ?
              AND dimension_name = ?
            """,
            (
                user_id,
                conversation_id,
                user_persona_id,
                character_id,
                resolved_scope_canonical,
                dimension_name,
            ),
        )
        cursor = await self._connection.execute(
            """
            SELECT id, payload_json, confidence
            FROM memory_objects
            WHERE user_id = ?
              AND object_type = ?
              AND status = ?
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND user_persona_id IS ?
              AND character_id IS ?
              AND COALESCE(scope_canonical, scope) = ?
              AND json_extract(payload_json, '$.dimension_name') = ?
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT 1
            """,
            (
                user_id,
                MemoryObjectType.INTERACTION_CONTRACT.value,
                MemoryStatus.ACTIVE.value,
                conversation_id,
                user_persona_id,
                character_id,
                resolved_scope_canonical,
                dimension_name,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            if commit:
                await self._connection.commit()
            return None

        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json_utils.loads(payload)
        value_json = payload.get("value_json", {})
        if not isinstance(value_json, dict):
            value_json = {}
        return await self.upsert_projection(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            scope=scope,
            dimension_name=dimension_name,
            value_json=value_json,
            confidence=float(row["confidence"]),
            source_memory_id=str(row["id"]),
            commit=commit,
        )

    async def _fetch_exact_projection(
        self,
        *,
        user_id: str,
        user_persona_id: str | None,
        character_id: str | None,
        conversation_id: str | None,
        scope_canonical: str,
        dimension_name: str,
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM contract_dimensions_current
            WHERE user_id = ?
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND user_persona_id IS ?
              AND character_id IS ?
              AND scope_canonical = ?
              AND dimension_name = ?
            """,
            (
                user_id,
                conversation_id,
                user_persona_id,
                character_id,
                scope_canonical,
                dimension_name,
            ),
        )
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    @staticmethod
    def _canonical_scope(scope: MemoryScope) -> str:
        if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
            return MemoryScope.CHAT.value
        if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
            return MemoryScope.CHARACTER.value
        return MemoryScope.USER.value

    @staticmethod
    def _context_clauses(
        *,
        user_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        del assistant_mode_id
        scope_expr = (
            "CASE "
            "WHEN scope_canonical IS NOT NULL THEN scope_canonical "
            "WHEN scope IN ('conversation', 'ephemeral_session') THEN 'chat' "
            "WHEN scope = 'workspace' THEN 'character' "
            "WHEN scope IN ('global_user', 'assistant_mode') THEN 'user' "
            "ELSE scope END"
        )
        clauses = [
            f"({scope_expr} = 'user' AND conversation_id IS NULL)",
        ]
        parameters: list[Any] = [user_id]
        if workspace_id is not None:
            clauses.append(
                f"({scope_expr} = 'character' "
                "AND (character_id = ? OR (character_id IS NULL AND workspace_id = ?)) "
                "AND conversation_id IS NULL)"
            )
            parameters.extend([workspace_id, workspace_id])
        if conversation_id is not None:
            clauses.append(f"({scope_expr} = 'chat' AND conversation_id = ?)")
            parameters.append(conversation_id)
        return clauses, parameters
