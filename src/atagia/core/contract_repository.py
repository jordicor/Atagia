"""SQLite repository helpers for interaction contract projections."""

from __future__ import annotations

from typing import Any

from atagia.core import json_utils
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.models.schemas_memory import (
    AssistantModeManifest,
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
    ) -> int:
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
    ) -> list[dict[str, Any]]:
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
        manifest = AssistantModeManifest.model_validate(json_utils.loads(row["memory_policy_json"]))
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
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                user_id,
                COALESCE(workspace_id, ''),
                COALESCE(conversation_id, ''),
                COALESCE(assistant_mode_id, ''),
                scope,
                dimension_name
            )
            DO UPDATE SET
                value_json = excluded.value_json,
                confidence = excluded.confidence,
                source_memory_id = excluded.source_memory_id,
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
                scope.value,
                dimension_name,
                _encode_json(value_json),
                confidence,
                source_memory_id,
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
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            scope=scope,
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
        commit: bool = True,
    ) -> dict[str, Any] | None:
        await self._connection.execute(
            """
            DELETE FROM contract_dimensions_current
            WHERE user_id = ?
              AND COALESCE(workspace_id, '') = COALESCE(?, '')
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND COALESCE(assistant_mode_id, '') = COALESCE(?, '')
              AND scope = ?
              AND dimension_name = ?
            """,
            (
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope.value,
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
              AND scope = ?
              AND COALESCE(workspace_id, '') = COALESCE(?, '')
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND COALESCE(assistant_mode_id, '') = COALESCE(?, '')
              AND json_extract(payload_json, '$.dimension_name') = ?
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT 1
            """,
            (
                user_id,
                MemoryObjectType.INTERACTION_CONTRACT.value,
                MemoryStatus.ACTIVE.value,
                scope.value,
                workspace_id,
                conversation_id,
                assistant_mode_id,
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
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        scope: MemoryScope,
        dimension_name: str,
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM contract_dimensions_current
            WHERE user_id = ?
              AND COALESCE(workspace_id, '') = COALESCE(?, '')
              AND COALESCE(conversation_id, '') = COALESCE(?, '')
              AND COALESCE(assistant_mode_id, '') = COALESCE(?, '')
              AND scope = ?
              AND dimension_name = ?
            """,
            (
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope.value,
                dimension_name,
            ),
        )
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    @staticmethod
    def _context_clauses(
        *,
        user_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses = [
            "(scope = 'global_user' AND assistant_mode_id IS NULL AND workspace_id IS NULL AND conversation_id IS NULL)",
            "(scope = 'assistant_mode' AND assistant_mode_id = ? AND workspace_id IS NULL AND conversation_id IS NULL)",
        ]
        parameters: list[Any] = [user_id, assistant_mode_id]
        if workspace_id is not None:
            clauses.append(
                "(scope = 'workspace' AND assistant_mode_id = ? AND workspace_id = ? AND conversation_id IS NULL)"
            )
            parameters.extend([assistant_mode_id, workspace_id])
        if conversation_id is not None:
            clauses.append(
                "(scope = 'conversation' AND assistant_mode_id = ? AND conversation_id = ?)"
            )
            clauses.append(
                "(scope = 'ephemeral_session' AND assistant_mode_id = ? AND conversation_id = ?)"
            )
            parameters.extend([assistant_mode_id, conversation_id, assistant_mode_id, conversation_id])
        return clauses, parameters
