"""SQLite repositories for the Step 3 persistence layer."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id, new_memory_id
from atagia.core.storage_backend import StorageBackend
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus


def _decode_json_columns(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in tuple(payload.items()):
        if key.endswith("_json") and isinstance(value, str):
            payload[key] = json.loads(value)
    return payload


def _encode_json(value: dict[str, Any] | list[Any] | None) -> str:
    if value is None:
        return json.dumps({}, ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _stable_string_union(existing: list[str], new_values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *new_values]:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


class BaseRepository:
    """Shared helpers for SQLite-backed repositories."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def begin(self) -> None:
        await self._connection.execute("BEGIN")

    async def commit(self) -> None:
        await self._connection.commit()

    async def rollback(self) -> None:
        await self._connection.rollback()

    async def _fetch_one(self, query: str, parameters: tuple[Any, ...]) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    async def _fetch_all(self, query: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        return [_decode_json_columns(row) for row in rows]


class UserRepository(BaseRepository):
    """Persistence operations for users."""

    async def create_user(self, user_id: str | None = None, external_ref: str | None = None) -> dict[str, Any]:
        resolved_user_id = user_id or generate_prefixed_id("usr")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (resolved_user_id, external_ref, timestamp, timestamp),
        )
        await self._connection.commit()
        return await self.get_user(resolved_user_id)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        )

    async def delete_user(self, user_id: str) -> None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE users
            SET deleted_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (timestamp, timestamp, user_id),
        )
        await self._connection.commit()


class WorkspaceRepository(BaseRepository):
    """Persistence operations for workspaces."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        storage_backend: StorageBackend | None = None,
    ) -> None:
        super().__init__(connection, clock)
        self._storage_backend = storage_backend

    async def create_workspace(
        self,
        workspace_id: str | None,
        user_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_workspace_id = workspace_id or generate_prefixed_id("wrk")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO workspaces(id, user_id, name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (resolved_workspace_id, user_id, name, _encode_json(metadata), timestamp, timestamp),
        )
        await self._connection.commit()
        return await self.get_workspace(resolved_workspace_id, user_id)

    async def get_workspace(self, workspace_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM workspaces
            WHERE id = ?
              AND user_id = ?
            """,
            (workspace_id, user_id),
        )

    async def list_workspaces(self, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM workspaces
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        )

    async def delete_workspace(self, workspace_id: str, user_id: str) -> None:
        await self._connection.execute(
            """
            DELETE FROM workspaces
            WHERE id = ?
              AND user_id = ?
            """,
            (workspace_id, user_id),
        )
        await self._connection.commit()
        if self._storage_backend is not None:
            await self._storage_backend.delete_context_views_for_user(user_id)


class ConversationRepository(BaseRepository):
    """Persistence operations for conversations."""

    async def create_conversation(
        self,
        conversation_id: str | None,
        user_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
        title: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_conversation_id = conversation_id or generate_prefixed_id("cnv")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO conversations(
                id,
                user_id,
                workspace_id,
                assistant_mode_id,
                title,
                status,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                resolved_conversation_id,
                user_id,
                workspace_id,
                assistant_mode_id,
                title,
                _encode_json(metadata),
                timestamp,
                timestamp,
            ),
        )
        await self._connection.commit()
        return await self.get_conversation(resolved_conversation_id, user_id)

    async def get_conversation(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
              AND user_id = ?
            """,
            (conversation_id, user_id),
        )

    async def list_conversations(
        self,
        user_id: str,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if workspace_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC, id ASC
                """,
                (user_id,),
            )

        return await self._fetch_all(
            """
            SELECT *
            FROM conversations
            WHERE user_id = ?
              AND workspace_id = ?
            ORDER BY updated_at DESC, id ASC
            """,
            (user_id, workspace_id),
        )

    async def update_conversation_status(self, conversation_id: str, user_id: str, status: str) -> None:
        await self._connection.execute(
            """
            UPDATE conversations
            SET status = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (status, self._timestamp(), conversation_id, user_id),
        )
        await self._connection.commit()


class MessageRepository(BaseRepository):
    """Persistence operations for messages plus Step 3 FTS search."""

    async def next_sequence(self, conversation_id: str) -> int:
        cursor = await self._connection.execute(
            """
            SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return int(row["next_seq"])

    async def create_message(
        self,
        message_id: str | None,
        conversation_id: str,
        role: str,
        seq: int | None,
        text: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        occurred_at: str | None = None,
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_message_id = message_id or generate_prefixed_id("msg")
        timestamp = self._timestamp()
        resolved_occurred_at = normalize_optional_timestamp(occurred_at) or timestamp
        if seq is None:
            await self._connection.execute(
                """
                INSERT INTO messages(
                    id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    metadata_json,
                    created_at,
                    occurred_at
                )
                SELECT
                    ?,
                    ?,
                    ?,
                    COALESCE(MAX(seq), 0) + 1,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                FROM messages
                WHERE conversation_id = ?
                """,
                (
                    resolved_message_id,
                    conversation_id,
                    role,
                    text,
                    token_count,
                    _encode_json(metadata),
                    timestamp,
                    resolved_occurred_at,
                    conversation_id,
                ),
            )
        else:
            await self._connection.execute(
                """
                INSERT INTO messages(
                    id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    metadata_json,
                    created_at,
                    occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_message_id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    _encode_json(metadata),
                    timestamp,
                    resolved_occurred_at,
                ),
            )
        if commit:
            await self._connection.commit()
        return await self._fetch_one(
            "SELECT * FROM messages WHERE id = ?",
            (resolved_message_id,),
        )

    async def get_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
            ORDER BY m.seq ASC
            LIMIT ?
            OFFSET ?
            """,
            (conversation_id, user_id, limit, offset),
        )

    async def get_message(self, message_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.id = ?
              AND c.user_id = ?
            """,
            (message_id, user_id),
        )

    async def search_messages(self, user_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT
                m.*,
                bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages AS m ON m._rowid = messages_fts.rowid
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND messages_fts MATCH ?
            ORDER BY rank ASC, m.seq ASC
            LIMIT ?
            """,
            (user_id, query, limit),
        )

class MemoryObjectRepository(BaseRepository):
    """Persistence operations for canonical memory objects."""

    async def get_memory_object(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )

    async def archive_memory_object(
        self,
        memory_id: str,
        user_id: str,
        *,
        commit: bool = True,
    ) -> bool:
        """Archive an active memory object and report whether a row was updated."""
        cursor = await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND status = ?
            """,
            (
                MemoryStatus.ARCHIVED.value,
                self._timestamp(),
                memory_id,
                user_id,
                MemoryStatus.ACTIVE.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return cursor.rowcount > 0

    async def create_memory_object(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        source_kind: MemorySourceKind,
        confidence: float,
        privacy_level: int,
        payload: dict[str, Any] | None = None,
        extraction_hash: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        stability: float = 0.5,
        vitality: float = 0.0,
        maya_score: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        valid_from: str | None = None,
        valid_to: str | None = None,
        memory_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_memory_id = memory_id or new_memory_id()
        timestamp = self._timestamp()
        parameters = (
            resolved_memory_id,
            user_id,
            workspace_id,
            conversation_id,
            assistant_mode_id,
            object_type.value,
            scope.value,
            canonical_text,
            extraction_hash,
            _encode_json(payload),
            source_kind.value,
            confidence,
            stability,
            vitality,
            maya_score,
            privacy_level,
            valid_from,
            valid_to,
            status.value,
            timestamp,
            timestamp,
        )
        try:
            await self._connection.execute(
                """
                INSERT INTO memory_objects(
                    id,
                    user_id,
                    workspace_id,
                    conversation_id,
                    assistant_mode_id,
                    object_type,
                    scope,
                    canonical_text,
                    extraction_hash,
                    payload_json,
                    source_kind,
                    confidence,
                    stability,
                    vitality,
                    maya_score,
                    privacy_level,
                    valid_from,
                    valid_to,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )
        except aiosqlite.IntegrityError:
            if extraction_hash is None:
                raise
            existing = await self.get_memory_object_by_extraction_hash(user_id, extraction_hash)
            if existing is None:
                raise
            return existing
        if commit:
            await self._connection.commit()
        return await self._fetch_one(
            "SELECT * FROM memory_objects WHERE id = ?",
            (resolved_memory_id,),
        )

    async def get_memory_object_by_extraction_hash(
        self,
        user_id: str,
        extraction_hash: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND extraction_hash = ?
            """,
            (user_id, extraction_hash),
        )

    async def refresh_memory_object_provenance(
        self,
        *,
        user_id: str,
        memory_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        source_message_ids: list[str],
        touch: bool = True,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")

        payload = existing.get("payload_json") or {}
        normalized_payload = dict(payload) if isinstance(payload, dict) else {}
        current_source_ids = [
            str(item).strip()
            for item in normalized_payload.get("source_message_ids", [])
            if str(item).strip()
        ]
        merged_source_ids = _stable_string_union(current_source_ids, source_message_ids)
        source_ids_changed = merged_source_ids != current_source_ids
        normalized_payload["source_message_ids"] = merged_source_ids
        if source_ids_changed:
            normalized_payload["confirmation_count"] = int(normalized_payload.get("confirmation_count", 0)) + 1

        identifiers_changed = any(
            existing.get(key) != value
            for key, value in {
                "assistant_mode_id": assistant_mode_id,
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
            }.items()
        )
        payload_changed = normalized_payload != payload
        if not (touch or identifiers_changed or payload_changed):
            return existing

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET assistant_mode_id = ?,
                workspace_id = ?,
                conversation_id = ?,
                payload_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                assistant_mode_id,
                workspace_id,
                conversation_id,
                _encode_json(normalized_payload),
                timestamp,
                memory_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(memory_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")
        return refreshed

    async def count_for_user_scopes(
        self,
        user_id: str,
        scopes: list[MemoryScope],
    ) -> int:
        return await self.count_for_context(
            user_id,
            scopes,
            workspace_id=None,
            conversation_id=None,
            assistant_mode_id=None,
        )

    async def count_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = [user_id]
        for scope in scopes:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append("(scope = 'global_user')")
            elif scope is MemoryScope.ASSISTANT_MODE and assistant_mode_id is not None:
                clauses.append("(scope = 'assistant_mode' AND assistant_mode_id = ?)")
                parameters.append(assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and workspace_id is not None:
                clauses.append("(scope = 'workspace' AND workspace_id = ?)")
                parameters.append(workspace_id)
            elif scope is MemoryScope.CONVERSATION and conversation_id is not None:
                clauses.append("(scope = 'conversation' AND conversation_id = ?)")
                parameters.append(conversation_id)
            elif scope is MemoryScope.EPHEMERAL_SESSION and conversation_id is not None:
                clauses.append("(scope = 'ephemeral_session' AND conversation_id = ?)")
                parameters.append(conversation_id)

        if not clauses:
            return 0

        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_objects
            WHERE user_id = ?
              AND status != 'deleted'
              AND (
                  {clauses}
              )
            """.format(clauses=" OR ".join(clauses)),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def get_state_snapshot(
        self,
        user_id: str,
        *,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        clauses, parameters = self._state_context_clauses(
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if not clauses:
            return {}

        rows = await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND object_type = ?
              AND status = ?
              AND ({clauses})
            ORDER BY updated_at DESC, id ASC
            """.format(clauses=" OR ".join(clauses)),
            (
                user_id,
                MemoryObjectType.STATE_SNAPSHOT.value,
                MemoryStatus.ACTIVE.value,
                *parameters,
            ),
        )

        resolved: dict[str, tuple[int, str, Any]] = {}
        for row in rows:
            scope_rank = self._state_scope_rank(row["scope"])
            updated_at = str(row["updated_at"])
            payload = row.get("payload_json") or {}
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                current = resolved.get(key)
                if current is None or (scope_rank, updated_at) > (current[0], current[1]):
                    resolved[key] = (scope_rank, updated_at, value)
        return {key: value for key, (_, _, value) in resolved.items()}

    async def has_memory_for_source_message(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        source_message_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> bool:
        clauses = [
            "mo.user_id = ?",
            "mo.object_type = ?",
            "mo.status != ?",
            "source_ids.value = ?",
        ]
        parameters: list[Any] = [
            user_id,
            object_type.value,
            MemoryStatus.DELETED.value,
            source_message_id,
        ]
        if assistant_mode_id is not None:
            clauses.append("mo.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if workspace_id is not None:
            clauses.append("mo.workspace_id = ?")
            parameters.append(workspace_id)
        if conversation_id is not None:
            clauses.append("mo.conversation_id = ?")
            parameters.append(conversation_id)
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM memory_objects AS mo
            JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
              ON 1 = 1
            WHERE {clauses}
            LIMIT 1
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return row is not None

    async def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        )

    async def list_for_source_message(
        self,
        *,
        user_id: str,
        source_message_id: str,
        assistant_mode_id: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = [
            "mo.user_id = ?",
            "source_ids.value = ?",
            "mo.status != ?",
        ]
        parameters: list[Any] = [user_id, source_message_id, MemoryStatus.DELETED.value]
        if assistant_mode_id is not None:
            clauses.append("mo.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if workspace_id is not None:
            clauses.append("mo.workspace_id = ?")
            parameters.append(workspace_id)
        if conversation_id is not None:
            clauses.append("mo.conversation_id = ?")
            parameters.append(conversation_id)
        return await self._fetch_all(
            """
            SELECT DISTINCT mo.*
            FROM memory_objects AS mo
            JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
              ON 1 = 1
            WHERE {clauses}
            ORDER BY mo.created_at ASC, mo.id ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def search_memory_objects(self, user_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT
                mo.*,
                bm25(memory_objects_fts) AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
            WHERE mo.user_id = ?
              AND memory_objects_fts MATCH ?
            ORDER BY rank ASC, mo.created_at DESC
            LIMIT ?
            """,
            (user_id, query, limit),
        )

    @staticmethod
    def _state_context_clauses(
        *,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses = ["(scope = 'global_user')"]
        parameters: list[Any] = []
        if assistant_mode_id is not None:
            clauses.append("(scope = 'assistant_mode' AND assistant_mode_id = ?)")
            parameters.append(assistant_mode_id)
            if workspace_id is not None:
                clauses.append("(scope = 'workspace' AND assistant_mode_id = ? AND workspace_id = ?)")
                parameters.extend([assistant_mode_id, workspace_id])
            if conversation_id is not None:
                clauses.append("(scope = 'conversation' AND assistant_mode_id = ? AND conversation_id = ?)")
                clauses.append(
                    "(scope = 'ephemeral_session' AND assistant_mode_id = ? AND conversation_id = ?)"
                )
                parameters.extend([assistant_mode_id, conversation_id, assistant_mode_id, conversation_id])
        return clauses, parameters

    @staticmethod
    def _state_scope_rank(scope_value: str) -> int:
        order = {
            MemoryScope.GLOBAL_USER.value: 0,
            MemoryScope.ASSISTANT_MODE.value: 1,
            MemoryScope.WORKSPACE.value: 2,
            MemoryScope.CONVERSATION.value: 3,
            MemoryScope.EPHEMERAL_SESSION.value: 4,
        }
        return order.get(scope_value, -1)
