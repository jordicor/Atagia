"""SQLite repositories for the Step 3 persistence layer."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.ids import generate_prefixed_id, new_memory_id
from atagia.core.storage_backend import StorageBackend
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    SummaryViewKind,
)

RETRIEVAL_ELIGIBLE_MEMORY_STATUSES: tuple[MemoryStatus, ...] = (MemoryStatus.ACTIVE,)


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


def _encode_language_codes(language_codes: list[str] | None) -> str | None:
    if not language_codes:
        return None
    normalized = sorted(
        {
            str(code).strip().lower()
            for code in language_codes
            if str(code).strip()
        }
    )
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False)


_HEAVY_MESSAGE_TOKEN_THRESHOLD = 512
_HEAVY_MESSAGE_CHAR_THRESHOLD = 4096


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_optional_text(value: Any, *, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if max_length is not None:
        normalized = normalized[:max_length].strip()
    return normalized or None


def _is_mechanically_heavy(text: str, token_count: int | None) -> bool:
    if token_count is not None and token_count >= _HEAVY_MESSAGE_TOKEN_THRESHOLD:
        return True
    return len(text) >= _HEAVY_MESSAGE_CHAR_THRESHOLD


def _build_context_placeholder(
    *,
    message_id: str,
    seq: int | None,
    role: str,
    content_kind: str,
    policy_reason: str,
) -> str:
    seq_part = str(seq) if seq is not None else "?"
    return (
        f"[Skipped message | id={message_id} seq={seq_part} role={role} "
        f"kind={content_kind} policy={policy_reason} ref={message_id}]"
    )


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


def summary_mirror_id(summary_view_id: str) -> str:
    """Return the deterministic mirror memory ID for a summary view."""
    return f"sum_mem_{summary_view_id}"


def _status_filter_clause(
    column_name: str,
    statuses: tuple[MemoryStatus, ...] | None,
) -> tuple[str, list[Any]]:
    if statuses is None:
        return "", []
    placeholders = ", ".join("?" for _ in statuses)
    return f"{column_name} IN ({placeholders})", [status.value for status in statuses]


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
        assistant_mode_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            parameters.append(workspace_id)
        if assistant_mode_id is not None:
            clauses.append("assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)

        return await self._fetch_all(
            """
            SELECT *
            FROM conversations
            WHERE {where_clause}
            ORDER BY updated_at DESC, id ASC
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
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

    @staticmethod
    def _derive_message_policy(
        *,
        message_id: str,
        role: str,
        seq: int | None,
        text: str,
        token_count: int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        explicit_content_kind = _normalize_optional_text(metadata.get("content_kind"), max_length=64)
        explicit_include_raw = "include_raw" in metadata
        explicit_policy_reason = _normalize_optional_text(metadata.get("policy_reason"), max_length=128)
        explicit_context_placeholder = _normalize_optional_text(
            metadata.get("context_placeholder"),
            max_length=300,
        )

        mechanical_heavy = _is_mechanically_heavy(text, token_count)
        heavy_content = _coerce_bool(metadata.get("heavy_content")) or mechanical_heavy
        artifact_backed = _coerce_bool(metadata.get("artifact_backed"))
        verbatim_required = _coerce_bool(metadata.get("verbatim_required"))
        skip_by_default = _coerce_bool(metadata.get("skip_by_default"))
        include_raw = _coerce_bool(metadata.get("include_raw"), True)
        requires_explicit_request = _coerce_bool(metadata.get("requires_explicit_request"))

        if heavy_content or artifact_backed or verbatim_required or skip_by_default:
            if not explicit_include_raw:
                include_raw = False
        if not include_raw:
            skip_by_default = True
        if heavy_content or artifact_backed or verbatim_required:
            skip_by_default = True if not explicit_include_raw else skip_by_default

        requires_explicit_request = (
            requires_explicit_request
            or skip_by_default
            or heavy_content
            or artifact_backed
            or verbatim_required
        )
        if explicit_include_raw and include_raw:
            requires_explicit_request = _coerce_bool(metadata.get("requires_explicit_request"))

        if explicit_content_kind is not None:
            content_kind = explicit_content_kind
        elif artifact_backed:
            content_kind = "artifact"
        elif heavy_content or skip_by_default:
            content_kind = "attachment"
        else:
            content_kind = "text"

        policy_reason = explicit_policy_reason
        if policy_reason is None:
            if artifact_backed:
                policy_reason = "artifact_backed"
            elif verbatim_required:
                policy_reason = "verbatim_required"
            elif heavy_content and mechanical_heavy:
                policy_reason = "mechanical_size_threshold"
            elif skip_by_default and not include_raw:
                policy_reason = "skip_by_default"
            elif not include_raw:
                policy_reason = "include_raw_false"

        context_placeholder = explicit_context_placeholder
        if context_placeholder is None and skip_by_default and seq is not None:
            context_placeholder = _build_context_placeholder(
                message_id=message_id,
                seq=seq,
                role=role,
                content_kind=content_kind,
                policy_reason=policy_reason or "skip_by_default",
            )

        return {
            "content_kind": content_kind,
            "include_raw": int(include_raw),
            "skip_by_default": int(skip_by_default),
            "heavy_content": int(heavy_content),
            "artifact_backed": int(artifact_backed),
            "verbatim_required": int(verbatim_required),
            "requires_explicit_request": int(requires_explicit_request),
            "context_placeholder": context_placeholder,
            "policy_reason": policy_reason,
        }

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
        resolved_occurred_at = normalize_optional_timestamp(occurred_at)
        message_policy = self._derive_message_policy(
            message_id=resolved_message_id,
            role=role,
            seq=seq,
            text=text,
            token_count=token_count,
            metadata=metadata,
        )
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
                    content_kind,
                    include_raw,
                    skip_by_default,
                    heavy_content,
                    artifact_backed,
                    verbatim_required,
                    requires_explicit_request,
                    context_placeholder,
                    policy_reason,
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
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
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
                    message_policy["content_kind"],
                    message_policy["include_raw"],
                    message_policy["skip_by_default"],
                    message_policy["heavy_content"],
                    message_policy["artifact_backed"],
                    message_policy["verbatim_required"],
                    message_policy["requires_explicit_request"],
                    message_policy["context_placeholder"],
                    message_policy["policy_reason"],
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
                    content_kind,
                    include_raw,
                    skip_by_default,
                    heavy_content,
                    artifact_backed,
                    verbatim_required,
                    requires_explicit_request,
                    context_placeholder,
                    policy_reason,
                    created_at,
                    occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_message_id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    _encode_json(metadata),
                    message_policy["content_kind"],
                    message_policy["include_raw"],
                    message_policy["skip_by_default"],
                    message_policy["heavy_content"],
                    message_policy["artifact_backed"],
                    message_policy["verbatim_required"],
                    message_policy["requires_explicit_request"],
                    message_policy["context_placeholder"],
                    message_policy["policy_reason"],
                    timestamp,
                    resolved_occurred_at,
                ),
            )
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            "SELECT * FROM messages WHERE id = ?",
            (resolved_message_id,),
        )
        if created is None:
            raise RuntimeError(f"Failed to create message {resolved_message_id}")
        if seq is None and created.get("skip_by_default"):
            generated_placeholder = _build_context_placeholder(
                message_id=resolved_message_id,
                seq=int(created["seq"]),
                role=role,
                content_kind=str(created.get("content_kind") or "text"),
                policy_reason=str(created.get("policy_reason") or "skip_by_default"),
            )
            if created.get("context_placeholder") != generated_placeholder:
                await self._connection.execute(
                    """
                    UPDATE messages
                    SET context_placeholder = ?
                    WHERE id = ?
                    """,
                    (generated_placeholder, resolved_message_id),
                )
                if commit:
                    await self._connection.commit()
                created["context_placeholder"] = generated_placeholder
        return created

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

    async def get_recent_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM (
                SELECT m.*
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.conversation_id = ?
                  AND c.user_id = ?
                ORDER BY m.seq DESC
                LIMIT ?
            ) AS recent_messages
            ORDER BY seq ASC
            """,
            (conversation_id, user_id, limit),
        )

    async def list_messages_for_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id),
        )

    async def get_messages_in_seq_range(
        self,
        conversation_id: str,
        user_id: str,
        start_seq: int,
        end_seq: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq BETWEEN ? AND ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq, end_seq),
        )

    async def get_messages_from_seq(
        self,
        conversation_id: str,
        user_id: str,
        start_seq: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq >= ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq),
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

    async def search_messages_with_privacy(
        self,
        *,
        user_id: str,
        query: str,
        privacy_ceiling: int,
        limit: int,
        allow_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Wave 1 batch 2 (1-C): privacy-filtered FTS search on messages.

        Security invariants enforced at the SQL layer, BEFORE ranking:
        - ``user_id`` filter: messages belong to the caller's conversations.
        - Mode privacy ceiling: conversations whose owning assistant_mode
          has a privacy_ceiling strictly greater than the current
          retrieval ceiling are excluded entirely. Therapy-mode messages
          can never surface inside a general-mode retrieval even though
          they share a user. The active conversation may opt into a
          narrow same-context bypass via ``allow_conversation_id``.
        - Consent gating: messages that are the verbatim source of a
          memory object still waiting for user confirmation are excluded
          so unconfirmed sensitive content never leaks through the raw
          channel.
        """
        if limit <= 0:
            return []
        return await self._fetch_all(
            """
            SELECT
                m.*,
                c.assistant_mode_id AS conversation_assistant_mode_id,
                am.privacy_ceiling AS mode_privacy_ceiling,
                bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages AS m ON m._rowid = messages_fts.rowid
            JOIN conversations AS c ON c.id = m.conversation_id
            JOIN assistant_modes AS am ON am.id = c.assistant_mode_id
            WHERE c.user_id = ?
              AND (am.privacy_ceiling <= ? OR (? IS NOT NULL AND c.id = ?))
              AND c.status = 'active'
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_objects AS pending
                  JOIN json_each(
                      json_extract(pending.payload_json, '$.source_message_ids')
                  ) AS pending_src ON pending_src.value = m.id
                  WHERE pending.user_id = ?
                    AND pending.status = 'pending_user_confirmation'
              )
              AND messages_fts MATCH ?
            ORDER BY rank ASC, m.seq ASC
            LIMIT ?
            """,
            (
                user_id,
                privacy_ceiling,
                allow_conversation_id,
                allow_conversation_id,
                user_id,
                query,
                limit,
            ),
        )

    async def fetch_message_window(
        self,
        *,
        conversation_id: str,
        user_id: str,
        center_seq: int,
        window_size: int,
    ) -> list[dict[str, Any]]:
        """Fetch up to ``window_size`` contiguous messages around ``center_seq``.

        The window is built by ``seq`` arithmetic only: ``start_seq`` and
        ``end_seq`` are derived from ``center_seq`` and ``window_size``
        without requiring the center row to exist. Messages are returned
        in ascending ``seq`` order so the list reads as a contiguous
        transcript. ``user_id`` is enforced via the owning conversation
        to preserve the per-user isolation invariant.
        """
        if window_size <= 0:
            return []
        half = max(0, (window_size - 1) // 2)
        start_seq = max(0, center_seq - half)
        end_seq = center_seq + (window_size - 1 - half)
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq BETWEEN ? AND ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq, end_seq),
        )

    async def sum_text_length_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> int:
        """Return the total message text length eligible for small-corpus mode.

        Messages inherit their scope from the owning conversation. The widest
        allowed scope determines how many conversations contribute: a
        `global_user` allowance sweeps every user conversation, while
        `assistant_mode`, `workspace`, `conversation`, and `ephemeral_session`
        progressively narrow the set.
        """
        clauses, parameters = self._message_scope_clauses(
            scopes,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return 0

        cursor = await self._connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(m.text)), 0) AS total_length
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND (
                  {clauses}
              )
            """.format(clauses=" OR ".join(clauses)),
            tuple([user_id, *parameters]),
        )
        row = await cursor.fetchone()
        return int(row["total_length"])

    async def list_eligible_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> list[dict[str, Any]]:
        """Return every message eligible for small-corpus composition.

        Messages are ordered by seq within each conversation, and across
        conversations by creation time, so callers get a deterministic
        transcript-shaped feed.
        """
        clauses, parameters = self._message_scope_clauses(
            scopes,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return []

        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND (
                  {clauses}
              )
            ORDER BY c.created_at ASC, m.conversation_id ASC, m.seq ASC
            """.format(clauses=" OR ".join(clauses)),
            tuple([user_id, *parameters]),
        )

    @staticmethod
    def _message_scope_clauses(
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scopes:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append("1 = 1")
            elif scope is MemoryScope.ASSISTANT_MODE:
                clauses.append("c.assistant_mode_id = ?")
                parameters.append(assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and workspace_id is not None:
                clauses.append("(c.assistant_mode_id = ? AND c.workspace_id = ?)")
                parameters.extend([assistant_mode_id, workspace_id])
            elif scope is MemoryScope.CONVERSATION:
                clauses.append("(c.assistant_mode_id = ? AND c.id = ?)")
                parameters.extend([assistant_mode_id, conversation_id])
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append("(c.assistant_mode_id = ? AND c.id = ?)")
                parameters.extend([assistant_mode_id, conversation_id])
        return clauses, parameters


class MemoryObjectRepository(BaseRepository):
    """Persistence operations for canonical memory objects."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(connection, clock)
        self._settings = settings or Settings.from_env()

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

    async def list_memory_objects_by_ids(
        self,
        user_id: str,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ", ".join("?" for _ in memory_ids)
        return await self._fetch_all(
            f"""
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, *memory_ids),
        )

    async def update_memory_object_status(
        self,
        *,
        memory_id: str,
        user_id: str,
        status: MemoryStatus,
        payload_updates: dict[str, Any] | None = None,
        expected_current_status: MemoryStatus | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            return None
        if expected_current_status is not None and existing["status"] != expected_current_status.value:
            return None

        payload = existing.get("payload_json")
        normalized_payload = dict(payload) if isinstance(payload, dict) else {}
        if payload_updates:
            normalized_payload.update(payload_updates)
        timestamp = self._timestamp()
        parameters: list[Any] = [
            status.value,
            _encode_json(normalized_payload),
            timestamp,
            memory_id,
            user_id,
        ]
        status_clause = ""
        if expected_current_status is not None:
            status_clause = " AND status = ?"
            parameters.append(expected_current_status.value)
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?,
                payload_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            {status_clause}
            """.format(status_clause=status_clause),
            tuple(parameters),
        )
        if commit:
            await self._connection.commit()
        return await self.get_memory_object(memory_id, user_id)

    async def upsert_summary_mirror(
        self,
        *,
        user_id: str,
        summary_view_id: str,
        summary_kind: SummaryViewKind,
        hierarchy_level: int,
        summary_text: str,
        source_object_ids: list[str],
        created_at: str,
        updated_at: str | None = None,
        index_text: str | None = None,
        scope: MemoryScope = MemoryScope.GLOBAL_USER,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        confidence: float = 0.72,
        stability: float = 0.82,
        vitality: float = 0.15,
        maya_score: float = 1.5,
        privacy_level: int = 0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        mirror_id = summary_mirror_id(summary_view_id)
        existing = await self.get_memory_object(mirror_id, user_id)
        normalized_source_ids = [
            str(item).strip()
            for item in source_object_ids
            if str(item).strip()
        ]
        normalized_payload = {
            **(payload or {}),
            "summary_view_id": summary_view_id,
            "summary_kind": summary_kind.value,
            "hierarchy_level": hierarchy_level,
            "source_object_ids": normalized_source_ids,
        }

        if existing is None:
            return await self.create_memory_object(
                user_id=user_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
                object_type=MemoryObjectType.SUMMARY_VIEW,
                scope=scope,
                canonical_text=summary_text,
                index_text=index_text,
                payload=normalized_payload,
                source_kind=MemorySourceKind.SUMMARIZED,
                confidence=confidence,
                stability=stability,
                vitality=vitality,
                maya_score=maya_score,
                privacy_level=privacy_level,
                memory_category=MemoryCategory.UNKNOWN,
                preserve_verbatim=False,
                status=status,
                memory_id=mirror_id,
                commit=commit,
            )

        await self._connection.execute(
            """
            UPDATE memory_objects
            SET workspace_id = ?,
                conversation_id = ?,
                assistant_mode_id = ?,
                scope = ?,
                canonical_text = ?,
                index_text = ?,
                payload_json = ?,
                source_kind = ?,
                confidence = ?,
                stability = ?,
                vitality = ?,
                maya_score = ?,
                privacy_level = ?,
                memory_category = ?,
                preserve_verbatim = ?,
                status = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                workspace_id,
                conversation_id,
                assistant_mode_id,
                scope.value,
                summary_text,
                index_text,
                _encode_json(normalized_payload),
                MemorySourceKind.SUMMARIZED.value,
                confidence,
                stability,
                vitality,
                maya_score,
                privacy_level,
                MemoryCategory.UNKNOWN.value,
                0,
                status.value,
                self._timestamp(),
                mirror_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(mirror_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown summary mirror memory_id: {mirror_id}")
        return refreshed

    async def create_memory_object(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
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
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        created, _was_created = await self._create_memory_object_impl(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            canonical_text=canonical_text,
            index_text=index_text,
            source_kind=source_kind,
            confidence=confidence,
            privacy_level=privacy_level,
            payload=payload,
            extraction_hash=extraction_hash,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            stability=stability,
            vitality=vitality,
            maya_score=maya_score,
            status=status,
            memory_category=memory_category,
            preserve_verbatim=preserve_verbatim,
            valid_from=valid_from,
            valid_to=valid_to,
            temporal_type=temporal_type,
            language_codes=language_codes,
            memory_id=memory_id,
            commit=commit,
        )
        return created

    async def create_memory_object_with_flag(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
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
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        return await self._create_memory_object_impl(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            canonical_text=canonical_text,
            index_text=index_text,
            source_kind=source_kind,
            confidence=confidence,
            privacy_level=privacy_level,
            payload=payload,
            extraction_hash=extraction_hash,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            stability=stability,
            vitality=vitality,
            maya_score=maya_score,
            status=status,
            memory_category=memory_category,
            preserve_verbatim=preserve_verbatim,
            valid_from=valid_from,
            valid_to=valid_to,
            temporal_type=temporal_type,
            language_codes=language_codes,
            memory_id=memory_id,
            commit=commit,
        )

    async def _create_memory_object_impl(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
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
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        resolved_memory_id = memory_id or new_memory_id()
        timestamp = self._timestamp()
        language_codes_json = _encode_language_codes(language_codes)
        parameters = (
            resolved_memory_id,
            user_id,
            workspace_id,
            conversation_id,
            assistant_mode_id,
            object_type.value,
            scope.value,
            canonical_text,
            index_text,
            extraction_hash,
            _encode_json(payload),
            source_kind.value,
            confidence,
            stability,
            vitality,
            maya_score,
            privacy_level,
            memory_category.value,
            int(preserve_verbatim),
            valid_from,
            valid_to,
            temporal_type,
            language_codes_json,
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
                    index_text,
                    extraction_hash,
                    payload_json,
                    source_kind,
                    confidence,
                    stability,
                    vitality,
                    maya_score,
                    privacy_level,
                    memory_category,
                    preserve_verbatim,
                    valid_from,
                    valid_to,
                    temporal_type,
                    language_codes_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )
        except aiosqlite.IntegrityError:
            if extraction_hash is None:
                raise
            existing = await self.get_memory_object_by_extraction_hash(user_id, extraction_hash)
            if existing is None:
                raise
            return existing, False
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            "SELECT * FROM memory_objects WHERE id = ?",
            (resolved_memory_id,),
        )
        if created is None:
            raise RuntimeError(f"Failed to create memory object {resolved_memory_id}")
        return created, True

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
        clauses, parameters = self._context_scope_clauses(
            scopes,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return 0

        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
            """.format(
                clauses=" OR ".join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                ]
            ),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def sum_canonical_text_length_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        privacy_ceiling: int,
    ) -> int:
        """Return the total canonical_text length for eligible active memories.

        The filter mirrors retrieval eligibility: user_id, allowed scopes,
        active status, and privacy ceiling. Used to detect small corpora
        where the full memory set fits inside the context budget.
        """
        clauses, parameters = self._context_scope_clauses(
            scopes,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return 0

        cursor = await self._connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(canonical_text)), 0) AS total_length
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
              AND privacy_level <= ?
            """.format(
                clauses=" OR ".join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                    privacy_ceiling,
                ]
            ),
        )
        row = await cursor.fetchone()
        return int(row["total_length"])

    async def list_eligible_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        privacy_ceiling: int,
    ) -> list[dict[str, Any]]:
        """Return eligible active memory objects for small-corpus composition."""
        clauses, parameters = self._context_scope_clauses(
            scopes,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return []

        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
              AND privacy_level <= ?
            ORDER BY updated_at DESC, id ASC
            """.format(
                clauses=" OR ".join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                    privacy_ceiling,
                ]
            ),
        )

    @staticmethod
    def _context_scope_clauses(
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
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
        return clauses, parameters

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
        now = self._clock.now()
        for row in rows:
            if self._is_expired_ephemeral_state_snapshot(row, now):
                continue
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
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> bool:
        clauses = [
            "mo.user_id = ?",
            "mo.object_type = ?",
            "source_ids.value = ?",
        ]
        parameters: list[Any] = [
            user_id,
            object_type.value,
            source_message_id,
        ]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
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

    async def list_for_user(
        self,
        user_id: str,
        *,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        status_clause, status_parameters = _status_filter_clause("status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE {clauses}
            ORDER BY created_at ASC, id ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def list_for_source_message(
        self,
        *,
        user_id: str,
        source_message_id: str,
        assistant_mode_id: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> list[dict[str, Any]]:
        clauses = [
            "mo.user_id = ?",
            "source_ids.value = ?",
        ]
        parameters: list[Any] = [user_id, source_message_id]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
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

    async def search_memory_objects(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> list[dict[str, Any]]:
        clauses = ["mo.user_id = ?", "memory_objects_fts MATCH ?"]
        parameters: list[Any] = [user_id, query]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT
                mo.*,
                bm25(memory_objects_fts) AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
            WHERE {clauses}
            ORDER BY rank ASC, mo.created_at DESC
            LIMIT ?
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
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

    def _is_expired_ephemeral_state_snapshot(
        self,
        row: dict[str, Any],
        reference: datetime,
    ) -> bool:
        if str(row.get("temporal_type", "unknown")) != "ephemeral":
            return False
        valid_from = self._parse_temporal_datetime(row.get("valid_from"), reference)
        if valid_from is None:
            return True
        valid_to = self._parse_temporal_datetime(row.get("valid_to"), reference)
        effective_end = valid_to or (
            valid_from + timedelta(hours=self._settings.ephemeral_scoring_hours)
        )
        return effective_end < reference

    @staticmethod
    def _parse_temporal_datetime(value: Any, reference: datetime) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None and reference.tzinfo is not None:
            parsed = parsed.replace(tzinfo=reference.tzinfo)
        return parsed
