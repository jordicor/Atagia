"""SQLite repository helpers for summary views."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.core.repositories import BaseRepository, _encode_json
from atagia.models.schemas_memory import SummaryViewKind


class _SummaryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str | None = None
    workspace_id: str | None = None
    source_message_start_seq: int = Field(ge=0)
    source_message_end_seq: int = Field(ge=0)
    summary_kind: SummaryViewKind
    summary_text: str = Field(min_length=1)
    source_object_ids_json: list[str] = Field(default_factory=list)
    maya_score: float
    model: str
    created_at: str

    @model_validator(mode="after")
    def validate_parent_reference(self) -> "_SummaryCreate":
        if self.conversation_id is None and self.workspace_id is None:
            raise ValueError("Summary views require conversation_id or workspace_id")
        return self


class SummaryRepository(BaseRepository):
    """Persistence helpers for non-canonical summary views."""

    async def create_summary(
        self,
        user_id: str,
        summary_data: dict[str, Any],
        *,
        commit: bool = True,
    ) -> str:
        payload = _SummaryCreate.model_validate(summary_data)
        await self._assert_summary_ownership(
            user_id=user_id,
            conversation_id=payload.conversation_id,
            workspace_id=payload.workspace_id,
        )
        await self._connection.execute(
            """
            INSERT INTO summary_views(
                id,
                conversation_id,
                workspace_id,
                source_message_start_seq,
                source_message_end_seq,
                summary_kind,
                summary_text,
                source_object_ids_json,
                maya_score,
                model,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.id,
                payload.conversation_id,
                payload.workspace_id,
                payload.source_message_start_seq,
                payload.source_message_end_seq,
                payload.summary_kind.value,
                payload.summary_text,
                _encode_json(payload.source_object_ids_json),
                payload.maya_score,
                payload.model,
                payload.created_at,
            ),
        )
        if commit:
            await self._connection.commit()
        return payload.id

    async def _assert_summary_ownership(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None,
    ) -> None:
        if conversation_id is not None:
            conversation = await self._fetch_one(
                """
                SELECT id
                FROM conversations
                WHERE id = ?
                  AND user_id = ?
                """,
                (conversation_id, user_id),
            )
            if conversation is None:
                raise ValueError("Summary conversation_id does not belong to user_id")
        if workspace_id is not None:
            workspace = await self._fetch_one(
                """
                SELECT id
                FROM workspaces
                WHERE id = ?
                  AND user_id = ?
                """,
                (workspace_id, user_id),
            )
            if workspace is None:
                raise ValueError("Summary workspace_id does not belong to user_id")

    async def get_summary(self, summary_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT sv.*
            FROM summary_views AS sv
            LEFT JOIN conversations AS c ON c.id = sv.conversation_id
            LEFT JOIN workspaces AS w ON w.id = sv.workspace_id
            WHERE sv.id = ?
              AND COALESCE(c.user_id, w.user_id) = ?
            """,
            (summary_id, user_id),
        )

    async def list_conversation_chunks(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT sv.*
            FROM summary_views AS sv
            JOIN conversations AS c ON c.id = sv.conversation_id
            WHERE c.user_id = ?
              AND sv.conversation_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.source_message_start_seq ASC, sv.id ASC
            LIMIT ?
            """,
            (user_id, conversation_id, SummaryViewKind.CONVERSATION_CHUNK.value, limit),
        )

    async def get_latest_conversation_chunk(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT sv.*
            FROM summary_views AS sv
            JOIN conversations AS c ON c.id = sv.conversation_id
            WHERE c.user_id = ?
              AND sv.conversation_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.source_message_end_seq DESC, sv.created_at DESC, sv.id DESC
            LIMIT 1
            """,
            (user_id, conversation_id, SummaryViewKind.CONVERSATION_CHUNK.value),
        )

    async def list_workspace_rollups(
        self,
        user_id: str,
        workspace_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT sv.*
            FROM summary_views AS sv
            JOIN workspaces AS w ON w.id = sv.workspace_id
            WHERE w.user_id = ?
              AND sv.workspace_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.created_at DESC, sv.id DESC
            LIMIT ?
            """,
            (user_id, workspace_id, SummaryViewKind.WORKSPACE_ROLLUP.value, limit),
        )

    async def get_latest_workspace_rollup(
        self,
        user_id: str,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT sv.*
            FROM summary_views AS sv
            JOIN workspaces AS w ON w.id = sv.workspace_id
            WHERE w.user_id = ?
              AND sv.workspace_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.created_at DESC, sv.id DESC
            LIMIT 1
            """,
            (user_id, workspace_id, SummaryViewKind.WORKSPACE_ROLLUP.value),
        )

    async def delete_old_rollups(
        self,
        user_id: str,
        workspace_id: str,
        *,
        keep_count: int = 3,
    ) -> int:
        rows = await self._fetch_all(
            """
            SELECT sv.id
            FROM summary_views AS sv
            JOIN workspaces AS w ON w.id = sv.workspace_id
            WHERE w.user_id = ?
              AND sv.workspace_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.created_at DESC, sv.id DESC
            LIMIT -1 OFFSET ?
            """,
            (user_id, workspace_id, SummaryViewKind.WORKSPACE_ROLLUP.value, keep_count),
        )
        if not rows:
            return 0
        summary_ids = [str(row["id"]) for row in rows]
        placeholders = ", ".join("?" for _ in summary_ids)
        await self._connection.execute(
            """
            DELETE FROM summary_views
            WHERE id IN ({placeholders})
            """.format(placeholders=placeholders),
            tuple(summary_ids),
        )
        await self._connection.commit()
        return len(summary_ids)
