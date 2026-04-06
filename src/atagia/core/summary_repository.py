"""SQLite repository helpers for summary views."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.core.repositories import BaseRepository, _encode_json, summary_mirror_id
from atagia.models.schemas_memory import SummaryViewKind

_SUMMARY_KIND_LEVELS = {
    SummaryViewKind.CONVERSATION_CHUNK: 0,
    SummaryViewKind.WORKSPACE_ROLLUP: 0,
    SummaryViewKind.CONTEXT_VIEW: 0,
    SummaryViewKind.EPISODE: 1,
    SummaryViewKind.THEMATIC_PROFILE: 2,
}


class _SummaryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str | None = None
    workspace_id: str | None = None
    source_message_start_seq: int | None = Field(default=None, ge=0)
    source_message_end_seq: int | None = Field(default=None, ge=0)
    summary_kind: SummaryViewKind
    hierarchy_level: int = Field(default=0, ge=0)
    summary_text: str = Field(min_length=1)
    source_object_ids_json: list[str] = Field(default_factory=list)
    maya_score: float
    model: str
    created_at: str

    @model_validator(mode="after")
    def validate_parent_reference(self) -> "_SummaryCreate":
        if (self.source_message_start_seq is None) != (self.source_message_end_seq is None):
            raise ValueError("Summary view message bounds must both be set or both be null")
        expected_level = _SUMMARY_KIND_LEVELS[self.summary_kind]
        if self.hierarchy_level != expected_level:
            raise ValueError(
                f"summary_kind {self.summary_kind.value} requires hierarchy_level={expected_level}"
            )
        if (
            self.summary_kind in {SummaryViewKind.CONVERSATION_CHUNK, SummaryViewKind.WORKSPACE_ROLLUP, SummaryViewKind.CONTEXT_VIEW}
            and self.conversation_id is None
            and self.workspace_id is None
        ):
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
                user_id,
                conversation_id,
                workspace_id,
                source_message_start_seq,
                source_message_end_seq,
                summary_kind,
                summary_text,
                source_object_ids_json,
                maya_score,
                model,
                hierarchy_level,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.id,
                user_id,
                payload.conversation_id,
                payload.workspace_id,
                payload.source_message_start_seq,
                payload.source_message_end_seq,
                payload.summary_kind.value,
                payload.summary_text,
                _encode_json(payload.source_object_ids_json),
                payload.maya_score,
                payload.model,
                payload.hierarchy_level,
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
            SELECT *
            FROM summary_views
            WHERE id = ?
              AND user_id = ?
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
            WHERE sv.user_id = ?
              AND sv.conversation_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.source_message_start_seq ASC, sv.id ASC
            LIMIT ?
            """,
            (user_id, conversation_id, SummaryViewKind.CONVERSATION_CHUNK.value, limit),
        )

    async def list_all_conversation_chunks(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT sv.*
            FROM summary_views AS sv
            WHERE sv.user_id = ?
              AND sv.conversation_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.source_message_start_seq ASC, sv.id ASC
            """,
            (user_id, conversation_id, SummaryViewKind.CONVERSATION_CHUNK.value),
        )

    async def list_all_user_conversation_chunks(self, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT sv.*
            FROM summary_views AS sv
            WHERE sv.user_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.created_at ASC, sv.id ASC
            """,
            (user_id, SummaryViewKind.CONVERSATION_CHUNK.value),
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
            WHERE sv.user_id = ?
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
            WHERE sv.user_id = ?
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
            WHERE sv.user_id = ?
              AND sv.workspace_id = ?
              AND sv.summary_kind = ?
            ORDER BY sv.created_at DESC, sv.id DESC
            LIMIT 1
            """,
            (user_id, workspace_id, SummaryViewKind.WORKSPACE_ROLLUP.value),
        )

    async def list_summaries_by_kind(
        self,
        user_id: str,
        summary_kind: SummaryViewKind,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT *
            FROM summary_views
            WHERE user_id = ?
              AND summary_kind = ?
            ORDER BY created_at DESC, id DESC
        """
        parameters: tuple[Any, ...]
        if limit is None:
            parameters = (user_id, summary_kind.value)
        else:
            query += "\nLIMIT ?"
            parameters = (user_id, summary_kind.value, limit)
        return await self._fetch_all(query, parameters)

    async def delete_summaries(
        self,
        user_id: str,
        summary_ids: list[str],
        *,
        commit: bool = True,
    ) -> int:
        if not summary_ids:
            return 0
        placeholders = ", ".join("?" for _ in summary_ids)
        mirror_ids = [summary_mirror_id(summary_id) for summary_id in summary_ids]
        mirror_placeholders = ", ".join("?" for _ in mirror_ids)
        started_transaction = False
        try:
            if commit:
                await self.begin()
                started_transaction = True
            await self._connection.execute(
                """
                DELETE FROM memory_objects
                WHERE user_id = ?
                  AND object_type = 'summary_view'
                  AND id IN ({mirror_placeholders})
                """.format(mirror_placeholders=mirror_placeholders),
                (user_id, *mirror_ids),
            )
            await self._connection.execute(
                """
                DELETE FROM summary_views
                WHERE user_id = ?
                  AND id IN ({placeholders})
                """.format(placeholders=placeholders),
                (user_id, *summary_ids),
            )
            if commit:
                await self.commit()
        except Exception:
            if started_transaction:
                await self.rollback()
            raise
        return len(summary_ids)

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
            WHERE sv.user_id = ?
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
        return await self.delete_summaries(user_id, summary_ids, commit=True)
