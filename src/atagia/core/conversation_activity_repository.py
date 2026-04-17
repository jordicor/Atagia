"""Persistence helpers for derived conversation activity stats."""

from __future__ import annotations

from typing import Any
from atagia.core.repositories import BaseRepository, _encode_json


class ConversationActivityRepository(BaseRepository):
    """CRUD operations for materialized conversation activity stats."""

    async def upsert_activity_stats(
        self,
        stats: dict[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        await self._connection.execute(
            """
            INSERT INTO conversation_activity_stats(
                user_id,
                conversation_id,
                workspace_id,
                assistant_mode_id,
                timezone,
                first_message_at,
                last_message_at,
                last_user_message_at,
                message_count,
                user_message_count,
                assistant_message_count,
                retrieval_count,
                active_day_count,
                recent_1d_message_count,
                recent_7d_message_count,
                recent_30d_message_count,
                weekday_histogram_json,
                hour_histogram_json,
                hour_of_week_histogram_json,
                return_interval_histogram_json,
                avg_return_interval_minutes,
                median_return_interval_minutes,
                p90_return_interval_minutes,
                main_thread_score,
                likely_soon_score,
                return_habit_confidence,
                schedule_pattern_kind,
                activity_version,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                workspace_id = excluded.workspace_id,
                assistant_mode_id = excluded.assistant_mode_id,
                timezone = excluded.timezone,
                first_message_at = excluded.first_message_at,
                last_message_at = excluded.last_message_at,
                last_user_message_at = excluded.last_user_message_at,
                message_count = excluded.message_count,
                user_message_count = excluded.user_message_count,
                assistant_message_count = excluded.assistant_message_count,
                retrieval_count = excluded.retrieval_count,
                active_day_count = excluded.active_day_count,
                recent_1d_message_count = excluded.recent_1d_message_count,
                recent_7d_message_count = excluded.recent_7d_message_count,
                recent_30d_message_count = excluded.recent_30d_message_count,
                weekday_histogram_json = excluded.weekday_histogram_json,
                hour_histogram_json = excluded.hour_histogram_json,
                hour_of_week_histogram_json = excluded.hour_of_week_histogram_json,
                return_interval_histogram_json = excluded.return_interval_histogram_json,
                avg_return_interval_minutes = excluded.avg_return_interval_minutes,
                median_return_interval_minutes = excluded.median_return_interval_minutes,
                p90_return_interval_minutes = excluded.p90_return_interval_minutes,
                main_thread_score = excluded.main_thread_score,
                likely_soon_score = excluded.likely_soon_score,
                return_habit_confidence = excluded.return_habit_confidence,
                schedule_pattern_kind = excluded.schedule_pattern_kind,
                activity_version = excluded.activity_version,
                updated_at = excluded.updated_at
            """,
            (
                stats["user_id"],
                stats["conversation_id"],
                stats.get("workspace_id"),
                stats["assistant_mode_id"],
                stats.get("timezone", "UTC"),
                stats.get("first_message_at"),
                stats.get("last_message_at"),
                stats.get("last_user_message_at"),
                int(stats.get("message_count", 0)),
                int(stats.get("user_message_count", 0)),
                int(stats.get("assistant_message_count", 0)),
                int(stats.get("retrieval_count", 0)),
                int(stats.get("active_day_count", 0)),
                int(stats.get("recent_1d_message_count", 0)),
                int(stats.get("recent_7d_message_count", 0)),
                int(stats.get("recent_30d_message_count", 0)),
                _encode_json(stats.get("weekday_histogram_json", [])),
                _encode_json(stats.get("hour_histogram_json", [])),
                _encode_json(stats.get("hour_of_week_histogram_json", [])),
                _encode_json(stats.get("return_interval_histogram_json", [])),
                stats.get("avg_return_interval_minutes"),
                stats.get("median_return_interval_minutes"),
                stats.get("p90_return_interval_minutes"),
                float(stats.get("main_thread_score", 0.0)),
                float(stats.get("likely_soon_score", 0.0)),
                float(stats.get("return_habit_confidence", 0.0)),
                stats.get("schedule_pattern_kind", "inactive"),
                int(stats.get("activity_version", 1)),
                stats["updated_at"],
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_activity_stats(
            user_id=str(stats["user_id"]),
            conversation_id=str(stats["conversation_id"]),
        ) or stats

    async def upsert_activity_stats_bulk(
        self,
        rows: list[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> int:
        if not rows:
            return 0
        for row in rows:
            await self.upsert_activity_stats(row, commit=False)
        if commit:
            await self._connection.commit()
        return len(rows)

    async def get_activity_stats(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversation_activity_stats
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )

    async def list_activity_stats(
        self,
        *,
        user_id: str,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        limit: int | None = None,
        as_of: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        del as_of
        clauses = ["cas.user_id = ?"]
        parameters: list[Any] = [user_id]
        if active_only:
            clauses.append("c.status = 'active'")
        if workspace_id is not None:
            clauses.append("cas.workspace_id = ?")
            parameters.append(workspace_id)
        if assistant_mode_id is not None:
            clauses.append("cas.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT cas.*
            FROM conversation_activity_stats AS cas
            JOIN conversations AS c ON c.id = cas.conversation_id
            WHERE {clauses}
            ORDER BY cas.likely_soon_score DESC, cas.main_thread_score DESC, cas.last_message_at DESC, cas.conversation_id ASC
            {limit_clause}
            """.format(
                clauses=" AND ".join(clauses),
                limit_clause=limit_clause,
            ),
            tuple(parameters),
        )

    async def delete_activity_stats_for_user(self, user_id: str) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM conversation_activity_stats
            WHERE user_id = ?
            """,
            (user_id,),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0)
