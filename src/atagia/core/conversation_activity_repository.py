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
                user_persona_id,
                platform_id,
                character_id,
                incognito,
                remember_across_chats,
                remember_across_devices,
                effective_policy_hash,
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
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                workspace_id = excluded.workspace_id,
                assistant_mode_id = excluded.assistant_mode_id,
                user_persona_id = excluded.user_persona_id,
                platform_id = excluded.platform_id,
                character_id = excluded.character_id,
                incognito = excluded.incognito,
                remember_across_chats = excluded.remember_across_chats,
                remember_across_devices = excluded.remember_across_devices,
                effective_policy_hash = excluded.effective_policy_hash,
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
                stats.get("assistant_mode_id"),
                stats.get("user_persona_id"),
                stats.get("platform_id"),
                stats.get("character_id"),
                1 if stats.get("incognito") else 0,
                1 if stats.get("remember_across_chats", True) else 0,
                1 if stats.get("remember_across_devices", True) else 0,
                stats.get("effective_policy_hash"),
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
        return (
            await self.get_activity_stats(
                user_id=str(stats["user_id"]),
                conversation_id=str(stats["conversation_id"]),
            )
            or stats
        )

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
        row = await self._fetch_one(
            """
            SELECT
                cas.*,
                c.workspace_id AS _live_workspace_id,
                c.assistant_mode_id AS _live_assistant_mode_id,
                c.user_persona_id AS _live_user_persona_id,
                c.platform_id AS _live_platform_id,
                c.character_id AS _live_character_id,
                c.incognito AS _live_incognito,
                u.remember_across_chats AS _live_remember_across_chats,
                u.remember_across_devices AS _live_remember_across_devices
            FROM conversation_activity_stats AS cas
            JOIN conversations AS c
              ON c.id = cas.conversation_id
             AND c.user_id = cas.user_id
            JOIN users AS u ON u.id = c.user_id
            WHERE cas.user_id = ?
              AND c.user_id = ?
              AND cas.conversation_id = ?
            """,
            (user_id, user_id, conversation_id),
        )
        return self._with_live_scope(row)

    async def list_activity_stats(
        self,
        *,
        user_id: str,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        limit: int | None = None,
        as_of: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        del as_of
        clauses, parameters = self._activity_filter_parts(
            user_id=user_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            active_only=active_only,
        )
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            parameters.append(limit)
        rows = await self._fetch_all(
            """
            SELECT
                cas.*,
                c.workspace_id AS _live_workspace_id,
                c.assistant_mode_id AS _live_assistant_mode_id,
                c.user_persona_id AS _live_user_persona_id,
                c.platform_id AS _live_platform_id,
                c.character_id AS _live_character_id,
                c.incognito AS _live_incognito,
                u.remember_across_chats AS _live_remember_across_chats,
                u.remember_across_devices AS _live_remember_across_devices
            FROM conversation_activity_stats AS cas
            JOIN conversations AS c
              ON c.id = cas.conversation_id
             AND c.user_id = cas.user_id
            JOIN users AS u ON u.id = c.user_id
            WHERE {clauses}
            ORDER BY cas.likely_soon_score DESC, cas.main_thread_score DESC, cas.last_message_at DESC, cas.conversation_id ASC
            {limit_clause}
            """.format(
                clauses=" AND ".join(clauses),
                limit_clause=limit_clause,
            ),
            tuple(parameters),
        )
        return [self._with_live_scope(row) for row in rows if row is not None]

    async def list_activity_membership_ids(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        active_only: bool = False,
    ) -> list[str]:
        """Return the complete live membership behind an activity read."""

        clauses, parameters = self._activity_filter_parts(
            user_id=user_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            active_only=active_only,
        )
        if conversation_id is not None:
            clauses.append("cas.conversation_id = ?")
            parameters.append(conversation_id)
        rows = await self._fetch_all(
            """
            SELECT cas.conversation_id
            FROM conversation_activity_stats AS cas
            JOIN conversations AS c
              ON c.id = cas.conversation_id
             AND c.user_id = cas.user_id
            WHERE {clauses}
            ORDER BY cas.conversation_id ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )
        return [str(row["conversation_id"]) for row in rows]

    @staticmethod
    def _activity_filter_parts(
        *,
        user_id: str,
        workspace_id: str | None,
        assistant_mode_id: str | None,
        namespace_filter: bool,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        active_only: bool,
    ) -> tuple[list[str], list[Any]]:
        clauses = ["cas.user_id = ?", "c.user_id = ?"]
        parameters: list[Any] = [user_id, user_id]
        if active_only:
            clauses.extend(["c.status = 'active'", "c.temporary = 0"])
        if workspace_id is not None:
            clauses.append("c.workspace_id = ?")
            parameters.append(workspace_id)
        if assistant_mode_id is not None:
            clauses.append("c.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if namespace_filter:
            clauses.extend(
                [
                    "c.user_persona_id IS ?",
                    "c.platform_id = ?",
                    "c.character_id IS ?",
                    "c.incognito = ?",
                ]
            )
            parameters.extend(
                [user_persona_id, platform_id, character_id, 1 if incognito else 0]
            )
        return clauses, parameters

    @staticmethod
    def _with_live_scope(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        resolved = dict(row)
        for field in (
            "workspace_id",
            "assistant_mode_id",
            "user_persona_id",
            "platform_id",
            "character_id",
            "incognito",
            "remember_across_chats",
            "remember_across_devices",
        ):
            resolved[field] = resolved.pop(f"_live_{field}")
        return resolved

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
