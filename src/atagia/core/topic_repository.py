"""SQLite persistence helpers for live conversation topic working sets."""

from __future__ import annotations

from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _encode_json
from atagia.models.schemas_memory import IntimacyBoundary

_TOPIC_SOURCE_REF_LIMIT = 8
_DEFAULT_REFRESH_MESSAGE_THRESHOLD = 4
_DEFAULT_STALE_MESSAGE_THRESHOLD = 10
_DEFAULT_REFRESH_TOKEN_THRESHOLD = 2000
_DEFAULT_STALE_TOKEN_THRESHOLD = 5000


class TopicRepository(BaseRepository):
    """Persistence operations for conversation-level Topic Working Sets."""

    async def create_topic(
        self,
        *,
        user_id: str,
        conversation_id: str,
        topic_id: str | None = None,
        parent_topic_id: str | None = None,
        status: str = "active",
        title: str,
        summary: str = "",
        active_goal: str | None = None,
        open_questions: list[str] | None = None,
        decisions: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        source_message_start_seq: int | None = None,
        source_message_end_seq: int | None = None,
        last_touched_seq: int | None = None,
        last_touched_at: str | None = None,
        confidence: float = 0.5,
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_topic_id = topic_id or generate_prefixed_id("tpc")
        timestamp = self._timestamp()
        touched_at = last_touched_at or timestamp
        await self._connection.execute(
            """
            INSERT INTO conversation_topics(
                id,
                user_id,
                conversation_id,
                parent_topic_id,
                status,
                title,
                summary,
                active_goal,
                open_questions_json,
                decisions_json,
                artifact_ids_json,
                source_message_start_seq,
                source_message_end_seq,
                last_touched_seq,
                last_touched_at,
                confidence,
                privacy_level,
                intimacy_boundary,
                intimacy_boundary_confidence,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_topic_id,
                user_id,
                conversation_id,
                parent_topic_id,
                status,
                title,
                summary,
                active_goal,
                _encode_json(open_questions or []),
                _encode_json(decisions or []),
                _encode_json(artifact_ids or []),
                source_message_start_seq,
                source_message_end_seq,
                last_touched_seq,
                touched_at,
                confidence,
                privacy_level,
                intimacy_boundary.value,
                float(intimacy_boundary_confidence),
                timestamp,
                timestamp,
            ),
        )
        await self.create_event(
            user_id=user_id,
            conversation_id=conversation_id,
            topic_id=resolved_topic_id,
            event_type="created",
            payload={"title": title, "status": status},
            commit=False,
        )
        if commit:
            await self._connection.commit()
        created = await self.get_topic(resolved_topic_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create conversation topic {resolved_topic_id}")
        return created

    async def get_topic(self, topic_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversation_topics
            WHERE id = ?
              AND user_id = ?
            """,
            (topic_id, user_id),
        )

    async def list_topics(
        self,
        *,
        user_id: str,
        conversation_id: str,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?", "conversation_id = ?"]
        parameters: list[Any] = [user_id, conversation_id]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(statuses)
        query = f"""
            SELECT *
            FROM conversation_topics
            WHERE {" AND ".join(clauses)}
            ORDER BY
                CASE status
                    WHEN 'active' THEN 0
                    WHEN 'parked' THEN 1
                    ELSE 2
                END,
                last_touched_at DESC,
                id ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return await self._fetch_all(query, tuple(parameters))

    async def update_topic(
        self,
        *,
        topic_id: str,
        user_id: str,
        status: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        active_goal: str | None = None,
        open_questions: list[str] | None = None,
        decisions: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        source_message_start_seq: int | None = None,
        source_message_end_seq: int | None = None,
        last_touched_seq: int | None = None,
        last_touched_at: str | None = None,
        confidence: float | None = None,
        privacy_level: int | None = None,
        intimacy_boundary: IntimacyBoundary | None = None,
        intimacy_boundary_confidence: float | None = None,
        event_type: str = "updated",
        event_payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_topic(topic_id, user_id)
        if existing is None:
            return None
        timestamp = self._timestamp()
        updates: list[str] = ["updated_at = ?"]
        parameters: list[Any] = [timestamp]
        field_updates = {
            "status": status,
            "title": title,
            "summary": summary,
            "active_goal": active_goal,
            "source_message_start_seq": source_message_start_seq,
            "source_message_end_seq": source_message_end_seq,
            "last_touched_seq": last_touched_seq,
            "last_touched_at": last_touched_at,
            "confidence": confidence,
            "privacy_level": privacy_level,
            "intimacy_boundary": intimacy_boundary.value if intimacy_boundary is not None else None,
            "intimacy_boundary_confidence": intimacy_boundary_confidence,
        }
        for field_name, value in field_updates.items():
            if value is None:
                continue
            updates.append(f"{field_name} = ?")
            parameters.append(value)
        json_updates = {
            "open_questions_json": open_questions,
            "decisions_json": decisions,
            "artifact_ids_json": artifact_ids,
        }
        for field_name, value in json_updates.items():
            if value is None:
                continue
            updates.append(f"{field_name} = ?")
            parameters.append(_encode_json(value))
        parameters.extend([topic_id, user_id])
        await self._connection.execute(
            f"""
            UPDATE conversation_topics
            SET {", ".join(updates)}
            WHERE id = ?
              AND user_id = ?
            """,
            tuple(parameters),
        )
        await self.create_event(
            user_id=user_id,
            conversation_id=str(existing["conversation_id"]),
            topic_id=topic_id,
            event_type=event_type,
            payload=event_payload or {},
            commit=False,
        )
        if commit:
            await self._connection.commit()
        return await self.get_topic(topic_id, user_id)

    async def create_event(
        self,
        *,
        user_id: str,
        conversation_id: str,
        event_type: str,
        topic_id: str | None = None,
        payload: dict[str, Any] | None = None,
        source_message_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        event_id = generate_prefixed_id("tpe")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO conversation_topic_events(
                id,
                user_id,
                conversation_id,
                topic_id,
                event_type,
                payload_json,
                source_message_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                user_id,
                conversation_id,
                topic_id,
                event_type,
                _encode_json(payload),
                source_message_id,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            """
            SELECT *
            FROM conversation_topic_events
            WHERE id = ?
              AND user_id = ?
            """,
            (event_id, user_id),
        )
        if created is None:
            raise RuntimeError(f"Failed to create conversation topic event {event_id}")
        return created

    async def link_source(
        self,
        *,
        user_id: str,
        topic_id: str,
        source_kind: str,
        source_id: str,
        relation_kind: str = "evidence",
        commit: bool = True,
    ) -> dict[str, Any]:
        topic = await self.get_topic(topic_id, user_id)
        if topic is None:
            raise ValueError(f"Topic {topic_id} does not belong to user {user_id}")
        source_link_id = generate_prefixed_id("tps")
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            INSERT OR IGNORE INTO conversation_topic_sources(
                id,
                user_id,
                topic_id,
                source_kind,
                source_id,
                relation_kind,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_link_id,
                user_id,
                topic_id,
                source_kind,
                source_id,
                relation_kind,
                timestamp,
            ),
        )
        if cursor.rowcount:
            await self.create_event(
                user_id=user_id,
                conversation_id=str(topic["conversation_id"]),
                topic_id=topic_id,
                event_type="source_linked",
                payload={
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "relation_kind": relation_kind,
                },
                commit=False,
            )
        if commit:
            await self._connection.commit()
        return await self._fetch_one(
            """
            SELECT *
            FROM conversation_topic_sources
            WHERE user_id = ?
              AND topic_id = ?
              AND source_kind = ?
              AND source_id = ?
              AND relation_kind = ?
            """,
            (user_id, topic_id, source_kind, source_id, relation_kind),
        )

    async def list_topic_sources(self, *, user_id: str, topic_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM conversation_topic_sources
            WHERE user_id = ?
              AND topic_id = ?
            ORDER BY created_at ASC, _rowid ASC
            """,
            (user_id, topic_id),
        )

    async def list_events(
        self,
        *,
        user_id: str,
        conversation_id: str,
        topic_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if topic_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM conversation_topic_events
                WHERE user_id = ?
                  AND conversation_id = ?
                ORDER BY created_at ASC, _rowid ASC
                """,
                (user_id, conversation_id),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM conversation_topic_events
            WHERE user_id = ?
              AND conversation_id = ?
              AND topic_id = ?
            ORDER BY created_at ASC, _rowid ASC
            """,
            (user_id, conversation_id, topic_id),
        )

    async def get_topic_snapshot(
        self,
        *,
        user_id: str,
        conversation_id: str,
        active_limit: int = 3,
        parked_limit: int = 5,
        current_seq_upper_bound: int | None = None,
        refresh_message_threshold: int = _DEFAULT_REFRESH_MESSAGE_THRESHOLD,
        stale_message_threshold: int = _DEFAULT_STALE_MESSAGE_THRESHOLD,
        refresh_token_threshold: int = _DEFAULT_REFRESH_TOKEN_THRESHOLD,
        stale_token_threshold: int = _DEFAULT_STALE_TOKEN_THRESHOLD,
    ) -> dict[str, Any]:
        active_topics = await self.list_topics(
            user_id=user_id,
            conversation_id=conversation_id,
            statuses=["active"],
            limit=active_limit,
        )
        parked_topics = await self.list_topics(
            user_id=user_id,
            conversation_id=conversation_id,
            statuses=["parked"],
            limit=parked_limit,
        )
        freshness = await self._topic_freshness(
            user_id=user_id,
            conversation_id=conversation_id,
            topics=[*active_topics, *parked_topics],
            current_seq_upper_bound=current_seq_upper_bound,
            refresh_message_threshold=refresh_message_threshold,
            stale_message_threshold=stale_message_threshold,
            refresh_token_threshold=refresh_token_threshold,
            stale_token_threshold=stale_token_threshold,
        )
        return {
            "active_topics": await self._compact_topics_with_sources(user_id, active_topics),
            "parked_topics": await self._compact_topics_with_sources(user_id, parked_topics),
            "freshness": freshness,
        }

    async def _compact_topics_with_sources(
        self,
        user_id: str,
        topics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        compact_topics: list[dict[str, Any]] = []
        for topic in topics:
            sources = await self.list_topic_sources(user_id=user_id, topic_id=str(topic["id"]))
            compact = _compact_topic(topic)
            compact["source_counts"] = _source_counts(sources)
            compact["source_refs"] = _compact_sources(sources)
            compact_topics.append(compact)
        return compact_topics

    async def _topic_freshness(
        self,
        *,
        user_id: str,
        conversation_id: str,
        topics: list[dict[str, Any]],
        current_seq_upper_bound: int | None,
        refresh_message_threshold: int,
        stale_message_threshold: int,
        refresh_token_threshold: int,
        stale_token_threshold: int,
    ) -> dict[str, Any]:
        event_processed_seq = await self._last_processed_seq_from_events(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        last_processed_seq = _max_optional_int(_last_processed_seq(topics), event_processed_seq)
        latest_message_seq = await self._latest_message_seq(
            user_id=user_id,
            conversation_id=conversation_id,
            current_seq_upper_bound=current_seq_upper_bound,
        )
        lag_stats = await self._lag_stats(
            user_id=user_id,
            conversation_id=conversation_id,
            after_seq=last_processed_seq or 0,
            current_seq_upper_bound=latest_message_seq,
        )
        last_processed_message_id = (
            await self._message_id_for_seq(
                user_id=user_id,
                conversation_id=conversation_id,
                seq=last_processed_seq,
            )
            if last_processed_seq is not None
            else None
        )
        status = _freshness_status(
            has_topics=bool(topics),
            latest_message_seq=latest_message_seq,
            lag_message_count=lag_stats["lag_message_count"],
            lag_token_count=lag_stats["lag_token_count"],
            refresh_message_threshold=refresh_message_threshold,
            stale_message_threshold=stale_message_threshold,
            refresh_token_threshold=refresh_token_threshold,
            stale_token_threshold=stale_token_threshold,
        )
        return {
            "status": status,
            "last_processed_seq": last_processed_seq,
            "last_processed_message_id": last_processed_message_id,
            "latest_message_seq": latest_message_seq,
            "lag_message_count": lag_stats["lag_message_count"],
            "lag_token_count": lag_stats["lag_token_count"],
            "refresh_message_threshold": refresh_message_threshold,
            "stale_message_threshold": stale_message_threshold,
            "refresh_token_threshold": refresh_token_threshold,
            "stale_token_threshold": stale_token_threshold,
        }

    async def _latest_message_seq(
        self,
        *,
        user_id: str,
        conversation_id: str,
        current_seq_upper_bound: int | None,
    ) -> int:
        clauses = ["m.conversation_id = ?", "c.user_id = ?"]
        parameters: list[Any] = [conversation_id, user_id]
        if current_seq_upper_bound is not None:
            clauses.append("m.seq <= ?")
            parameters.append(current_seq_upper_bound)
        cursor = await self._connection.execute(
            f"""
            SELECT COALESCE(MAX(m.seq), 0) AS latest_seq
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE {" AND ".join(clauses)}
            """,
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return int(row["latest_seq"])

    async def _last_processed_seq_from_events(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> int | None:
        cursor = await self._connection.execute(
            """
            SELECT MAX(
                CAST(json_extract(payload_json, '$.processed_through_seq') AS INTEGER)
            ) AS last_processed_seq
            FROM conversation_topic_events
            WHERE user_id = ?
              AND conversation_id = ?
              AND json_extract(payload_json, '$.source') = 'offline_topic_working_set_updater'
            """,
            (user_id, conversation_id),
        )
        row = await cursor.fetchone()
        value = row["last_processed_seq"]
        if value is None:
            return None
        return int(value)

    async def _lag_stats(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_seq: int,
        current_seq_upper_bound: int,
    ) -> dict[str, int]:
        cursor = await self._connection.execute(
            """
            SELECT
                COUNT(*) AS lag_message_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN m.token_count IS NOT NULL AND m.token_count > 0
                                THEN m.token_count
                            ELSE CAST((LENGTH(m.text) + 3) / 4 AS INTEGER)
                        END
                    ),
                    0
                ) AS lag_token_count
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq > ?
              AND m.seq <= ?
            """,
            (conversation_id, user_id, after_seq, current_seq_upper_bound),
        )
        row = await cursor.fetchone()
        return {
            "lag_message_count": int(row["lag_message_count"]),
            "lag_token_count": int(row["lag_token_count"]),
        }

    async def _message_id_for_seq(
        self,
        *,
        user_id: str,
        conversation_id: str,
        seq: int,
    ) -> str | None:
        cursor = await self._connection.execute(
            """
            SELECT m.id
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq = ?
            """,
            (conversation_id, user_id, seq),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return str(row["id"])


def _compact_topic(topic: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": topic["id"],
        "status": topic["status"],
        "title": topic["title"],
        "summary": topic["summary"],
        "active_goal": topic.get("active_goal"),
        "open_questions": topic.get("open_questions_json") or [],
        "decisions": topic.get("decisions_json") or [],
        "artifact_ids": topic.get("artifact_ids_json") or [],
        "source_message_start_seq": topic.get("source_message_start_seq"),
        "source_message_end_seq": topic.get("source_message_end_seq"),
        "last_touched_seq": topic.get("last_touched_seq"),
        "last_touched_at": topic.get("last_touched_at"),
        "confidence": topic.get("confidence"),
        "privacy_level": topic.get("privacy_level"),
        "intimacy_boundary": topic.get("intimacy_boundary") or IntimacyBoundary.ORDINARY.value,
        "intimacy_boundary_confidence": topic.get("intimacy_boundary_confidence") or 0.0,
    }


def _last_processed_seq(topics: list[dict[str, Any]]) -> int | None:
    candidates: list[int] = []
    for topic in topics:
        for field_name in ("last_touched_seq", "source_message_end_seq"):
            value = topic.get(field_name)
            if isinstance(value, int):
                candidates.append(value)
            elif isinstance(value, str) and value.isdigit():
                candidates.append(int(value))
    if not candidates:
        return None
    return max(candidates)


def _max_optional_int(left: int | None, right: int | None) -> int | None:
    candidates = [value for value in (left, right) if value is not None]
    if not candidates:
        return None
    return max(candidates)


def _freshness_status(
    *,
    has_topics: bool,
    latest_message_seq: int,
    lag_message_count: int,
    lag_token_count: int,
    refresh_message_threshold: int,
    stale_message_threshold: int,
    refresh_token_threshold: int,
    stale_token_threshold: int,
) -> str:
    if not has_topics:
        return "missing" if latest_message_seq > 0 else "fresh"
    if lag_message_count >= stale_message_threshold or lag_token_count >= stale_token_threshold:
        return "stale"
    if lag_message_count >= refresh_message_threshold or lag_token_count >= refresh_token_threshold:
        return "slightly_stale"
    return "fresh"


def _source_counts(sources: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        source_kind = str(source.get("source_kind") or "")
        counts[source_kind] = counts.get(source_kind, 0) + 1
    return counts


def _compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "source_kind": str(source.get("source_kind") or ""),
            "source_id": str(source.get("source_id") or ""),
            "relation_kind": str(source.get("relation_kind") or ""),
        }
        for source in sources[:_TOPIC_SOURCE_REF_LIMIT]
    ]
