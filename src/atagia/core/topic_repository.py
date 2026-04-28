"""SQLite persistence helpers for live conversation topic working sets."""

from __future__ import annotations

from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _encode_json

_TOPIC_SOURCE_REF_LIMIT = 8


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
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        last_touched_seq: int | None = None,
        last_touched_at: str | None = None,
        confidence: float | None = None,
        privacy_level: int | None = None,
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
            "last_touched_seq": last_touched_seq,
            "last_touched_at": last_touched_at,
            "confidence": confidence,
            "privacy_level": privacy_level,
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
        return {
            "active_topics": await self._compact_topics_with_sources(user_id, active_topics),
            "parked_topics": await self._compact_topics_with_sources(user_id, parked_topics),
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
        "last_touched_seq": topic.get("last_touched_seq"),
        "last_touched_at": topic.get("last_touched_at"),
        "confidence": topic.get("confidence"),
        "privacy_level": topic.get("privacy_level"),
    }


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
