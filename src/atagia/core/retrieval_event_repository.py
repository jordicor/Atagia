"""Repositories for retrieval traces, feedback, and admin audit logging."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id, new_retrieval_id
from atagia.core.repositories import BaseRepository, MemoryObjectRepository, _encode_json


class MemoryFeedbackOwnershipError(ValueError):
    """Raised when feedback references an event or memory outside the user's scope."""


class MemoryFeedbackMismatchError(ValueError):
    """Raised when feedback references a memory not returned in the retrieval event."""


class RetrievalEventRepository(BaseRepository):
    """Persistence operations for retrieval event traces."""

    async def create_event(self, event: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        event_id = str(event.get("id") or new_retrieval_id())
        timestamp = str(event.get("created_at") or self._timestamp())
        await self._connection.execute(
            """
            INSERT INTO retrieval_events(
                id,
                user_id,
                conversation_id,
                request_message_id,
                response_message_id,
                assistant_mode_id,
                user_persona_id,
                platform_id,
                character_id,
                mode,
                incognito,
                remember_across_chats,
                remember_across_devices,
                retrieval_plan_json,
                selected_memory_ids_json,
                context_view_json,
                outcome_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event["user_id"],
                event["conversation_id"],
                event["request_message_id"],
                event.get("response_message_id"),
                event["assistant_mode_id"],
                event.get("user_persona_id"),
                event.get("platform_id"),
                event.get("character_id"),
                event.get("mode"),
                1 if event.get("incognito") else 0,
                1 if event.get("remember_across_chats", True) else 0,
                1 if event.get("remember_across_devices", True) else 0,
                _encode_json(event["retrieval_plan_json"]),
                _encode_json(event.get("selected_memory_ids_json", [])),
                _encode_json(event.get("context_view_json", {})),
                _encode_json(event.get("outcome_json", {})),
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self.get_event(event_id, str(event["user_id"]))
        if created is None:
            raise RuntimeError("Failed to create retrieval event row")
        return created

    async def get_event(self, event_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM retrieval_events
            WHERE id = ?
              AND user_id = ?
            """,
            (event_id, user_id),
        )

    async def list_events(
        self,
        user_id: str,
        conversation_id: str | None,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if conversation_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM retrieval_events
                WHERE user_id = ?
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                OFFSET ?
                """,
                (user_id, limit, offset),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM retrieval_events
            WHERE user_id = ?
              AND conversation_id = ?
            ORDER BY created_at DESC, id ASC
            LIMIT ?
            OFFSET ?
            """,
            (user_id, conversation_id, limit, offset),
        )

    async def list_events_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM retrieval_events
            WHERE user_id = ?
              AND conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, conversation_id),
        )

    async def update_outcome_fields(
        self,
        event_id: str,
        user_id: str,
        updates: dict[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        event = await self.get_event(event_id, user_id)
        if event is None:
            return None
        outcome = event.get("outcome_json")
        current_outcome = dict(outcome) if isinstance(outcome, dict) else {}
        current_outcome.update(updates)
        await self._connection.execute(
            """
            UPDATE retrieval_events
            SET outcome_json = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                _encode_json(current_outcome),
                event_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_event(event_id, user_id)


class MemoryFeedbackRepository(BaseRepository):
    """Persistence operations for memory usefulness feedback."""

    async def create_feedback(
        self,
        retrieval_event_id: str | None,
        memory_id: str | None,
        user_id: str,
        feedback_type: str,
        score: float | None,
        metadata: dict[str, Any] | None = None,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        mode: str | None = None,
    ) -> dict[str, Any]:
        selected_memory_ids: set[str] | None = None
        namespace_required = conversation_id is not None and platform_id is not None
        if retrieval_event_id is not None:
            event_row = await self._fetch_one(
                """
                SELECT *
                FROM retrieval_events
                WHERE id = ?
                  AND user_id = ?
                """,
                (retrieval_event_id, user_id),
            )
            if event_row is None:
                raise MemoryFeedbackOwnershipError(
                    f"Retrieval event {retrieval_event_id} does not belong to user {user_id}"
                )
            if namespace_required and (
                event_row.get("conversation_id") != conversation_id
                or event_row.get("user_persona_id") != user_persona_id
                or event_row.get("platform_id") != platform_id
                or event_row.get("character_id") != character_id
                or bool(event_row.get("incognito")) != bool(incognito)
            ):
                raise MemoryFeedbackOwnershipError(
                    f"Retrieval event {retrieval_event_id} is outside the requested namespace"
                )
            raw_selected_ids = event_row.get("selected_memory_ids_json") or []
            if isinstance(raw_selected_ids, list):
                selected_memory_ids = {str(item) for item in raw_selected_ids}
            else:
                selected_memory_ids = set()
        if memory_id is not None:
            if namespace_required:
                memory = await MemoryObjectRepository(
                    self._connection,
                    self._clock,
                ).get_visible_memory_object(
                    memory_id,
                    user_id,
                    conversation_id=conversation_id,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    remember_across_chats=remember_across_chats,
                    remember_across_devices=remember_across_devices,
                    sensitivity_gates_enabled=True,
                )
                if memory is None:
                    raise MemoryFeedbackOwnershipError(
                        f"Memory object {memory_id} is outside the requested namespace"
                    )
            else:
                cursor = await self._connection.execute(
                    """
                    SELECT 1
                    FROM memory_objects
                    WHERE id = ?
                      AND user_id = ?
                    """,
                    (memory_id, user_id),
                )
                if await cursor.fetchone() is None:
                    raise MemoryFeedbackOwnershipError(
                        f"Memory object {memory_id} does not belong to user {user_id}"
                    )
            if selected_memory_ids is not None and memory_id not in selected_memory_ids:
                raise MemoryFeedbackMismatchError(
                    f"Memory object {memory_id} was not selected in retrieval event {retrieval_event_id}"
                )

        feedback_id = generate_prefixed_id("fbk")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO memory_feedback_events(
                id,
                user_id,
                retrieval_event_id,
                memory_id,
                feedback_type,
                score,
                metadata_json,
                created_at,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                mode,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                policy_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                user_id,
                retrieval_event_id,
                memory_id,
                feedback_type,
                score,
                _encode_json(metadata),
                timestamp,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                mode,
                1 if incognito else 0,
                1 if remember_across_chats else 0,
                1 if remember_across_devices else 0,
                _encode_json(
                    {
                        "source": "memory_feedback",
                        "conversation_id": conversation_id,
                        "mode": mode,
                    }
                ),
            ),
        )
        await self._connection.commit()
        created = await self._fetch_one(
            """
            SELECT *
            FROM memory_feedback_events
            WHERE id = ?
              AND user_id = ?
            """,
            (feedback_id, user_id),
        )
        if created is None:
            raise RuntimeError("Failed to create memory feedback row")
        return created

    async def list_feedback(self, memory_id: str, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_feedback_events
            WHERE memory_id = ?
              AND user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (memory_id, user_id),
        )


class AdminAuditRepository(BaseRepository):
    """Persistence operations for lightweight admin read auditing."""

    async def create_audit_entry(
        self,
        *,
        admin_user_id: str,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audit_id = generate_prefixed_id("aud")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO admin_audit_log(
                id,
                admin_user_id,
                action,
                target_type,
                target_id,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                admin_user_id,
                action,
                target_type,
                target_id,
                _encode_json(metadata),
                timestamp,
            ),
        )
        await self._connection.commit()
        created = await self._fetch_one(
            """
            SELECT *
            FROM admin_audit_log
            WHERE id = ?
            """,
            (audit_id,),
        )
        if created is None:
            raise RuntimeError("Failed to create admin audit row")
        return created

    async def list_entries(
        self,
        admin_user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if admin_user_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM admin_audit_log
                ORDER BY created_at ASC, _rowid ASC
                LIMIT ?
                """,
                (limit,),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM admin_audit_log
            WHERE admin_user_id = ?
            ORDER BY created_at ASC, _rowid ASC
            LIMIT ?
            """,
            (admin_user_id, limit),
        )


def build_logging_repositories(
    connection: aiosqlite.Connection,
    clock: Clock,
) -> tuple[RetrievalEventRepository, MemoryFeedbackRepository, AdminAuditRepository]:
    """Return the Step 10 logging repositories sharing one connection/clock."""
    return (
        RetrievalEventRepository(connection, clock),
        MemoryFeedbackRepository(connection, clock),
        AdminAuditRepository(connection, clock),
    )
