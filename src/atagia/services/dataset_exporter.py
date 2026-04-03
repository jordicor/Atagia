"""Conversation export helpers for replay and research workflows."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.repositories import ConversationRepository, MessageRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_replay import (
    ConversationExport,
    ExportedMessage,
    ExportedRetrievalTrace,
)


class DatasetExporter:
    """Export conversations and retrieval traces as JSON-serializable payloads."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock
        self._conversation_repository = ConversationRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)
        self._retrieval_event_repository = RetrievalEventRepository(connection, clock)

    async def export_conversation(
        self,
        conversation_id: str,
        user_id: str,
        include_retrieval_traces: bool = True,
        include_memory_snapshots: bool = False,
    ) -> ConversationExport:
        del include_memory_snapshots
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ValueError("Conversation not found for user")

        messages = await self._message_repository.get_messages(conversation_id, user_id, limit=5000, offset=0)
        exported_messages = [
            ExportedMessage(
                message_id=str(message["id"]),
                seq=int(message["seq"]),
                role=str(message["role"]),
                content=str(message["text"]),
                occurred_at=resolve_message_occurred_at(message),
                created_at=str(message["created_at"]),
            )
            for message in messages
        ]

        retrieval_traces = None
        if include_retrieval_traces:
            message_seq_by_id = {
                str(message["id"]): int(message["seq"])
                for message in messages
            }
            event_rows = await self._retrieval_event_repository.list_events(user_id, conversation_id, limit=5000, offset=0)
            retrieval_traces = [
                ExportedRetrievalTrace(
                    retrieval_event_id=str(event["id"]),
                    request_message_seq=message_seq_by_id.get(str(event["request_message_id"]), 0),
                    detected_needs=[
                        str(need_type)
                        for need_type in (event.get("outcome_json") or {}).get("detected_needs", [])
                    ],
                    retrieval_plan=dict(event.get("retrieval_plan_json") or {}),
                    selected_memory_ids=[
                        str(memory_id)
                        for memory_id in (event.get("selected_memory_ids_json") or [])
                    ],
                    scored_candidates=[
                        dict(candidate)
                        for candidate in (event.get("outcome_json") or {}).get("scored_candidates", [])
                        if isinstance(candidate, dict)
                    ],
                    context_view=dict(event.get("context_view_json") or {}),
                    outcome=dict(event.get("outcome_json") or {}),
                )
                for event in sorted(
                    event_rows,
                    key=lambda event: (
                        message_seq_by_id.get(str(event["request_message_id"]), 0),
                        str(event["id"]),
                    ),
                )
            ]

        return ConversationExport(
            conversation_id=conversation_id,
            user_id=user_id,
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            workspace_id=conversation.get("workspace_id"),
            messages=exported_messages,
            retrieval_traces=retrieval_traces,
            exported_at=self._clock.now().isoformat(),
        )
