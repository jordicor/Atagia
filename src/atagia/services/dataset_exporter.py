"""Conversation export helpers for replay and research workflows."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MessageRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_replay import (
    ConversationExport,
    ConversationExportKind,
    ExportAnonymizationMode,
    ExportedMessage,
    ExportedRetrievalTrace,
)
from atagia.services.export_anonymizer import (
    ExportAnonymizationProjection,
    ExportAnonymizer,
)
from atagia.services.llm_client import LLMClient
from atagia.services.model_resolution import resolve_component_model


class ConversationExportNotFoundError(ValueError):
    """Raised when the requested conversation does not belong to the user."""


class UnsafeConversationExportRequestError(ValueError):
    """Raised when the export request violates the allowed safety policy."""


class AnonymizedExportDisabledError(UnsafeConversationExportRequestError):
    """Raised when anonymized exports are requested without explicit opt-in."""


@dataclass(frozen=True, slots=True)
class _ExportIdMap:
    user_id: str
    conversation_id: str
    workspace_id: str | None
    message_ids: dict[str, str]


class DatasetExporter:
    """Export conversations and retrieval traces as JSON-serializable payloads."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        llm_client: LLMClient[Any] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._llm_client = llm_client
        self._settings = settings
        self._conversation_repository = ConversationRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)
        self._retrieval_event_repository = RetrievalEventRepository(connection, clock)

    async def export_conversation(
        self,
        conversation_id: str,
        user_id: str,
        include_retrieval_traces: bool = True,
        include_memory_snapshots: bool = False,
        include_intimacy_context: bool = False,
        anonymization_mode: ExportAnonymizationMode = ExportAnonymizationMode.RAW,
    ) -> ConversationExport:
        del include_memory_snapshots
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ConversationExportNotFoundError("Conversation not found for user")
        intimacy_boundary_counts = await self._conversation_intimacy_boundary_counts(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if intimacy_boundary_counts and not include_intimacy_context:
            raise UnsafeConversationExportRequestError(
                "Conversation export contains intimacy-bound context; set include_intimacy_context=true explicitly"
            )
        if anonymization_mode != ExportAnonymizationMode.RAW and include_retrieval_traces:
            raise UnsafeConversationExportRequestError(
                "Retrieval traces are not available for anonymized projection exports"
            )

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
        if anonymization_mode != ExportAnonymizationMode.RAW:
            return await self._export_anonymized_projection(
                conversation_id=conversation_id,
                user_id=user_id,
                workspace_id=self._coerce_workspace_id(conversation.get("workspace_id")),
                assistant_mode_id=str(conversation["assistant_mode_id"]),
                messages=exported_messages,
                intimacy_boundary_counts=intimacy_boundary_counts,
                anonymization_mode=anonymization_mode,
            )

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
            export_kind=ConversationExportKind.RAW_REPLAY,
            replay_compatible=True,
            workspace_id=conversation.get("workspace_id"),
            messages=exported_messages,
            retrieval_traces=retrieval_traces,
            intimacy_boundary_counts=intimacy_boundary_counts,
            exported_at=self._clock.now().isoformat(),
        )

    async def _export_anonymized_projection(
        self,
        *,
        conversation_id: str,
        user_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
        messages: list[ExportedMessage],
        intimacy_boundary_counts: dict[str, int],
        anonymization_mode: ExportAnonymizationMode,
    ) -> ConversationExport:
        if self._settings is None or not self._settings.allow_admin_export_anonymization:
            raise AnonymizedExportDisabledError(
                "Anonymized admin export requires ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION=true"
            )
        if self._llm_client is None:
            raise UnsafeConversationExportRequestError(
                "Anonymized export requires a configured LLM client"
            )
        anonymizer = ExportAnonymizer(
            self._llm_client,
            model=self._resolve_anonymization_model(),
        )
        projection = await anonymizer.anonymize_messages(messages, anonymization_mode)
        export_ids = self._pseudonymized_ids(
            workspace_id=workspace_id,
            messages=messages,
        )
        return ConversationExport(
            conversation_id=export_ids.conversation_id,
            user_id=export_ids.user_id,
            assistant_mode_id=assistant_mode_id,
            export_kind=ConversationExportKind.ANONYMIZED_PROJECTION,
            replay_compatible=False,
            workspace_id=export_ids.workspace_id,
            messages=self._anonymized_messages(messages, export_ids, projection, anonymization_mode),
            retrieval_traces=None,
            intimacy_boundary_counts=intimacy_boundary_counts,
            exported_at=None,
            anonymization=projection.summary,
        )

    async def _conversation_intimacy_boundary_counts(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        cursor = await self._connection.execute(
            """
            SELECT mo.intimacy_boundary, COUNT(DISTINCT mo.id) AS boundary_count
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND mo.intimacy_boundary != 'ordinary'
              AND (
                mo.conversation_id = ?
                OR EXISTS (
                    SELECT 1
                    FROM json_each(mo.payload_json, '$.source_message_ids') AS source_ids
                    JOIN messages AS m ON m.id = source_ids.value
                    WHERE m.conversation_id = ?
                )
              )
            GROUP BY mo.intimacy_boundary
            """,
            (user_id, conversation_id, conversation_id),
        )
        for row in await cursor.fetchall():
            boundary = str(row["intimacy_boundary"])
            counts[boundary] = counts.get(boundary, 0) + int(row["boundary_count"])
        for table in (
            "summary_views",
            "conversation_topics",
            "artifacts",
            "verbatim_pins",
        ):
            cursor = await self._connection.execute(
                f"""
                SELECT intimacy_boundary, COUNT(*) AS boundary_count
                FROM {table}
                WHERE user_id = ?
                  AND conversation_id = ?
                  AND intimacy_boundary != 'ordinary'
                GROUP BY intimacy_boundary
                """,
                (user_id, conversation_id),
            )
            for row in await cursor.fetchall():
                boundary = str(row["intimacy_boundary"])
                counts[boundary] = counts.get(boundary, 0) + int(row["boundary_count"])
        return dict(sorted(counts.items()))

    def _resolve_anonymization_model(self) -> str:
        if self._settings is None:
            raise UnsafeConversationExportRequestError(
                "Anonymized export requires runtime settings"
            )
        return resolve_component_model(self._settings, "export_anonymizer")

    @staticmethod
    def _coerce_workspace_id(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _pseudonymized_ids(
        *,
        workspace_id: str | None,
        messages: list[ExportedMessage],
    ) -> _ExportIdMap:
        export_salt = secrets.token_hex(4)
        message_ids = {
            message.message_id: f"anon_message_{index:04d}_{export_salt}"
            for index, message in enumerate(messages, start=1)
        }
        return _ExportIdMap(
            user_id=f"anon_user_0001_{export_salt}",
            conversation_id=f"anon_conversation_0001_{export_salt}",
            workspace_id=f"anon_workspace_0001_{export_salt}" if workspace_id is not None else None,
            message_ids=message_ids,
        )

    @staticmethod
    def _anonymized_messages(
        original_messages: list[ExportedMessage],
        export_ids: _ExportIdMap,
        projection: ExportAnonymizationProjection,
        mode: ExportAnonymizationMode,
    ) -> list[ExportedMessage]:
        selected_messages = (
            projection.strict_messages
            if mode == ExportAnonymizationMode.STRICT
            else projection.readable_messages
        )
        return [
            ExportedMessage(
                message_id=export_ids.message_ids[message.message_id],
                seq=message.seq,
                role=message.role,
                content=selected_messages[message.message_id],
                occurred_at=None,
                created_at=None,
            )
            for message in original_messages
        ]
