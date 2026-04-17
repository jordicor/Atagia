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
        anonymization_mode: ExportAnonymizationMode = ExportAnonymizationMode.RAW,
    ) -> ConversationExport:
        del include_memory_snapshots
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ConversationExportNotFoundError("Conversation not found for user")
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
            exported_at=None,
            anonymization=projection.summary,
        )

    def _resolve_anonymization_model(self) -> str:
        if self._settings is None:
            raise UnsafeConversationExportRequestError(
                "Anonymized export requires runtime settings"
            )
        for candidate in (
            self._settings.llm_classifier_model,
            self._settings.llm_extraction_model,
            self._settings.llm_chat_model,
        ):
            if candidate is not None and candidate.strip():
                return candidate.strip()
        raise UnsafeConversationExportRequestError(
            "Anonymized export requires at least one configured LLM model"
        )

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
