"""Tests for conversation dataset export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.models.schemas_replay import ConversationExportKind, ExportAnonymizationMode
from atagia.services.dataset_exporter import (
    AnonymizedExportDisabledError,
    DatasetExporter,
    UnsafeConversationExportRequestError,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class ExportAnonymizationProvider(LLMProvider):
    name = "dataset-export-anonymizer"

    def __init__(self) -> None:
        self.rewrite_calls = 0
        self.verify_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if purpose == "export_anonymization_rewrite":
            self.rewrite_calls += 1
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "entities": [
                            {
                                "placeholder": "[person_001]",
                                "readable_label": "Person 1",
                                "source_forms": ["Maria"],
                            },
                            {
                                "placeholder": "[place_001]",
                                "readable_label": "Place 1",
                                "source_forms": ["Barcelona"],
                            },
                        ],
                        "messages": [
                            {
                                "message_id": "msg_1",
                                "strict_content": "[person_001] lives in [place_001] and needs retry help.",
                                "readable_content": "Person 1 lives in Place 1 and needs retry help.",
                            },
                            {
                                "message_id": "msg_2",
                                "strict_content": "Try the guard for [person_001] in [place_001].",
                                "readable_content": "Try the guard for Person 1 in Place 1.",
                            },
                        ],
                    }
                ),
            )
        if purpose == "export_anonymization_verify":
            self.verify_calls += 1
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "approved": True,
                        "remaining_identifiers": [],
                        "unsafe_descriptive_clues": [],
                        "reasoning": "Projection is safe for export.",
                    }
                ),
            )
        raise AssertionError(f"Unexpected completion purpose in dataset exporter test: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embedding is not expected in dataset exporter test: {request.metadata}")


def _settings(*, allow_admin_export_anonymization: bool) -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="chat-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_admin_export_anonymization=allow_admin_export_anonymization,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("ws_ops", "usr_1", "Ops")
    await conversations.create_conversation("cnv_1", "usr_1", "ws_ops", "coding_debug", "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other Chat")
    await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "Maria lives in Barcelona and needs retry help.",
        8,
        {},
        "2023-05-08T13:56:00",
    )
    await messages.create_message(
        "msg_2",
        "cnv_1",
        "assistant",
        2,
        "Try the guard for Maria in Barcelona.",
        8,
        {},
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Retry memory",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id="mem_1",
    )
    await events.create_event(
        {
            "id": "ret_1",
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry"]},
            "selected_memory_ids_json": ["mem_1"],
            "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
            "outcome_json": {
                "detected_needs": ["follow_up_failure"],
                "scored_candidates": [{"memory_id": "mem_1", "final_score": 0.9}],
            },
            "created_at": "2026-04-05T14:00:00+00:00",
        }
    )
    return connection, clock


@pytest.mark.asyncio
async def test_export_conversation_with_and_without_retrieval_traces() -> None:
    connection, clock = await _build_runtime()
    try:
        exporter = DatasetExporter(connection, clock)

        with_traces = await exporter.export_conversation("cnv_1", "usr_1", include_retrieval_traces=True)
        without_traces = await exporter.export_conversation("cnv_1", "usr_1", include_retrieval_traces=False)

        assert with_traces.export_kind is ConversationExportKind.RAW_REPLAY
        assert with_traces.replay_compatible is True
        assert with_traces.messages[0].content == "Maria lives in Barcelona and needs retry help."
        assert [message.seq for message in with_traces.messages] == [1, 2]
        assert with_traces.messages[0].occurred_at == "2023-05-08T13:56:00"
        assert with_traces.messages[1].occurred_at is None
        assert with_traces.retrieval_traces is not None
        assert with_traces.retrieval_traces[0].retrieval_event_id == "ret_1"
        assert without_traces.retrieval_traces is None
        json.dumps(with_traces.model_dump(mode="json"))
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_content"),
    [
        (ExportAnonymizationMode.STRICT, "[person_001] lives in [place_001] and needs retry help."),
        (ExportAnonymizationMode.READABLE, "Person 1 lives in Place 1 and needs retry help."),
    ],
)
async def test_export_conversation_anonymized_projection_modes(
    mode: ExportAnonymizationMode,
    expected_content: str,
) -> None:
    connection, clock = await _build_runtime()
    provider = ExportAnonymizationProvider()
    llm_client = LLMClient(provider_name=provider.name, providers=[provider])
    try:
        exporter = DatasetExporter(
            connection,
            clock,
            llm_client=llm_client,
            settings=_settings(allow_admin_export_anonymization=True),
        )
        exported = await exporter.export_conversation(
            "cnv_1",
            "usr_1",
            include_retrieval_traces=False,
            anonymization_mode=mode,
        )

        assert exported.export_kind is ConversationExportKind.ANONYMIZED_PROJECTION
        assert exported.replay_compatible is False
        assert re.fullmatch(r"anon_user_0001_[0-9a-f]{8}", exported.user_id)
        assert re.fullmatch(r"anon_conversation_0001_[0-9a-f]{8}", exported.conversation_id)
        assert re.fullmatch(r"anon_workspace_0001_[0-9a-f]{8}", exported.workspace_id)
        assert re.fullmatch(r"anon_message_0001_[0-9a-f]{8}", exported.messages[0].message_id)
        assert exported.messages[0].content == expected_content
        assert exported.messages[0].created_at is None
        assert exported.messages[0].occurred_at is None
        assert exported.exported_at is None
        assert exported.retrieval_traces is None
        assert exported.anonymization is not None
        assert exported.anonymization.mode is mode
        assert exported.anonymization.entity_count == 2
        assert all("Maria" not in message.content for message in exported.messages)
        assert all("Barcelona" not in message.content for message in exported.messages)
        assert provider.rewrite_calls == 1
        assert provider.verify_calls == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_anonymized_export_requires_explicit_setting() -> None:
    connection, clock = await _build_runtime()
    provider = ExportAnonymizationProvider()
    llm_client = LLMClient(provider_name=provider.name, providers=[provider])
    try:
        exporter = DatasetExporter(
            connection,
            clock,
            llm_client=llm_client,
            settings=_settings(allow_admin_export_anonymization=False),
        )
        with pytest.raises(
            AnonymizedExportDisabledError,
            match="ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION=true",
        ):
            await exporter.export_conversation(
                "cnv_1",
                "usr_1",
                include_retrieval_traces=False,
                anonymization_mode=ExportAnonymizationMode.STRICT,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_anonymized_export_rejects_retrieval_traces() -> None:
    connection, clock = await _build_runtime()
    try:
        exporter = DatasetExporter(connection, clock)
        with pytest.raises(
            UnsafeConversationExportRequestError,
            match="Retrieval traces are not available",
        ):
            await exporter.export_conversation(
                "cnv_1",
                "usr_1",
                include_retrieval_traces=True,
                anonymization_mode=ExportAnonymizationMode.STRICT,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_export_conversation_verifies_user_ownership() -> None:
    connection, clock = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="Conversation not found for user"):
            await DatasetExporter(connection, clock).export_conversation("cnv_1", "usr_2")
    finally:
        await connection.close()
