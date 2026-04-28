"""Tests for verbatim pin service behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.verbatim_pin_service import VerbatimPinService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class NoopProvider(LLMProvider):
    name = "noop-verbatim-pin-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM should not be called in verbatim pin tests: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings should not be called in verbatim pin tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-verbatim-pin.db"),
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
        llm_extraction_model="test-model",
        llm_scoring_model="test-model",
        llm_classifier_model="test-model",
        llm_chat_model="openai/test-model",
        llm_ingest_model="openai/test-model",
        llm_retrieval_model="openai/test-model",
        llm_component_models={"intent_classifier": "openai/test-model"},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppRuntime:
    provider = NoopProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    runtime.clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    return runtime


@pytest.mark.asyncio
async def test_verbatim_pin_service_resolves_message_memory_object_and_text_span_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "alpha beta gamma",
                3,
                {},
                "2026-03-30T11:00:00+00:00",
            )
            await memories.create_memory_object(
                user_id="usr_1",
                workspace_id=None,
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="memory object snapshot",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.8,
                privacy_level=0,
                memory_id="mem_1",
            )

            message_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                privacy_level=0,
                created_by="usr_1",
            )
            memory_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MEMORY_OBJECT,
                target_id="mem_1",
                privacy_level=0,
                created_by="usr_1",
            )
            span_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.TEXT_SPAN,
                target_id="msg_1",
                target_span_start=6,
                target_span_end=10,
                privacy_level=0,
                created_by="usr_1",
            )

            assert message_pin["canonical_text"] == "alpha beta gamma"
            assert memory_pin["canonical_text"] == "memory object snapshot"
            assert span_pin["canonical_text"] == "beta"
            assert message_pin["index_text"] == "alpha beta gamma"
            assert span_pin["index_text"] == "beta"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_verbatim_pin_service_search_respects_conversation_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat A")
            await conversations.create_conversation("cnv_2", "usr_1", None, "coding_debug", "Chat B")
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "Scope locked phrase",
                3,
                {},
            )
            await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                privacy_level=0,
                created_by="usr_1",
            )

            same_conversation = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="scope locked phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
            )
            other_conversation = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="scope locked phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_2",
                limit=10,
            )

            assert [row["target_id"] for row in same_conversation] == ["msg_1"]
            assert other_conversation == []
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_verbatim_pin_service_uses_safe_index_text_for_high_privacy_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")

            created = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                canonical_text="Bank card PIN: 4512",
                index_text="bank card PIN",
                privacy_level=3,
                created_by="usr_1",
            )

            assert created["canonical_text"] == "Bank card PIN: 4512"
            assert created["index_text"] == "bank card PIN"

            matching_rows = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="bank card PIN",
                privacy_ceiling=3,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
                as_of="2026-03-30T12:00:00+00:00",
            )
            assert [row["id"] for row in matching_rows] == [created["id"]]

            assert await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="4512",
                privacy_ceiling=3,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
                as_of="2026-03-30T12:00:00+00:00",
            ) == []

            updated = await service.update_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created["id"],
                status=VerbatimPinStatus.ARCHIVED,
            )
            assert updated is not None
            assert updated["status"] == VerbatimPinStatus.ARCHIVED.value

            deleted = await service.delete_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created["id"],
            )
            assert deleted is not None
            assert deleted["status"] == VerbatimPinStatus.DELETED.value
        finally:
            await connection.close()
    finally:
        await runtime.close()
