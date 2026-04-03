"""Tests for the library-mode Atagia engine."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atagia import Atagia
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)

_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class EngineProvider(LLMProvider):
    name = "engine-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps([]),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    [
                        {"memory_id": memory_id, "llm_applicability": 0.5}
                        for memory_id in memory_ids
                    ]
                ),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        if purpose == "memory_extraction":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "evidences": [],
                        "beliefs": [],
                        "contract_signals": [],
                        "state_updates": [],
                        "nothing_durable": True,
                    }
                ),
            )
        if purpose == "contract_projection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "signals": [],
                        "nothing_durable": True,
                    }
                ),
            )
        if purpose == "consequence_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_consequence": False,
                        "action_description": "",
                        "outcome_description": "",
                        "outcome_sentiment": "neutral",
                        "confidence": 0.0,
                        "likely_action_message_id": None,
                    }
                ),
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in engine tests: {request.model}")


class FailingEngineProvider(EngineProvider):
    def __init__(self, fail_purpose: str) -> None:
        super().__init__()
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == self._fail_purpose:
            raise LLMError(f"Injected failure for {self._fail_purpose}")
        return await super().complete(request)


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: EngineProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )


@pytest.mark.asyncio
async def test_engine_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    assert engine.runtime is not None

    await engine.close()
    assert engine.runtime is None


@pytest.mark.asyncio
async def test_engine_setup_respects_env_context_cache_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "false")
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.context_cache_enabled is False
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_setup_respects_chunking_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_CHUNKING_ENABLED", "true")
    engine = Atagia(
        db_path=":memory:",
        llm_provider="openai",
        llm_api_key="test-openai-key",
        chunking_enabled=False,
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.chunking_enabled is False
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_create_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_user("usr_1")
        await engine.create_workspace("usr_1", "wrk_1", "Workspace")
        conversation_id = await engine.create_conversation(
            "usr_1",
            "cnv_1",
            workspace_id="wrk_1",
            assistant_mode_id="coding_debug",
        )

        connection = await engine.runtime.open_connection()
        try:
            users = UserRepository(connection, engine.runtime.clock)
            workspaces = WorkspaceRepository(connection, engine.runtime.clock)
            conversations = ConversationRepository(connection, engine.runtime.clock)
            assert await users.get_user("usr_1") is not None
            assert await workspaces.get_workspace("wrk_1", "usr_1") is not None
            assert await conversations.get_conversation(conversation_id, "usr_1") is not None
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
        )

        assert isinstance(context.system_prompt, str)
        assert context.system_prompt
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[-1]["role"] == "user"
            assert stored_messages[-1]["text"] == "Please help me debug this retry loop."
            assert stored_messages[-1]["occurred_at"] == "2023-05-08T13:56:00"
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_add_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine.db",
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
            occurred_at="2023-05-09T14:10:00",
        )
        assert await engine.flush(timeout_seconds=5.0) is True

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[-1]["role"] == "assistant"
            assert stored_messages[-1]["text"] == "Check the retry guard first."
            assert stored_messages[-1]["occurred_at"] == "2023-05-09T14:10:00"
        finally:
            await connection.close()

        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction") == 2
        assert purposes.count("contract_projection") == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    async with engine:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        assert context.system_prompt

    assert engine._closed is True
    assert engine.runtime is None


@pytest.mark.asyncio
async def test_engine_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        result = await engine.chat(
            user_id="usr_1",
            conversation_id="cnv_1",
            mode="coding_debug",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
        )

        assert result.response_text == "Check the retry guard first."
        assert result.retrieval_event_id is not None
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[0]["occurred_at"] == "2023-05-08T13:56:00"
            assert stored_messages[1]["occurred_at"] == stored_messages[1]["created_at"]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_cache_hit_exposes_observability_without_retrieval_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        first = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        second = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="continue",
        )

        assert first.from_cache is False
        assert second.from_cache is True
        assert second.need_detection_skipped is True
        assert second.detected_needs == []
        connection = await engine.runtime.open_connection()
        try:
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            assert await events.list_events("usr_1", "cnv_1", limit=10) == []
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_add_response_invalidates_stable_context_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is not None

        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
        )

        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_invalidates_stable_context_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is not None

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Check the retry guard first.",
        )

        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-flush.db",
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ablation_switches_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        llm_provider="openai",
        llm_api_key="test-openai-key",
        skip_belief_revision=True,
        skip_compaction=True,
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.skip_belief_revision is True
        assert engine.runtime.settings.skip_compaction is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest.db",
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Check the retry guard first.",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
        assert not any(
            request.metadata.get("purpose") in {"need_detection", "applicability_scoring"}
            for request in provider.requests
        )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [message["role"] for message in stored_messages[-2:]] == ["user", "assistant"]
            assert stored_messages[-2]["text"] == "Please help me debug this retry loop."
            assert stored_messages[-2]["occurred_at"] == "2023-05-08T13:56:00"
            assert stored_messages[-1]["text"] == "Check the retry guard first."
        finally:
            await connection.close()

        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction") == 2
        assert purposes.count("contract_projection") == 1
        assert any(
            request.metadata.get("purpose") == "memory_extraction"
            and "<message_timestamp>2023-05-08T13:56:00</message_timestamp>" in request.messages[1].content
            for request in provider.requests
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_rolls_back_user_message_when_retrieval_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingEngineProvider("need_detection")
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", llm_provider="openai", llm_api_key="test-openai-key")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        with pytest.raises(LLMError):
            await engine.get_context(
                user_id="usr_1",
                conversation_id="cnv_1",
                message="Please help me debug this retry loop.",
            )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            assert await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0) == []
        finally:
            await connection.close()
    finally:
        await engine.close()
