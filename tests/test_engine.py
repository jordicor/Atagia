"""Tests for the library-mode Atagia engine."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia import Atagia
from atagia.core.clock import FrozenClock
from atagia.core.mind_repository import DEFAULT_MIND_ID
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.chat_support import default_operational_profile_snapshot
from atagia.services.errors import MessageIdConflictError, SourceSequenceConflictError
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)

_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


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
                output_text=json.dumps(
                    {
                        "needs": [],
                        "temporal_range": None,
                        "sub_queries": ["retry loop"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "retry loop",
                                "fts_phrase": "retry loop",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
        if purpose == "applicability_scoring":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "scores": [
                            {"score_key": score_key, "llm_applicability": 0.5}
                            for _memory_id, score_key in candidate_keys
                        ],
                    }
                ),
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": True,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "consent_confirmation_intent":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": "confirm"}),
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


def test_engine_build_settings_preserves_llm_debug_io_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    debug_dir = tmp_path / "llm-debug"
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO", "true")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_DIR", str(debug_dir))
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_PURPOSES", "applicability_scoring")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_RAW", "true")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_MAX_CHARS", "12345")
    engine = Atagia(
        db_path=tmp_path / "debug-settings.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    settings = engine._build_settings()

    assert settings.llm_debug_io_enabled is True
    assert settings.llm_debug_io_dir == str(debug_dir)
    assert settings.llm_debug_io_purposes == ("applicability_scoring",)
    assert settings.llm_debug_io_raw is True
    assert settings.llm_debug_io_max_chars == 12345


def test_engine_build_settings_preserves_memory_bm25_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_MEMORY_FTS_CANONICAL_BM25_WEIGHT", "9.0")
    monkeypatch.setenv("ATAGIA_MEMORY_FTS_INDEX_BM25_WEIGHT", "0.2")
    engine = Atagia(db_path=tmp_path / "bm25-settings.db")

    settings = engine._build_settings()

    assert settings.memory_fts_canonical_bm25_weight == 9.0
    assert settings.memory_fts_index_bm25_weight == 0.2


def test_engine_build_settings_preserves_embedding_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec")
    monkeypatch.setenv("ATAGIA_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("ATAGIA_EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "17")
    monkeypatch.setenv("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "3")
    engine = Atagia(db_path=tmp_path / "embedding-settings.db")

    settings = engine._build_settings()

    assert settings.embedding_backend == "sqlite_vec"
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    assert settings.embedding_vector_limit_cap == 17
    assert settings.embedding_search_overfetch_multiplier == 3


def test_engine_build_settings_preserves_topic_working_set_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_MESSAGE_LAG", "11")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_STALE_MESSAGE_LAG", "13")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_TOKEN_LAG", "1700")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_STALE_TOKEN_LAG", "2300")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_BATCH_MESSAGES", "5")
    engine = Atagia(db_path=tmp_path / "topic-settings.db")

    settings = engine._build_settings()

    assert settings.topic_working_set_enabled is False
    assert settings.topic_working_set_refresh_message_lag == 11
    assert settings.topic_working_set_stale_message_lag == 13
    assert settings.topic_working_set_refresh_token_lag == 1700
    assert settings.topic_working_set_stale_token_lag == 2300
    assert settings.topic_working_set_refresh_batch_messages == 5


def test_engine_build_settings_allows_explicit_embedding_backend_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec")
    engine = Atagia(
        db_path=tmp_path / "embedding-settings.db",
        embedding_backend="none",
    )

    settings = engine._build_settings()

    assert settings.embedding_backend == "none"


def test_engine_build_settings_resolves_resource_env_from_external_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATAGIA_MIGRATIONS_PATH", "./migrations")
    monkeypatch.setenv("ATAGIA_MANIFESTS_PATH", "./manifests")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILES_PATH", "./operational_profiles")

    engine = Atagia(db_path=tmp_path / "external-cwd.db")
    settings = engine._build_settings()

    assert settings.migrations_dir().exists()
    assert settings.manifests_dir().exists()
    assert settings.operational_profiles_dir().exists()


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: EngineProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    # The engine tests assert the full retrieval pipeline runs (including
    # need detection), so disable the small-corpus shortcut for the duration
    # of the test. Individual tests can still override via setenv.
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


def _normal_operational_profile_token(engine: Atagia) -> str:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    return default_operational_profile_snapshot(
        loader=engine.runtime.operational_profile_loader,
        settings=engine.runtime.settings,
    ).token


@pytest.mark.asyncio
async def test_engine_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.context_cache_enabled is False
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_setup_respects_env_graph_projection_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_GRAPH_PROJECTION_ENABLED", "true")
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.graph_projection_enabled is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_setup_respects_chunking_disable_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_DISABLE_CHUNKING_EXTRACTION", "false")
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        disable_chunking_extraction=True,
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.disable_chunking_extraction is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_create_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
async def test_engine_lifecycle_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_close",
            assistant_mode_id="coding_debug",
            temporary=True,
            temporary_ttl_seconds=3600,
            purge_on_close=False,
        )
        closed = await engine.close_conversation("usr_1", "cnv_close")
        assert closed["status"] == "closed"

        await engine.create_conversation("usr_1", "cnv_memory", assistant_mode_id="coding_debug")
        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            await memories.create_memory_object(
                user_id="usr_1",
                conversation_id="cnv_memory",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="Original lifecycle memory.",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                memory_id="mem_lifecycle",
            )
        finally:
            await connection.close()

        edited = await engine.edit_memory("usr_1", "mem_lifecycle", "Updated lifecycle memory.")
        assert edited["canonical_text"] == "Updated lifecycle memory."
        memory_report = await engine.delete_memory("usr_1", "mem_lifecycle", hard=True, confirmation="HARD_DELETE_MEMORY")
        assert memory_report.deleted_memories == 1

        conversation_report = await engine.delete_conversation(
            "usr_1",
            "cnv_memory",
            confirmation="DELETE_CONVERSATION",
        )
        assert conversation_report.conversation_id == "cnv_memory"

        await engine.create_conversation("usr_1", "cnv_erase", assistant_mode_id="coding_debug")
        erase_report = await engine.erase_user_data("usr_1", confirmation="ERASE_ALL_DATA")
        assert erase_report.deleted_conversations == 3
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_lifecycle_deletes_only_targeted_retrieval_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        for conversation_id in ("cnv_mem_1", "cnv_mem_2", "cnv_del_1", "cnv_del_2"):
            await engine.create_conversation("usr_1", conversation_id, assistant_mode_id="coding_debug")

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            for conversation_id, message_id in (
                ("cnv_mem_1", "msg_mem_1"),
                ("cnv_mem_2", "msg_mem_2"),
                ("cnv_del_1", "msg_del_1"),
                ("cnv_del_2", "msg_del_2"),
            ):
                await messages.create_message(message_id, conversation_id, "user", 1, "hello", 1, {})
            for conversation_id, memory_id, text in (
                ("cnv_mem_1", "mem_target", "target memory"),
                ("cnv_mem_2", "mem_other", "other memory"),
            ):
                await memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id=conversation_id,
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text=text,
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id=memory_id,
                )
            await memories.create_memory_object(
                user_id="usr_1",
                conversation_id=None,
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.USER,
                canonical_text="broad memory sourced by a deleted chat",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                payload={"source_message_ids": ["msg_del_1"]},
                memory_id="mem_del_broad",
            )
            for event_id, conversation_id, message_id, selected_ids in (
                ("evt_mem_target", "cnv_mem_1", "msg_mem_1", ["mem_target"]),
                ("evt_mem_other", "cnv_mem_2", "msg_mem_2", ["mem_other"]),
                ("evt_del_target", "cnv_del_1", "msg_del_1", []),
                ("evt_cross_selected", "cnv_del_2", "msg_del_2", ["mem_del_broad"]),
                ("evt_del_other", "cnv_del_2", "msg_del_2", []),
            ):
                await events.create_event(
                    {
                        "id": event_id,
                        "user_id": "usr_1",
                        "conversation_id": conversation_id,
                        "request_message_id": message_id,
                        "assistant_mode_id": "coding_debug",
                        "retrieval_plan_json": {},
                        "selected_memory_ids_json": selected_ids,
                        "context_view_json": {"event_id": event_id},
                        "outcome_json": {},
                    }
                )
        finally:
            await connection.close()

        await engine.delete_memory(
            "usr_1",
            "mem_target",
            hard=True,
            confirmation="HARD_DELETE_MEMORY",
        )
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_del_2",
            workspace_id=None,
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
        )
        await engine.runtime.storage_backend.set_context_view(
            cache_key,
            {"user_id": "usr_1", "conversation_id": "cnv_del_2"},
            ttl_seconds=60,
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is not None
        await engine.delete_conversation(
            "usr_1",
            "cnv_del_1",
            confirmation="DELETE_CONVERSATION",
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None

        connection = await engine.runtime.open_connection()
        try:
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            remaining_ids = {
                row["id"] for row in await events.list_events("usr_1", None, limit=20)
            }
            assert remaining_ids == {"evt_mem_other", "evt_del_other"}
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
            message_id="aurvek:msg:ctx-1",
        )

        assert isinstance(context.system_prompt, str)
        assert context.system_prompt
        assert context.request_message_id == "aurvek:msg:ctx-1"
        assert context.recent_transcript == []
        assert context.recent_transcript_trace is not None
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[-1]["role"] == "user"
            assert stored_messages[-1]["id"] == "aurvek:msg:ctx-1"
            assert stored_messages[-1]["text"] == "Please help me debug this retry loop."
            assert stored_messages[-1]["occurred_at"] == "2023-05-08T13:56:00"
        finally:
            await connection.close()

        duplicate_context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
            message_id="aurvek:msg:ctx-1",
        )

        assert duplicate_context.request_message_id == "aurvek:msg:ctx-1"
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [message["id"] for message in stored_messages] == ["aurvek:msg:ctx-1"]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_includes_recent_transcript_without_fts_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
            "usr_1",
            "cnv_1",
            "user",
            "Yesterday I went to the bank on Carrer Major.",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What time does that usually close?",
        )

        transcript_texts = [entry.text for entry in context.recent_transcript]
        assert transcript_texts == ["Yesterday I went to the bank on Carrer Major."]
        assert "What time does that usually close?" not in transcript_texts
        assert "<recent_transcript_json>" in context.system_prompt
        assert "Carrer Major" in context.system_prompt
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_uses_recent_transcript_budget_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        recent_transcript_budget_tokens=30000,
    )

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
            message="What did we decide?",
        )

        assert context.recent_transcript_trace is not None
        assert context.recent_transcript_trace.budget_tokens == 30000
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_can_disable_recent_transcript_for_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
            "usr_1",
            "cnv_1",
            "user",
            "This prior sentence should not be injected as raw transcript.",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did I just say?",
        )

        assert context.recent_transcript == []
        assert context.recent_transcript_omissions == []
        assert context.recent_transcript_trace is None
        assert context.assistant_guidance == []
        assert "<recent_transcript_json>" not in context.system_prompt
        assert "This prior sentence should not be injected" not in context.system_prompt
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
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
            message_id="aurvek:msg:assistant-1",
        )
        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
            occurred_at="2023-05-09T14:10:00",
            message_id="aurvek:msg:assistant-1",
        )
        assert await engine.flush(timeout_seconds=5.0) is True

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[-1]["role"] == "assistant"
            assert stored_messages[-1]["id"] == "aurvek:msg:assistant-1"
            assert stored_messages[-1]["text"] == "Check the retry guard first."
            assert stored_messages[-1]["occurred_at"] == "2023-05-09T14:10:00"
            assert [message["role"] for message in stored_messages] == [
                "user",
                "assistant",
            ]
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
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        engine.runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
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
            assert stored_messages[1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
            assert stored_messages[1]["created_at"] == "2026-03-31T04:00:00+00:00"
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_chat_can_disable_raw_recent_transcript_for_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
            "usr_1",
            "cnv_1",
            "user",
            "This prior chat message must not reach the chat model as transcript.",
        )

        await engine.chat(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Answer from retrieved memory only.",
        )

        chat_request = next(
            request
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        )
        chat_prompt = "\n".join(message.content for message in chat_request.messages)
        assert "This prior chat message must not reach" not in chat_prompt
        assert [message.role for message in chat_request.messages] == ["system", "user"]
        assert chat_request.messages[-1].content == "Answer from retrieved memory only."
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_cache_hit_exposes_observability_without_retrieval_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
        assert [entry.text for entry in second.recent_transcript] == [
            "Please help me debug this retry loop."
        ]
        assert "continue" not in [entry.text for entry in second.recent_transcript]
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
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
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
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

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
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
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
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
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
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        engine.runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
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
            assert stored_messages[-1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
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
async def test_engine_ingest_message_is_idempotent_by_message_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest-idempotent.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert [message["id"] for message in stored_messages] == ["aurvek:msg:1"]
        finally:
            await connection.close()
        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction") == 1
        assert purposes.count("contract_projection") == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_source_seq_preserves_backfill_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest-source-seq.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="First message.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
            source_seq=1,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Third message arrived before retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:3",
            source_seq=3,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Second message retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:2",
            source_seq=2,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Second message retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:2",
            source_seq=2,
        )

        with pytest.raises(SourceSequenceConflictError, match="source_seq already exists"):
            await engine.ingest_message(
                user_id="usr_1",
                conversation_id="cnv_1",
                role="user",
                text="Different message for occupied source seq.",
                mode="coding_debug",
                platform_id="aurvek",
                message_id="aurvek:msg:other",
                source_seq=2,
            )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert [(message["id"], message["seq"]) for message in stored_messages] == [
                ("aurvek:msg:1", 1),
                ("aurvek:msg:2", 2),
                ("aurvek:msg:3", 3),
            ]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_message_id_conflict_rejects_incompatible_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-message-id-conflict.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Original content.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )

        with pytest.raises(MessageIdConflictError, match="different role or text"):
            await engine.ingest_message(
                user_id="usr_1",
                conversation_id="cnv_1",
                role="user",
                text="Changed content.",
                mode="coding_debug",
                platform_id="aurvek",
                message_id="aurvek:msg:1",
            )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_marks_large_plain_text_skip_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-heavy-message.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        skip_compaction=True,
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        huge_text = "large biography segment " * 800
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text=huge_text,
        )

        assert await engine.flush(timeout_seconds=5.0) is True

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            stored = stored_messages[-1]
            assert stored["text"] == huge_text
            assert stored["include_raw"] == 0
            assert stored["skip_by_default"] == 1
            assert stored["heavy_content"] == 1
            assert stored["requires_explicit_request"] == 1
            assert stored["policy_reason"] == "mechanical_size_threshold"
            assert "large biography segment" not in stored["context_placeholder"]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_rolls_back_user_message_when_scoring_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingEngineProvider("applicability_scoring")
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        # Seed a memory so the shortlist is non-empty and scoring is invoked.
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            await memories.create_memory_object(
                user_id="usr_1",
                workspace_id=None,
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="retry loop websocket backoff",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
            )
        finally:
            await connection.close()

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


@pytest.mark.asyncio
async def test_engine_get_context_degrades_when_need_detector_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingEngineProvider("need_detection")
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(db_path=":memory:", openai_api_key="test-openai-key", llm_forced_global_model="openai/test-model")

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        # Need detector failure should no longer break retrieval. The pipeline
        # falls back to the base search and the user message is persisted.
        result = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        assert result is not None

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert [row["text"] for row in stored] == [
                "Please help me debug this retry loop.",
            ]
        finally:
            await connection.close()
    finally:
        await engine.close()
