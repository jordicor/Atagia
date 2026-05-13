"""Tests for MCP server helper logic."""

from __future__ import annotations

# ruff: noqa: E402

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

mcp_available = pytest.importorskip("mcp", reason="mcp package not installed")

from atagia import Atagia
from atagia.core.clock import FrozenClock
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.space_repository import SpaceRepository
from atagia.models.schemas_jobs import EXTRACT_STREAM_NAME, JobEnvelope, WORKER_GROUP_NAME
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    SpaceBoundaryMode,
)
from atagia.mcp_server import (
    _add_memory_impl,
    _delete_conversation_impl,
    _delete_memory_impl,
    _edit_memory_impl,
    _get_context_impl,
    _list_memories_impl,
    _search_memories_impl,
    lifespan,
)
from atagia.services.errors import DeletionConfirmationError, MemoryNotFoundError
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.chat_support import default_operational_profile_snapshot
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


class MCPProvider(LLMProvider):
    name = "mcp-tests"

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
        raise AssertionError(f"Embeddings are not used in MCP tests: {request.model}")


class TrackingEmbeddingIndex(EmbeddingIndex):
    def __init__(self) -> None:
        self.deleted_memory_ids: list[str] = []

    @property
    def vector_limit(self) -> int:
        return 1

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        return None

    async def search(self, query: str, user_id: str, top_k: int):
        return []

    async def delete(self, memory_id: str) -> None:
        self.deleted_memory_ids.append(memory_id)


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: MCPProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    # The MCP tests assert the full retrieval pipeline runs (including need
    # detection), so disable the small-corpus shortcut for the duration of
    # the test.
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


def _normal_operational_profile_token(engine: Atagia) -> str:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    return default_operational_profile_snapshot(
        loader=engine.runtime.operational_profile_loader,
        settings=engine.runtime.settings,
    ).token


async def _seed_memory(
    engine: Atagia,
    *,
    memory_id: str,
    text: str,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    conversation_id: str | None = "cnv_1",
    platform_id: str = "web",
    space_id: str | None = None,
    space_boundary_mode: SpaceBoundaryMode | None = None,
    embodiment_id: str | None = None,
    realm_id: str | None = None,
) -> None:
    runtime = engine.runtime
    if runtime is None:
        raise AssertionError("Engine runtime should be initialized before seeding memories")
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        scope_canonical = {
            MemoryScope.CONVERSATION: MemoryScope.CHAT.value,
            MemoryScope.EPHEMERAL_SESSION: MemoryScope.CHAT.value,
            MemoryScope.WORKSPACE: MemoryScope.CHARACTER.value,
            MemoryScope.GLOBAL_USER: MemoryScope.USER.value,
        }.get(scope)
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id=conversation_id,
            assistant_mode_id="coding_debug",
            object_type=object_type,
            scope=scope,
            canonical_text=text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=privacy_level,
            status=status,
            memory_id=memory_id,
            platform_id=platform_id,
            scope_canonical=scope_canonical,
            space_id=space_id,
            space_boundary_mode=space_boundary_mode.value if space_boundary_mode is not None else None,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_mcp_get_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        first = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "mcp",
                "Please help me debug this retry loop.",
                conversation_id="cnv_1",
                mode="coding_debug",
                user_persona_id="persona_mcp",
                character_id="char_mcp",
            )
        )
        second = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "mcp",
                "continue",
                conversation_id="cnv_1",
                mode="coding_debug",
                user_persona_id="persona_mcp",
                character_id="char_mcp",
            )
        )

        assert first["system_prompt"]
        assert first["conversation_id"] == "cnv_1"
        assert second["conversation_id"] == "cnv_1"
        assert sum(request.metadata.get("purpose") == "need_detection" for request in provider.requests) == 2
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="char_mcp",
            operational_profile_token=_normal_operational_profile_token(engine),
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(connection, runtime.clock).get_conversation(
                "cnv_1",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["user_persona_id"] == "persona_mcp"
            assert conversation["character_id"] == "char_mcp"
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_lifespan_reads_embodiment_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_DB_PATH", str(tmp_path / "atagia-mcp-env.db"))
    monkeypatch.setenv("ATAGIA_USER_ID", "usr_1")
    monkeypatch.setenv("ATAGIA_PLATFORM_ID", "mcp")
    monkeypatch.setenv("ATAGIA_EMBODIMENT_ID", "body_env")
    monkeypatch.setenv("ATAGIA_REALM_ID", "realm_env")

    async with lifespan(None) as context:
        assert context.user_id == "usr_1"
        assert context.platform_id == "mcp"
        assert context.embodiment_id == "body_env"
        assert context.realm_id == "realm_env"


@pytest.mark.asyncio
async def test_mcp_context_and_add_memory_propagate_embodiment_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await _get_context_impl(
            engine,
            "usr_1",
            "mcp",
            "Please help me debug this retry loop.",
            conversation_id="cnv_body",
            mode="coding_debug",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            embodiment_id="body_mcp",
        )
        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that this chat is on the headset.",
            conversation_id="cnv_body",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            embodiment_id="body_mcp",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(connection, runtime.clock).get_conversation(
                "cnv_body",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_embodiment_id"] == "body_mcp"
            messages = await MessageRepository(connection, runtime.clock).get_messages(
                "cnv_body",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert len(messages) >= 2
            assert {message["active_embodiment_id"] for message in messages} == {
                "body_mcp"
            }
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_context_and_add_memory_propagate_realm_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await _get_context_impl(
            engine,
            "usr_1",
            "mcp",
            "Please help me debug this retry loop.",
            conversation_id="cnv_realm",
            mode="coding_debug",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            realm_id="realm_mcp",
        )
        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that this chat is in a story realm.",
            conversation_id="cnv_realm",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            realm_id="realm_mcp",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(connection, runtime.clock).get_conversation(
                "cnv_realm",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_realm_id"] == "realm_mcp"
            messages = await MessageRepository(connection, runtime.clock).get_messages(
                "cnv_realm",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert len(messages) >= 2
            assert {message["active_realm_id"] for message in messages} == {
                "realm_mcp"
            }
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_memory_tools_can_apply_env_embodiment_to_existing_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_env",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        await _seed_memory(
            engine,
            memory_id="mem_body_env",
            text="environmentbodytoken memory belongs to the headset",
            scope=MemoryScope.GLOBAL_USER,
            conversation_id=None,
            platform_id="mcp",
            embodiment_id="body_env",
        )

        before_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_env",
                platform_id="mcp",
            )
        )
        with_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_env",
                platform_id="mcp",
                embodiment_id="body_env",
            )
        )
        search_with_env = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "environmentbodytoken",
                conversation_id="cnv_env",
                platform_id="mcp",
                embodiment_id="body_env",
            )
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(connection, runtime.clock).get_conversation(
                "cnv_env",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_embodiment_id"] == "body_env"
        finally:
            await connection.close()

        assert {memory["id"] for memory in before_env} == set()
        assert {memory["id"] for memory in with_env} == {"mem_body_env"}
        assert {memory["id"] for memory in search_with_env} == {"mem_body_env"}
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_memory_tools_can_apply_env_realm_to_existing_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_realm_env",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        await _seed_memory(
            engine,
            memory_id="mem_realm_env",
            text="environmentrealmtoken memory belongs to the story realm",
            scope=MemoryScope.GLOBAL_USER,
            conversation_id=None,
            platform_id="mcp",
            realm_id="realm_env",
        )

        before_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
            )
        )
        with_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
                realm_id="realm_env",
            )
        )
        search_with_env = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "environmentrealmtoken",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
                realm_id="realm_env",
            )
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(connection, runtime.clock).get_conversation(
                "cnv_realm_env",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_realm_id"] == "realm_env"
        finally:
            await connection.close()

        assert {memory["id"] for memory in before_env} == set()
        assert {memory["id"] for memory in with_env} == {"mem_realm_env"}
        assert {memory["id"] for memory in search_with_env} == {"mem_realm_env"}
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_search_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
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
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="retry loop websocket backoff",
        )
        await _seed_memory(
            engine,
            memory_id="mem_archived",
            text="archived retry memory",
        )
        await _seed_memory(
            engine,
            memory_id="mem_pending",
            text="pending retry credential",
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            engine,
            memory_id="mem_private",
            text="private retry credential",
            privacy_level=2,
        )
        await _seed_memory(
            engine,
            memory_id="mem_declined",
            text="declined retry credential",
            status=MemoryStatus.DECLINED,
        )
        await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_archived",
            conversation_id="cnv_1",
            platform_id="web",
        )

        results = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "retry",
                limit=10,
                conversation_id="cnv_1",
                platform_id="web",
            )
        )

        assert results
        assert results[0]["id"] == "mem_1"
        assert "retry loop websocket backoff" in results[0]["text"]
        assert all(result["id"] != "mem_archived" for result in results)
        assert all(result["id"] != "mem_pending" for result in results)
        assert all(result["id"] != "mem_private" for result in results)
        assert all(result["id"] != "mem_declined" for result in results)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_add_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        engine.runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="default_assistant",
            operational_profile_token=_normal_operational_profile_token(engine),
        )
        await engine.runtime.storage_backend.set_context_view(
            cache_key,
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:other",
            {"user_id": "usr_2", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )

        confirmation = await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that the retry loop needs a backoff.",
            conversation_id="cnv_1",
        )

        assert "Stored memory candidate message" in confirmation
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[-1]["role"] == "user"
            assert stored_messages[-1]["text"] == "Please remember that the retry loop needs a backoff."
            assert stored_messages[-1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
        finally:
            await connection.close()
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:other") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_2",
        }
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_add_memory_carries_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
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
            platform_id="mcp",
        )

        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that offline mode should keep answers short.",
            conversation_id="cnv_1",
            operational_profile="offline",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        messages = await runtime.storage_backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test-consumer",
            count=1,
            block_ms=0,
        )
        if messages:
            envelope = JobEnvelope.model_validate(messages[0].payload)
            assert envelope.operational_profile is not None
            assert envelope.operational_profile.profile_id == "offline"
        else:
            connection = await runtime.open_connection()
            try:
                cursor = await connection.execute(
                    """
                    SELECT metadata_json
                    FROM worker_job_runs
                    WHERE user_id = ?
                      AND conversation_id = ?
                      AND job_type = ?
                    """,
                    ("usr_1", "cnv_1", "extract_memory_candidates"),
                )
                rows = await cursor.fetchall()
            finally:
                await connection.close()
            assert any(
                json.loads(row["metadata_json"]).get("operational_profile") == "offline"
                for row in rows
            )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_delete_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        tracking_embeddings = TrackingEmbeddingIndex()
        if engine.runtime is None:
            raise AssertionError("Engine runtime should be initialized")
        engine.runtime.embedding_index = tracking_embeddings
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="retry loop websocket backoff",
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:1",
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:2",
            {"user_id": "usr_1", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:3",
            {"user_id": "usr_2", "conversation_id": "cnv_3"},
            ttl_seconds=60,
        )

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_1",
            conversation_id="cnv_1",
            platform_id="web",
        )

        assert confirmation == "Archived memory mem_1."
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            memory = await memories.get_memory_object("mem_1", "usr_1")
            assert memory is not None
            assert memory["status"] == "archived"
        finally:
            await connection.close()
        assert tracking_embeddings.deleted_memory_ids == ["mem_1"]
        assert await engine.runtime.storage_backend.get_context_view("ctx:1") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:2") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:3") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_3",
        }
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_destructive_deletes_require_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
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
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="destructive delete confirmation",
        )

        with pytest.raises(DeletionConfirmationError):
            await _delete_memory_impl(
                engine,
                "usr_1",
                "mem_1",
                hard=True,
                conversation_id="cnv_1",
                platform_id="web",
            )
        with pytest.raises(DeletionConfirmationError):
            await _delete_conversation_impl(
                engine,
                "usr_1",
                "cnv_1",
                platform_id="web",
            )

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_1",
            hard=True,
            confirmation="HARD_DELETE_MEMORY",
            conversation_id="cnv_1",
            platform_id="web",
        )
        assert confirmation == "Hard-deleted memory mem_1."
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_list_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
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
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_evidence",
            text="retry loop websocket backoff",
            object_type=MemoryObjectType.EVIDENCE,
        )
        await _seed_memory(
            engine,
            memory_id="mem_belief",
            text="The issue is likely in the retry guard.",
            object_type=MemoryObjectType.BELIEF,
        )
        await _seed_memory(
            engine,
            memory_id="mem_archived",
            text="Old archived memory",
            object_type=MemoryObjectType.EVIDENCE,
        )
        await _seed_memory(
            engine,
            memory_id="mem_pending",
            text="Pending memory",
            object_type=MemoryObjectType.EVIDENCE,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            engine,
            memory_id="mem_private",
            text="Private memory",
            object_type=MemoryObjectType.EVIDENCE,
            privacy_level=2,
        )
        await _seed_memory(
            engine,
            memory_id="mem_declined",
            text="Declined memory",
            object_type=MemoryObjectType.EVIDENCE,
            status=MemoryStatus.DECLINED,
        )
        await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_archived",
            conversation_id="cnv_1",
            platform_id="web",
        )

        all_memories = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_1",
                platform_id="web",
            )
        )
        belief_memories = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                memory_type=MemoryObjectType.BELIEF.value,
                conversation_id="cnv_1",
                platform_id="web",
            )
        )

        assert len(all_memories) == 3
        assert {memory["id"] for memory in all_memories} == {
            "mem_evidence",
            "mem_belief",
            "mem_archived",
        }
        assert next(memory for memory in all_memories if memory["id"] == "mem_archived")["status"] == (
            MemoryStatus.ARCHIVED.value
        )
        assert all(memory["id"] != "mem_pending" for memory in all_memories)
        assert all(memory["id"] != "mem_private" for memory in all_memories)
        assert all(memory["id"] != "mem_declined" for memory in all_memories)
        assert len(belief_memories) == 1
        assert belief_memories[0]["id"] == "mem_belief"
        assert belief_memories[0]["type"] == MemoryObjectType.BELIEF.value
    finally:
        await engine.close()


@pytest.mark.parametrize(
    ("space_id", "boundary_mode"),
    [
        ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
        ("space_severed", SpaceBoundaryMode.SEVERANCE),
    ],
)
@pytest.mark.asyncio
async def test_mcp_memory_tools_enforce_space_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    space_id: str,
    boundary_mode: SpaceBoundaryMode,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should be initialized")
        connection = await runtime.open_connection()
        try:
            await SpaceRepository(connection, runtime.clock).resolve_space(
                owner_user_id="usr_1",
                space_id=space_id,
                boundary_mode=boundary_mode,
                display_name=space_id,
                source_kind="explicit",
                source_id=space_id,
            )
        finally:
            await connection.close()
        await engine.create_conversation(
            "usr_1",
            "cnv_outside",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await engine.create_conversation(
            "usr_1",
            "cnv_inside",
            assistant_mode_id="coding_debug",
            platform_id="web",
            space_id=space_id,
        )
        for operation in ("list", "edit", "delete"):
            await _seed_memory(
                engine,
                memory_id=f"mem_{space_id}_{operation}",
                text=f"{operation} mcp boundary token inside {space_id}",
                scope=MemoryScope.GLOBAL_USER,
                conversation_id=None,
                platform_id="web",
                space_id=space_id,
                space_boundary_mode=boundary_mode,
            )

        outside_list = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        )
        outside_search = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "boundary",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        )
        outside_ids = {memory["id"] for memory in outside_list}
        outside_search_ids = {memory["id"] for memory in outside_search}
        assert f"mem_{space_id}_list" not in outside_ids
        assert f"mem_{space_id}_list" not in outside_search_ids

        with pytest.raises(MemoryNotFoundError):
            await _edit_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_edit",
                "Outside MCP edit must not land.",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        with pytest.raises(MemoryNotFoundError):
            await _delete_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_delete",
                conversation_id="cnv_outside",
                platform_id="web",
            )

        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            edit_memory = await memories.get_memory_object(f"mem_{space_id}_edit", "usr_1")
            delete_memory = await memories.get_memory_object(f"mem_{space_id}_delete", "usr_1")
            assert edit_memory is not None
            assert delete_memory is not None
            assert edit_memory["canonical_text"] == f"edit mcp boundary token inside {space_id}"
            assert delete_memory["status"] == MemoryStatus.ACTIVE.value
        finally:
            await connection.close()

        inside_list = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        inside_search = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "boundary",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        inside_ids = {memory["id"] for memory in inside_list}
        inside_search_ids = {memory["id"] for memory in inside_search}
        assert f"mem_{space_id}_list" in inside_ids
        assert f"mem_{space_id}_list" in inside_search_ids

        edited_payload = json.loads(
            await _edit_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_edit",
                "Inside MCP edit is allowed.",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        assert edited_payload == {
            "id": f"mem_{space_id}_edit",
            "canonical_text": "Inside MCP edit is allowed.",
        }

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            f"mem_{space_id}_delete",
            conversation_id="cnv_inside",
            platform_id="web",
        )
        assert confirmation == f"Archived memory mem_{space_id}_delete."
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            archived_memory = await memories.get_memory_object(f"mem_{space_id}_delete", "usr_1")
            assert archived_memory is not None
            assert archived_memory["status"] == MemoryStatus.ARCHIVED.value
        finally:
            await connection.close()
    finally:
        await engine.close()
