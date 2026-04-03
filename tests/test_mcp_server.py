"""Tests for MCP server helper logic."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

mcp_available = pytest.importorskip("mcp", reason="mcp package not installed")

from atagia import Atagia
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.mcp_server import (
    _add_memory_impl,
    _delete_memory_impl,
    _get_context_impl,
    _list_memories_impl,
    _search_memories_impl,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


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


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: MCPProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )


async def _seed_memory(
    engine: Atagia,
    *,
    memory_id: str,
    text: str,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
) -> None:
    runtime = engine.runtime
    if runtime is None:
        raise AssertionError("Engine runtime should be initialized before seeding memories")
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=object_type,
            scope=MemoryScope.CONVERSATION,
            canonical_text=text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id=memory_id,
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
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        first = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "Please help me debug this retry loop.",
                conversation_id="cnv_1",
                mode="coding_debug",
            )
        )
        second = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "continue",
                conversation_id="cnv_1",
                mode="coding_debug",
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
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
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
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation("usr_1", "cnv_1", assistant_mode_id="coding_debug")
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
        await _delete_memory_impl(engine, "usr_1", "mem_archived")

        results = json.loads(await _search_memories_impl(engine, "usr_1", "retry", limit=10))

        assert results
        assert results[0]["id"] == "mem_1"
        assert "retry loop websocket backoff" in results[0]["text"]
        assert all(result["id"] != "mem_archived" for result in results)
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
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation("usr_1", "cnv_1", assistant_mode_id="coding_debug")
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
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
async def test_mcp_delete_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation("usr_1", "cnv_1", assistant_mode_id="coding_debug")
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

        confirmation = await _delete_memory_impl(engine, "usr_1", "mem_1")

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
        assert await engine.runtime.storage_backend.get_context_view("ctx:1") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:2") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:3") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_3",
        }
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
        llm_provider="openai",
        llm_api_key="test-openai-key",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation("usr_1", "cnv_1", assistant_mode_id="coding_debug")
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
        await _delete_memory_impl(engine, "usr_1", "mem_archived")

        all_memories = json.loads(await _list_memories_impl(engine, "usr_1"))
        belief_memories = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                memory_type=MemoryObjectType.BELIEF.value,
            )
        )

        assert len(all_memories) == 2
        assert {memory["id"] for memory in all_memories} == {"mem_evidence", "mem_belief"}
        assert all(memory["id"] != "mem_archived" for memory in all_memories)
        assert len(belief_memories) == 1
        assert belief_memories[0]["id"] == "mem_belief"
        assert belief_memories[0]["type"] == MemoryObjectType.BELIEF.value
    finally:
        await engine.close()
