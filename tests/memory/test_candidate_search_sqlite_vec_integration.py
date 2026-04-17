"""Integration tests for CandidateSearch using the real sqlite-vec backend."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus, RetrievalPlan
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMProvider,
)
from atagia.services.sqlite_vec_backend import SQLiteVecBackend

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
HAS_SQLITE_VEC = find_spec("sqlite_vec") is not None


class DeterministicEmbeddingProvider(LLMProvider):
    name = "candidate-search-sqlite-vec-tests"
    supports_embedding_dimensions = True

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = dict(vectors_by_text)

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError("Completions are not used in sqlite-vec candidate search tests")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[
                LLMEmbeddingVector(index=index, values=self.vectors_by_text[text])
                for index, text in enumerate(request.input_texts)
            ],
        )


def _settings(database_path: str) -> Settings:
    return Settings(
        sqlite_path=database_path,
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
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        embedding_backend="sqlite_vec",
        embedding_model="text-embedding-3-small",
        embedding_dimension=2,
    )


def _plan(*, vector_limit: int = 10, max_candidates: int = 10) -> RetrievalPlan:
    resolved_fts_queries = ["retry websocket"]
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=resolved_fts_queries,
        sub_query_plans=[
            {
                "text": resolved_fts_queries[0],
                "fts_queries": resolved_fts_queries,
            }
        ],
        query_type="default",
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.WORKSPACE, MemoryScope.ASSISTANT_MODE],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=vector_limit,
        max_candidates=max_candidates,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=False,
    )


async def _seed_memory(
    memories: MemoryObjectRepository,
    backend: SQLiteVecBackend,
    *,
    memory_id: str,
    user_id: str,
    canonical_text: str,
    workspace_id: str,
    conversation_id: str,
) -> None:
    await memories.create_memory_object(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id=memory_id,
    )
    await backend.upsert(
        memory_id,
        canonical_text,
        {
            "user_id": user_id,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "scope": MemoryScope.CONVERSATION.value,
            "created_at": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc).isoformat(),
        },
    )


async def _build_runtime(tmp_path: Path, vectors_by_text: dict[str, list[float]]):
    database_path = str(tmp_path / "candidate-search-sqlite-vec.db")
    search_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    embedding_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(search_connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(search_connection, clock)
    workspaces = WorkspaceRepository(search_connection, clock)
    conversations = ConversationRepository(search_connection, clock)
    memories = MemoryObjectRepository(search_connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await workspaces.create_workspace("wrk_2", "usr_2", "Other Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", "wrk_2", "coding_debug", "User Two")
    backend = SQLiteVecBackend(
        embedding_connection,
        LLMClient(
            provider_name=DeterministicEmbeddingProvider.name,
            providers=[DeterministicEmbeddingProvider(vectors_by_text)],
        ),
        _settings(database_path),
    )
    await backend.initialize()
    search = CandidateSearch(search_connection, clock, embedding_index=backend)
    return search_connection, embedding_connection, memories, search, backend


pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec is not installed")


@pytest.mark.asyncio
async def test_candidate_search_returns_embedding_matches_via_real_sqlite_vec(tmp_path: Path) -> None:
    search_connection, embedding_connection, memories, search, backend = await _build_runtime(
        tmp_path,
        {
            "automatic reconnect fallback": [1.0, 0.0],
            "retry websocket": [1.0, 0.0],
        },
    )
    try:
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_local",
            user_id="usr_1",
            canonical_text="automatic reconnect fallback",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(_plan(), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_local"]
        assert candidates[0]["retrieval_sources"] == ["embedding"]
    finally:
        await search_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_candidate_search_keeps_user_b_neighbors_out_of_user_a_results(tmp_path: Path) -> None:
    search_connection, embedding_connection, memories, search, backend = await _build_runtime(
        tmp_path,
        {
            "user a retry memory": [0.9, 0.1],
            "user b retry memory": [1.0, 0.0],
            "retry websocket": [1.0, 0.0],
        },
    )
    try:
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_user_a",
            user_id="usr_1",
            canonical_text="user a retry memory",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_user_b",
            user_id="usr_2",
            canonical_text="user b retry memory",
            workspace_id="wrk_2",
            conversation_id="cnv_2",
        )

        candidates = await search.search(_plan(), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_user_a"]
    finally:
        await search_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_candidate_search_overfetch_survives_cross_user_starvation_pressure(tmp_path: Path) -> None:
    vectors_by_text = {"retry websocket": [1.0, 0.0], "usr_1 survivor": [0.9, 0.1]}
    for index in range(80):
        vectors_by_text[f"usr_2 outranker {index}"] = [1.0, 0.0]
    search_connection, embedding_connection, memories, search, backend = await _build_runtime(
        tmp_path,
        vectors_by_text,
    )
    try:
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_survivor",
            user_id="usr_1",
            canonical_text="usr_1 survivor",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        for index in range(80):
            await _seed_memory(
                memories,
                backend,
                memory_id=f"mem_outranker_{index}",
                user_id="usr_2",
                canonical_text=f"usr_2 outranker {index}",
                workspace_id="wrk_2",
                conversation_id="cnv_2",
            )

        candidates = await search.search(_plan(vector_limit=10), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_survivor"]
        assert candidates[0]["retrieval_sources"] == ["embedding"]
    finally:
        await search_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_candidate_search_rrf_deduplicates_real_fts_and_embedding_hits(tmp_path: Path) -> None:
    search_connection, embedding_connection, memories, search, backend = await _build_runtime(
        tmp_path,
        {
            "retry websocket exact hit": [1.0, 0.0],
            "semantic-only fallback": [0.8, 0.2],
            "retry websocket": [1.0, 0.0],
        },
    )
    try:
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_both",
            user_id="usr_1",
            canonical_text="retry websocket exact hit",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            backend,
            memory_id="mem_embedding_only",
            user_id="usr_1",
            canonical_text="semantic-only fallback",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(_plan(), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_both", "mem_embedding_only"]
        assert candidates[0]["retrieval_sources"] == ["fts", "embedding"]
        assert candidates[1]["retrieval_sources"] == ["embedding"]
    finally:
        await search_connection.close()
        await embedding_connection.close()
