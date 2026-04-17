"""Tests for embedding-backed candidate search."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus, RetrievalPlan
from atagia.services.embeddings import EmbeddingMatch

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class FakeEmbeddingIndex:
    vector_limit = 1

    def __init__(self, matches: list[EmbeddingMatch]) -> None:
        self.matches = list(matches)
        self.calls: list[tuple[str, str, int]] = []

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise AssertionError("upsert() is not used in candidate search tests")

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        self.calls.append((query, user_id, top_k))
        return list(self.matches)

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used in candidate search tests")


async def _build_runtime(matches: list[EmbeddingMatch]):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    embedding_index = FakeEmbeddingIndex(matches)
    search = CandidateSearch(connection, clock, embedding_index=embedding_index)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await workspaces.create_workspace("wrk_3", "usr_1", "Secondary Workspace")
    await workspaces.create_workspace("wrk_2", "usr_2", "Other Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", "wrk_2", "coding_debug", "User Two")
    return connection, memories, search, embedding_index


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    user_id: str,
    canonical_text: str,
    scope: MemoryScope,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    assistant_mode_id: str | None = "coding_debug",
    temporal_type: str = "unknown",
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> None:
    await memories.create_memory_object(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=MemoryObjectType.EVIDENCE,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        temporal_type=temporal_type,
        valid_from=valid_from,
        valid_to=valid_to,
        memory_id=memory_id,
    )


def _plan(
    *,
    vector_limit: int = 5,
    max_candidates: int = 10,
    fts_queries: list[str] | None = None,
) -> RetrievalPlan:
    resolved_fts_queries = fts_queries or ["retry loop"]
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


@pytest.mark.asyncio
async def test_search_by_embedding_returns_candidates_with_similarity_scores() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [EmbeddingMatch(memory_id="mem_1", score=0.88)]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            user_id="usr_1",
            canonical_text="Retry loop workaround",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert len(candidates) == 1
        assert candidates[0]["id"] == "mem_1"
        assert candidates[0]["similarity_score"] == pytest.approx(0.88)
        assert candidates[0]["position_rank"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_filters_by_user_and_scope() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_ok", score=0.9, position_rank=1),
            EmbeddingMatch(memory_id="mem_other_user", score=0.8, position_rank=2),
            EmbeddingMatch(memory_id="mem_other_workspace", score=0.7, position_rank=3),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_ok",
            user_id="usr_1",
            canonical_text="Retry loop workaround",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_user",
            user_id="usr_2",
            canonical_text="Other user memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_2",
            conversation_id="cnv_2",
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_workspace",
            user_id="usr_1",
            canonical_text="Wrong workspace memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_3",
            conversation_id=None,
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_ok"]
        assert candidates[0]["position_rank"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_returns_empty_when_vector_limit_is_zero() -> None:
    connection, _memories, search, embedding_index = await _build_runtime(
        [EmbeddingMatch(memory_id="mem_1", score=0.88)]
    )
    try:
        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=0),
            embedding_index=embedding_index,
        )

        assert candidates == []
        assert embedding_index.calls == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_preserves_backend_rank_after_plan_filtering() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_filtered", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_survivor", score=0.91, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_filtered",
            user_id="usr_1",
            canonical_text="Filtered workspace memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_3",
            conversation_id=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_survivor",
            user_id="usr_1",
            canonical_text="Retry loop survivor",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_survivor"]
        assert candidates[0]["position_rank"] == 2
        assert candidates[0]["embedding_position_rank"] == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_enforces_vector_limit_after_overfetch() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_2", score=0.91, position_rank=2),
            EmbeddingMatch(memory_id="mem_3", score=0.90, position_rank=3),
        ]
    )
    try:
        for index in range(1, 4):
            await _seed_memory(
                memories,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                canonical_text=f"retry loop candidate {index}",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
            )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=1),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_overfetches_and_demotes_stale_ephemeral_hits() -> None:
    connection, memories, search, embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_stale_ephemeral", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_current", score=0.91, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_stale_ephemeral",
            user_id="usr_1",
            canonical_text="retry loop stale state",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            temporal_type="ephemeral",
            valid_from="2026-04-03T09:00:00+00:00",
        )
        await _seed_memory(
            memories,
            memory_id="mem_current",
            user_id="usr_1",
            canonical_text="retry loop durable memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            temporal_type="permanent",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=1),
            embedding_index=embedding_index,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_current"]
        assert embedding_index.calls == [("retry loop", "usr_1", 2)]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_merged_fts_and_embedding_search_deduplicates_correctly() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_2", score=0.82, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            user_id="usr_1",
            canonical_text="retry loop fastapi websocket",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_2",
            user_id="usr_1",
            canonical_text="Fallback memory for semantic match",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(_plan(), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_1", "mem_2"]
        assert [candidate["id"] for candidate in candidates].count("mem_1") == 1
        assert candidates[0]["retrieval_sources"] == ["fts", "embedding"]
        assert "fts_rank" in candidates[0]
        assert candidates[0]["embedding_similarity_score"] == pytest.approx(0.95)
        assert "embedding_distance" in candidates[0]
        assert candidates[1]["retrieval_sources"] == ["embedding"]
        assert candidates[0]["rrf_score"] > candidates[1]["rrf_score"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_merged_search_respects_max_candidates_cap() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.91),
            EmbeddingMatch(memory_id="mem_2", score=0.88),
            EmbeddingMatch(memory_id="mem_3", score=0.87),
        ]
    )
    try:
        for index in range(1, 4):
            await _seed_memory(
                memories,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                canonical_text=f"semantic memory {index}",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
            )

            candidates = await search.search(_plan(vector_limit=5, max_candidates=2), "usr_1")

        assert len(candidates) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rrf_uses_best_fts_rank_once_across_query_rewrites() -> None:
    connection, memories, search, _embedding_index = await _build_runtime([])
    try:
        await _seed_memory(
            memories,
            memory_id="mem_shared",
            user_id="usr_1",
            canonical_text="retry loop fastapi websocket",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_broad",
            user_id="usr_1",
            canonical_text="retry loop fallback",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(
            _plan(fts_queries=["retry loop fastapi", "retry loop"], vector_limit=0),
            "usr_1",
        )

        shared = next(candidate for candidate in candidates if candidate["id"] == "mem_shared")

        assert shared["retrieval_sources"] == ["fts"]
        assert shared["channel_ranks"] == {
            "fts": 1,
            "embedding": None,
            "consequence": None,
            "raw_message": None,
        }
        assert shared["matched_sub_queries"] == ["retry loop fastapi"]
        assert shared["subquery_ranks"]["retry loop fastapi"] >= 1
        assert shared["rrf_score_raw"] == pytest.approx(
            1.0 / (60 + int(shared["position_rank"]))
        )
        assert shared["rrf_score"] == pytest.approx(
            shared["rrf_score_raw"] / (1.0 / 61.0)
        )
    finally:
        await connection.close()


def test_rank_fusion_prefers_exact_single_hit_for_default_queries() -> None:
    search = CandidateSearch(None, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))

    _raw_exact, exact_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=3,
        query_type="default",
    )
    _raw_generic, generic_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1, 2],
        max_lists=3,
        query_type="default",
    )

    assert exact_score > generic_score


def test_rank_fusion_keeps_coverage_bias_for_broad_list_queries() -> None:
    search = CandidateSearch(None, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))

    _raw_exact, exact_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=3,
        query_type="broad_list",
    )
    _raw_generic, generic_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1, 2],
        max_lists=3,
        query_type="broad_list",
    )

    assert generic_score > exact_score
