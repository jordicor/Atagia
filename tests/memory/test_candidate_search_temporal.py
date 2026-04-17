"""Tests for temporal-aware candidate search ordering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalPlan,
    TemporalQueryRange,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    return connection, memories, search


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    temporal_type: str,
    valid_from: str | None,
    valid_to: str | None,
    canonical_text: str = "Tokyo trip planning details",
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
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
    max_candidates: int,
    fts_queries: list[str] | None = None,
    temporal_query_range: TemporalQueryRange | None = None,
    include_temporal_query_range: bool = True,
) -> RetrievalPlan:
    resolved_temporal_query_range = temporal_query_range
    if resolved_temporal_query_range is None and include_temporal_query_range:
        resolved_temporal_query_range = TemporalQueryRange(
            start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 30, 23, 59, 59, 999999, tzinfo=timezone.utc),
        )
    resolved_fts_queries = fts_queries or ["tokyo trip"]
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
        vector_limit=0,
        max_candidates=max_candidates,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
        temporal_query_range=resolved_temporal_query_range,
        consequence_search_enabled=False,
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=False,
    )


@pytest.mark.asyncio
async def test_temporal_query_prefers_overlap_and_unknown_before_non_overlap() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_overlap",
            temporal_type="bounded",
            valid_from="2026-04-10T00:00:00+00:00",
            valid_to="2026-04-20T23:59:59.999999+00:00",
        )
        await _seed_memory(
            memories,
            memory_id="mem_unknown",
            temporal_type="unknown",
            valid_from=None,
            valid_to=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_non_overlap",
            temporal_type="bounded",
            valid_from="2026-06-01T00:00:00+00:00",
            valid_to="2026-06-30T23:59:59.999999+00:00",
        )

        candidates = await search.search(_plan(max_candidates=2), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_overlap", "mem_unknown"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_temporal_query_keeps_non_overlap_only_after_overlap_and_unknown() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_overlap",
            temporal_type="bounded",
            valid_from="2026-04-10T00:00:00+00:00",
            valid_to="2026-04-20T23:59:59.999999+00:00",
        )
        await _seed_memory(
            memories,
            memory_id="mem_unknown",
            temporal_type="unknown",
            valid_from=None,
            valid_to=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_non_overlap",
            temporal_type="bounded",
            valid_from="2026-06-01T00:00:00+00:00",
            valid_to="2026-06-30T23:59:59.999999+00:00",
        )

        candidates = await search.search(_plan(max_candidates=3), "usr_1")

        assert [candidate["id"] for candidate in candidates] == [
            "mem_overlap",
            "mem_unknown",
            "mem_non_overlap",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_temporal_query_treats_fresh_ephemeral_as_overlap() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_ephemeral",
            temporal_type="ephemeral",
            valid_from="2026-04-10T10:00:00+00:00",
            valid_to=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_non_overlap",
            temporal_type="bounded",
            valid_from="2026-06-01T00:00:00+00:00",
            valid_to="2026-06-30T23:59:59.999999+00:00",
        )

        candidates = await search.search(
            _plan(
                max_candidates=1,
                temporal_query_range=TemporalQueryRange(
                    start=datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 4, 10, 23, 59, 59, 999999, tzinfo=timezone.utc),
                ),
            ),
            "usr_1",
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_ephemeral"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_temporal_query_demotes_expired_ephemeral_after_horizon() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_ephemeral",
            temporal_type="ephemeral",
            valid_from="2026-04-01T00:00:00+00:00",
            valid_to=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_unknown",
            temporal_type="unknown",
            valid_from=None,
            valid_to=None,
        )

        candidates = await search.search(
            _plan(
                max_candidates=1,
                temporal_query_range=TemporalQueryRange(
                    start=datetime(2026, 4, 3, 0, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 4, 3, 23, 59, 59, 999999, tzinfo=timezone.utc),
                ),
            ),
            "usr_1",
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_unknown"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_non_temporal_search_demotes_stale_ephemeral_before_final_truncation() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_stale_ephemeral",
            temporal_type="ephemeral",
            valid_from="2026-04-02T09:00:00+00:00",
            valid_to=None,
            canonical_text="airport",
        )
        await _seed_memory(
            memories,
            memory_id="mem_permanent",
            temporal_type="permanent",
            valid_from=None,
            valid_to=None,
            canonical_text="airport checklist details",
        )

        candidates = await search.search(
            _plan(
                max_candidates=1,
                fts_queries=["airport"],
                include_temporal_query_range=False,
            ),
            "usr_1",
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_permanent"]
    finally:
        await connection.close()
