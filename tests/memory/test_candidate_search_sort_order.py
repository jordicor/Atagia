"""Tests for CandidateSearch._sort_candidates: relevance-first pool ordering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.retrieval_planner import build_retrieval_fts_queries
from atagia.models.schemas_memory import (
    MemoryScope,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)
from atagia.memory.token_document_frequency import TokenDocumentFrequencyCache

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)


def _plan(*, retrieval_levels: list[int] | None = None) -> RetrievalPlan:
    sub_query = "pottery studio"
    return RetrievalPlan(
        original_query=sub_query,
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=build_retrieval_fts_queries(sub_query),
        sub_query_plans=[
            PlannedSubQuery(
                text=sub_query,
                fts_queries=build_retrieval_fts_queries(sub_query),
            )
        ],
        query_type="default",
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=retrieval_levels or [0],
        temporal_query_range=None,
        require_evidence_regrounding=False,
        skip_retrieval=False,
    )


def _candidate(
    *,
    memory_id: str,
    rrf_score: float,
    hierarchy_level: int | None = None,
    updated_at: str,
) -> dict[str, Any]:
    is_summary = hierarchy_level is not None
    return {
        "id": memory_id,
        "object_type": "summary_view" if is_summary else "evidence",
        "payload_json": {"hierarchy_level": hierarchy_level} if is_summary else {},
        "scope": "global_user",
        "temporal_type": "persistent",
        "rrf_score": rrf_score,
        "updated_at": updated_at,
    }


@pytest.mark.asyncio
async def test_sort_candidates_relevance_beats_level_bucket() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        search = CandidateSearch(
            connection,
            FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
            token_document_frequency_cache=TokenDocumentFrequencyCache(),
        )
        plan = _plan(retrieval_levels=[0, 1, 2])
        strong_summary = _candidate(
            memory_id="mem_l2",
            rrf_score=0.95,
            hierarchy_level=2,
            updated_at="2026-03-01T00:00:00+00:00",
        )
        weak_fact = _candidate(
            memory_id="mem_l0",
            rrf_score=0.10,
            updated_at="2026-03-02T00:00:00+00:00",
        )

        ordered = search._sort_candidates([weak_fact, strong_summary], plan)

        assert [candidate["id"] for candidate in ordered] == ["mem_l2", "mem_l0"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sort_candidates_level_breaks_relevance_tie() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        search = CandidateSearch(
            connection,
            FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
            token_document_frequency_cache=TokenDocumentFrequencyCache(),
        )
        plan = _plan(retrieval_levels=[0, 1, 2])
        summary = _candidate(
            memory_id="mem_l1",
            rrf_score=0.50,
            hierarchy_level=1,
            updated_at="2026-03-01T00:00:00+00:00",
        )
        fact = _candidate(
            memory_id="mem_l0",
            rrf_score=0.50,
            updated_at="2026-03-02T00:00:00+00:00",
        )

        ordered = search._sort_candidates([summary, fact], plan)

        assert [candidate["id"] for candidate in ordered] == ["mem_l0", "mem_l1"]
    finally:
        await connection.close()
