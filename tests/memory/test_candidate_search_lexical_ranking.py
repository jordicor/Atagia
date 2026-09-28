"""Tests for lexical ranking: bm25-first SQL order, DF token cleanup, RRF kind weights."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.memory.retrieval_planner import (
    build_retrieval_fts_queries,
    filter_fts_query_tokens,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    SummaryViewKind,
    TemporalQueryRange,
)
from atagia.memory.token_document_frequency import (
    TOKEN_DOC_RATIO_MAX_RATIO,
    TokenDocumentFrequencyCache,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"


async def _build_candidate_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(
        connection,
        clock,
        token_document_frequency_cache=TokenDocumentFrequencyCache(),
    )
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "User Two")
    return connection, messages, memories, search


def _plan(
    *,
    sub_query: str,
    scope_filter: list[MemoryScope],
    max_candidates: int = 10,
    query_type: str = "default",
    retrieval_levels: list[int] | None = None,
    temporal_query_range: TemporalQueryRange | None = None,
) -> RetrievalPlan:
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
        query_type=query_type,
        scope_filter=scope_filter,
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=max_candidates,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=retrieval_levels or [0],
        temporal_query_range=temporal_query_range,
        require_evidence_regrounding=False,
        skip_retrieval=False,
    )


async def _create_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    scope: MemoryScope = MemoryScope.GLOBAL_USER,
    conversation_id: str | None = None,
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=scope,
        conversation_id=conversation_id,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        memory_id=memory_id,
    )


def test_filter_drops_corpus_ubiquitous_tokens_from_and_query() -> None:
    assert (
        filter_fts_query_tokens(
            "what is the name",
            token_doc_ratios={"what": 0.9, "is": 0.8, "the": 0.95, "name": 0.1},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == "name"
    )


def test_filter_leaves_or_query_untouched_with_corpus_stats() -> None:
    # OR queries are exempt from the DF filter: each OR term is an independent
    # match path (often the protagonist's name in small corpora), and noisy-OR
    # cost is handled downstream by the RRF down-weight instead.
    assert (
        filter_fts_query_tokens(
            "what OR is OR pottery",
            token_doc_ratios={"what": 0.9, "is": 0.8, "pottery": 0.05},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == "what OR is OR pottery"
    )


def test_filter_returns_empty_when_every_token_is_ubiquitous() -> None:
    assert (
        filter_fts_query_tokens(
            "what is the",
            token_doc_ratios={"what": 0.9, "is": 0.8, "the": 0.95},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == ""
    )


def test_filter_keeps_phrase_and_prefix_queries_unchanged() -> None:
    assert (
        filter_fts_query_tokens(
            '"exact phrase"',
            token_doc_ratios={"exact": 0.9},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == '"exact phrase"'
    )
    assert (
        filter_fts_query_tokens(
            "alph* bet*",
            token_doc_ratios={"alph": 0.9},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == "alph* bet*"
    )


def test_filter_with_empty_corpus_mapping_leaves_or_untouched() -> None:
    assert (
        filter_fts_query_tokens(
            "a OR bb OR ccc",
            token_doc_ratios={},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == "a OR bb OR ccc"
    )


def test_filter_with_empty_corpus_mapping_leaves_query_untouched() -> None:
    assert (
        filter_fts_query_tokens(
            "what is the",
            token_doc_ratios={},
            max_doc_ratio=TOKEN_DOC_RATIO_MAX_RATIO,
        )
        == "what is the"
    )


@pytest.mark.asyncio
async def test_bm25_rank_precedes_scope_buckets_in_fts_sql_order() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await _create_memory(
            memories,
            memory_id="mem_conv",
            canonical_text="zebra apple banana grape melon kiwi plum peach apricot",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
        )
        await _create_memory(
            memories,
            memory_id="mem_global",
            canonical_text="zebra zebra zebra zebra zebra",
        )

        candidates = await search.search(
            _plan(
                sub_query="zebra",
                scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
            ),
            user_id="usr_1",
        )

        fts_ranks = {
            str(candidate["id"]): candidate["channel_ranks"]["fts"]
            for candidate in candidates
        }
        # The conversation-scope bucket sorts first in SQL CASE terms, but
        # bm25 rank leads the ORDER BY, so the stronger lexical match takes
        # rank 1 despite sitting in the later scope bucket.
        assert fts_ranks["mem_global"] == 1
        assert fts_ranks["mem_conv"] == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_bm25_first_order_keeps_top_lexical_match_under_limit() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await _create_memory(
            memories,
            memory_id="mem_conv_1",
            canonical_text="zebra apple banana grape melon kiwi plum peach apricot",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
        )
        await _create_memory(
            memories,
            memory_id="mem_conv_2",
            canonical_text="zebra lemon cherry fig date elderberry guava papaya",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
        )
        await _create_memory(
            memories,
            memory_id="mem_global",
            canonical_text="zebra zebra zebra zebra zebra",
        )
        # A temporal range disables the overfetch multiplier, so the SQL
        # LIMIT equals max_candidates and bucket-first ordering would cut
        # the global-scope memory before ranking ever sees it.
        plan = _plan(
            sub_query="zebra",
            scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
            max_candidates=2,
            temporal_query_range=TemporalQueryRange(
                start=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
            ),
        )

        candidates = await search.search(plan, user_id="usr_1")

        candidate_ids = [str(candidate["id"]) for candidate in candidates]
        assert "mem_global" in candidate_ids
        assert len(candidate_ids) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_df_filtered_and_query_avoids_zero_row_collapse_and_or_noise() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        for index in range(11):
            await _create_memory(
                memories,
                memory_id=f"mem_filler_{index}",
                canonical_text=f"what is the name of entry number {index}",
            )
        await _create_memory(
            memories,
            memory_id="mem_pottery",
            canonical_text="the pottery studio is called clay haven",
        )
        plan = _plan(
            sub_query="What is the name of the pottery studio?",
            scope_filter=[MemoryScope.GLOBAL_USER],
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        candidate_ids = [str(candidate["id"]) for candidate in candidates]
        # The precise memory ranks first; OR-only fillers follow down-weighted.
        assert candidate_ids[0] == "mem_pottery"
        memory_lane_entries = [
            entry for entry in fts_query_audit if entry.get("source") is None
        ]
        # The stopword-shaped AND variants carry no informative token, so they
        # are skipped; the OR fallback executes UNFILTERED (recall first) and
        # still finds the precise row on top via the precise-token matches.
        assert [entry["query"] for entry in memory_lane_entries] == [
            "what OR is OR the OR name OR of OR pottery OR studio"
        ]
        assert memory_lane_entries[0]["raw_rows"] == 12
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_broad_or_only_matches_get_reduced_rrf_weight() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await _create_memory(
            memories,
            memory_id="mem_precise",
            canonical_text="alpha beta gamma report",
        )
        await _create_memory(
            memories,
            memory_id="mem_broad_only",
            canonical_text="alpha notes",
        )

        candidates = await search.search(
            _plan(sub_query="alpha beta gamma", scope_filter=[MemoryScope.GLOBAL_USER]),
            user_id="usr_1",
        )

        by_id = {str(candidate["id"]): candidate for candidate in candidates}
        assert set(by_id) == {"mem_precise", "mem_broad_only"}
        precise = by_id["mem_precise"]
        broad_only = by_id["mem_broad_only"]
        match_modes = {
            str(match.get("match_mode"))
            for match in broad_only.get("fts_query_matches", [])
        }
        assert match_modes == {"explicit_or"}
        # A rank from the noisy broad OR counts less than a precise-AND
        # rank, but the candidate still contributes (weight is not zero).
        assert broad_only["rrf_score"] > 0.0
        assert broad_only["rrf_score"] < 0.75 * precise["rrf_score"]
        assert candidates[0]["id"] == "mem_precise"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_missing_position_rank_falls_back_to_enumeration_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection, _messages, _memories, search = await _build_candidate_runtime()
    try:
        aggregated: dict[str, dict[str, object]] = {}
        plan = _plan(sub_query="alpha", scope_filter=[MemoryScope.GLOBAL_USER])
        with caplog.at_level(logging.WARNING, logger="atagia.memory.candidate_search"):
            search._merge_channel_candidates(
                aggregated,
                [{"id": "mem_x"}, {"id": "mem_y"}],
                channel="fts",
                plan=plan,
            )

        assert aggregated["mem_x"]["channel_ranks"]["fts"] == 1
        assert aggregated["mem_y"]["channel_ranks"]["fts"] == 2
        warnings = [
            record for record in caplog.records if "missing position_rank" in record.message
        ]
        assert len(warnings) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_episode_and_theme_levels_become_candidates_for_broad_queries() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await _create_memory(
            memories,
            memory_id="mem_fact",
            canonical_text="The pottery budget was approved at 2800 dollars.",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_episode_1",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=1,
            summary_text="Episode: the pottery budget discussion settled on 2800 dollars.",
            source_object_ids=["mem_fact"],
            created_at="2026-03-30T20:00:00+00:00",
            index_text="pottery budget episode",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            payload={},
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_theme_1",
            summary_kind=SummaryViewKind.THEMATIC_PROFILE,
            hierarchy_level=2,
            summary_text="Theme: pottery budgeting across the whole project.",
            source_object_ids=["mem_fact"],
            created_at="2026-03-30T20:00:00+00:00",
            index_text="pottery budget theme",
            scope=MemoryScope.CONVERSATION,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            payload={},
        )

        broad_candidates = await search.search(
            _plan(
                sub_query="pottery budget",
                scope_filter=[MemoryScope.CONVERSATION],
                query_type="broad_list",
                retrieval_levels=[0, 1, 2],
            ),
            user_id="usr_1",
        )
        default_candidates = await search.search(
            _plan(
                sub_query="pottery budget",
                scope_filter=[MemoryScope.CONVERSATION],
                retrieval_levels=[0],
            ),
            user_id="usr_1",
        )

        broad_ids = [str(candidate["id"]) for candidate in broad_candidates]
        default_ids = [str(candidate["id"]) for candidate in default_candidates]
        assert "sum_mem_sum_episode_1" in broad_ids
        assert "sum_mem_sum_theme_1" in broad_ids
        assert "mem_fact" in broad_ids
        assert "sum_mem_sum_episode_1" not in default_ids
        assert "sum_mem_sum_theme_1" not in default_ids
        assert "mem_fact" in default_ids
    finally:
        await connection.close()
