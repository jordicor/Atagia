"""Tests for retrieval pipeline tracing (Wave 0-B instrumentation)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from atagia.models.schemas_memory import (
    CandidateSearchTrace,
    ComposedContext,
    CompositionTrace,
    NeedDetectionTrace,
    RetrievalTrace,
    ScoredCandidate,
    ScoringTrace,
    SubQuerySearchCount,
)
from atagia.models.schemas_replay import PipelineResult


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestNeedDetectionTrace:
    def test_minimal_construction(self) -> None:
        trace = NeedDetectionTrace(duration_ms=1.5)
        assert trace.detected_needs == []
        assert trace.sub_queries == []
        assert trace.sparse_hints == []
        assert trace.query_type == "default"
        assert trace.temporal_range is None
        assert trace.retrieval_levels == [0]
        assert trace.degraded_mode is False
        assert trace.duration_ms == 1.5

    def test_full_construction(self) -> None:
        trace = NeedDetectionTrace(
            detected_needs=["ambiguity", "contradiction"],
            sub_queries=["What is X?", "How does Y work?"],
            sparse_hints=["X definition", "Y mechanism"],
            query_type="slot_fill",
            temporal_range="2026-01-01T00:00:00/2026-01-31T23:59:59",
            retrieval_levels=[0, 1],
            degraded_mode=False,
            duration_ms=42.3,
        )
        assert len(trace.detected_needs) == 2
        assert trace.query_type == "slot_fill"
        assert trace.temporal_range is not None

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            NeedDetectionTrace(duration_ms=1.0, unknown_field="bad")

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(Exception):
            NeedDetectionTrace(duration_ms=-1.0)


class TestCandidateSearchTrace:
    def test_minimal_construction(self) -> None:
        trace = CandidateSearchTrace(duration_ms=5.0)
        assert trace.fts_candidates_count == 0
        assert trace.embedding_candidates_count == 0
        assert trace.consequence_candidates_count == 0
        assert trace.raw_message_candidates_count == 0
        assert trace.entity_candidates_count == 0
        assert trace.total_before_fusion == 0
        assert trace.total_after_fusion == 0
        assert trace.per_subquery_counts == []

    def test_full_construction(self) -> None:
        trace = CandidateSearchTrace(
            fts_candidates_count=10,
            embedding_candidates_count=5,
            consequence_candidates_count=2,
            raw_message_candidates_count=0,
            entity_candidates_count=0,
            total_before_fusion=17,
            total_after_fusion=12,
            per_subquery_counts=[
                SubQuerySearchCount(subquery="test query", fts=10, embedding=5),
            ],
            duration_ms=15.7,
        )
        assert trace.total_before_fusion == 17
        assert trace.total_after_fusion == 12

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            CandidateSearchTrace(duration_ms=1.0, unknown_field="bad")


class TestScoringTrace:
    def test_minimal_construction(self) -> None:
        trace = ScoringTrace(duration_ms=3.0)
        assert trace.candidates_received == 0
        assert trace.candidates_scored == 0
        assert trace.candidates_rejected == 0
        assert trace.rejection_reasons == {}
        assert trace.top_score == 0.0
        assert trace.median_score == 0.0
        assert trace.min_score == 0.0

    def test_full_construction(self) -> None:
        trace = ScoringTrace(
            candidates_received=20,
            candidates_scored=15,
            candidates_rejected=5,
            rejection_reasons={"scope_mismatch": 3, "privacy_ceiling": 2},
            top_score=0.95,
            median_score=0.72,
            min_score=0.45,
            duration_ms=120.5,
        )
        assert trace.candidates_rejected == 5


class TestCompositionTrace:
    def test_minimal_construction(self) -> None:
        trace = CompositionTrace(duration_ms=2.0)
        assert trace.candidates_selected == 0
        assert trace.token_budget_total == 0
        assert trace.token_budget_used == 0
        assert trace.contract_tokens == 0
        assert trace.workspace_tokens == 0
        assert trace.memory_tokens == 0
        assert trace.state_tokens == 0
        assert trace.diversity_penalties_applied == 0
        assert trace.support_level == "UNKNOWN"

    def test_full_construction(self) -> None:
        trace = CompositionTrace(
            candidates_selected=5,
            token_budget_total=4096,
            token_budget_used=3200,
            contract_tokens=500,
            workspace_tokens=200,
            memory_tokens=2000,
            state_tokens=500,
            diversity_penalties_applied=2,
            support_level="SUFFICIENT",
            duration_ms=8.3,
        )
        assert trace.token_budget_used == 3200


class TestRetrievalTrace:
    def test_minimal_construction(self) -> None:
        trace = RetrievalTrace(
            query_text="test query",
            user_id="u_123",
            conversation_id="c_456",
            timestamp_iso="2026-04-09T12:00:00Z",
        )
        assert trace.small_corpus_mode is False
        assert trace.degraded_mode is False
        assert trace.need_detection is None
        assert trace.candidate_search is None
        assert trace.scoring is None
        assert trace.composition is None
        assert trace.total_duration_ms == 0.0

    def test_degraded_mode_flag_can_be_set(self) -> None:
        trace = RetrievalTrace(
            query_text="q",
            user_id="u",
            conversation_id="c",
            timestamp_iso="2026-04-09T12:00:00Z",
            degraded_mode=True,
        )
        assert trace.degraded_mode is True
        assert trace.small_corpus_mode is False

    def test_small_corpus_and_degraded_are_independent(self) -> None:
        trace = RetrievalTrace(
            query_text="q",
            user_id="u",
            conversation_id="c",
            timestamp_iso="2026-04-09T12:00:00Z",
            small_corpus_mode=True,
            degraded_mode=False,
        )
        assert trace.small_corpus_mode is True
        assert trace.degraded_mode is False

    def test_full_construction(self) -> None:
        trace = RetrievalTrace(
            query_text="test query",
            user_id="u_123",
            conversation_id="c_456",
            timestamp_iso="2026-04-09T12:00:00Z",
            small_corpus_mode=False,
            need_detection=NeedDetectionTrace(
                detected_needs=["ambiguity"],
                sub_queries=["test query"],
                sparse_hints=["test query"],
                query_type="default",
                retrieval_levels=[0],
                duration_ms=10.0,
            ),
            candidate_search=CandidateSearchTrace(
                fts_candidates_count=5,
                total_after_fusion=5,
                duration_ms=20.0,
            ),
            scoring=ScoringTrace(
                candidates_received=5,
                candidates_scored=3,
                candidates_rejected=2,
                top_score=0.9,
                median_score=0.7,
                min_score=0.5,
                duration_ms=30.0,
            ),
            composition=CompositionTrace(
                candidates_selected=3,
                token_budget_total=4096,
                token_budget_used=2000,
                duration_ms=5.0,
            ),
            total_duration_ms=65.0,
        )
        assert trace.need_detection is not None
        assert trace.candidate_search is not None
        assert trace.scoring is not None
        assert trace.composition is not None
        assert trace.total_duration_ms == 65.0

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            RetrievalTrace(
                query_text="q",
                user_id="u",
                conversation_id="c",
                timestamp_iso="2026-04-09T12:00:00Z",
                bad_field=True,
            )

    def test_serialization_roundtrip(self) -> None:
        trace = RetrievalTrace(
            query_text="roundtrip test",
            user_id="u_rt",
            conversation_id="c_rt",
            timestamp_iso="2026-04-09T12:00:00Z",
            need_detection=NeedDetectionTrace(
                detected_needs=["loop"],
                sub_queries=["roundtrip test"],
                sparse_hints=["roundtrip test"],
                query_type="broad_list",
                retrieval_levels=[0, 1],
                duration_ms=5.0,
            ),
            total_duration_ms=10.0,
        )
        json_data = trace.model_dump(mode="json")
        restored = RetrievalTrace.model_validate(json_data)
        assert restored.query_text == trace.query_text
        assert restored.need_detection is not None
        assert restored.need_detection.detected_needs == ["loop"]
        assert restored.total_duration_ms == 10.0


# ---------------------------------------------------------------------------
# PipelineResult trace field tests
# ---------------------------------------------------------------------------


class TestPipelineResultTraceField:
    @staticmethod
    def _minimal_pipeline_result(
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        from atagia.models.schemas_memory import (
            MemoryScope,
            MemoryStatus,
            PlannedSubQuery,
            RetrievalPlan,
        )

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="test",
                    fts_queries=["test"],
                ),
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )
        composed = ComposedContext(
            total_tokens_estimate=0,
            budget_tokens=4096,
            items_included=0,
            items_dropped=0,
        )
        return PipelineResult(
            retrieval_plan=plan,
            composed_context=composed,
            trace=trace,
        )

    def test_trace_none_by_default(self) -> None:
        result = self._minimal_pipeline_result()
        assert result.trace is None

    def test_trace_populated_when_provided(self) -> None:
        trace = RetrievalTrace(
            query_text="test",
            user_id="u_1",
            conversation_id="c_1",
            timestamp_iso="2026-04-09T12:00:00Z",
            total_duration_ms=50.0,
        )
        result = self._minimal_pipeline_result(trace=trace)
        assert result.trace is not None
        assert result.trace.query_text == "test"

    def test_pipeline_result_serialization_with_trace(self) -> None:
        trace = RetrievalTrace(
            query_text="ser test",
            user_id="u_s",
            conversation_id="c_s",
            timestamp_iso="2026-04-09T12:00:00Z",
            need_detection=NeedDetectionTrace(
                sub_queries=["ser test"],
                sparse_hints=["ser test"],
                duration_ms=1.0,
            ),
            total_duration_ms=5.0,
        )
        result = self._minimal_pipeline_result(trace=trace)
        json_data = result.model_dump(mode="json")
        assert json_data["trace"] is not None
        assert json_data["trace"]["query_text"] == "ser test"

    def test_pipeline_result_serialization_without_trace(self) -> None:
        result = self._minimal_pipeline_result()
        json_data = result.model_dump(mode="json")
        assert json_data["trace"] is None

    def test_pipeline_result_exposes_small_corpus_and_degraded_flags(self) -> None:
        result = self._minimal_pipeline_result()
        assert result.small_corpus_mode is False
        assert result.degraded_mode is False


# ---------------------------------------------------------------------------
# Trace builder unit tests (static methods on RetrievalPipeline)
# ---------------------------------------------------------------------------


class TestTraceBuilders:
    def test_build_candidate_search_trace(self) -> None:
        from atagia.models.schemas_memory import (
            MemoryScope,
            MemoryStatus,
            PlannedSubQuery,
            RetrievalPlan,
        )
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="alpha query",
                    fts_queries=["alpha"],
                ),
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )
        raw_candidates = [
            {
                "id": "mem_1",
                "channel_ranks": {"fts": 1},
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "mem_2",
                "channel_ranks": {"fts": 2, "embedding": 1},
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "mem_3",
                "channel_ranks": {"consequence": 1},
                "matched_sub_queries": [],
            },
        ]
        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates, plan, 25.0
        )
        assert trace.fts_candidates_count == 2
        assert trace.embedding_candidates_count == 1
        assert trace.consequence_candidates_count == 1
        assert trace.total_before_fusion == 4
        assert trace.total_after_fusion == 3
        assert trace.duration_ms == 25.0
        assert len(trace.per_subquery_counts) == 1
        assert trace.per_subquery_counts[0].subquery == "alpha query"
        assert trace.per_subquery_counts[0].fts == 2
        assert trace.per_subquery_counts[0].embedding == 1

    def test_build_candidate_search_trace_counts_actual_channel_contributions(self) -> None:
        from atagia.models.schemas_memory import PlannedSubQuery, RetrievalPlan
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="alpha query",
                    fts_queries=["alpha"],
                ),
            ],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )
        raw_candidates = [
            {
                "id": "mem_fts",
                "channel_ranks": {
                    "fts": 1,
                    "embedding": None,
                    "consequence": None,
                    "raw_message": None,
                },
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "rmw_raw",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": None,
                    "raw_message": 1,
                },
                "matched_sub_queries": ["alpha query"],
                "is_raw_message_window": True,
            },
            # Raw window that entered via the marker only — channel rank is None
            # but is_raw_message_window is True. The fallback must still count it.
            {
                "id": "rmw_marker_only",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": None,
                    "raw_message": None,
                },
                "matched_sub_queries": ["alpha query"],
                "is_raw_message_window": True,
            },
        ]

        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates, plan, 25.0
        )

        assert trace.fts_candidates_count == 1
        assert trace.raw_message_candidates_count == 2
        assert trace.embedding_candidates_count == 0
        assert trace.consequence_candidates_count == 0
        assert trace.per_subquery_counts[0].fts == 1
        assert trace.per_subquery_counts[0].raw_message == 2
        assert trace.per_subquery_counts[0].embedding == 0

    def test_build_candidate_search_trace_zero_channel_stays_zero(self) -> None:
        from atagia.models.schemas_memory import PlannedSubQuery, RetrievalPlan
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="alpha query",
                    fts_queries=["alpha"],
                ),
            ],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )
        raw_candidates = [
            {
                "id": "mem_fts",
                "channel_ranks": {
                    "fts": 1,
                    "embedding": None,
                    "consequence": None,
                    "raw_message": None,
                },
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "mem_consequence",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": 1,
                    "raw_message": None,
                },
                "matched_sub_queries": [],
            },
        ]

        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates, plan, 25.0
        )

        assert trace.embedding_candidates_count == 0
        assert trace.raw_message_candidates_count == 0
        assert trace.per_subquery_counts[0].embedding == 0
        assert trace.per_subquery_counts[0].raw_message == 0

    def test_build_scoring_trace_with_candidates(self) -> None:
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        raw = [{"id": f"mem_{i}"} for i in range(10)]
        filtered = raw[:7]
        scored = [
            ScoredCandidate(
                memory_id=f"mem_{i}",
                memory_object={},
                llm_applicability=0.5,
                retrieval_score=0.3,
                vitality_boost=0.0,
                confirmation_boost=0.0,
                need_boost=0.0,
                penalty=0.0,
                final_score=score,
            )
            for i, score in enumerate([0.9, 0.7, 0.5, 0.3, 0.1])
        ]
        trace = RetrievalPipeline._build_scoring_trace(
            raw, filtered, scored, 50.0
        )
        assert trace.candidates_received == 10
        assert trace.candidates_scored == 5
        assert trace.candidates_rejected == 3
        assert trace.top_score == 0.9
        assert trace.min_score == 0.1
        assert trace.duration_ms == 50.0
        # Median of [0.1, 0.3, 0.5, 0.7, 0.9] = 0.5
        assert trace.median_score == 0.5

    def test_build_scoring_trace_empty(self) -> None:
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        trace = RetrievalPipeline._build_scoring_trace([], [], [], 1.0)
        assert trace.candidates_received == 0
        assert trace.candidates_scored == 0
        assert trace.top_score == 0.0
        assert trace.median_score == 0.0

    def test_build_scoring_trace_even_median(self) -> None:
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        scored = [
            ScoredCandidate(
                memory_id=f"mem_{i}",
                memory_object={},
                llm_applicability=0.5,
                retrieval_score=0.3,
                vitality_boost=0.0,
                confirmation_boost=0.0,
                need_boost=0.0,
                penalty=0.0,
                final_score=score,
            )
            for i, score in enumerate([0.2, 0.4, 0.6, 0.8])
        ]
        trace = RetrievalPipeline._build_scoring_trace(
            [{}] * 4, [{}] * 4, scored, 1.0
        )
        # Median of [0.2, 0.4, 0.6, 0.8] = (0.4 + 0.6) / 2 = 0.5
        assert trace.median_score == pytest.approx(0.5)
