"""Tests for retrieval pipeline tracing (Wave 0-B instrumentation)."""

from __future__ import annotations

import pytest

from atagia.models.schemas_memory import (
    CandidateSearchTrace,
    ComposedContext,
    CompositionTrace,
    CrossConversationRawPolicyTrace,
    DirectVsIndirectProvenanceTrace,
    FacetSupportObligationTrace,
    FacetSupportTrace,
    NeedDetectionTrace,
    ProvenanceEvidenceTrace,
    RequestRuntimeDiagnosticsTrace,
    RetrievalCustodyTrace,
    RetrievalSufficiencyDiagnostic,
    RetrievalTrace,
    RuntimeAliasGroupTrace,
    RuntimeAliasSurfaceTrace,
    ScoredCandidate,
    ScoringTrace,
    SubQuerySearchCount,
    TokenBudgetTrace,
    TopicWorkingSetTrace,
    ContentLanguageProfileTraceRow,
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
            content_language_profile=[
                ContentLanguageProfileTraceRow(
                    language_code="EN",
                    memory_count=3,
                    last_seen_at="2026-04-05T12:00:00+00:00",
                ),
                ContentLanguageProfileTraceRow(
                    language_code="unknown",
                    memory_count=1,
                ),
            ],
            query_type="slot_fill",
            temporal_range="2026-01-01T00:00:00/2026-01-31T23:59:59",
            retrieval_levels=[0, 1],
            degraded_mode=False,
            duration_ms=42.3,
        )
        assert len(trace.detected_needs) == 2
        assert trace.query_type == "slot_fill"
        assert trace.temporal_range is not None
        assert [row.language_code for row in trace.content_language_profile] == [
            "en",
            "unknown",
        ]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            NeedDetectionTrace(duration_ms=1.0, unknown_field="bad")

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(Exception):
            NeedDetectionTrace(duration_ms=-1.0)

    def test_alias_groups_are_content_minimal_non_evidential_diagnostics(self) -> None:
        trace = NeedDetectionTrace(
            duration_ms=1.0,
            alias_groups=[
                RuntimeAliasGroupTrace(
                    sub_query_text="amlodipine dose",
                    anchor_type="concept",
                    original_surface="amlodipino",
                    normalized_surface="amlodipino",
                    anchor_confidence=0.88,
                    aliases=[
                        RuntimeAliasSurfaceTrace(
                            surface="amlodipine",
                            alias_kind="translation",
                            alias_language="en",
                            confidence=0.84,
                        )
                    ],
                )
            ],
        )

        payload = trace.model_dump(mode="json")
        alias_group = payload["alias_groups"][0]
        alias = alias_group["aliases"][0]
        assert alias_group["sub_query_text"] == "amlodipine dose"
        assert alias_group["original_surface"] == "amlodipino"
        assert alias["surface"] == "amlodipine"
        assert alias["alias_kind"] == "translation"
        assert alias["alias_language"] == "en"
        assert alias["confidence"] == 0.84
        assert alias["non_evidential"] is True
        restored = NeedDetectionTrace.model_validate(payload)
        assert restored.alias_groups[0].aliases[0].surface == "amlodipine"

    def test_content_language_profile_trace_is_content_free_metadata(self) -> None:
        trace = NeedDetectionTrace(
            duration_ms=1.0,
            content_language_profile=[
                ContentLanguageProfileTraceRow(
                    language_code=" ES ",
                    memory_count=4,
                    last_seen_at="2026-04-05T12:00:00+00:00",
                )
            ],
        )

        payload = trace.model_dump(mode="json")
        assert payload["content_language_profile"] == [
            {
                "language_code": "es",
                "memory_count": 4,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
        with pytest.raises(Exception):
            ContentLanguageProfileTraceRow(
                language_code=" ",
                memory_count=1,
            )
        with pytest.raises(Exception):
            ContentLanguageProfileTraceRow(
                language_code="en",
                memory_count=1,
                snippet="not allowed",
            )

    def test_alias_group_rejects_evidential_alias_marker(self) -> None:
        with pytest.raises(Exception):
            RuntimeAliasSurfaceTrace(
                surface="amlodipine",
                alias_kind="translation",
                confidence=0.84,
                non_evidential=False,
            )

    def test_alias_trace_rejects_blank_required_surfaces(self) -> None:
        with pytest.raises(Exception):
            RuntimeAliasSurfaceTrace(
                surface="  ",
                alias_kind="translation",
                confidence=0.84,
            )
        with pytest.raises(Exception):
            RuntimeAliasGroupTrace(
                sub_query_text="  ",
                anchor_type="concept",
                original_surface="amlodipino",
                anchor_confidence=0.88,
            )
        with pytest.raises(Exception):
            RuntimeAliasGroupTrace(
                sub_query_text="amlodipine dose",
                anchor_type="concept",
                original_surface="  ",
                anchor_confidence=0.88,
            )

    def test_alias_trace_reuses_contract_literals(self) -> None:
        with pytest.raises(Exception):
            RuntimeAliasSurfaceTrace(
                surface="amlodipine",
                alias_kind="freeform_alias",
                confidence=0.84,
            )
        with pytest.raises(Exception):
            RuntimeAliasGroupTrace(
                sub_query_text="amlodipine dose",
                anchor_type="freeform_anchor",
                original_surface="amlodipino",
                anchor_confidence=0.88,
            )


class TestCandidateSearchTrace:
    def test_minimal_construction(self) -> None:
        trace = CandidateSearchTrace(duration_ms=5.0)
        assert trace.fts_candidates_count == 0
        assert trace.fact_facet_candidates_count == 0
        assert trace.embedding_candidates_count == 0
        assert trace.consequence_candidates_count == 0
        assert trace.verbatim_evidence_search_candidates_count == 0
        assert trace.entity_candidates_count == 0
        assert trace.total_before_fusion == 0
        assert trace.total_after_fusion == 0
        assert trace.per_subquery_counts == []

    def test_full_construction(self) -> None:
        trace = CandidateSearchTrace(
            fts_candidates_count=10,
            embedding_candidates_count=5,
            consequence_candidates_count=2,
            verbatim_evidence_search_candidates_count=0,
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
        assert trace.applicability_gate_mode == "off"
        assert trace.eligible_candidate_count == 0
        assert trace.ineligible_reason_counts == {}
        assert trace.llm_applicability_skipped_count == 0
        assert trace.estimated_calls_saved == 0
        assert trace.gate_reason == "mode_off"

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

    def test_build_scoring_trace_with_applicability_gate_metadata(self) -> None:
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        trace = RetrievalPipeline._build_scoring_trace(
            raw_candidates=[{"id": "mem_1"}, {"id": "mem_2"}],
            filtered_candidates=[{"id": "mem_1"}, {"id": "mem_2"}],
            scored_candidates=[],
            duration_ms=4.0,
            scoring_candidates=[
                {
                    "id": "mem_1",
                    "_applicability_gate": {
                        "mode": "shadow",
                        "eligible": True,
                        "reason": "eligible_source_backed_exact",
                        "ineligible_reason": "",
                        "llm_skipped": False,
                        "estimated_calls_saved": 1,
                        "shadow_disagreement": True,
                        "shadow_harmful_disagreement": False,
                    },
                },
                {
                    "id": "mem_2",
                    "_applicability_gate": {
                        "mode": "shadow",
                        "eligible": False,
                        "reason": "",
                        "ineligible_reason": "close_tie",
                        "llm_skipped": False,
                        "estimated_calls_saved": 0,
                        "shadow_disagreement": False,
                        "shadow_harmful_disagreement": False,
                    },
                },
            ],
        )

        assert trace.applicability_gate_mode == "shadow"
        assert trace.eligible_candidate_count == 1
        assert trace.ineligible_reason_counts == {"close_tie": 1}
        assert trace.llm_applicability_skipped_count == 0
        assert trace.shadow_disagreement_count == 1
        assert trace.shadow_harmful_disagreement_count == 0
        assert trace.estimated_calls_saved == 1
        assert trace.gate_reason == "shadow_disagreement"


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


class TestRetrievalSufficiencyDiagnostic:
    def test_minimal_construction(self) -> None:
        diagnostic = RetrievalSufficiencyDiagnostic(
            state="insufficient_no_candidates",
            confidence=0.95,
            rationale_codes=["raw_candidates_empty"],
            would_expand_channels=["fts", "embedding"],
            would_abstain=True,
        )

        assert diagnostic.state == "insufficient_no_candidates"
        assert diagnostic.candidate_count == 0
        assert diagnostic.would_abstain is True

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            RetrievalSufficiencyDiagnostic(
                state="retrieval_sufficient",
                confidence=0.8,
                bad_field=True,
            )

    def test_rejects_unknown_state(self) -> None:
        with pytest.raises(Exception):
            RetrievalSufficiencyDiagnostic(
                state="free_form_state",
                confidence=0.8,
            )


class TestStabilityDiagnostics:
    def test_facet_support_trace_serializes_obligations(self) -> None:
        trace = FacetSupportTrace(
            obligations=[
                FacetSupportObligationTrace(
                    id="f1",
                    description="original budget",
                    status="covered",
                    selected_memory_ids=["mem_budget"],
                    composed_memory_ids=["mem_budget"],
                    support_verdict="supported",
                )
            ]
        )

        payload = trace.model_dump(mode="json")
        assert payload["obligations"][0]["status"] == "covered"
        assert FacetSupportTrace.model_validate(payload).obligations[0].id == "f1"

    def test_provenance_trace_counts_direct_and_raw_cross_recovery(self) -> None:
        trace = DirectVsIndirectProvenanceTrace(
            evidence=[
                ProvenanceEvidenceTrace(
                    memory_id="vew_prior",
                    recovery_channels=[
                        "verbatim_evidence_search",
                        "raw_cross_conversation",
                    ],
                    proof_source="raw_source_span",
                    joined_to_base=True,
                    selected=True,
                    matched_subquery_indexes=[0],
                    scope_canonical="user",
                    conversation_id="prior",
                )
            ],
            direct_recovery_count=1,
            raw_cross_conversation_count=1,
        )

        assert trace.evidence[0].proof_source == "raw_source_span"
        assert trace.raw_cross_conversation_count == 1

    def test_token_budget_and_cross_raw_policy_traces_are_bounded(self) -> None:
        token_trace = TokenBudgetTrace(
            context_tokens=1200,
            token_budget_total=4096,
            token_budget_used=1200,
            answer_max_tokens=2048,
            postcondition_retry_count=1,
        )
        policy_trace = CrossConversationRawPolicyTrace(
            enabled=True,
            activation_reason="exact_recall_multi_obligation",
            candidate_count=3,
            selected_count=2,
            max_windows_per_subquery=8,
            policy_filters=["user_id", "scope", "privacy_ceiling"],
        )

        assert token_trace.postcondition_retry_count == 1
        assert policy_trace.violation_count == 0

    def test_custody_and_runtime_diagnostics_are_text_free_counts(self) -> None:
        custody = RetrievalCustodyTrace(
            raw_candidate_count=4,
            candidate_count_by_channel={"fts": 3, "verbatim_evidence_search": 1},
            post_user_id_candidate_count=4,
            post_scope_coordinate_lifecycle_candidate_count=3,
            scored_candidate_count=2,
            selected_candidate_count=1,
            selected_evidence_ids=["mem_1"],
            source_window_ids=["vew_conv_1_3"],
            selected_source_window_ids=[],
            drop_counts_by_stage={"composer": 1},
            drop_counts_by_reason={"not_selected_after_scoring": 1},
        )
        runtime = RequestRuntimeDiagnosticsTrace(
            stage_timings_ms={"candidate_search": 12.5},
            db_query_count=7,
            db_query_count_by_operation={"SELECT": 7},
            hydration_timings_ms={"contract_lookup": 1.2},
            lock_wait_count=0,
            sqlite_busy_count=0,
        )

        assert custody.candidate_count_by_channel["fts"] == 3
        assert custody.source_window_ids == ["vew_conv_1_3"]
        assert runtime.db_query_count == 7
        with pytest.raises(Exception):
            RetrievalCustodyTrace(raw_candidate_count=-1)
        with pytest.raises(Exception):
            RequestRuntimeDiagnosticsTrace(db_query_count=-1)


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
        assert trace.retrieval_sufficiency is None
        assert trace.facet_support is None
        assert trace.direct_vs_indirect_provenance is None
        assert trace.token_budget is None
        assert trace.cross_conversation_raw_policy is None
        assert trace.custody.raw_candidate_count == 0
        assert trace.runtime_diagnostics.db_query_count == 0
        assert trace.topic_snapshot.active_topics == []
        assert trace.topic_snapshot.parked_topics == []
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
            retrieval_sufficiency=RetrievalSufficiencyDiagnostic(
                state="retrieval_sufficient",
                confidence=0.8,
                rationale_codes=["scored_candidates_available"],
                scored_candidate_count=3,
                top_score=0.9,
            ),
            total_duration_ms=65.0,
        )
        assert trace.need_detection is not None
        assert trace.candidate_search is not None
        assert trace.scoring is not None
        assert trace.composition is not None
        assert trace.retrieval_sufficiency is not None
        assert trace.retrieval_sufficiency.state == "retrieval_sufficient"
        assert trace.total_duration_ms == 65.0

    def test_topic_snapshot_serializes_compact_working_set(self) -> None:
        trace = RetrievalTrace(
            query_text="q",
            user_id="u",
            conversation_id="c",
            timestamp_iso="2026-04-09T12:00:00Z",
            topic_snapshot=TopicWorkingSetTrace.model_validate(
                {
                    "active_topics": [
                        {
                            "id": "tpc_active",
                            "status": "active",
                            "title": "Active work",
                            "summary": "Current thread",
                            "active_goal": "Keep tracing only",
                            "open_questions": ["What evidence was missed?"],
                            "decisions": ["Do not boost rankings yet."],
                            "artifact_ids": ["art_1"],
                            "source_counts": {"message": 2, "artifact": 1},
                            "source_refs": [
                                {
                                    "source_kind": "artifact",
                                    "source_id": "art_1",
                                    "relation_kind": "evidence",
                                }
                            ],
                            "last_touched_seq": 7,
                            "last_touched_at": "2026-04-09T12:00:00Z",
                            "confidence": 0.7,
                            "privacy_level": 1,
                        }
                    ],
                    "parked_topics": [],
                }
            ),
        )

        payload = trace.model_dump(mode="json")
        assert payload["topic_snapshot"]["active_topics"][0]["id"] == "tpc_active"
        restored = RetrievalTrace.model_validate(payload)
        assert restored.topic_snapshot.active_topics[0].open_questions == [
            "What evidence was missed?"
        ]
        assert restored.topic_snapshot.active_topics[0].source_counts == {
            "message": 2,
            "artifact": 1,
        }

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
        assert result.retrieval_sufficiency is None

    def test_pipeline_result_exposes_retrieval_sufficiency(self) -> None:
        diagnostic = RetrievalSufficiencyDiagnostic(
            state="retrieval_sufficient",
            confidence=0.8,
            rationale_codes=["scored_candidates_available"],
        )
        result = self._minimal_pipeline_result().model_copy(
            update={"retrieval_sufficiency": diagnostic}
        )
        assert result.retrieval_sufficiency is diagnostic


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
                    fts_queries=["alpha", "alpha OR beta"],
                    fts_query_kinds=["default_and", "broad_or"],
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
                "fts_query_matches": [
                    {
                        "subquery": "alpha query",
                        "query": "alpha",
                        "kind": "default_and",
                        "match_mode": "implicit_and",
                    }
                ],
            },
            {
                "id": "mem_2",
                "channel_ranks": {"fts": 2, "embedding": 1},
                "matched_sub_queries": ["alpha query"],
                "fts_query_matches": [
                    {
                        "subquery": "alpha query",
                        "query": "alpha OR beta",
                        "kind": "broad_or",
                        "match_mode": "explicit_or",
                    }
                ],
            },
            {
                "id": "mem_3",
                "channel_ranks": {"consequence": 1},
                "matched_sub_queries": [],
            },
            {
                "id": "mff_1",
                "channel_ranks": {"fact_facet": 1},
                "matched_sub_queries": ["alpha query"],
                "is_fact_facet_candidate": True,
            },
        ]
        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates,
            plan,
            25.0,
            [
                {
                    "subquery": "alpha query",
                    "query": "alpha",
                    "kind": "default_and",
                    "match_mode": "implicit_and",
                    "raw_rows": 3,
                },
                {
                    "subquery": "alpha query",
                    "query": "alpha OR beta",
                    "kind": "broad_or",
                    "match_mode": "explicit_or",
                    "raw_rows": 2,
                },
            ],
        )
        assert trace.fts_candidates_count == 2
        assert trace.fact_facet_candidates_count == 1
        assert trace.embedding_candidates_count == 1
        assert trace.consequence_candidates_count == 1
        assert trace.total_before_fusion == 5
        assert trace.total_after_fusion == 4
        assert trace.duration_ms == 25.0
        assert len(trace.per_subquery_counts) == 1
        assert trace.per_subquery_counts[0].subquery == "alpha query"
        assert trace.per_subquery_counts[0].fts == 2
        assert trace.per_subquery_counts[0].fact_facet == 1
        assert trace.per_subquery_counts[0].embedding == 1
        assert trace.per_subquery_counts[0].fts_queries == ["alpha", "alpha OR beta"]
        assert trace.per_subquery_counts[0].fts_query_kinds == [
            "default_and",
            "broad_or",
        ]
        assert [
            execution.model_dump()
            for execution in trace.per_subquery_counts[0].fts_query_executions
        ] == [
            {
                "query": "alpha",
                "kind": "default_and",
                "match_mode": "implicit_and",
                "source": "planned",
                "non_evidential": True,
                "raw_rows": 3,
                "candidates": 1,
            },
            {
                "query": "alpha OR beta",
                "kind": "broad_or",
                "match_mode": "explicit_or",
                "source": "planned",
                "non_evidential": True,
                "raw_rows": 2,
                "candidates": 1,
            },
        ]

    def test_build_candidate_search_trace_distinguishes_raw_fts_rows_from_final_candidates(self) -> None:
        from atagia.models.schemas_memory import PlannedSubQuery, RetrievalPlan
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="alpha query",
                    fts_queries=["alpha"],
                    fts_query_kinds=["default_and"],
                ),
            ],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )

        trace = RetrievalPipeline._build_candidate_search_trace(
            [],
            plan,
            12.0,
            [
                {
                    "subquery": "alpha query",
                    "query": "alpha",
                    "kind": "default_and",
                    "match_mode": "implicit_and",
                    "raw_rows": 2,
                },
            ],
        )

        assert trace.fts_candidates_count == 0
        assert trace.per_subquery_counts[0].fts == 0
        assert [
            execution.model_dump()
            for execution in trace.per_subquery_counts[0].fts_query_executions
        ] == [
            {
                "query": "alpha",
                "kind": "default_and",
                "match_mode": "implicit_and",
                "source": "planned",
                "non_evidential": True,
                "raw_rows": 2,
                "candidates": 0,
            }
        ]

    def test_build_candidate_search_trace_includes_audit_only_fts_executions(self) -> None:
        from atagia.models.schemas_memory import PlannedSubQuery, RetrievalPlan
        from atagia.services.retrieval_pipeline import RetrievalPipeline

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            conversation_id="c_test",
            sub_query_plans=[
                PlannedSubQuery(
                    text="alpha query",
                    fts_queries=["alpha"],
                    fts_query_kinds=["default_and"],
                ),
            ],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=3,
        )
        raw_candidates = [
            {
                "id": "mem_1",
                "channel_ranks": {"fts": 1},
                "matched_sub_queries": ["alpha query"],
                "fts_query_matches": [
                    {
                        "subquery": "alpha query",
                        "query": "alpina",
                        "kind": "corpus_near_or",
                        "match_mode": "implicit_and",
                    }
                ],
            }
        ]

        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates,
            plan,
            12.0,
            [
                {
                    "subquery": "alpha query",
                    "query": "alpha",
                    "kind": "default_and",
                    "match_mode": "implicit_and",
                    "raw_rows": 0,
                },
                {
                    "subquery": "alpha query",
                    "query": "alpina",
                    "kind": "corpus_near_or",
                    "match_mode": "implicit_and",
                    "raw_rows": 1,
                },
            ],
        )

        assert [
            execution.model_dump()
            for execution in trace.per_subquery_counts[0].fts_query_executions
        ] == [
            {
                "query": "alpha",
                "kind": "default_and",
                "match_mode": "implicit_and",
                "source": "planned",
                "non_evidential": True,
                "raw_rows": 0,
                "candidates": 0,
            },
            {
                "query": "alpina",
                "kind": "corpus_near_or",
                "match_mode": "implicit_and",
                "source": "dynamic",
                "non_evidential": True,
                "raw_rows": 1,
                "candidates": 1,
            },
        ]

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
                    "verbatim_evidence_search": None,
                },
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "rmw_raw",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": None,
                    "verbatim_evidence_search": 1,
                },
                "matched_sub_queries": ["alpha query"],
                "is_verbatim_evidence_window": True,
            },
            # Raw window that entered via the marker only — channel rank is None
            # but is_verbatim_evidence_window is True. The fallback must still count it.
            {
                "id": "rmw_marker_only",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": None,
                    "verbatim_evidence_search": None,
                },
                "matched_sub_queries": ["alpha query"],
                "is_verbatim_evidence_window": True,
            },
        ]

        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates, plan, 25.0
        )

        assert trace.fts_candidates_count == 1
        assert trace.verbatim_evidence_search_candidates_count == 2
        assert trace.embedding_candidates_count == 0
        assert trace.consequence_candidates_count == 0
        assert trace.per_subquery_counts[0].fts == 1
        assert trace.per_subquery_counts[0].verbatim_evidence_search == 2
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
                    "verbatim_evidence_search": None,
                },
                "matched_sub_queries": ["alpha query"],
            },
            {
                "id": "mem_consequence",
                "channel_ranks": {
                    "fts": None,
                    "embedding": None,
                    "consequence": 1,
                    "verbatim_evidence_search": None,
                },
                "matched_sub_queries": [],
            },
        ]

        trace = RetrievalPipeline._build_candidate_search_trace(
            raw_candidates, plan, 25.0
        )

        assert trace.embedding_candidates_count == 0
        assert trace.verbatim_evidence_search_candidates_count == 0
        assert trace.per_subquery_counts[0].embedding == 0
        assert trace.per_subquery_counts[0].verbatim_evidence_search == 0

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
