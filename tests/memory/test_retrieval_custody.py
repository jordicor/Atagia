"""Tests for safe core retrieval custody records."""

from __future__ import annotations

import json

from atagia.memory.retrieval_custody import build_candidate_custody
from atagia.models.schemas_memory import RetrievalPlan, ScoredCandidate


def _plan() -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=["project budget"],
        sub_query_plans=[
            {
                "text": "project budget",
                "fts_queries": ["project budget"],
            }
        ],
        query_type="default",
        scope_filter=[],
        status_filter=[],
        vector_limit=0,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=False,
    )


def test_candidate_custody_uses_safe_fields_without_raw_text_values() -> None:
    candidate = {
        "id": "mem_1",
        "object_type": "evidence",
        "scope": "conversation",
        "status": "active",
        "privacy_level": 0,
        "retrieval_level": 0,
        "source_kind": "extracted",
        "temporal_type": "unknown",
        "canonical_text": "POISON canonical raw memory text",
        "index_text": "POISON index text",
        "payload_json": {
            "source_excerpt_messages": ["POISON source excerpt"],
            "nested": {"secret": "POISON nested payload"},
        },
        "matched_sub_queries": ["project budget"],
        "subquery_ranks": {"project budget": 1},
        "channel_ranks": {"fts": 1},
        "retrieval_sources": ["fts"],
        "rrf_score": 0.75,
    }
    scored = ScoredCandidate(
        memory_id="mem_1",
        memory_object={
            **candidate,
            "canonical_text": "POISON scored memory text",
        },
        llm_applicability=0.9,
        retrieval_score=0.7,
        vitality_boost=0.0,
        confirmation_boost=0.0,
        need_boost=0.1,
        penalty=0.0,
        final_score=0.73,
    )

    custody = build_candidate_custody(
        raw_candidates=[candidate],
        filtered_candidates=[candidate],
        shortlist=[candidate],
        scored_candidates=[scored],
        selected_memory_ids=["mem_1"],
        retrieval_plan=_plan(),
        filter_reasons_by_id={},
    )

    assert custody == [
        {
            "schema_version": 2,
            "candidate_id": "mem_1",
            "candidate_kind": "evidence",
            "fusion_position": 1,
            "channels": ["fts"],
            "channel_ranks": {"fts": 1},
            "retrieval_sources": ["fts"],
            "matched_subquery_indexes": [0],
            "matched_subquery_count": 1,
            "subquery_ranks": {"0": 1},
            "fused_score": 0.75,
            "scope": "conversation",
            "status": "active",
            "privacy_level": 0,
            "intimacy_boundary": "ordinary",
            "intimacy_boundary_confidence": 0.0,
            "retrieval_level": 0,
            "source_kind": "extracted",
            "temporal_type": "unknown",
            "filter_reason": None,
            "shortlisted": True,
            "shortlist_rank": 1,
            "shortlist_status": "shortlisted",
            "scored": True,
            "score_rank": 1,
            "score_status": "scored",
            "scorer": {
                "llm_applicability": 0.9,
                "retrieval_score": 0.7,
                "vitality_boost": 0.0,
                "confirmation_boost": 0.0,
                "need_boost": 0.1,
                "penalty": 0.0,
                "final_score": 0.73,
            },
            "composer_decision": "selected",
            "selected": True,
            "selection_rank": 1,
        }
    ]
    serialized = json.dumps(custody)
    assert "POISON" not in serialized
    assert "project budget" not in serialized


def test_candidate_custody_separates_filter_shortlist_score_and_composer_status() -> None:
    kept = {
        "id": "mem_kept",
        "object_type": "evidence",
        "scope": "conversation",
        "status": "active",
        "privacy_level": 0,
    }
    filtered = {
        "id": "mem_filtered",
        "object_type": "belief",
        "scope": "ephemeral_session",
        "status": "active",
        "privacy_level": 0,
    }

    custody = build_candidate_custody(
        raw_candidates=[kept, filtered],
        filtered_candidates=[kept],
        shortlist=[],
        scored_candidates=[],
        selected_memory_ids=[],
        retrieval_plan=_plan(),
        filter_reasons_by_id={"mem_filtered": "policy_filtered_scope"},
    )

    by_id = {record["candidate_id"]: record for record in custody}
    assert by_id["mem_filtered"]["filter_reason"] == "policy_filtered_scope"
    assert by_id["mem_filtered"]["shortlist_status"] == "filtered_before_shortlist"
    assert by_id["mem_filtered"]["score_status"] == "filtered_before_scoring"
    assert by_id["mem_filtered"]["composer_decision"] == "not_scored"
    assert by_id["mem_kept"]["filter_reason"] is None
    assert by_id["mem_kept"]["shortlist_status"] == "not_shortlisted"
    assert by_id["mem_kept"]["score_status"] == "not_shortlisted"
