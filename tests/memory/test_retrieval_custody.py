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


def _trace_for(plan: RetrievalPlan, candidate: dict[str, object]) -> dict[str, object]:
    custody = build_candidate_custody(
        raw_candidates=[candidate],
        filtered_candidates=[],
        shortlist=[],
        scored_candidates=[],
        selected_memory_ids=[],
        retrieval_plan=plan,
        filter_reasons_by_id={},
    )
    return custody[0]["coordinate_trace_v1"]


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
            "scope_canonical": "conversation",
            "user_persona_id": None,
            "platform_id": None,
            "character_id": None,
            "conversation_id": None,
            "status": "active",
            "sensitivity": None,
            "platform_locked": None,
            "privacy_level": 0,
            "intimacy_boundary": "ordinary",
            "intimacy_boundary_confidence": 0.0,
            "retrieval_level": 0,
            "source_kind": "extracted",
            "temporal_type": "unknown",
            "coordinate_trace_v1": {
                "hard_partition": {
                    "user_id": "request_user",
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "required_user_match",
                },
                "presence": {
                    "active_presence_id": None,
                    "candidate_active_presence_id": None,
                    "candidate_source_presence_id": None,
                    "candidate_presence_cluster_id": None,
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "allowed_unscoped_presence",
                },
                "space": {
                    "active_space_id": None,
                    "active_space_boundary_mode": None,
                    "candidate_space_id": None,
                    "candidate_space_boundary_mode": None,
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "allowed_unscoped_space",
                },
                "mind": {
                    "active_mind_id": None,
                    "mind_topology": "unimind",
                    "candidate_memory_owner_id": None,
                    "candidate_source_mind_id": None,
                    "relation": None,
                    "grant_kind": None,
                    "grant_target_kind": None,
                    "grant_target_id": None,
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "allowed_unowned_mind",
                },
                "embodiment": {
                    "active_embodiment_id": None,
                    "cross_embodiment_mode": "direct_if_same_body",
                    "candidate_embodiment_id": None,
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "allowed_unscoped_embodiment",
                },
                "realm": {
                    "active_realm_id": None,
                    "cross_realm_mode": "none",
                    "candidate_realm_id": None,
                    "relation": None,
                    "bridge_mode": None,
                    "allowed": True,
                    "decision": "allowed",
                    "reason": "allowed_unscoped_realm",
                },
                "policy": {
                    "scope_filter": [],
                    "status_filter": [],
                    "privacy_ceiling": 1,
                    "retrieval_levels": [0],
                    "raw_context_access_mode": "normal",
                },
            },
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


def test_candidate_custody_includes_coordinate_trace_without_payload_text() -> None:
    plan = _plan().model_copy(
        update={
            "active_space_id": "space_dev",
            "active_space_boundary_mode": "severance",
            "active_mind_id": "ojocentauri",
            "mind_topology": "ojocentauri",
            "active_embodiment_id": "drone",
            "active_realm_id": "realm_real",
        }
    )
    candidate = {
        "id": "mem_granted",
        "object_type": "evidence",
        "scope": "user",
        "status": "active",
        "privacy_level": 0,
        "space_id": "space_dev",
        "space_boundary_mode": "severance",
        "memory_owner_id": "mind_alpha",
        "mind_relation": "granted",
        "mind_grant_kind": "read",
        "mind_grant_target_kind": "mind",
        "mind_grant_target_id": "mind_alpha",
        "embodiment_id": "drone",
        "realm_id": "realm_aincrad",
        "realm_relation": "cross",
        "realm_bridge_mode": "attributed",
        "canonical_text": "POISON granted memory",
    }

    custody = build_candidate_custody(
        raw_candidates=[candidate],
        filtered_candidates=[candidate],
        shortlist=[],
        scored_candidates=[],
        selected_memory_ids=[],
        retrieval_plan=plan,
        filter_reasons_by_id={},
    )

    trace = custody[0]["coordinate_trace_v1"]
    assert trace["space"]["decision"] == "allowed"
    assert trace["space"]["reason"] == "allowed_same_space"
    assert trace["mind"]["decision"] == "allowed"
    assert trace["mind"]["reason"] == "allowed_by_overseer_grant"
    assert trace["mind"]["grant_kind"] == "read"
    assert trace["embodiment"]["decision"] == "allowed"
    assert trace["embodiment"]["reason"] == "allowed_same_embodiment"
    assert trace["realm"]["decision"] == "allowed"
    assert trace["realm"]["reason"] == "allowed_by_realm_bridge_attributed"
    assert "POISON" not in json.dumps(trace)


def test_coordinate_trace_marks_space_severance_block_as_blocked() -> None:
    plan = _plan().model_copy(
        update={
            "active_space_id": "space_a",
            "active_space_boundary_mode": "severance",
        }
    )
    trace = _trace_for(
        plan,
        {
            "id": "mem_other_space",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "space_id": "space_b",
            "space_boundary_mode": "focus",
        },
    )

    assert trace["space"]["allowed"] is False
    assert trace["space"]["decision"] == "blocked"
    assert trace["space"]["reason"] == "blocked_by_space_severance"


def test_coordinate_trace_marks_tagged_and_focus_space_visibility_as_allowed() -> None:
    focus_plan = _plan().model_copy(
        update={
            "active_space_id": "space_a",
            "active_space_boundary_mode": "focus",
        }
    )
    tagged_candidate_trace = _trace_for(
        focus_plan,
        {
            "id": "mem_tagged_other_space",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "space_id": "space_b",
            "space_boundary_mode": "tagged",
        },
    )
    assert tagged_candidate_trace["space"]["allowed"] is True
    assert tagged_candidate_trace["space"]["reason"] == "allowed_by_tagged_space_visibility"

    tagged_plan = _plan().model_copy(
        update={
            "active_space_id": "space_a",
            "active_space_boundary_mode": "tagged",
        }
    )
    focus_candidate_trace = _trace_for(
        tagged_plan,
        {
            "id": "mem_focus_other_space",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "space_id": "space_b",
            "space_boundary_mode": "focus",
        },
    )
    assert focus_candidate_trace["space"]["allowed"] is True
    assert focus_candidate_trace["space"]["reason"] == "allowed_by_active_tagged_space"


def test_coordinate_trace_marks_other_embodiment_as_blocked() -> None:
    plan = _plan().model_copy(update={"active_embodiment_id": "desktop"})
    trace = _trace_for(
        plan,
        {
            "id": "mem_drone",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "embodiment_id": "drone",
        },
    )

    assert trace["embodiment"]["allowed"] is False
    assert trace["embodiment"]["decision"] == "blocked"
    assert trace["embodiment"]["reason"] == "blocked_by_embodiment_boundary"


def test_coordinate_trace_marks_realm_bridge_eligibility() -> None:
    plan = _plan().model_copy(update={"active_realm_id": "realm_real"})
    no_bridge_trace = _trace_for(
        plan,
        {
            "id": "mem_aincrad",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "realm_id": "realm_aincrad",
        },
    )
    assert no_bridge_trace["realm"]["allowed"] is False
    assert no_bridge_trace["realm"]["reason"] == "blocked_by_realm_no_bridge"

    bridge_trace = _trace_for(
        plan,
        {
            "id": "mem_aincrad_bridge",
            "object_type": "evidence",
            "scope": "user",
            "status": "active",
            "privacy_level": 0,
            "realm_id": "realm_aincrad",
            "realm_relation": "cross",
            "realm_bridge_mode": "attributed",
        },
    )
    assert bridge_trace["realm"]["allowed"] is True
    assert bridge_trace["realm"]["decision"] == "allowed"
    assert bridge_trace["realm"]["reason"] == "allowed_by_realm_bridge_attributed"
