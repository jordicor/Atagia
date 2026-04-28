"""Tests for text-free retrieval sufficiency diagnostics."""

from __future__ import annotations

import json

from atagia.memory.retrieval_diagnostics import build_retrieval_sufficiency_diagnostic
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    ScoredCandidate,
)


def _plan(
    *,
    raw_context_access_mode: str = "normal",
    exact_recall_mode: bool = False,
    require_evidence_regrounding: bool = False,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="What should I use?",
        assistant_mode_id="general_qa",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["diagnostic"],
        sub_query_plans=[
            PlannedSubQuery(
                text="diagnostic",
                fts_queries=["diagnostic"],
            )
        ],
        raw_context_access_mode=raw_context_access_mode,  # type: ignore[arg-type]
        scope_filter=[MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=2,
        exact_recall_mode=exact_recall_mode,
        require_evidence_regrounding=require_evidence_regrounding,
    )


def _candidate(
    memory_id: str,
    *,
    object_type: str = MemoryObjectType.EVIDENCE.value,
    rrf_score: float = 0.1,
    payload_json: dict[str, object] | None = None,
    retrieval_sources: list[str] | None = None,
    is_artifact_chunk: bool = False,
    is_raw_message_window: bool = False,
    tension_score: float = 0.0,
) -> dict[str, object]:
    return {
        "id": memory_id,
        "object_type": object_type,
        "scope": MemoryScope.CONVERSATION.value,
        "status": MemoryStatus.ACTIVE.value,
        "privacy_level": 0,
        "canonical_text": "secret text must never appear in diagnostics",
        "payload_json": payload_json or {},
        "rrf_score": rrf_score,
        "retrieval_sources": retrieval_sources or ["fts"],
        "channel_ranks": {
            "fts": 1,
            "embedding": None,
            "raw_message": 1 if is_raw_message_window else None,
            "artifact_chunk": 1 if is_artifact_chunk else None,
            "consequence": None,
            "verbatim_pin": None,
        },
        "is_artifact_chunk": is_artifact_chunk,
        "is_raw_message_window": is_raw_message_window,
        "tension_score": tension_score,
    }


def _scored(candidate: dict[str, object], *, final_score: float = 0.8) -> ScoredCandidate:
    return ScoredCandidate(
        memory_id=str(candidate["id"]),
        memory_object=dict(candidate),
        llm_applicability=final_score,
        retrieval_score=0.5,
        vitality_boost=0.0,
        confirmation_boost=0.0,
        need_boost=0.0,
        penalty=0.0,
        final_score=final_score,
    )


def _diagnostic(
    *,
    raw_candidates: list[dict[str, object]],
    filtered_candidates: list[dict[str, object]] | None = None,
    shortlist: list[dict[str, object]] | None = None,
    scored_candidates: list[ScoredCandidate] | None = None,
    plan: RetrievalPlan | None = None,
):
    return build_retrieval_sufficiency_diagnostic(
        raw_candidates=raw_candidates,
        filtered_candidates=list(filtered_candidates or raw_candidates),
        shortlist=list(shortlist or raw_candidates),
        scored_candidates=list(scored_candidates or []),
        retrieval_plan=plan or _plan(),
        contradiction_tension_threshold=0.5,
    )


def test_no_candidates_reports_insufficient_no_candidates() -> None:
    diagnostic = _diagnostic(raw_candidates=[])

    assert diagnostic.state == "insufficient_no_candidates"
    assert diagnostic.would_abstain is True
    assert diagnostic.rationale_codes == ["raw_candidates_empty"]
    assert diagnostic.would_expand_channels == ["fts", "embedding"]


def test_raw_candidates_without_scores_report_insufficient_no_scored_candidates() -> None:
    raw = [_candidate("mem_raw")]

    diagnostic = _diagnostic(raw_candidates=raw, scored_candidates=[])

    assert diagnostic.state == "insufficient_no_scored_candidates"
    assert diagnostic.rationale_codes == ["scored_candidates_empty"]
    assert diagnostic.candidate_count == 1
    assert diagnostic.scored_candidate_count == 0


def test_artifact_mode_without_artifact_candidates_requests_artifact_channel() -> None:
    candidate = _candidate("mem_evidence")

    diagnostic = _diagnostic(
        raw_candidates=[candidate],
        scored_candidates=[_scored(candidate)],
        plan=_plan(raw_context_access_mode="artifact"),
    )

    assert diagnostic.state == "insufficient_need_artifact"
    assert diagnostic.would_expand_channels == ["artifact_chunk"]
    assert diagnostic.artifact_candidate_count == 0


def test_exact_recall_without_direct_evidence_requests_raw_message_channel() -> None:
    summary = _candidate(
        "sum_episode",
        object_type=MemoryObjectType.SUMMARY_VIEW.value,
        payload_json={
            "hierarchy_level": 1,
            "summary_kind": "episode",
            "source_object_ids": ["mem_missing"],
        },
    )

    diagnostic = _diagnostic(
        raw_candidates=[summary],
        scored_candidates=[_scored(summary)],
        plan=_plan(exact_recall_mode=True),
    )

    assert diagnostic.state == "insufficient_need_more_raw_evidence"
    assert diagnostic.would_expand_channels == ["raw_message"]
    assert diagnostic.direct_evidence_candidate_count == 0


def test_unsupported_high_score_summary_only_is_not_sufficient() -> None:
    summary = _candidate(
        "sum_episode",
        object_type=MemoryObjectType.SUMMARY_VIEW.value,
        payload_json={
            "hierarchy_level": 1,
            "summary_kind": "episode",
            "source_object_ids": ["mem_missing"],
        },
    )

    diagnostic = _diagnostic(
        raw_candidates=[summary],
        scored_candidates=[_scored(summary, final_score=0.99)],
        plan=_plan(),
    )

    assert diagnostic.state == "insufficient_summary_support"
    assert diagnostic.unsupported_summary_candidate_count == 1
    assert diagnostic.top_score == 0.99


def test_high_tension_belief_reports_contradictory_candidates() -> None:
    belief = _candidate(
        "mem_belief",
        object_type=MemoryObjectType.BELIEF.value,
        tension_score=0.7,
    )

    diagnostic = _diagnostic(
        raw_candidates=[belief],
        scored_candidates=[_scored(belief, final_score=0.8)],
    )

    assert diagnostic.state == "contradictory_candidates"
    assert diagnostic.contradictory_candidate_count == 1
    assert diagnostic.would_abstain is False


def test_healthy_scored_evidence_reports_retrieval_sufficient() -> None:
    evidence = _candidate("mem_evidence")

    diagnostic = _diagnostic(
        raw_candidates=[evidence],
        scored_candidates=[_scored(evidence, final_score=0.8)],
    )

    assert diagnostic.state == "retrieval_sufficient"
    assert diagnostic.rationale_codes == ["scored_candidates_available"]
    assert diagnostic.direct_evidence_candidate_count == 1


def test_low_top_score_reports_retrieval_insufficient() -> None:
    evidence = _candidate("mem_evidence")

    diagnostic = _diagnostic(
        raw_candidates=[evidence],
        scored_candidates=[_scored(evidence, final_score=0.2)],
    )

    assert diagnostic.state == "retrieval_insufficient"
    assert diagnostic.rationale_codes == ["top_score_below_floor"]


def test_diagnostic_serialization_is_text_free() -> None:
    evidence = _candidate("mem_evidence")

    diagnostic = _diagnostic(
        raw_candidates=[evidence],
        scored_candidates=[_scored(evidence, final_score=0.8)],
    )

    serialized = json.dumps(diagnostic.model_dump(mode="json"))
    assert "secret text" not in serialized
    assert "canonical_text" not in serialized
