"""Text-free shadow diagnostics for retrieval sufficiency."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import (
    MemoryObjectType,
    RawContextAccessMode,
    RetrievalPlan,
    RetrievalSufficiencyDiagnostic,
    RetrievalSufficiencyRationaleCode,
    ScoredCandidate,
)

_SUFFICIENCY_SCORE_FLOOR = 0.30
_HIERARCHICAL_SUMMARY_LEVELS = {1, 2}


def build_retrieval_sufficiency_diagnostic(
    *,
    raw_candidates: list[dict[str, Any]],
    filtered_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    scored_candidates: list[ScoredCandidate],
    retrieval_plan: RetrievalPlan,
    contradiction_tension_threshold: float,
) -> RetrievalSufficiencyDiagnostic:
    """Build a deterministic shadow diagnostic without raw text or LLM calls."""
    scored_objects = [dict(candidate.memory_object) for candidate in scored_candidates]
    top_score = max((float(candidate.final_score) for candidate in scored_candidates), default=0.0)
    raw_message_count = _count_channel_candidates(raw_candidates, "raw_message")
    artifact_count = _count_channel_candidates(raw_candidates, "artifact_chunk")
    direct_evidence_count = sum(1 for candidate in scored_objects if _is_direct_evidence(candidate))
    summary_count = sum(1 for candidate in scored_objects if _is_summary_candidate(candidate))
    unsupported_summary_count = sum(
        1
        for candidate in scored_objects
        if _is_unsupported_hierarchical_summary(
            candidate,
            scored_ids={str(scored.memory_id) for scored in scored_candidates},
            shortlist_ids={str(candidate.get("id") or "") for candidate in shortlist},
        )
    )
    contradictory_count = sum(
        1
        for candidate in scored_objects
        if _is_contradictory_belief(
            candidate,
            tension_threshold=contradiction_tension_threshold,
        )
    )

    rationale_codes: list[RetrievalSufficiencyRationaleCode] = []
    would_expand_channels: list[str] = []
    state = "retrieval_sufficient"
    confidence = _confidence_from_score(top_score)
    would_abstain = False

    if not raw_candidates:
        state = "insufficient_no_candidates"
        confidence = 0.95
        rationale_codes.append("raw_candidates_empty")
        would_expand_channels.extend(["fts", "embedding"])
        would_abstain = True
    elif not scored_candidates:
        state = "insufficient_no_scored_candidates"
        confidence = 0.9
        rationale_codes.append("scored_candidates_empty")
        would_expand_channels.extend(["fts", "embedding"])
        would_abstain = True
    elif _artifact_support_missing(retrieval_plan, artifact_count):
        state = "insufficient_need_artifact"
        confidence = 0.82
        rationale_codes.append("artifact_requested_no_artifact_candidates")
        would_expand_channels.append("artifact_chunk")
        would_abstain = True
    elif _raw_evidence_missing(
        retrieval_plan,
        direct_evidence_count=direct_evidence_count,
        raw_message_count=raw_message_count,
    ):
        state = "insufficient_need_more_raw_evidence"
        confidence = 0.82
        rationale_codes.append("raw_evidence_requested_no_direct_candidates")
        would_expand_channels.append("raw_message")
        would_abstain = True
    elif _summary_support_missing(
        scored_objects,
        unsupported_summary_count=unsupported_summary_count,
        direct_evidence_count=direct_evidence_count,
    ):
        state = "insufficient_summary_support"
        confidence = 0.8
        rationale_codes.extend(_summary_support_rationale(scored_objects))
        would_expand_channels.append("raw_message")
        would_abstain = True
    elif contradictory_count:
        state = "contradictory_candidates"
        confidence = 0.7
        rationale_codes.append("contradictory_belief_candidate")
    elif top_score < _SUFFICIENCY_SCORE_FLOOR:
        state = "retrieval_insufficient"
        confidence = 0.65
        rationale_codes.append("top_score_below_floor")
        would_expand_channels.extend(["fts", "embedding"])
        would_abstain = True
    else:
        rationale_codes.append("scored_candidates_available")

    return RetrievalSufficiencyDiagnostic(
        state=state,
        confidence=confidence,
        rationale_codes=_dedupe(rationale_codes),
        would_expand_channels=_dedupe(would_expand_channels),
        would_abstain=would_abstain,
        candidate_count=len(raw_candidates),
        filtered_candidate_count=len(filtered_candidates),
        shortlist_count=len(shortlist),
        scored_candidate_count=len(scored_candidates),
        top_score=top_score,
        direct_evidence_candidate_count=direct_evidence_count,
        summary_candidate_count=summary_count,
        raw_message_candidate_count=raw_message_count,
        artifact_candidate_count=artifact_count,
        unsupported_summary_candidate_count=unsupported_summary_count,
        contradictory_candidate_count=contradictory_count,
    )


def _count_channel_candidates(candidates: list[dict[str, Any]], channel: str) -> int:
    return sum(1 for candidate in candidates if _has_channel(candidate, channel))


def _has_channel(candidate: dict[str, Any], channel: str) -> bool:
    if channel == "raw_message" and candidate.get("is_raw_message_window"):
        return True
    if channel == "artifact_chunk" and candidate.get("is_artifact_chunk"):
        return True
    retrieval_sources = candidate.get("retrieval_sources")
    if isinstance(retrieval_sources, list) and channel in {str(item) for item in retrieval_sources}:
        return True
    channel_ranks = candidate.get("channel_ranks")
    return isinstance(channel_ranks, dict) and channel_ranks.get(channel) is not None


def _is_direct_evidence(candidate: dict[str, Any]) -> bool:
    if candidate.get("is_raw_message_window") or candidate.get("is_artifact_chunk"):
        return True
    return str(candidate.get("object_type")) in {
        MemoryObjectType.EVIDENCE.value,
        MemoryObjectType.INTERACTION_CONTRACT.value,
        MemoryObjectType.STATE_SNAPSHOT.value,
    }


def _is_summary_candidate(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("object_type")) == MemoryObjectType.SUMMARY_VIEW.value


def _is_unsupported_hierarchical_summary(
    candidate: dict[str, Any],
    *,
    scored_ids: set[str],
    shortlist_ids: set[str],
) -> bool:
    if not _is_summary_candidate(candidate):
        return False
    payload_json = candidate.get("payload_json") or {}
    if not isinstance(payload_json, dict):
        return True
    try:
        hierarchy_level = int(payload_json.get("hierarchy_level", -1))
    except (TypeError, ValueError):
        return True
    if hierarchy_level not in _HIERARCHICAL_SUMMARY_LEVELS:
        return False
    source_ids = {
        str(item).strip()
        for item in payload_json.get("source_object_ids", [])
        if str(item).strip()
    }
    if not source_ids:
        return True
    support_ids = scored_ids | shortlist_ids
    return source_ids.isdisjoint(support_ids)


def _is_contradictory_belief(
    candidate: dict[str, Any],
    *,
    tension_threshold: float,
) -> bool:
    if str(candidate.get("object_type")) != MemoryObjectType.BELIEF.value:
        return False
    try:
        tension_score = float(candidate.get("tension_score") or 0.0)
    except (TypeError, ValueError):
        return False
    return tension_score >= tension_threshold


def _artifact_support_missing(retrieval_plan: RetrievalPlan, artifact_count: int) -> bool:
    return retrieval_plan.raw_context_access_mode == "artifact" and artifact_count == 0


def _raw_evidence_missing(
    retrieval_plan: RetrievalPlan,
    *,
    direct_evidence_count: int,
    raw_message_count: int,
) -> bool:
    raw_mode: RawContextAccessMode = retrieval_plan.raw_context_access_mode
    if raw_mode != "verbatim" and not retrieval_plan.exact_recall_mode and not retrieval_plan.require_evidence_regrounding:
        return False
    return direct_evidence_count == 0 and raw_message_count == 0


def _summary_support_missing(
    scored_objects: list[dict[str, Any]],
    *,
    unsupported_summary_count: int,
    direct_evidence_count: int,
) -> bool:
    if unsupported_summary_count <= 0:
        return False
    if unsupported_summary_count == len(scored_objects):
        return True
    top_candidate = scored_objects[0] if scored_objects else {}
    return _is_summary_candidate(top_candidate) and direct_evidence_count == 0


def _summary_support_rationale(
    scored_objects: list[dict[str, Any]],
) -> list[RetrievalSufficiencyRationaleCode]:
    if len(scored_objects) == 1:
        return ["unsupported_summary_only"]
    top_candidate = scored_objects[0] if scored_objects else {}
    if _is_summary_candidate(top_candidate):
        return ["top_candidate_unsupported_summary"]
    return ["unsupported_summary_only"]


def _confidence_from_score(score: float) -> float:
    return max(0.5, min(0.99, float(score)))


def _dedupe[T](values: list[T]) -> list[T]:
    deduped: list[T] = []
    seen: set[T] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
