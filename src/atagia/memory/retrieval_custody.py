"""Safe per-candidate retrieval custody records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from atagia.models.schemas_memory import RetrievalPlan, ScoredCandidate


def build_candidate_custody(
    *,
    raw_candidates: list[dict[str, Any]],
    filtered_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    scored_candidates: list[ScoredCandidate],
    selected_memory_ids: list[str],
    retrieval_plan: RetrievalPlan,
    filter_reasons_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build text-free custody records for retrieval candidates."""
    filter_reasons = filter_reasons_by_id or {}
    candidate_rows = _ordered_candidate_rows(
        raw_candidates=raw_candidates,
        shortlist=shortlist,
        scored_candidates=scored_candidates,
    )
    filtered_ids = _candidate_id_set(filtered_candidates)
    shortlist_ranks = _rank_by_candidate_id(shortlist)
    scored_by_id = {candidate.memory_id: candidate for candidate in scored_candidates}
    score_ranks = {
        candidate.memory_id: rank
        for rank, candidate in enumerate(scored_candidates, start=1)
    }
    selection_ranks = {
        memory_id: rank
        for rank, memory_id in enumerate(selected_memory_ids, start=1)
    }
    subquery_indexes = {
        sub_query.text: index
        for index, sub_query in enumerate(retrieval_plan.sub_query_plans)
    }

    custody: list[dict[str, Any]] = []
    for fusion_position, candidate in candidate_rows:
        candidate_id = str(candidate.get("id") or "")
        filter_reason = filter_reasons.get(candidate_id)
        shortlisted = candidate_id in shortlist_ranks
        scored = scored_by_id.get(candidate_id)
        selected = candidate_id in selection_ranks
        record: dict[str, Any] = {
            "schema_version": 2,
            "candidate_id": candidate_id,
            "candidate_kind": _candidate_kind(candidate),
            "fusion_position": fusion_position,
            "channels": _candidate_channels(candidate),
            "channel_ranks": _safe_rank_map(candidate.get("channel_ranks")),
            "retrieval_sources": _safe_str_list(candidate.get("retrieval_sources")),
            "matched_subquery_indexes": _matched_subquery_indexes(
                candidate.get("matched_sub_queries"),
                subquery_indexes,
            ),
            "matched_subquery_count": _safe_list_count(candidate.get("matched_sub_queries")),
            "subquery_ranks": _subquery_rank_indexes(
                candidate.get("subquery_ranks"),
                subquery_indexes,
            ),
            "fused_score": _safe_float(candidate.get("rrf_score")),
            "scope": _safe_optional_str(candidate.get("scope")),
            "status": _safe_optional_str(candidate.get("status")),
            "privacy_level": _safe_int(candidate.get("privacy_level")),
            "retrieval_level": _safe_int(candidate.get("retrieval_level")),
            "source_kind": _safe_optional_str(candidate.get("source_kind")),
            "temporal_type": _safe_optional_str(candidate.get("temporal_type")),
            "filter_reason": filter_reason,
            "shortlisted": shortlisted,
            "shortlist_rank": shortlist_ranks.get(candidate_id),
            "shortlist_status": _shortlist_status(
                candidate_id,
                shortlisted=shortlisted,
                filter_reason=filter_reason,
                filtered_ids=filtered_ids,
            ),
            "scored": scored is not None,
            "score_rank": score_ranks.get(candidate_id),
            "score_status": _score_status(
                shortlisted=shortlisted,
                scored=scored is not None,
                filter_reason=filter_reason,
            ),
            "scorer": _scorer_record(scored),
            "composer_decision": _composer_decision(
                selected=selected,
                scored=scored is not None,
            ),
            "selected": selected,
            "selection_rank": selection_ranks.get(candidate_id),
        }
        custody.append(record)
    return custody


def _ordered_candidate_rows(
    *,
    raw_candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    scored_candidates: list[ScoredCandidate],
) -> list[tuple[int | None, dict[str, Any]]]:
    rows: list[tuple[int | None, dict[str, Any]]] = []
    seen: set[str] = set()
    for position, candidate in enumerate(raw_candidates, start=1):
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in seen:
            continue
        rows.append((position, candidate))
        seen.add(candidate_id)

    for candidate in shortlist:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in seen:
            continue
        rows.append((None, candidate))
        seen.add(candidate_id)

    for scored in scored_candidates:
        candidate_id = scored.memory_id
        if candidate_id in seen:
            continue
        memory_object = scored.memory_object
        rows.append((None, memory_object if isinstance(memory_object, dict) else {"id": candidate_id}))
        seen.add(candidate_id)
    return rows


def _candidate_id_set(candidates: Iterable[dict[str, Any]]) -> set[str]:
    return {str(candidate.get("id") or "") for candidate in candidates}


def _rank_by_candidate_id(candidates: list[dict[str, Any]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("id") or "")
        ranks.setdefault(candidate_id, rank)
    return ranks


def _candidate_kind(candidate: dict[str, Any]) -> str:
    if candidate.get("is_verbatim_pin"):
        return "verbatim_pin"
    if candidate.get("is_artifact_chunk"):
        return "artifact_chunk"
    if candidate.get("is_raw_message_window"):
        return "raw_message_window"
    return str(candidate.get("object_type") or candidate.get("candidate_kind") or "memory_object")


def _candidate_channels(candidate: dict[str, Any]) -> list[str]:
    channels = set(_safe_str_list(candidate.get("retrieval_sources")))
    channel_ranks = candidate.get("channel_ranks")
    if isinstance(channel_ranks, dict):
        channels.update(
            str(channel)
            for channel, rank in channel_ranks.items()
            if rank is not None
        )
    if candidate.get("is_verbatim_pin"):
        channels.add("verbatim_pin")
    if candidate.get("is_artifact_chunk"):
        channels.add("artifact_chunk")
    if candidate.get("is_raw_message_window"):
        channels.add("raw_message")
    return sorted(channels)


def _safe_rank_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    ranks: dict[str, int] = {}
    for key, raw_rank in value.items():
        rank = _safe_int(raw_rank)
        if rank is not None:
            ranks[str(key)] = rank
    return ranks


def _safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if str(item)})


def _matched_subquery_indexes(
    value: Any,
    subquery_indexes: dict[str, int],
) -> list[int]:
    if not isinstance(value, list):
        return []
    indexes = {
        subquery_indexes[item]
        for item in value
        if isinstance(item, str) and item in subquery_indexes
    }
    return sorted(indexes)


def _safe_list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _subquery_rank_indexes(
    value: Any,
    subquery_indexes: dict[str, int],
) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    ranks: dict[str, int] = {}
    for subquery_text, raw_rank in value.items():
        if not isinstance(subquery_text, str) or subquery_text not in subquery_indexes:
            continue
        rank = _safe_int(raw_rank)
        if rank is not None:
            ranks[str(subquery_indexes[subquery_text])] = rank
    return ranks


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _shortlist_status(
    candidate_id: str,
    *,
    shortlisted: bool,
    filter_reason: str | None,
    filtered_ids: set[str],
) -> str:
    if shortlisted:
        return "shortlisted"
    if filter_reason is not None or candidate_id not in filtered_ids:
        return "filtered_before_shortlist"
    return "not_shortlisted"


def _score_status(
    *,
    shortlisted: bool,
    scored: bool,
    filter_reason: str | None,
) -> str:
    if scored:
        return "scored"
    if filter_reason is not None:
        return "filtered_before_scoring"
    if shortlisted:
        return "llm_score_missing"
    return "not_shortlisted"


def _composer_decision(*, selected: bool, scored: bool) -> str:
    if selected:
        return "selected"
    if scored:
        return "not_selected_after_scoring"
    return "not_scored"


def _scorer_record(scored: ScoredCandidate | None) -> dict[str, Any] | None:
    if scored is None:
        return None
    return {
        "llm_applicability": scored.llm_applicability,
        "retrieval_score": scored.retrieval_score,
        "vitality_boost": scored.vitality_boost,
        "confirmation_boost": scored.confirmation_boost,
        "need_boost": scored.need_boost,
        "penalty": scored.penalty,
        "final_score": scored.final_score,
    }
