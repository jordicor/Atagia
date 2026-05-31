"""Safe per-candidate retrieval custody records.

Composer-stage rejections use generic composer reasons; ``missing_source_span``
was retired with C2.2 instead of adding composer-to-custody plumbing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from atagia.memory.embodiment_policy import candidate_allows_embodiment_boundary
from atagia.memory.mind_policy import candidate_allows_mind_boundary
from atagia.memory.realm_policy import candidate_allows_realm_boundary
from atagia.memory.space_policy import candidate_allows_space_boundary
from atagia.models.schemas_memory import RetrievalPlan, ScoredCandidate

_SOURCE_BACKED_OBJECT_TYPES = {"evidence", "interaction_contract", "state_snapshot"}
_SOURCE_BACKED_SOURCE_KINDS = {"verbatim", "extracted"}
_SUMMARY_ONLY_SOURCE_KINDS = {"summarized", "composed"}
_STALE_STATUSES = {"superseded", "archived", "deleted"}
# Same 0..1 scale as final_score; 0.5 is the minimum plausibly useful score for
# surfacing a selected-vs-rejected custody discrepancy during trace review.
_HIGH_VALUE_SCORE_FLOOR = 0.5


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
    candidate_stage_flags: dict[str, tuple[bool, bool]] = {}
    subquery_indexes = {
        sub_query.text: index
        for index, sub_query in enumerate(retrieval_plan.sub_query_plans)
    }

    selected_summary_only_count = 0
    selected_source_backed_count = 0
    selected_scores = [
        scored_by_id[memory_id].final_score
        for memory_id in selection_ranks
        if memory_id in scored_by_id
    ]
    selected_min_score = min(selected_scores) if selected_scores else None
    for _, candidate in candidate_rows:
        candidate_id = str(candidate.get("id") or "")
        source_backed = _is_source_backed_candidate(candidate)
        summary_only = _is_summary_only_candidate(candidate)
        candidate_stage_flags[candidate_id] = (source_backed, summary_only)
        if candidate_id not in selection_ranks:
            continue
        if source_backed:
            selected_source_backed_count += 1
        if summary_only:
            selected_summary_only_count += 1

    custody: list[dict[str, Any]] = []
    for fusion_position, candidate in candidate_rows:
        candidate_id = str(candidate.get("id") or "")
        filter_reason = filter_reasons.get(candidate_id)
        shortlisted = candidate_id in shortlist_ranks
        scored = scored_by_id.get(candidate_id)
        selected = candidate_id in selection_ranks
        source_backed, summary_only = candidate_stage_flags[candidate_id]
        shortlist_status = _shortlist_status(
            candidate_id,
            shortlisted=shortlisted,
            filter_reason=filter_reason,
            filtered_ids=filtered_ids,
        )
        score_status = _score_status(
            shortlisted=shortlisted,
            scored=scored is not None,
            filter_reason=filter_reason,
        )
        composer_decision = _composer_decision(
            selected=selected,
            scored=scored is not None,
        )
        drop_stage, drop_reason = _drop_stage_and_reason(
            selected=selected,
            candidate_id=candidate_id,
            filtered_ids=filtered_ids,
            filter_reason=filter_reason,
            shortlisted=shortlisted,
            scored=scored is not None,
            shortlist_status=shortlist_status,
            score_status=score_status,
            composer_decision=composer_decision,
        )
        eviction_reason = _eviction_reason(
            candidate=candidate,
            selected=selected,
            source_backed=source_backed,
            filter_reason=filter_reason,
            shortlist_status=shortlist_status,
            score_status=score_status,
            scored=scored,
            selected_summary_only_count=selected_summary_only_count,
            selected_source_backed_count=selected_source_backed_count,
            selected_min_score=selected_min_score,
        )
        high_value_rejected = _is_high_value_rejected(
            selected=selected,
            source_backed=source_backed,
            scored=scored,
        )
        record: dict[str, Any] = {
            "schema_version": 3,
            "candidate_id": candidate_id,
            "candidate_kind": _candidate_kind(candidate),
            "source_window_id": _source_window_id(candidate, candidate_id),
            "source_window_message_ids": _source_window_message_ids(candidate),
            "source_backed": source_backed,
            "summary_only": summary_only,
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
            "scope_canonical": _safe_optional_str(
                candidate.get("scope_canonical") or candidate.get("scope")
            ),
            "user_persona_id": _safe_optional_str(candidate.get("user_persona_id")),
            "platform_id": _safe_optional_str(candidate.get("platform_id")),
            "character_id": _safe_optional_str(candidate.get("character_id")),
            "conversation_id": _safe_optional_str(candidate.get("conversation_id")),
            "status": _safe_optional_str(candidate.get("status")),
            "sensitivity": _safe_optional_str(candidate.get("sensitivity")),
            "platform_locked": _safe_bool(candidate.get("platform_locked")),
            "privacy_level": _safe_int(candidate.get("privacy_level")),
            "intimacy_boundary": _safe_optional_str(
                candidate.get("intimacy_boundary") or "ordinary"
            ),
            "intimacy_boundary_confidence": _safe_float(
                candidate.get("intimacy_boundary_confidence", 0.0) or 0.0
            ),
            "retrieval_level": _safe_int(candidate.get("retrieval_level")),
            "source_kind": _safe_optional_str(candidate.get("source_kind")),
            "temporal_type": _safe_optional_str(candidate.get("temporal_type")),
            "coordinate_trace_v1": build_coordinate_trace(candidate, retrieval_plan),
            "filter_reason": filter_reason,
            "drop_stage": drop_stage,
            "drop_reason": drop_reason,
            "eviction_reason": eviction_reason,
            "high_value_rejected": high_value_rejected,
            "shortlisted": shortlisted,
            "shortlist_rank": shortlist_ranks.get(candidate_id),
            "shortlist_status": shortlist_status,
            "scored": scored is not None,
            "score_rank": score_ranks.get(candidate_id),
            "score_status": score_status,
            "scorer": _scorer_record(scored),
            "composer_decision": composer_decision,
            "selected": selected,
            "rendered": selected,
            "selection_rank": selection_ranks.get(candidate_id),
        }
        surface_class = _fact_facet_surface_class(candidate)
        if surface_class is not None:
            record["surface_class"] = surface_class
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
    if candidate.get("is_fact_facet_candidate"):
        return "fact_facet"
    if candidate.get("is_verbatim_evidence_window"):
        return "verbatim_evidence_search_window"
    return str(candidate.get("object_type") or candidate.get("candidate_kind") or "memory_object")


def _fact_facet_surface_class(candidate: dict[str, Any]) -> str | None:
    if not candidate.get("is_fact_facet_candidate"):
        return None
    surface_class = _safe_optional_str(
        candidate.get("fact_facet_surface_class") or candidate.get("surface_class")
    )
    if surface_class:
        return surface_class
    payload = candidate.get("payload_json")
    if isinstance(payload, dict):
        surface_class = _safe_optional_str(payload.get("surface_class"))
        if surface_class:
            return surface_class
        fact_facet = payload.get("fact_facet")
        if isinstance(fact_facet, dict):
            return _safe_optional_str(fact_facet.get("surface_class"))
    return None


def _is_source_backed_candidate(candidate: dict[str, Any]) -> bool:
    """Lenient custody-stage classifier for source-backed observability counts."""
    if _is_summary_only_candidate(candidate):
        return False
    if (
        candidate.get("is_verbatim_pin")
        or candidate.get("is_artifact_chunk")
        or candidate.get("is_fact_facet_candidate")
        or candidate.get("is_verbatim_evidence_window")
    ):
        return True
    object_type = str(candidate.get("object_type") or candidate.get("candidate_kind") or "")
    source_kind = str(candidate.get("source_kind") or "")
    if (
        object_type in _SOURCE_BACKED_OBJECT_TYPES
        and source_kind in _SOURCE_BACKED_SOURCE_KINDS
    ):
        return True
    if object_type in {"artifact_chunk", "raw_source_span"}:
        return True
    return _has_source_reference(candidate)


def _is_summary_only_candidate(candidate: dict[str, Any]) -> bool:
    """Custody-stage classifier for candidates that need source recovery."""
    object_type = str(candidate.get("object_type") or candidate.get("candidate_kind") or "")
    source_kind = str(candidate.get("source_kind") or "")
    if object_type == "summary_view":
        return True
    if source_kind in _SUMMARY_ONLY_SOURCE_KINDS:
        return True
    return False


def _has_source_reference(candidate: dict[str, Any]) -> bool:
    direct_keys = {
        "source_message_id",
        "source_span_id",
        "source_hash",
        "source_memory_id",
    }
    if any(_safe_optional_str(candidate.get(key)) for key in direct_keys):
        return True
    list_keys = {
        "source_message_ids",
        "source_memory_ids",
        "source_object_ids",
        "source_span_ids",
    }
    for key in list_keys:
        value = candidate.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    payload = candidate.get("payload_json")
    if not isinstance(payload, dict):
        return False
    for key in direct_keys:
        if _safe_optional_str(payload.get(key)):
            return True
    for key in list_keys:
        value = payload.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    return False


def _eviction_reason(
    *,
    candidate: dict[str, Any],
    selected: bool,
    source_backed: bool,
    filter_reason: str | None,
    shortlist_status: str,
    score_status: str,
    scored: ScoredCandidate | None,
    selected_summary_only_count: int,
    selected_source_backed_count: int,
    selected_min_score: float | None,
) -> str | None:
    if selected:
        return None
    if filter_reason is not None:
        return "policy_filtered"
    if str(candidate.get("status") or "") in _STALE_STATUSES:
        return "stale_or_superseded"
    if score_status == "llm_score_missing":
        return "low_applicability"
    if shortlist_status == "not_shortlisted":
        return "lower_score"
    if scored is not None:
        if (
            source_backed
            and selected_source_backed_count == 0
            and selected_summary_only_count > 0
        ):
            return "summary_preferred"
        if (
            selected_min_score is not None
            and float(scored.final_score) < selected_min_score
        ):
            return "lower_score"
        if selected_min_score is not None:
            return "budget_exhausted"
        return "composer_strategy"
    return "unknown"


def _is_high_value_rejected(
    *,
    selected: bool,
    source_backed: bool,
    scored: ScoredCandidate | None,
) -> bool:
    if selected or not source_backed:
        return False
    return scored is not None and float(scored.final_score) >= _HIGH_VALUE_SCORE_FLOOR


def _source_window_id(candidate: dict[str, Any], candidate_id: str) -> str | None:
    if not candidate.get("is_verbatim_evidence_window"):
        return None
    return candidate_id or None


def _source_window_message_ids(candidate: dict[str, Any]) -> list[str]:
    if not candidate.get("is_verbatim_evidence_window"):
        return []
    raw_ids = candidate.get("verbatim_evidence_window_message_ids")
    if not isinstance(raw_ids, list):
        payload = candidate.get("payload_json")
        if isinstance(payload, dict):
            raw_ids = payload.get("source_message_ids")
    if not isinstance(raw_ids, list):
        return []
    return sorted({str(message_id) for message_id in raw_ids if str(message_id)})


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
    if candidate.get("is_fact_facet_candidate"):
        channels.add("fact_facet")
    if candidate.get("is_verbatim_evidence_window"):
        channels.add("verbatim_evidence_search")
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


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _safe_optional_str(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def build_coordinate_trace(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    """Return text-free coordinate diagnostics for inspector surfaces."""

    return {
        "hard_partition": {
            "user_id": "request_user",
            "allowed": True,
            "decision": "allowed",
            "reason": "required_user_match",
        },
        "presence": {
            "active_presence_id": _safe_optional_str(retrieval_plan.active_presence_id),
            "candidate_active_presence_id": _safe_optional_str(candidate.get("active_presence_id")),
            "candidate_source_presence_id": _safe_optional_str(candidate.get("source_presence_id")),
            "candidate_presence_cluster_id": _safe_optional_str(candidate.get("presence_cluster_id")),
            **_presence_gate(candidate, retrieval_plan),
        },
        "space": {
            "active_space_id": _safe_optional_str(retrieval_plan.active_space_id),
            "active_space_boundary_mode": _safe_optional_str(retrieval_plan.active_space_boundary_mode),
            "candidate_space_id": _safe_optional_str(candidate.get("space_id")),
            "candidate_space_boundary_mode": _safe_optional_str(candidate.get("space_boundary_mode")),
            **_space_gate(candidate, retrieval_plan),
        },
        "mind": {
            "active_mind_id": _safe_optional_str(retrieval_plan.active_mind_id),
            "mind_topology": _safe_optional_str(retrieval_plan.mind_topology),
            "candidate_memory_owner_id": _safe_optional_str(candidate.get("memory_owner_id")),
            "candidate_source_mind_id": _safe_optional_str(candidate.get("source_mind_id")),
            "relation": _safe_optional_str(candidate.get("mind_relation")),
            "grant_kind": _safe_optional_str(candidate.get("mind_grant_kind")),
            "grant_target_kind": _safe_optional_str(candidate.get("mind_grant_target_kind")),
            "grant_target_id": _safe_optional_str(candidate.get("mind_grant_target_id")),
            **_mind_gate(candidate, retrieval_plan),
        },
        "embodiment": {
            "active_embodiment_id": _safe_optional_str(retrieval_plan.active_embodiment_id),
            "cross_embodiment_mode": _safe_optional_str(retrieval_plan.cross_embodiment_mode),
            "candidate_embodiment_id": _safe_optional_str(
                candidate.get("embodiment_id") or candidate.get("active_embodiment_id")
            ),
            **_embodiment_gate(candidate, retrieval_plan),
        },
        "realm": {
            "active_realm_id": _safe_optional_str(retrieval_plan.active_realm_id),
            "cross_realm_mode": _safe_optional_str(retrieval_plan.cross_realm_mode),
            "candidate_realm_id": _safe_optional_str(
                candidate.get("realm_id") or candidate.get("active_realm_id")
            ),
            "relation": _safe_optional_str(candidate.get("realm_relation")),
            "bridge_mode": _safe_optional_str(candidate.get("realm_bridge_mode")),
            **_realm_gate(candidate, retrieval_plan),
        },
        "policy": {
            "scope_filter": _enum_values(retrieval_plan.scope_filter),
            "status_filter": _enum_values(retrieval_plan.status_filter),
            "privacy_ceiling": retrieval_plan.privacy_ceiling,
            "retrieval_levels": list(retrieval_plan.retrieval_levels),
            "raw_context_access_mode": str(retrieval_plan.raw_context_access_mode),
        },
    }


def _presence_gate(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    return _gate_result(
        True,
        _presence_reason(
            active_value=retrieval_plan.active_presence_id,
            candidate_value=candidate.get("active_presence_id") or candidate.get("source_presence_id"),
        ),
    )


def _presence_reason(*, active_value: Any, candidate_value: Any) -> str:
    return _coordinate_reason(
        active_value,
        candidate_value,
        scoped_label="allowed_same_presence",
        unscoped_label="allowed_unscoped_presence",
        cross_label="allowed_cross_presence_attributed",
        missing_active_label="allowed_without_active_presence",
    )


def _space_gate(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    allowed = candidate_allows_space_boundary(candidate, retrieval_plan)
    candidate_space_id = _safe_optional_str(candidate.get("space_id"))
    candidate_mode = _space_mode(candidate.get("space_boundary_mode"))
    active_space_id = _safe_optional_str(retrieval_plan.active_space_id)
    active_mode = _space_mode(retrieval_plan.active_space_boundary_mode)
    if allowed:
        if candidate_space_id is None:
            reason = "allowed_unscoped_space"
        elif active_space_id is None:
            reason = "allowed_without_active_space_focus_or_tagged"
        elif candidate_space_id == active_space_id:
            reason = "allowed_same_space"
        elif active_mode == "tagged" and candidate_mode in {"focus", "tagged"}:
            reason = "allowed_by_active_tagged_space"
        elif candidate_mode == "tagged":
            reason = "allowed_by_tagged_space_visibility"
        else:
            reason = "allowed_by_space_boundary"
        return _gate_result(True, reason)

    if active_space_id is None:
        reason = "blocked_without_active_space"
    elif active_mode == "severance":
        reason = "blocked_by_space_severance"
    elif candidate_mode == "privacy_vault":
        reason = "blocked_by_space_privacy_vault"
    else:
        reason = "blocked_by_space_boundary"
    return _gate_result(False, reason)


def _mind_gate(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    allowed = candidate_allows_mind_boundary(candidate, retrieval_plan)
    candidate_owner_id = _safe_optional_str(
        candidate.get("memory_owner_id") or candidate.get("active_mind_id")
    )
    active_mind_id = _safe_optional_str(retrieval_plan.active_mind_id)
    topology = _safe_optional_str(retrieval_plan.mind_topology) or "unimind"
    if allowed:
        if candidate_owner_id is None:
            reason = "allowed_unowned_mind"
        elif active_mind_id is not None and candidate_owner_id == active_mind_id:
            reason = "allowed_same_mind"
        elif _safe_optional_str(candidate.get("mind_relation")) == "granted":
            reason = "allowed_by_overseer_grant"
        else:
            reason = "allowed_by_mind_boundary"
        return _gate_result(True, reason)

    if active_mind_id is None:
        reason = "blocked_without_active_mind"
    elif topology == "ojocentauri":
        reason = "blocked_missing_overseer_grant"
    else:
        reason = "blocked_by_mind_boundary"
    return _gate_result(False, reason)


def _embodiment_gate(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    allowed = candidate_allows_embodiment_boundary(candidate, retrieval_plan)
    candidate_embodiment_id = _safe_optional_str(
        candidate.get("embodiment_id") or candidate.get("active_embodiment_id")
    )
    active_embodiment_id = _safe_optional_str(retrieval_plan.active_embodiment_id)
    if allowed:
        if candidate_embodiment_id is None:
            reason = "allowed_unscoped_embodiment"
        elif active_embodiment_id is not None and candidate_embodiment_id == active_embodiment_id:
            reason = "allowed_same_embodiment"
        else:
            reason = "allowed_by_embodiment_boundary"
        return _gate_result(True, reason)

    if active_embodiment_id is None:
        reason = "blocked_without_active_embodiment"
    else:
        reason = "blocked_by_embodiment_boundary"
    return _gate_result(False, reason)


def _realm_gate(candidate: dict[str, Any], retrieval_plan: RetrievalPlan) -> dict[str, Any]:
    allowed = candidate_allows_realm_boundary(candidate, retrieval_plan)
    candidate_realm_id = _safe_optional_str(
        candidate.get("realm_id") or candidate.get("active_realm_id")
    )
    active_realm_id = _safe_optional_str(retrieval_plan.active_realm_id)
    bridge_relation = _safe_optional_str(candidate.get("realm_relation"))
    bridge_mode = _safe_optional_str(candidate.get("realm_bridge_mode"))
    if allowed:
        if candidate_realm_id is None:
            reason = "allowed_unscoped_realm"
        elif active_realm_id is not None and candidate_realm_id == active_realm_id:
            reason = "allowed_same_realm"
        elif bridge_relation == "cross" and bridge_mode is not None:
            reason = f"allowed_by_realm_bridge_{bridge_mode}"
        else:
            reason = "allowed_by_realm_boundary"
        return _gate_result(True, reason)

    if active_realm_id is None:
        reason = "blocked_without_active_realm"
    elif bridge_relation == "cross":
        reason = "blocked_by_realm_bridge_missing_or_invalid"
    else:
        reason = "blocked_by_realm_no_bridge"
    return _gate_result(False, reason)


def _gate_result(allowed: bool, reason: str) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "decision": "allowed" if allowed else "blocked",
        "reason": reason,
    }


def _coordinate_reason(
    active_value: Any,
    candidate_value: Any,
    *,
    scoped_label: str,
    unscoped_label: str,
    cross_label: str,
    missing_active_label: str,
) -> str:
    active = _safe_optional_str(active_value)
    candidate = _safe_optional_str(candidate_value)
    if candidate is None:
        return unscoped_label
    if active is not None and candidate == active:
        return scoped_label
    if active is None:
        return missing_active_label
    return cross_label


def _space_mode(value: Any) -> str:
    normalized = _safe_optional_str(value) or "focus"
    if normalized in {"focus", "severance", "privacy_vault", "tagged"}:
        return normalized
    return "focus"


def _enum_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(getattr(value, "value", value)) for value in values]


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


def _drop_stage_and_reason(
    *,
    selected: bool,
    candidate_id: str,
    filtered_ids: set[str],
    filter_reason: str | None,
    shortlisted: bool,
    scored: bool,
    shortlist_status: str,
    score_status: str,
    composer_decision: str,
) -> tuple[str | None, str | None]:
    if selected:
        return None, None
    if filter_reason is not None:
        return "post_scope_coordinate_lifecycle", filter_reason
    if candidate_id not in filtered_ids:
        return "post_scope_coordinate_lifecycle", shortlist_status
    if not shortlisted:
        return "shortlist", shortlist_status
    if not scored:
        return "post_applicability_rerank", score_status
    return "composer", composer_decision


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
