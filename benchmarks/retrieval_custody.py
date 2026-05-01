"""Per-candidate retrieval custody helpers for benchmark traces."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_replay import PipelineResult


def build_retrieval_custody(
    *,
    pipeline_result: PipelineResult,
    selected_memory_ids: list[str],
    user_id: str | None = None,
    retrieval_event_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-candidate custody records from a pipeline result."""
    core_custody = getattr(pipeline_result, "candidate_custody", None)
    if core_custody:
        return [
            _with_optional_edge_fields(
                dict(record),
                user_id=user_id,
                retrieval_event_id=retrieval_event_id,
            )
            for record in core_custody
            if isinstance(record, dict)
        ]

    selected = set(selected_memory_ids)
    scored_by_id = {
        candidate.memory_id: candidate
        for candidate in pipeline_result.scored_candidates
    }
    custody: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(pipeline_result.raw_candidates, start=1):
        candidate_id = str(raw_candidate.get("id") or "")
        scored = scored_by_id.get(candidate_id)
        channel_ranks = raw_candidate.get("channel_ranks") or {}
        channels = _candidate_channels(raw_candidate, channel_ranks)
        selected_for_context = candidate_id in selected
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_kind": _candidate_kind(raw_candidate),
            "fusion_position": index,
            "channels": sorted(channels),
            "channel_ranks": {
                str(channel): rank
                for channel, rank in channel_ranks.items()
                if rank is not None
            },
            "matched_sub_queries": [
                str(value)
                for value in raw_candidate.get("matched_sub_queries") or []
            ],
            "fused_score": raw_candidate.get("rrf_score"),
            "scored": scored is not None,
            "filter_reason": None if scored is not None else "not_scored_or_filtered",
            "scorer": _scorer_record(scored),
            "composer_decision": "selected" if selected_for_context else "not_selected",
            "selected": selected_for_context,
        }
        custody.append(
            _with_optional_edge_fields(
                record,
                user_id=user_id,
                retrieval_event_id=retrieval_event_id,
            )
        )
    return custody


def _with_optional_edge_fields(
    record: dict[str, Any],
    *,
    user_id: str | None,
    retrieval_event_id: str | None,
) -> dict[str, Any]:
    if user_id is not None:
        record["user_id"] = user_id
    if retrieval_event_id is not None:
        record["retrieval_event_id"] = retrieval_event_id
    return record


def _candidate_channels(
    raw_candidate: dict[str, Any],
    channel_ranks: dict[str, Any],
) -> list[str]:
    channels = [
        str(channel)
        for channel, rank in channel_ranks.items()
        if rank is not None
    ]
    if raw_candidate.get("is_verbatim_pin") and "verbatim_pin" not in channels:
        channels.append("verbatim_pin")
    if raw_candidate.get("is_artifact_chunk") and "artifact_chunk" not in channels:
        channels.append("artifact_chunk")
    if (
        raw_candidate.get("is_verbatim_evidence_window")
        and "verbatim_evidence_search" not in channels
    ):
        channels.append("verbatim_evidence_search")
    return channels


def _candidate_kind(candidate: dict[str, Any]) -> str:
    if candidate.get("is_verbatim_pin"):
        return "verbatim_pin"
    if candidate.get("is_artifact_chunk"):
        return "artifact_chunk"
    if candidate.get("is_verbatim_evidence_window"):
        return "verbatim_evidence_search_window"
    return str(candidate.get("object_type") or candidate.get("candidate_kind") or "memory_object")


def _scorer_record(scored: Any | None) -> dict[str, Any] | None:
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
