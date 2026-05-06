"""Small aggregate summaries for benchmark retrieval custody records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def summarize_retrieval_custody(
    custody_groups: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, object]:
    """Summarize per-candidate custody records without copying raw text."""
    candidate_count = 0
    selected_count = 0
    channel_counts: Counter[str] = Counter()
    selected_channel_counts: Counter[str] = Counter()
    candidate_kind_counts: Counter[str] = Counter()
    composer_decision_counts: Counter[str] = Counter()
    filter_reason_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    selected_scope_counts: Counter[str] = Counter()
    sensitivity_counts: Counter[str] = Counter()
    selected_sensitivity_counts: Counter[str] = Counter()
    platform_locked_count = 0

    for group in custody_groups:
        for record in group:
            candidate_count += 1
            channels = record.get("channels")
            if not isinstance(channels, list):
                channels = []
            normalized_channels = [str(channel) for channel in channels]
            channel_counts.update(normalized_channels)
            candidate_kind = record.get("candidate_kind")
            if candidate_kind is not None:
                candidate_kind_counts[str(candidate_kind)] += 1
            composer_decision = record.get("composer_decision")
            if composer_decision is not None:
                composer_decision_counts[str(composer_decision)] += 1
            filter_reason = record.get("filter_reason")
            if filter_reason is not None:
                filter_reason_counts[str(filter_reason)] += 1
            scope = record.get("scope_canonical") or record.get("scope")
            if scope is not None:
                scope_counts[str(scope)] += 1
            sensitivity = record.get("sensitivity")
            if sensitivity is not None:
                sensitivity_counts[str(sensitivity)] += 1
            if record.get("platform_locked") is True:
                platform_locked_count += 1
            if record.get("selected") is True:
                selected_count += 1
                selected_channel_counts.update(normalized_channels)
                if scope is not None:
                    selected_scope_counts[str(scope)] += 1
                if sensitivity is not None:
                    selected_sensitivity_counts[str(sensitivity)] += 1

    summary: dict[str, object] = {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "channel_counts": dict(sorted(channel_counts.items())),
        "selected_channel_counts": dict(sorted(selected_channel_counts.items())),
        "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
        "composer_decision_counts": dict(sorted(composer_decision_counts.items())),
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
    }
    if scope_counts:
        summary["scope_counts"] = dict(sorted(scope_counts.items()))
        summary["selected_scope_counts"] = dict(sorted(selected_scope_counts.items()))
    if sensitivity_counts:
        summary["sensitivity_counts"] = dict(sorted(sensitivity_counts.items()))
        summary["selected_sensitivity_counts"] = dict(sorted(selected_sensitivity_counts.items()))
    if platform_locked_count:
        summary["platform_locked_count"] = platform_locked_count
    return summary


def format_retrieval_custody_summary(value: object) -> str:
    """Return a compact terminal line for aggregate retrieval custody counters."""
    if not isinstance(value, dict):
        return "Retrieval custody: unavailable"
    candidate_count = _int_summary_value(value, "candidate_count")
    selected_count = _int_summary_value(value, "selected_count")
    parts = [
        f"Retrieval custody: candidates={candidate_count} selected={selected_count}",
        f"channels={_format_count_mapping(value.get('channel_counts'))}",
        f"selected_channels={_format_count_mapping(value.get('selected_channel_counts'))}",
    ]
    if value.get("scope_counts"):
        parts.append(f"scopes={_format_count_mapping(value.get('scope_counts'))}")
    if value.get("sensitivity_counts"):
        parts.append(f"sensitivity={_format_count_mapping(value.get('sensitivity_counts'))}")
    parts.extend(
        [
            f"kinds={_format_count_mapping(value.get('candidate_kind_counts'))}",
            f"decisions={_format_count_mapping(value.get('composer_decision_counts'))}",
            f"filters={_format_count_mapping(value.get('filter_reason_counts'))}",
        ]
    )
    return " ".join(parts)


def _format_count_mapping(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for key in sorted(value):
        try:
            amount = int(value[key])
        except (TypeError, ValueError):
            continue
        if amount:
            parts.append(f"{key}={amount}")
    return " ".join(parts) if parts else "none"


def _int_summary_value(value: dict[str, object], key: str) -> int:
    try:
        return int(value.get(key, 0))
    except (TypeError, ValueError):
        return 0
