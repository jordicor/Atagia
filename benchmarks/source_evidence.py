"""Utilities for passing official benchmark evidence into graders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

SourceEvidence = dict[str, str]


def source_evidence_from_turns(
    *,
    evidence_turn_ids: Sequence[str],
    turns: Iterable[Any],
    conversation_id: str,
) -> list[SourceEvidence]:
    """Return official source turns in benchmark evidence order."""
    if not evidence_turn_ids:
        return []

    evidence_order = {turn_id: index for index, turn_id in enumerate(evidence_turn_ids)}
    evidence_items: list[SourceEvidence] = []
    for turn in turns:
        turn_id = str(getattr(turn, "turn_id", "") or "")
        if turn_id not in evidence_order:
            continue
        speaker = str(
            getattr(turn, "speaker", "")
            or getattr(turn, "role", "")
            or ""
        )
        item: SourceEvidence = {
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "timestamp": str(getattr(turn, "timestamp", "") or ""),
            "speaker": speaker,
            "role": str(getattr(turn, "role", "") or ""),
            "text": str(getattr(turn, "text", "") or ""),
        }
        session_id = str(getattr(turn, "session_id", "") or "")
        if session_id:
            item["session_id"] = session_id
        evidence_items.append(item)

    evidence_items.sort(key=lambda item: evidence_order.get(item["turn_id"], 0))
    return evidence_items
