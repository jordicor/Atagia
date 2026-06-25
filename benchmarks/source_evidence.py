"""Utilities for passing official benchmark evidence into graders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re
from typing import Any

SourceEvidence = dict[str, Any]

_EVIDENCE_CITATION_PATTERN = re.compile(r"\bD:?(?P<session>\d+):0*(?P<turn>\d+)\b")


def normalize_evidence_turn_ids(raw_evidence: Any) -> list[str]:
    """Normalize structured benchmark citation fields into turn ids.

    LoCoMo occasionally stores multiple citations in one string, such as
    ``"D8:6; D9:17"`` or ``"D9:1 D4:4 D4:6"``. This helper only parses the
    mechanical citation grammar and preserves order; it does not infer missing
    or semantically related evidence.
    """
    if raw_evidence is None:
        return []
    values = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        matches = list(_EVIDENCE_CITATION_PATTERN.finditer(text))
        turn_ids = [
            f"D{match.group('session')}:{int(match.group('turn'))}"
            for match in matches
        ]
        if not turn_ids:
            turn_ids = [text]
        for turn_id in turn_ids:
            if turn_id in seen:
                continue
            normalized.append(turn_id)
            seen.add(turn_id)
    return normalized


def missing_evidence_turn_ids(
    *,
    evidence_turn_ids: Sequence[str],
    turns: Iterable[Any],
) -> list[str]:
    """Return official evidence turn ids that do not resolve to raw turns."""
    available_turn_ids = {
        str(getattr(turn, "turn_id", "") or "")
        for turn in turns
    }
    return [
        turn_id
        for turn_id in evidence_turn_ids
        if turn_id not in available_turn_ids
    ]


def validate_evidence_turn_ids(
    *,
    evidence_turn_ids: Sequence[str],
    turns: Iterable[Any],
    dataset_name: str,
    question_id: str,
    conversation_id: str,
    require_non_empty: bool = False,
) -> None:
    """Raise an actionable error when official evidence ids are unresolved."""
    if require_non_empty and not evidence_turn_ids:
        raise ValueError(
            f"{dataset_name} question {question_id} in conversation "
            f"{conversation_id} has no evidence_turn_ids"
        )
    missing = missing_evidence_turn_ids(
        evidence_turn_ids=evidence_turn_ids,
        turns=turns,
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{dataset_name} question {question_id} in conversation "
            f"{conversation_id} references unresolved evidence_turn_ids: "
            f"{missing_text}"
        )


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
        metadata = getattr(turn, "metadata", None)
        if isinstance(metadata, dict):
            blip_caption = str(metadata.get("blip_caption") or "").strip()
            if blip_caption:
                item["blip_caption"] = blip_caption
        attachment_texts = _attachment_texts(getattr(turn, "attachments", []) or [])
        if attachment_texts:
            item["attachment_text"] = "\n\n".join(attachment_texts)
        evidence_items.append(item)

    evidence_items.sort(key=lambda item: evidence_order.get(item["turn_id"], 0))
    return evidence_items


def _attachment_texts(raw_attachments: Any) -> list[str]:
    """Extract prompt-safe text from benchmark attachment payloads."""
    if not isinstance(raw_attachments, list):
        return []
    texts: list[str] = []
    for attachment in raw_attachments:
        if not isinstance(attachment, dict):
            continue
        content_text = str(attachment.get("content_text") or "").strip()
        if content_text:
            texts.append(content_text)
    return texts
