"""Community corrections loader for LoCoMo benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmarks.source_evidence import normalize_evidence_turn_ids

_QUESTION_ID_PATTERN = re.compile(r"^locomo_(\d+)_qa(\d+)$")


def load_community_corrections(
    errors_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    """Load dial481/locomo-audit errors.json and convert to Atagia format.

    Returns a corrections dict keyed by our question_id format
    (e.g. "conv-26:q57"). Answer corrections and citation/evidence
    corrections are both preserved because source-aware judging treats the
    evidence packet as the official oracle.
    """
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    sample_ids = _load_sample_ids(dataset_path)

    corrections: dict[str, Any] = {}
    for entry in errors:
        match = _QUESTION_ID_PATTERN.match(entry.get("question_id", ""))
        if match is None:
            continue

        conv_index = int(match.group(1))
        question_index = int(match.group(2))

        if conv_index >= len(sample_ids):
            continue

        correct_answer = entry.get("correct_answer", "")
        corrected_evidence_turn_ids = normalize_evidence_turn_ids(
            entry.get("correct_evidence")
        )
        if not correct_answer and not corrected_evidence_turn_ids:
            continue

        sample_id = sample_ids[conv_index]
        our_question_id = f"{sample_id}:q{question_index + 1}"

        correction: dict[str, Any] = {
            "original_ground_truth": entry.get("golden_answer", ""),
            "reason": entry.get("reasoning", ""),
            "error_type": entry.get("error_type", ""),
            "source": "community/dial481/locomo-audit",
        }
        if correct_answer:
            correction["corrected_ground_truth"] = correct_answer
        if corrected_evidence_turn_ids:
            correction["corrected_evidence_turn_ids"] = corrected_evidence_turn_ids
            correction["original_evidence_turn_ids"] = normalize_evidence_turn_ids(
                entry.get("cited_evidence")
            )

        corrections[our_question_id] = correction

    return corrections


def _load_sample_ids(dataset_path: Path) -> list[str]:
    """Extract sample_ids from the LoCoMo dataset file."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        str(sample.get("sample_id") or f"locomo-{index:02d}")
        for index, sample in enumerate(raw, start=1)
    ]
