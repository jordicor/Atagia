"""Community corrections loader for LoCoMo benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_QUESTION_ID_PATTERN = re.compile(r"^locomo_(\d+)_qa(\d+)$")

# Citation-only errors don't affect scoring
_CITATION_ONLY_TYPES = {"WRONG_CITATION"}


def load_community_corrections(
    errors_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    """Load dial481/locomo-audit errors.json and convert to Atagia format.

    Returns a corrections dict keyed by our question_id format
    (e.g. "conv-26:q57"). Only score-corrupting errors are included;
    citation-only errors are skipped.
    """
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    sample_ids = _load_sample_ids(dataset_path)

    corrections: dict[str, Any] = {}
    for entry in errors:
        if entry.get("error_type") in _CITATION_ONLY_TYPES:
            continue

        match = _QUESTION_ID_PATTERN.match(entry.get("question_id", ""))
        if match is None:
            continue

        conv_index = int(match.group(1))
        question_index = int(match.group(2))

        if conv_index >= len(sample_ids):
            continue

        correct_answer = entry.get("correct_answer", "")
        if not correct_answer:
            continue

        sample_id = sample_ids[conv_index]
        our_question_id = f"{sample_id}:q{question_index + 1}"

        corrections[our_question_id] = {
            "corrected_ground_truth": correct_answer,
            "original_ground_truth": entry.get("golden_answer", ""),
            "reason": entry.get("reasoning", ""),
            "error_type": entry.get("error_type", ""),
            "source": "community/dial481/locomo-audit",
        }

    return corrections


def _load_sample_ids(dataset_path: Path) -> list[str]:
    """Extract sample_ids from the LoCoMo dataset file."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        str(sample.get("sample_id") or f"locomo-{index:02d}")
        for index, sample in enumerate(raw, start=1)
    ]
