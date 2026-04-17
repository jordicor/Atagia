"""Tests for community corrections loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.corrections import load_community_corrections


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    """Minimal LoCoMo dataset with two conversations."""
    dataset = [
        {"sample_id": "conv-26", "conversation": {}, "qa": []},
        {"sample_id": "conv-42", "conversation": {}, "qa": []},
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


@pytest.fixture
def errors_file(tmp_path: Path) -> Path:
    """Community audit errors with mixed types."""
    errors = [
        {
            "question_id": "locomo_0_qa56",
            "question": "What symbols are important to Caroline?",
            "golden_answer": "Rainbow flag, transgender symbol",
            "category": 1,
            "error_type": "HALLUCINATION",
            "cited_evidence": ["D14:15"],
            "correct_evidence": ["D14:15"],
            "reasoning": "Transgender symbol not in transcript.",
            "correct_answer": "Rainbow flag, eagle",
        },
        {
            "question_id": "locomo_0_qa10",
            "question": "Some question",
            "golden_answer": "Original answer",
            "category": 2,
            "error_type": "WRONG_CITATION",
            "cited_evidence": ["D3:1"],
            "correct_evidence": ["D3:2"],
            "reasoning": "Citation points to wrong turn.",
            "correct_answer": "Original answer",
        },
        {
            "question_id": "locomo_1_qa3",
            "question": "Question from conv-42",
            "golden_answer": "Wrong date",
            "category": 3,
            "error_type": "TEMPORAL_ERROR",
            "cited_evidence": ["D1:5"],
            "correct_evidence": ["D1:5"],
            "reasoning": "Date calculation is off.",
            "correct_answer": "Correct date",
        },
    ]
    path = tmp_path / "errors.json"
    path.write_text(json.dumps(errors), encoding="utf-8")
    return path


def test_converts_question_ids_to_atagia_format(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert "conv-26:q57" in corrections
    assert corrections["conv-26:q57"]["corrected_ground_truth"] == "Rainbow flag, eagle"


def test_maps_second_conversation(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert "conv-42:q4" in corrections
    assert corrections["conv-42:q4"]["corrected_ground_truth"] == "Correct date"


def test_filters_citation_only_errors(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    # WRONG_CITATION entry (locomo_0_qa10 -> conv-26:q11) should be excluded
    assert "conv-26:q11" not in corrections


def test_total_count_excludes_citation_only(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert len(corrections) == 2


def test_preserves_metadata_fields(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    entry = corrections["conv-26:q57"]
    assert entry["original_ground_truth"] == "Rainbow flag, transgender symbol"
    assert entry["error_type"] == "HALLUCINATION"
    assert entry["source"] == "community/dial481/locomo-audit"


def test_our_corrections_override_community(
    errors_file: Path, dataset_file: Path
) -> None:
    """Verify merge order: our corrections take precedence."""
    community = load_community_corrections(errors_file, dataset_file)
    our_corrections = {
        "conv-26:q57": {
            "corrected_ground_truth": "Our corrected version",
        }
    }
    merged = {**community, **our_corrections}

    assert merged["conv-26:q57"]["corrected_ground_truth"] == "Our corrected version"
