"""Tests for loading independent synthetic LoCoMo correction fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.corrections import load_community_corrections


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    """Write two fictional conversations using the LoCoMo sample contract."""
    dataset = [
        {"sample_id": "fixture-orchid", "conversation": {}, "qa": []},
        {"sample_id": "fixture-comet", "conversation": {}, "qa": []},
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


@pytest.fixture
def errors_file(tmp_path: Path) -> Path:
    """Write one answer correction and one citation-only correction."""
    errors = [
        {
            "question_id": "locomo_0_qa0",
            "question": "What color marker labels the prism crate?",
            "golden_answer": "The marker is amber.",
            "category": 1,
            "correct_answer": "The marker is cobalt.",
            "cited_evidence": ["D71:101"],
            "correct_evidence": ["D71:102"],
            "reasoning": "The corrected inventory note names cobalt.",
            "error_type": "answer_mismatch",
        },
        {
            "question_id": "locomo_1_qa1",
            "question": "Where is the folded star chart stored?",
            "golden_answer": "The chart is in archive bay four.",
            "category": 2,
            "correct_answer": "",
            "cited_evidence": ["D72:201"],
            "correct_evidence": ["D72:203"],
            "reasoning": "The answer is correct, but the cited turn is not.",
            "error_type": "citation_mismatch",
        },
    ]
    path = tmp_path / "errors.json"
    path.write_text(json.dumps(errors), encoding="utf-8")
    return path


def test_converts_question_ids_to_atagia_format(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert "fixture-orchid:q1" in corrections
    assert corrections["fixture-orchid:q1"]["corrected_ground_truth"] == (
        "The marker is cobalt."
    )


def test_maps_second_conversation(errors_file: Path, dataset_file: Path) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert "fixture-comet:q2" in corrections
    assert corrections["fixture-comet:q2"]["original_ground_truth"] == (
        "The chart is in archive bay four."
    )


def test_preserves_citation_only_errors_as_evidence_corrections(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    entry = corrections["fixture-comet:q2"]
    assert "corrected_ground_truth" not in entry
    assert entry["corrected_evidence_turn_ids"] == ["D72:203"]
    assert entry["original_evidence_turn_ids"] == ["D72:201"]


def test_total_count_includes_citation_only(
    errors_file: Path, dataset_file: Path
) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    assert len(corrections) == 2


def test_preserves_metadata_fields(errors_file: Path, dataset_file: Path) -> None:
    corrections = load_community_corrections(errors_file, dataset_file)

    entry = corrections["fixture-orchid:q1"]
    assert entry["original_ground_truth"] == "The marker is amber."
    assert entry["error_type"] == "answer_mismatch"
    assert entry["reason"] == "The corrected inventory note names cobalt."
    assert entry["original_evidence_turn_ids"] == ["D71:101"]


def test_our_corrections_override_community(
    errors_file: Path, dataset_file: Path
) -> None:
    """Our local correction wins when dictionaries are merged in that order."""
    community = load_community_corrections(errors_file, dataset_file)
    our_corrections = {
        "fixture-orchid:q1": {
            "corrected_ground_truth": "The marker is silver.",
        }
    }
    merged = {**community, **our_corrections}

    assert merged["fixture-orchid:q1"]["corrected_ground_truth"] == (
        "The marker is silver."
    )
