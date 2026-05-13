from __future__ import annotations

import json
from pathlib import Path

from benchmarks.atagia_bench.runner import load_holdout_question_ids
from benchmarks.graph_stress.adapter import (
    GRAPH_STRESS_DATA_DIR,
    REQUIRED_GRAPH_STRESS_TAGS,
    load_graph_stress_dataset,
    validate_graph_stress_dataset,
)


def test_graph_stress_dataset_loads_and_covers_required_tags() -> None:
    dataset = load_graph_stress_dataset()

    summary = validate_graph_stress_dataset(dataset)

    assert dataset.name == "atagia-bench-v0"
    assert summary["persona_count"] == 1
    assert summary["conversation_count"] == 4
    assert summary["question_count"] == 8
    assert summary["tag_counts"]["graph_stress"] == 8
    for tag in REQUIRED_GRAPH_STRESS_TAGS:
        assert summary["tag_counts"][tag] >= 1


def test_graph_stress_questions_reference_existing_turns() -> None:
    dataset = load_graph_stress_dataset()
    persona = dataset.personas[0]
    turn_ids = {
        turn.turn_id
        for conversation in persona.conversations
        for turn in conversation.turns
    }

    for question in persona.questions:
        assert question.evidence_turn_ids
        assert set(question.evidence_turn_ids).issubset(turn_ids)


def test_graph_stress_manifest_matches_dataset() -> None:
    manifest_path = GRAPH_STRESS_DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_graph_stress_dataset()
    summary = validate_graph_stress_dataset(dataset)

    assert manifest["manifest_kind"] == "graph_stress_dataset_manifest"
    assert manifest["question_count"] == summary["question_count"]
    assert sorted(manifest["required_graph_stress_tags"]) == sorted(REQUIRED_GRAPH_STRESS_TAGS)
    assert manifest["dataset_contract"] == "atagia_bench_compatible"


def test_graph_stress_holdout_ids_are_dataset_questions() -> None:
    dataset = load_graph_stress_dataset()
    question_ids = {
        question.question_id
        for persona_data in dataset.personas
        for question in persona_data.questions
    }
    holdout_path = GRAPH_STRESS_DATA_DIR / "holdout_v0.json"

    holdout_ids = load_holdout_question_ids(holdout_path)

    assert holdout_ids == ["graph-morgan-q04", "graph-morgan-q08"]
    assert set(holdout_ids).issubset(question_ids)


def test_graph_stress_data_dir_is_repo_local() -> None:
    assert isinstance(GRAPH_STRESS_DATA_DIR, Path)
    assert GRAPH_STRESS_DATA_DIR.name == "data"
    assert (GRAPH_STRESS_DATA_DIR / "personas.json").exists()
