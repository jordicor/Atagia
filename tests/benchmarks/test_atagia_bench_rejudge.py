"""Tests for the Atagia-bench rejudge gate tool (no LLM calls).

Exercises report loading, the llm_judge vs non-llm_judge split against a fully
synthetic dataset, and the dry-run cost projection path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.atagia_bench.adapter import (
    AtagiaBenchAdapter,
    AtagiaBenchConversation,
    AtagiaBenchDataset,
    AtagiaBenchPersona,
    AtagiaBenchPersonaData,
    AtagiaBenchQuestion,
)
from benchmarks.atagia_bench.rejudge import (
    RejudgeConfig,
    _resolve_report_paths,
    load_reports,
    run_rejudge,
)
from benchmarks.scorer import JudgeProtocol


def _dataset() -> AtagiaBenchDataset:
    persona = AtagiaBenchPersona(
        persona_id="synthetic_persona",
        display_name="Synthetic Persona",
        age=36,
        occupation="test fixture author",
        profile="A synthetic persona used only by rejudge unit tests.",
        modes=["default"],
        conversation_count=1,
        test_scenarios=["recall", "abstention"],
    )
    conversation = AtagiaBenchConversation(
        conversation_id="synthetic-persona-conv-01",
        mode="default",
        timestamp_base="2026-01-01T00:00:00Z",
        turns=[
            {
                "turn_id": "synthetic-persona-01-t01",
                "role": "user",
                "text": "My cat is named Juniper and I recently moved to Porto.",
                "timestamp": "2026-01-01T09:00:00Z",
            },
            {
                "turn_id": "synthetic-persona-01-t02",
                "role": "assistant",
                "text": "I will remember Juniper and Porto.",
                "timestamp": "2026-01-01T09:00:05Z",
            },
        ],
    )
    questions = [
        AtagiaBenchQuestion(
            question_id="synthetic-q01",
            question_text="What is the user's cat named?",
            ground_truth="Juniper",
            answer_type="llm_judge",
            category_tags=["recall"],
            evidence_turn_ids=["synthetic-persona-01-t01"],
            grader="llm_judge",
        ),
        AtagiaBenchQuestion(
            question_id="synthetic-q02",
            question_text="Where did the user recently move?",
            ground_truth="Porto",
            answer_type="llm_judge",
            category_tags=["recall"],
            evidence_turn_ids=["synthetic-persona-01-t01"],
            grader="llm_judge",
        ),
        AtagiaBenchQuestion(
            question_id="synthetic-q03",
            question_text="Return the configured synthetic value.",
            ground_truth="fixture-value",
            answer_type="exact_match",
            category_tags=["deterministic"],
            grader="exact_match",
        ),
        AtagiaBenchQuestion(
            question_id="synthetic-q04",
            question_text="What private value was never provided?",
            ground_truth="ABSTAIN.",
            answer_type="privacy_check",
            category_tags=["abstention"],
            grader="abstention",
        ),
    ]
    return AtagiaBenchDataset(
        personas=[
            AtagiaBenchPersonaData(
                persona=persona,
                conversations=[conversation],
                questions=questions,
            )
        ]
    )


def _write_report(path: Path) -> None:
    """Write a synthetic report covering protocol-sensitive grader splits."""
    payload = {
        "benchmark_name": "atagia-bench-v0",
        "per_question": [
            {
                "question_id": "synthetic-q01",
                "persona_id": "synthetic_persona",
                "prediction": "The cat is named Juniper.",
                "ground_truth": "Outdated stored gold.",
                "answer_type": "llm_judge",
                "grade": {"grader_name": "llm_judge (kimi)", "passed": True, "score": 1.0},
            },
            {
                "question_id": "synthetic-q02",
                "persona_id": "synthetic_persona",
                "prediction": "The user moved to Porto.",
                "ground_truth": "Porto",
                "answer_type": "llm_judge",
                "grade": {"grader_name": "llm_judge (kimi)", "passed": True, "score": 1.0},
            },
            {
                "question_id": "synthetic-q03",
                "persona_id": "synthetic_persona",
                "prediction": "fixture-value",
                "ground_truth": "fixture-value",
                "answer_type": "exact_match",
                "grade": {"grader_name": "exact_match", "passed": True, "score": 1.0},
            },
            {
                "question_id": "synthetic-q04",
                "persona_id": "synthetic_persona",
                "prediction": "I cannot disclose that.",
                "ground_truth": "ABSTAIN.",
                "answer_type": "privacy_check",
                "grade": {"grader_name": "abstention", "passed": True, "score": 1.0},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_report_paths_file_and_missing(tmp_path: Path) -> None:
    report = tmp_path / "atagia-bench-report-x.json"
    _write_report(report)
    assert _resolve_report_paths([str(report)]) == [report]
    with pytest.raises(FileNotFoundError):
        _resolve_report_paths([str(tmp_path / "nope.json")])


def test_load_reports_splits_llm_judge_from_others(tmp_path: Path) -> None:
    report = tmp_path / "atagia-bench-report-x.json"
    _write_report(report)
    dataset = _dataset()
    loaded = load_reports(RejudgeConfig(report_paths=[report]), dataset)

    llm_ids = {r.question_id for r in loaded.llm_judge_records}
    assert llm_ids == {"synthetic-q01", "synthetic-q02"}
    assert all(r.prediction for r in loaded.llm_judge_records)

    # The stored gold is kept for flip auditing; grading uses the CURRENT
    # dataset gold, so gold corrections take effect on rejudge.
    first_record = next(
        r for r in loaded.llm_judge_records if r.question_id == "synthetic-q01"
    )
    assert first_record.stored_ground_truth == "Outdated stored gold."
    dataset_gold = {
        question.question_id: question.ground_truth
        for persona in dataset.personas
        for question in persona.questions
    }
    assert dataset_gold["synthetic-q01"] != first_record.stored_ground_truth

    non_llm = {s.question_id: s.measurement_layer for s in loaded.non_llm_judge}
    assert non_llm == {
        "synthetic-q03": "memory_content",
        "synthetic-q04": "product_behavior",
    }


def test_load_reports_fails_fast_on_dataset_drift(tmp_path: Path) -> None:
    report = tmp_path / "atagia-bench-report-x.json"
    payload = {
        "benchmark_name": "atagia-bench-v0",
        "per_question": [
            {
                "question_id": "ghost-q99",
                "persona_id": "synthetic_persona",
                "prediction": "x",
                "ground_truth": "y",
                "answer_type": "llm_judge",
                "grade": {"grader_name": "llm_judge", "passed": True, "score": 1.0},
            }
        ],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    dataset = _dataset()
    with pytest.raises(ValueError, match="not in the current Atagia-bench dataset"):
        load_reports(RejudgeConfig(report_paths=[report]), dataset)


@pytest.mark.asyncio
async def test_dry_run_projects_without_llm_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "atagia-bench-report-x.json"
    _write_report(report)
    dataset = _dataset()
    monkeypatch.setattr(
        AtagiaBenchAdapter,
        "load",
        lambda self, persona_ids=None: dataset,
    )
    result = await run_rejudge(
        RejudgeConfig(
            report_paths=[report],
            protocol=JudgeProtocol.MEMORY_QUALITY,
            dry_run=True,
        )
    )
    assert result["dry_run"] is True
    assert result["llm_judge_questions"] == 2
    assert result["non_llm_judge_questions"] == 2
    assert result["projection"]["llm_calls_planned"] == 2
    assert result["projection"]["projected_cost_usd"] >= 0.0
