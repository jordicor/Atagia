"""Tests for benchmark report diff tooling."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.report_diff import build_benchmark_diff, save_benchmark_diff


def _question_result(
    *,
    question_id: str,
    category: int,
    prediction: str,
    score: int,
    memories_used: int,
    retrieval_time_ms: float,
) -> QuestionResult:
    return QuestionResult(
        question=BenchmarkQuestion(
            question_text=f"Question {question_id}?",
            ground_truth="answer",
            category=category,
            evidence_turn_ids=["D1:1"],
            question_id=question_id,
        ),
        prediction=prediction,
        score_result=ScoreResult(
            score=score,
            reasoning=f"score={score}",
            judge_model="judge-model",
        ),
        memories_used=memories_used,
        retrieval_time_ms=retrieval_time_ms,
    )


def _report(*, accuracy: float, results: list[QuestionResult], timestamp: str) -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=accuracy,
        category_breakdown={1: accuracy},
        conversations=[
            ConversationReport(
                conversation_id="conv-test-1",
                results=results,
                accuracy=accuracy,
                category_breakdown={1: accuracy},
            )
        ],
        total_questions=len(results),
        total_correct=sum(result.score_result.score for result in results),
        ablation_config=None,
        timestamp=timestamp,
        model_info={
            "provider": "openai",
            "answer_model": "answer-model",
            "judge_model": "judge-model",
            "assistant_mode_id": "general_qa",
        },
        duration_seconds=1.0,
    )


def test_build_benchmark_diff_tracks_question_flips() -> None:
    before = _report(
        accuracy=0.5,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="wrong",
                score=0,
                memories_used=1,
                retrieval_time_ms=10.0,
            ),
            _question_result(
                question_id="conv-test-1:q2",
                category=1,
                prediction="answer",
                score=1,
                memories_used=2,
                retrieval_time_ms=11.0,
            ),
        ],
        timestamp="2026-04-07T00:00:00+00:00",
    )
    after = _report(
        accuracy=0.5,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="answer",
                score=1,
                memories_used=3,
                retrieval_time_ms=9.5,
            ),
            _question_result(
                question_id="conv-test-1:q2",
                category=1,
                prediction="wrong",
                score=0,
                memories_used=1,
                retrieval_time_ms=12.0,
            ),
        ],
        timestamp="2026-04-07T01:00:00+00:00",
    )

    diff = build_benchmark_diff(
        before,
        after,
        before_label="baseline.json",
        after_label="candidate.json",
    )

    assert diff.overall_accuracy_delta == 0.0
    assert diff.improved_questions == 1
    assert diff.regressed_questions == 1
    assert diff.unchanged_questions == 0
    question_status = {
        item.question_id: item.status
        for item in diff.conversations[0].question_diffs
    }
    assert question_status == {
        "conv-test-1:q1": "improved",
        "conv-test-1:q2": "regressed",
    }


def test_save_benchmark_diff_writes_json(tmp_path: Path) -> None:
    before = _report(
        accuracy=0.0,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="wrong",
                score=0,
                memories_used=0,
                retrieval_time_ms=10.0,
            )
        ],
        timestamp="2026-04-07T00:00:00+00:00",
    )
    after = _report(
        accuracy=1.0,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="answer",
                score=1,
                memories_used=1,
                retrieval_time_ms=9.0,
            )
        ],
        timestamp="2026-04-07T01:00:00+00:00",
    )
    diff = build_benchmark_diff(
        before,
        after,
        before_label="before.json",
        after_label="after.json",
    )

    output_path = save_benchmark_diff(diff, tmp_path / "diff.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["overall_accuracy_delta"] == 1.0
    assert payload["improved_questions"] == 1
    assert payload["conversations"][0]["question_diffs"][0]["status"] == "improved"
