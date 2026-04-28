"""Tests for benchmark report diff tooling."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.report_diff import build_benchmark_diff, format_diff_summary, save_benchmark_diff


def _question_result(
    *,
    question_id: str,
    category: int,
    prediction: str,
    score: int,
    memories_used: int,
    retrieval_time_ms: float,
    trace: dict | None = None,
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
        trace=trace or {},
    )


def _report(
    *,
    accuracy: float,
    results: list[QuestionResult],
    timestamp: str,
    warning_counts: dict[str, int] | None = None,
) -> BenchmarkReport:
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
            "warning_counts": warning_counts or {},
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
                trace={
                    "diagnosis_bucket": "retrieval_or_ranking_miss",
                    "sufficiency_diagnostic": "retrieval_insufficient",
                    "selected_memory_ids": ["mem_old"],
                    "selected_evidence_memory_ids": [],
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_old",
                            "candidate_kind": "memory",
                            "channels": ["fts"],
                            "selected": False,
                            "composer_decision": "rejected",
                            "filter_reason": "low_score",
                        }
                    ],
                },
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
        warning_counts={"failed_questions": 1, "tracebacks": 2},
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
                trace={
                    "diagnosis_bucket": "passed",
                    "sufficiency_diagnostic": "retrieval_sufficient",
                    "selected_memory_ids": ["mem_new"],
                    "selected_evidence_memory_ids": ["mem_new"],
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_new",
                            "candidate_kind": "memory",
                            "channels": ["embedding"],
                            "selected": True,
                            "composer_decision": "selected",
                            "filter_reason": None,
                        }
                    ],
                },
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
        warning_counts={"failed_questions": 1, "tracebacks": 1, "provider_rate_limits": 1},
    )

    diff = build_benchmark_diff(
        before,
        after,
        before_label="baseline.json",
        after_label="candidate.json",
        before_report_sha256="hash-before",
        after_report_sha256="hash-after",
    )

    assert diff.overall_accuracy_delta == 0.0
    assert diff.before_report_sha256 == "hash-before"
    assert diff.after_report_sha256 == "hash-after"
    assert diff.warning_count_deltas == {
        "failed_questions": 0,
        "provider_rate_limits": 1,
        "tracebacks": -1,
    }
    assert diff.before_diagnosis_bucket_counts == {
        "retrieval_or_ranking_miss": 1,
        "unknown": 1,
    }
    assert diff.after_diagnosis_bucket_counts == {
        "passed": 1,
        "unknown": 1,
    }
    assert diff.diagnosis_bucket_count_deltas == {
        "passed": 1,
        "retrieval_or_ranking_miss": -1,
        "unknown": 0,
    }
    assert diff.before_sufficiency_diagnostic_counts == {
        "retrieval_insufficient": 1,
        "unknown": 1,
    }
    assert diff.after_sufficiency_diagnostic_counts == {
        "retrieval_sufficient": 1,
        "unknown": 1,
    }
    assert diff.sufficiency_diagnostic_count_deltas == {
        "retrieval_insufficient": -1,
        "retrieval_sufficient": 1,
        "unknown": 0,
    }
    assert diff.before_retrieval_custody_summary == {
        "candidate_count": 1,
        "selected_count": 0,
        "channel_counts": {"fts": 1},
        "selected_channel_counts": {},
        "candidate_kind_counts": {"memory": 1},
        "composer_decision_counts": {"rejected": 1},
        "filter_reason_counts": {"low_score": 1},
    }
    assert diff.after_retrieval_custody_summary == {
        "candidate_count": 1,
        "selected_count": 1,
        "channel_counts": {"embedding": 1},
        "selected_channel_counts": {"embedding": 1},
        "candidate_kind_counts": {"memory": 1},
        "composer_decision_counts": {"selected": 1},
        "filter_reason_counts": {},
    }
    assert diff.improved_questions == 1
    assert diff.regressed_questions == 1
    assert diff.unchanged_questions == 0
    assert diff.before_retrieval_time_ms.model_dump() == {
        "count": 2,
        "mean": 10.5,
        "min": 10.0,
        "max": 11.0,
    }
    assert diff.after_retrieval_time_ms.model_dump() == {
        "count": 2,
        "mean": 10.75,
        "min": 9.5,
        "max": 12.0,
    }
    assert diff.retrieval_time_delta_ms.model_dump() == {
        "count": 2,
        "mean": 0.25,
        "min": -0.5,
        "max": 1.0,
    }
    assert diff.before_memories_used.model_dump() == {
        "count": 2,
        "mean": 1.5,
        "min": 1.0,
        "max": 2.0,
    }
    assert diff.after_memories_used.model_dump() == {
        "count": 2,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert diff.memories_used_delta.model_dump() == {
        "count": 2,
        "mean": 0.5,
        "min": -1.0,
        "max": 2.0,
    }
    question_status = {
        item.question_id: item.status
        for item in diff.conversations[0].question_diffs
    }
    assert question_status == {
        "conv-test-1:q1": "improved",
        "conv-test-1:q2": "regressed",
    }
    first_diff = diff.conversations[0].question_diffs[0]
    assert first_diff.before_diagnosis_bucket == "retrieval_or_ranking_miss"
    assert first_diff.after_diagnosis_bucket == "passed"
    assert first_diff.after_selected_evidence_memory_ids == ["mem_new"]
    assert diff.conversations[0].retrieval_time_delta_ms.model_dump() == {
        "count": 2,
        "mean": 0.25,
        "min": -0.5,
        "max": 1.0,
    }
    assert diff.conversations[0].memories_used_delta.model_dump() == {
        "count": 2,
        "mean": 0.5,
        "min": -1.0,
        "max": 2.0,
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
    assert payload["retrieval_time_delta_ms"] == {
        "count": 1,
        "mean": -1.0,
        "min": -1.0,
        "max": -1.0,
    }
    assert payload["memories_used_delta"] == {
        "count": 1,
        "mean": 1.0,
        "min": 1.0,
        "max": 1.0,
    }
    assert payload["conversations"][0]["question_diffs"][0]["status"] == "improved"


def test_format_diff_summary_includes_observability_sections() -> None:
    before = _report(
        accuracy=0.0,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="wrong",
                score=0,
                memories_used=1,
                retrieval_time_ms=10.0,
                trace={
                    "diagnosis_bucket": "retrieval_or_ranking_miss",
                    "sufficiency_diagnostic": "retrieval_insufficient",
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_old",
                            "candidate_kind": "memory",
                            "channels": ["fts"],
                            "selected": False,
                            "composer_decision": "rejected",
                            "filter_reason": "low_score",
                        }
                    ],
                },
            )
        ],
        timestamp="2026-04-07T00:00:00+00:00",
        warning_counts={"failed_questions": 1},
    )
    after = _report(
        accuracy=1.0,
        results=[
            _question_result(
                question_id="conv-test-1:q1",
                category=1,
                prediction="answer",
                score=1,
                memories_used=2,
                retrieval_time_ms=9.0,
                trace={
                    "diagnosis_bucket": "passed",
                    "sufficiency_diagnostic": "retrieval_sufficient",
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_new",
                            "candidate_kind": "summary_view",
                            "channels": ["embedding"],
                            "selected": True,
                            "composer_decision": "selected",
                            "filter_reason": None,
                        }
                    ],
                },
            )
        ],
        timestamp="2026-04-07T01:00:00+00:00",
        warning_counts={"failed_questions": 0},
    )
    diff = build_benchmark_diff(
        before,
        after,
        before_label="before.json",
        after_label="after.json",
    )

    summary = format_diff_summary(diff)

    assert "LoCoMo Diff Report" in summary
    assert "Accuracy: 0.0% -> 100.0% (+100.0%)" in summary
    assert "Correct: 0/1 -> 1/1 (+1)" in summary
    assert "Retrieval custody:" in summary
    assert "Before: candidates=1 selected=0 channels=fts=1" in summary
    assert "After:  candidates=1 selected=1 channels=embedding=1" in summary
    assert "Warning deltas: failed_questions=-1" in summary
    assert "Diagnosis deltas: passed=+1, retrieval_or_ranking_miss=-1" in summary
    assert "Sufficiency deltas: retrieval_insufficient=-1, retrieval_sufficient=+1" in summary
    assert "Category changes (>=1% delta):" in summary
    assert "  Cat 1: +100.0%" in summary
    assert "[PASS] conv-test-1:q1: 0 -> 1 (cat 1)" in summary


def test_build_benchmark_diff_derives_failed_counts_when_warning_counts_missing() -> None:
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
    before.model_info = {}
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
    after.model_info = {}

    diff = build_benchmark_diff(
        before,
        after,
        before_label="before.json",
        after_label="after.json",
    )

    assert diff.before_warning_counts["failed_questions"] == 1
    assert diff.after_warning_counts["failed_questions"] == 0
    assert diff.warning_count_deltas["failed_questions"] == -1
