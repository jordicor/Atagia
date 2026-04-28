"""Tests for benchmark report aggregation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.report_aggregate import (
    build_combined_report,
    format_combined_report_summary,
    save_combined_report,
    save_combined_run_manifest,
)


def _result(
    question_id: str,
    score: int,
    *,
    trace: dict[str, object] | None = None,
) -> QuestionResult:
    return QuestionResult(
        question=BenchmarkQuestion(
            question_text=f"Question {question_id}?",
            ground_truth="answer",
            category=1,
            evidence_turn_ids=[],
            question_id=question_id,
        ),
        prediction="answer" if score else "wrong",
        score_result=ScoreResult(
            score=score,
            reasoning="ok",
            judge_model="judge-model",
        ),
        memories_used=0,
        retrieval_time_ms=1.0,
        trace=trace or {},
    )


def _report(
    conversation_id: str,
    question_id: str,
    score: int,
    *,
    benchmark_db_path: str | None = None,
    selection: dict[str, object] | None = None,
    trace: dict[str, object] | None = None,
) -> BenchmarkReport:
    result = _result(question_id, score, trace=trace)
    model_info: dict[str, object] = {"warning_counts": {"failed_questions": 1 - score}}
    if selection is not None:
        model_info["selection"] = selection
    return BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=float(score),
        category_breakdown={1: float(score)},
        conversations=[
            ConversationReport(
                conversation_id=conversation_id,
                results=[result],
                accuracy=float(score),
                category_breakdown={1: float(score)},
                metadata=(
                    {"benchmark_db_path": benchmark_db_path}
                    if benchmark_db_path is not None
                    else {}
                ),
            )
        ],
        total_questions=1,
        total_correct=score,
        ablation_config=None,
        timestamp="2026-04-26T00:00:00+00:00",
        model_info=model_info,
        duration_seconds=2.0,
    )


def test_combined_report_merges_report_shards_and_warning_counts() -> None:
    combined = build_combined_report(
        [
            _report(
                "conv-a",
                "conv-a:q1",
                1,
                selection={
                    "selected_conversation_ids": ["conv-a"],
                    "planned_question_count": 1,
                },
            ),
            _report("conv-b", "conv-b:q1", 0),
        ],
        source_paths=["a.json", "b.json"],
        source_hashes=["hash-a", "hash-b"],
    )

    assert combined.total_questions == 2
    assert combined.total_correct == 1
    assert combined.overall_accuracy == 0.5
    assert combined.model_info["warning_counts"]["failed_questions"] == 1
    assert combined.model_info["warning_counts"]["tracebacks"] == 0
    assert combined.model_info["warning_counts"]["failed_worker_jobs"] == 0
    assert combined.model_info["source_reports"] == ["a.json", "b.json"]
    assert combined.model_info["source_report_sha256"] == {
        "a.json": "hash-a",
        "b.json": "hash-b",
    }
    assert combined.model_info["source_selection"] == {
        "a.json": {
            "selected_conversation_ids": ["conv-a"],
            "planned_question_count": 1,
        }
    }
    assert combined.model_info["source_accuracy_summary"]["overall_accuracy"] == {
        "count": 2,
        "mean": 0.5,
        "min": 0.0,
        "max": 1.0,
    }


def test_combined_report_duplicate_strategy_last_replaces_question() -> None:
    combined = build_combined_report(
        [
            _report("conv-a", "conv-a:q1", 0),
            _report("conv-a", "conv-a:q1", 1),
        ],
        source_paths=["old.json", "new.json"],
        duplicate_strategy="last",
    )

    assert combined.total_questions == 1
    assert combined.total_correct == 1
    assert combined.model_info["duplicate_question_ids"] == ["conv-a:q1"]


def test_combined_report_derives_failed_questions_when_warning_counts_are_missing() -> None:
    source_report = _report("conv-a", "conv-a:q1", 0)
    source_report.model_info = {}

    combined = build_combined_report(
        [source_report],
        source_paths=["old-report.json"],
    )

    assert combined.model_info["warning_counts"]["failed_questions"] == 1
    assert combined.model_info["warning_counts"]["tracebacks"] == 0


def test_combined_report_summarizes_replay_accuracy_by_retained_db() -> None:
    combined = build_combined_report(
        [
            _report("conv-a", "conv-a:q1", 0, benchmark_db_path="dbs/conv-a/benchmark.db"),
            _report("conv-a", "conv-a:q2", 1, benchmark_db_path="dbs/conv-a/benchmark.db"),
            _report("conv-b", "conv-b:q1", 1, benchmark_db_path="dbs/conv-b/benchmark.db"),
        ],
        source_paths=["run-a.json", "run-b.json", "run-c.json"],
    )

    retained_summary = combined.model_info["retained_db_accuracy_summary"]
    assert retained_summary["dbs/conv-a/benchmark.db"] == {
        "sample_count": 2,
        "source_reports": ["run-a.json", "run-b.json"],
        "conversation_ids": ["conv-a"],
        "accuracy": {"count": 2, "mean": 0.5, "min": 0.0, "max": 1.0},
        "total_correct": {"count": 2, "mean": 0.5, "min": 0.0, "max": 1.0},
        "total_questions": {"count": 2, "mean": 1.0, "min": 1.0, "max": 1.0},
    }
    assert retained_summary["dbs/conv-b/benchmark.db"]["accuracy"] == {
        "count": 1,
        "mean": 1.0,
        "min": 1.0,
        "max": 1.0,
    }


def test_combined_report_summarizes_trace_diagnostics() -> None:
    combined = build_combined_report(
        [
            _report(
                "conv-a",
                "conv-a:q1",
                0,
                trace={
                    "diagnosis_bucket": "missing_extraction",
                    "sufficiency_diagnostic": "missing_memory_extraction",
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_1",
                            "candidate_kind": "memory",
                            "channels": ["fts"],
                            "selected": True,
                            "composer_decision": "selected",
                            "filter_reason": None,
                        }
                    ],
                },
            ),
            _report(
                "conv-b",
                "conv-b:q1",
                1,
                trace={
                    "diagnosis_bucket": "answer_policy_or_grading",
                    "sufficiency_diagnostic": "retrieval_sufficient",
                },
            ),
            _report("conv-c", "conv-c:q1", 1),
        ],
        source_paths=["run-a.json", "run-b.json", "run-c.json"],
    )

    assert combined.model_info["diagnosis_bucket_counts"] == {
        "answer_policy_or_grading": 1,
        "missing_extraction": 1,
        "unknown": 1,
    }
    assert combined.model_info["sufficiency_diagnostic_counts"] == {
        "missing_memory_extraction": 1,
        "retrieval_sufficient": 1,
        "unknown": 1,
    }
    assert combined.model_info["retrieval_custody_summary"] == {
        "candidate_count": 1,
        "selected_count": 1,
        "channel_counts": {"fts": 1},
        "selected_channel_counts": {"fts": 1},
        "candidate_kind_counts": {"memory": 1},
        "composer_decision_counts": {"selected": 1},
        "filter_reason_counts": {},
    }


def test_format_combined_report_summary_includes_observability() -> None:
    combined = build_combined_report(
        [
            _report(
                "conv-a",
                "conv-a:q1",
                1,
                trace={
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_1",
                            "candidate_kind": "memory",
                            "channels": ["embedding"],
                            "selected": True,
                            "composer_decision": "selected",
                            "filter_reason": None,
                        }
                    ],
                },
            ),
            _report("conv-b", "conv-b:q1", 0),
        ],
        source_paths=["run-a.json", "run-b.json"],
    )

    summary = format_combined_report_summary(
        combined,
        report_path="combined.json",
        custody_path="combined-failed-custody.json",
        manifest_path="combined-run-manifest.json",
    )

    assert "Combined Benchmark Report" in summary
    assert "Accuracy: 50.0% (1/2)" in summary
    assert "Source reports: 2" in summary
    assert "Warning counts: failed_questions=1" in summary
    assert "Retrieval custody: candidates=1 selected=1" in summary
    assert "channels=embedding=1" in summary
    assert "Report saved to: combined.json" in summary


def test_combined_run_manifest_records_artifact_hashes(tmp_path: Path) -> None:
    combined = build_combined_report(
        [
            _report(
                "conv-a",
                "conv-a:q1",
                1,
                selection={
                    "selected_conversation_ids": ["conv-a"],
                    "planned_question_count": 1,
                },
            ),
            _report("conv-b", "conv-b:q1", 0),
        ],
        source_paths=["a.json", "b.json"],
        source_hashes=["hash-a", "hash-b"],
    )
    report_path = save_combined_report(combined, tmp_path / "combined.json")
    custody_path = tmp_path / "combined-failed-custody.json"
    custody_payload = b'{"total_failed_questions": 1}'
    custody_path.write_bytes(custody_payload)

    manifest_path = save_combined_run_manifest(
        combined,
        report_path=report_path,
        custody_path=custody_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_kind"] == "combined_benchmark_run_manifest"
    assert manifest["report_path"] == str(report_path)
    assert manifest["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert manifest["custody_path"] == str(custody_path)
    assert manifest["custody_sha256"] == hashlib.sha256(custody_payload).hexdigest()
    assert manifest["source_reports"] == ["a.json", "b.json"]
    assert manifest["source_report_sha256"] == {"a.json": "hash-a", "b.json": "hash-b"}
    assert manifest["source_selection"] == {
        "a.json": {
            "selected_conversation_ids": ["conv-a"],
            "planned_question_count": 1,
        }
    }
    assert manifest["result_summary"]["total_questions"] == 2
    assert manifest["result_summary"]["retrieval_time_ms"] == {
        "count": 2,
        "mean": 1.0,
        "min": 1.0,
        "max": 1.0,
    }
    assert manifest["result_summary"]["memories_used"] == {
        "count": 2,
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    assert manifest["warning_counts"]["failed_questions"] == 1
    assert manifest["retrieval_custody_summary"] == {
        "candidate_count": 0,
        "selected_count": 0,
        "channel_counts": {},
        "selected_channel_counts": {},
        "candidate_kind_counts": {},
        "composer_decision_counts": {},
        "filter_reason_counts": {},
    }
    assert "git" in manifest
