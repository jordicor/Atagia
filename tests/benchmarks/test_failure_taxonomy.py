"""Tests for text-free benchmark failure taxonomy reports."""

from __future__ import annotations

import json

from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaQuestionResult
from benchmarks.base import (
    BenchmarkQuestion,
    BenchmarkReport,
    ConversationReport,
    QuestionResult,
    ScoreResult,
)
from benchmarks.failure_taxonomy import (
    build_failure_taxonomy_report,
    failure_taxonomy_manifest_summary,
    format_failure_taxonomy_summary,
)


def test_failure_taxonomy_maps_locomo_diagnostics_without_raw_text(tmp_path) -> None:
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=0.0,
        category_breakdown={1: 0.0},
        conversations=[
            ConversationReport(
                conversation_id="conv_1",
                results=[
                    QuestionResult(
                        question=BenchmarkQuestion(
                            question_text="RAW QUESTION TEXT",
                            ground_truth="RAW GROUND TRUTH",
                            category=1,
                            evidence_turn_ids=["D1:1"],
                            question_id="q1",
                        ),
                        prediction="RAW PREDICTION",
                        score_result=ScoreResult(
                            score=0,
                            reasoning="RAW JUDGE REASON",
                            judge_model="judge",
                        ),
                        memories_used=0,
                        retrieval_time_ms=12.0,
                        trace={
                            "diagnosis_bucket": "missing_extraction",
                            "sufficiency_diagnostic": "missing_memory_extraction",
                            "evidence_turn_ids": ["D1:1"],
                            "evidence_memory_ids": [],
                            "selected_memory_ids": [],
                            "selected_evidence_memory_ids": [],
                            "retrieval_custody": [],
                        },
                    )
                ],
                accuracy=0.0,
                category_breakdown={1: 0.0},
            )
        ],
        total_questions=1,
        total_correct=0,
        timestamp="2026-04-26T00:00:00+00:00",
        duration_seconds=1.0,
    )
    source = tmp_path / "report.json"
    source.write_text("{}", encoding="utf-8")

    taxonomy = build_failure_taxonomy_report(report, source_report=str(source))

    assert taxonomy.total_failed_questions == 1
    assert taxonomy.taxonomy_counts == {"extraction": 1}
    assert taxonomy.items[0].taxonomy_bucket == "extraction"
    assert taxonomy.items[0].question_id == "q1"
    assert taxonomy.items[0].evidence_turn_count == 1
    serialized = json.dumps(taxonomy.model_dump(mode="json"))
    assert "RAW QUESTION TEXT" not in serialized
    assert "RAW GROUND TRUTH" not in serialized
    assert "RAW PREDICTION" not in serialized
    assert "RAW JUDGE REASON" not in serialized


def test_failure_taxonomy_maps_technical_execution_failures(tmp_path) -> None:
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=0.0,
        category_breakdown={1: 0.0},
        conversations=[
            ConversationReport(
                conversation_id="conv_1",
                results=[
                    QuestionResult(
                        question=BenchmarkQuestion(
                            question_text="RAW QUESTION TEXT",
                            ground_truth="RAW GROUND TRUTH",
                            category=1,
                            question_id="q1",
                        ),
                        prediction="",
                        score_result=ScoreResult(
                            score=0,
                            reasoning="RAW ERROR",
                            judge_model="judge",
                        ),
                        memories_used=0,
                        retrieval_time_ms=0.0,
                        trace={
                            "diagnosis_bucket": "retrieval_failed",
                            "sufficiency_diagnostic": "retrieval_failed",
                        },
                    )
                ],
                accuracy=0.0,
                category_breakdown={1: 0.0},
            )
        ],
        total_questions=1,
        total_correct=0,
        timestamp="2026-04-26T00:00:00+00:00",
        duration_seconds=1.0,
    )
    source = tmp_path / "report.json"
    source.write_text("{}", encoding="utf-8")

    taxonomy = build_failure_taxonomy_report(report, source_report=str(source))

    assert taxonomy.taxonomy_counts == {"execution_failure": 1}
    assert taxonomy.items[0].diagnosis_bucket == "retrieval_failed"
    assert taxonomy.items[0].taxonomy_bucket == "execution_failure"


def test_failure_taxonomy_uses_exact_mappings_and_sanitizes_free_form_diagnostics(tmp_path) -> None:
    report = AtagiaBenchReport(
        timestamp="2026-04-26T00:00:00+00:00",
        run_duration_seconds=1.0,
        config={"provider": "static", "answer_model": "static-model"},
        personas_used=["mini"],
        total_questions=1,
        total_passed=0,
        pass_rate=0.0,
        avg_score=0.0,
        critical_error_count=0,
        per_question=[
            AtagiaQuestionResult(
                question_id="q1",
                question_text="RAW QUESTION TEXT",
                ground_truth="RAW GROUND TRUTH",
                prediction="RAW PREDICTION",
                answer_type="fact",
                category_tags=["smoke"],
                grade=GradeResult(
                    passed=False,
                    score=0.0,
                    reason="RAW GRADE REASON",
                    grader_name="exact_match",
                ),
                memories_used=1,
                retrieval_time_ms=8.0,
                conversation_id="conv_1",
                persona_id="mini",
                trace={
                    "diagnosis_bucket": "retrieval no candidates in free form",
                    "sufficiency_diagnostic": "needs raw evidence maybe",
                    "retrieval_custody": [
                        {
                            "candidate_id": "mem_1",
                            "candidate_kind": "external_memory",
                            "channels": ["static_memory"],
                            "composer_decision": "not_selected",
                            "selected": False,
                        }
                    ],
                },
            )
        ],
        per_category=[],
    )
    source = tmp_path / "atagia-report.json"
    source.write_text("{}", encoding="utf-8")

    taxonomy = build_failure_taxonomy_report(report, source_report=str(source))

    assert taxonomy.taxonomy_counts == {"unknown": 1}
    assert taxonomy.items[0].diagnosis_bucket == "unknown"
    assert taxonomy.items[0].sufficiency_diagnostic == "unknown"
    assert taxonomy.items[0].retrieval_custody_summary["candidate_count"] == 1
    assert failure_taxonomy_manifest_summary(taxonomy) == {
        "total_failed_questions": 1,
        "taxonomy_counts": {"unknown": 1},
        "diagnosis_counts": {"unknown": 1},
        "sufficiency_counts": {"unknown": 1},
    }
    assert format_failure_taxonomy_summary(taxonomy) == (
        "Failure taxonomy: failed=1 buckets=unknown=1"
    )
    serialized = json.dumps(taxonomy.model_dump(mode="json"))
    assert "retrieval no candidates in free form" not in serialized
    assert "needs raw evidence maybe" not in serialized
    assert "RAW" not in serialized
