from __future__ import annotations

from pathlib import Path

from benchmarks.atagia_bench.__main__ import _format_report_summary
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.report_diff import build_diff, format_diff_summary
from benchmarks.atagia_bench.runner import (
    AtagiaBenchReport,
    AtagiaQuestionResult,
    CategoryStats,
)


def _result(
    *,
    question_id: str,
    passed: bool,
    score: float,
    memories_used: int = 1,
    retrieval_time_ms: float = 1.0,
    trace: dict[str, object] | None = None,
) -> AtagiaQuestionResult:
    return AtagiaQuestionResult(
        question_id=question_id,
        question_text=f"Question {question_id}?",
        ground_truth="answer",
        prediction="answer" if passed else "wrong",
        answer_type="fact",
        category_tags=["smoke"],
        evidence_turn_ids=["turn_1"],
        grade=GradeResult(
            passed=passed,
            score=score,
            reason="ok" if passed else "miss",
            grader_name="exact_match",
        ),
        memories_used=memories_used,
        retrieval_time_ms=retrieval_time_ms,
        conversation_id="cnv_1",
        persona_id="persona_1",
        trace=trace or {},
    )


def _report(
    *,
    result: AtagiaQuestionResult,
    pass_rate: float,
    warning_counts: dict[str, int] | None = None,
) -> AtagiaBenchReport:
    return AtagiaBenchReport(
        timestamp="2026-04-26T00:00:00+00:00",
        run_duration_seconds=1.0,
        config={"provider": "static", "warning_counts": warning_counts or {}},
        personas_used=["persona_1"],
        total_questions=1,
        total_passed=1 if result.grade.passed else 0,
        pass_rate=pass_rate,
        avg_score=result.grade.score,
        critical_error_count=0,
        per_question=[result],
        per_category=[
            CategoryStats(
                category="smoke",
                count=1,
                pass_count=1 if result.grade.passed else 0,
                pass_rate=pass_rate,
                avg_score=result.grade.score,
            )
        ],
    )


def test_atagia_bench_diff_carries_diagnostics_and_selected_ids() -> None:
    before = _report(
        result=_result(
            question_id="q1",
            passed=False,
            score=0.0,
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
        pass_rate=0.0,
        warning_counts={"failed_questions": 1, "tracebacks": 1},
    )
    after = _report(
        result=_result(
            question_id="q1",
            passed=True,
            score=1.0,
            memories_used=3,
            retrieval_time_ms=8.5,
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
        pass_rate=1.0,
        warning_counts={"failed_questions": 0, "provider_rate_limits": 1},
    )

    diff = build_diff(
        before,
        after,
        before_report_sha256="hash-before",
        after_report_sha256="hash-after",
    )
    question_diff = diff.question_diffs[0]

    assert diff.before_report_sha256 == "hash-before"
    assert diff.after_report_sha256 == "hash-after"
    assert diff.warning_count_deltas == {
        "failed_questions": -1,
        "provider_rate_limits": 1,
        "tracebacks": -1,
    }
    assert diff.before_diagnosis_bucket_counts == {"retrieval_or_ranking_miss": 1}
    assert diff.after_diagnosis_bucket_counts == {"passed": 1}
    assert diff.diagnosis_bucket_count_deltas == {
        "passed": 1,
        "retrieval_or_ranking_miss": -1,
    }
    assert diff.before_sufficiency_diagnostic_counts == {"retrieval_insufficient": 1}
    assert diff.after_sufficiency_diagnostic_counts == {"retrieval_sufficient": 1}
    assert diff.sufficiency_diagnostic_count_deltas == {
        "retrieval_insufficient": -1,
        "retrieval_sufficient": 1,
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
    summary = format_diff_summary(diff)
    assert "Retrieval custody:" in summary
    assert "Before: candidates=1 selected=0" in summary
    assert "After:  candidates=1 selected=1" in summary
    assert question_diff.status == "improved"
    assert diff.before_retrieval_time_ms == {
        "count": 1,
        "mean": 10.0,
        "min": 10.0,
        "max": 10.0,
    }
    assert diff.after_retrieval_time_ms == {
        "count": 1,
        "mean": 8.5,
        "min": 8.5,
        "max": 8.5,
    }
    assert diff.retrieval_time_delta_ms == {
        "count": 1,
        "mean": -1.5,
        "min": -1.5,
        "max": -1.5,
    }
    assert diff.before_memories_used == {
        "count": 1,
        "mean": 1.0,
        "min": 1.0,
        "max": 1.0,
    }
    assert diff.after_memories_used == {
        "count": 1,
        "mean": 3.0,
        "min": 3.0,
        "max": 3.0,
    }
    assert diff.memories_used_delta == {
        "count": 1,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
    }
    assert question_diff.before_retrieval_time_ms == 10.0
    assert question_diff.after_retrieval_time_ms == 8.5
    assert question_diff.retrieval_time_delta_ms == -1.5
    assert question_diff.memories_used_delta == 2
    assert question_diff.before_diagnosis_bucket == "retrieval_or_ranking_miss"
    assert question_diff.after_diagnosis_bucket == "passed"
    assert question_diff.before_sufficiency_diagnostic == "retrieval_insufficient"
    assert question_diff.after_sufficiency_diagnostic == "retrieval_sufficient"
    assert question_diff.before_selected_memory_ids == ["mem_old"]
    assert question_diff.after_selected_evidence_memory_ids == ["mem_new"]


def test_atagia_bench_diff_derives_failed_counts_when_warning_counts_missing() -> None:
    before = _report(
        result=_result(question_id="q1", passed=False, score=0.0),
        pass_rate=0.0,
    )
    before.config = {}
    after = _report(
        result=_result(question_id="q1", passed=True, score=1.0),
        pass_rate=1.0,
    )
    after.config = {}

    diff = build_diff(before, after)

    assert diff.before_warning_counts["failed_questions"] == 1
    assert diff.after_warning_counts["failed_questions"] == 0
    assert diff.warning_count_deltas["failed_questions"] == -1


def test_format_atagia_bench_report_summary_includes_retrieval_custody(
    tmp_path: Path,
) -> None:
    report = _report(
        result=_result(question_id="q1", passed=True, score=1.0),
        pass_rate=1.0,
    )
    report.config["retrieval_custody_summary"] = {
        "candidate_count": 2,
        "selected_count": 1,
        "channel_counts": {"fts": 2},
        "selected_channel_counts": {"fts": 1},
        "candidate_kind_counts": {"memory": 2},
        "composer_decision_counts": {"selected": 1, "rejected": 1},
        "filter_reason_counts": {"low_score": 1},
    }

    summary = _format_report_summary(
        report=report,
        report_path=tmp_path / "atagia-bench-report.json",
    )

    assert "Atagia-bench v0 Results" in summary
    assert "Retrieval custody: candidates=2 selected=1" in summary
    assert "channels=fts=2" in summary
