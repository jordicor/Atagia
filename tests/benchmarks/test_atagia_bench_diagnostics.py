"""Tests for Atagia-bench diagnostic bucketing."""

from __future__ import annotations

from benchmarks.atagia_bench.adapter import AtagiaBenchQuestion
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import AtagiaBenchRunner, AtagiaQuestionResult
from benchmarks.trusted_eval import trusted_evaluation_ablation
from atagia.models.schemas_memory import RetrievalTrace
from atagia.models.schemas_replay import AblationConfig


def _bucket(
    *,
    passed: bool = False,
    has_evidence_turns: bool = True,
    evidence_message_count: int = 1,
    evidence_memory_count: int = 1,
    active_evidence_count: int = 1,
    candidate_count: int = 1,
    selected_memory_count: int = 1,
    selected_evidence_count: int = 1,
) -> str:
    return AtagiaBenchRunner._diagnosis_bucket(
        passed=passed,
        has_evidence_turns=has_evidence_turns,
        evidence_message_count=evidence_message_count,
        evidence_memory_count=evidence_memory_count,
        active_evidence_count=active_evidence_count,
        candidate_count=candidate_count,
        selected_memory_count=selected_memory_count,
        selected_evidence_count=selected_evidence_count,
    )


def test_diagnosis_bucket_passed() -> None:
    assert _bucket(passed=True, evidence_memory_count=0) == "passed"


def test_diagnosis_bucket_missing_extraction() -> None:
    assert _bucket(evidence_memory_count=0) == "missing_extraction"


def test_diagnosis_bucket_memory_not_active() -> None:
    assert _bucket(evidence_memory_count=2, active_evidence_count=0) == "memory_not_active"


def test_diagnosis_bucket_retrieval_no_candidates() -> None:
    assert _bucket(candidate_count=0) == "retrieval_no_candidates"


def test_diagnosis_bucket_composition_selected_none() -> None:
    assert _bucket(selected_memory_count=0) == "composition_selected_none"


def test_diagnosis_bucket_retrieval_or_ranking_miss() -> None:
    assert _bucket(selected_evidence_count=0) == "retrieval_or_ranking_miss"


def test_diagnosis_bucket_answer_policy_or_grading() -> None:
    assert _bucket() == "answer_policy_or_grading"


def _trace(*, raw_context_access_mode: str = "normal") -> RetrievalTrace:
    return RetrievalTrace(
        query_text="Where is the evidence?",
        user_id="usr_1",
        conversation_id="cnv_1",
        timestamp_iso="2026-04-26T06:55:00Z",
        raw_context_access_mode=raw_context_access_mode,
    )


def test_sufficiency_diagnostic_passed() -> None:
    assert (
        AtagiaBenchRunner._sufficiency_diagnostic("passed", retrieval_trace=_trace())
        == "retrieval_sufficient"
    )


def test_sufficiency_diagnostic_missing_artifact_support() -> None:
    assert (
        AtagiaBenchRunner._sufficiency_diagnostic(
            "retrieval_no_candidates",
            retrieval_trace=_trace(raw_context_access_mode="artifact"),
        )
        == "missing_artifact_support"
    )


def test_sufficiency_diagnostic_ranking_miss() -> None:
    assert (
        AtagiaBenchRunner._sufficiency_diagnostic(
            "retrieval_or_ranking_miss",
            retrieval_trace=_trace(),
        )
        == "retrieval_insufficient"
    )


def test_warning_counts_include_failure_buckets_and_degraded_traces() -> None:
    result = AtagiaQuestionResult(
        question_id="q1",
        question_text="Question?",
        ground_truth="Answer",
        prediction="Wrong",
        answer_type="fact",
        grade=GradeResult(
            passed=False,
            score=0.0,
            reason="miss",
            grader_name="exact_match",
        ),
        memories_used=0,
        retrieval_time_ms=1.0,
        conversation_id="cnv_1",
        persona_id="persona_1",
        trace={
            "diagnosis_bucket": "retrieval_no_candidates",
            "retrieval_trace": {
                "degraded_mode": True,
                "need_detection": {"degraded_mode": True},
            },
            "retrieval_custody": [
                {
                    "candidate_kind": "evidence",
                    "channels": ["fts"],
                    "selected": True,
                }
            ],
        },
    )

    counts = AtagiaBenchRunner._aggregate_warning_counts([result])
    custody_summary = AtagiaBenchRunner._aggregate_retrieval_custody_summary([result])

    assert counts["failed_questions"] == 1
    assert counts["retrieval_no_candidates"] == 1
    assert counts["degraded_retrievals"] == 1
    assert counts["need_detection_degraded"] == 1
    assert counts["structured_output_retries"] == 0
    assert counts["failed_worker_jobs"] == 0
    assert custody_summary["candidate_count"] == 1
    assert custody_summary["selected_channel_counts"] == {"fts": 1}


def test_trusted_evaluation_ablation_raises_privacy_ceiling() -> None:
    ablation = trusted_evaluation_ablation(None)

    assert ablation.override_retrieval_params == {"privacy_ceiling": 3}


def test_trusted_evaluation_ablation_preserves_existing_overrides() -> None:
    ablation = trusted_evaluation_ablation(
        AblationConfig(
            skip_compaction=True,
            override_retrieval_params={"max_candidates": 10, "privacy_ceiling": 1},
        )
    )

    assert ablation.skip_compaction is True
    assert ablation.override_retrieval_params == {
        "max_candidates": 10,
        "privacy_ceiling": 3,
    }


def test_filter_questions_accepts_category_and_question_id() -> None:
    questions = [
        AtagiaBenchQuestion(
            question_id="q1",
            question_text="Question 1?",
            ground_truth="one",
            answer_type="short",
            category_tags=["alpha"],
            grader="exact_match",
        ),
        AtagiaBenchQuestion(
            question_id="q2",
            question_text="Question 2?",
            ground_truth="two",
            answer_type="short",
            category_tags=["beta"],
            grader="exact_match",
        ),
    ]

    filtered = AtagiaBenchRunner._filter_questions(
        questions,
        {"alpha", "beta"},
        {"q2"},
    )

    assert [question.question_id for question in filtered] == ["q2"]
