"""Tests for the LoCoMo failure-funnel report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.funnel_report import (
    _accumulate,
    _classify_stage,
    _gold_contained,
    _is_abstention,
    _s4_is_full,
    build_funnel_report,
)
from benchmarks.locomo.night_run_artifacts import QuestionRecord, iter_report_records


def _record(
    *,
    qid: str,
    category: int,
    strict_score: int,
    trace: dict[str, Any],
    prediction: str = "",
    ground_truth: str = "",
) -> QuestionRecord:
    return QuestionRecord(
        conversation_id="conv-x",
        question_id=qid,
        category=category,
        question_text="q",
        ground_truth=ground_truth,
        evidence_turn_ids=[],
        prediction=prediction,
        strict_score=strict_score,
        strict_reasoning="",
        strict_judge_model="kimi",
        trace=trace,
    )


def _s1_trace() -> dict[str, Any]:
    return {"evidence_memory_ids": [], "critical_evidence_custody": {"items": []}}


def _s2_trace() -> dict[str, Any]:
    return {
        "evidence_memory_ids": ["m1"],
        "selected_evidence_memory_ids": [],
        "critical_evidence_custody": {
            "items": [
                {"memory_id": "m1", "in_raw_candidates": False, "scored": False, "selected": False}
            ]
        },
    }


def _s3_trace() -> dict[str, Any]:
    return {
        "evidence_memory_ids": ["m1"],
        "selected_evidence_memory_ids": [],
        "critical_evidence_custody": {
            "items": [
                {"memory_id": "m1", "in_raw_candidates": True, "scored": True, "selected": False}
            ]
        },
        "retrieval_custody": [
            {"candidate_id": "m1", "eviction_reason": "budget_exhausted", "score_rank": 1}
        ],
    }


def _s4_full_trace() -> dict[str, Any]:
    return {
        "evidence_memory_ids": ["m1"],
        "evidence_message_ids": ["msgA"],
        "selected_evidence_memory_ids": ["m1"],
        "critical_evidence_custody": {
            "items": [
                {
                    "memory_id": "m1",
                    "in_raw_candidates": True,
                    "scored": True,
                    "selected": True,
                    "source_message_ids": ["msgA"],
                }
            ]
        },
    }


def _s4_partial_trace() -> dict[str, Any]:
    return {
        "evidence_memory_ids": ["m1", "m2"],
        "evidence_message_ids": ["msgA", "msgB"],
        "selected_evidence_memory_ids": ["m1"],
        "critical_evidence_custody": {
            "items": [
                {
                    "memory_id": "m1",
                    "in_raw_candidates": True,
                    "scored": True,
                    "selected": True,
                    "source_message_ids": ["msgA"],
                },
                {
                    "memory_id": "m2",
                    "in_raw_candidates": True,
                    "scored": True,
                    "selected": False,
                    "source_message_ids": ["msgB"],
                },
            ]
        },
        "retrieval_custody": [
            {"candidate_id": "m2", "eviction_reason": "lower_score", "score_rank": 12}
        ],
    }


def test_stage_classification() -> None:
    assert _classify_stage(_record(qid="q1", category=4, strict_score=1, trace={})) is None
    assert _classify_stage(_record(qid="q2", category=1, strict_score=0, trace=_s1_trace())) == "S1"
    assert _classify_stage(_record(qid="q3", category=1, strict_score=0, trace=_s2_trace())) == "S2"
    assert _classify_stage(_record(qid="q4", category=1, strict_score=0, trace=_s3_trace())) == "S3"
    assert _classify_stage(_record(qid="q5", category=1, strict_score=0, trace=_s4_full_trace())) == "S4"
    assert _classify_stage(_record(qid="q6", category=1, strict_score=0, trace=_s4_partial_trace())) == "S4"


def test_s4_full_vs_partial() -> None:
    assert _s4_is_full(_record(qid="a", category=1, strict_score=0, trace=_s4_full_trace())) is True
    assert _s4_is_full(_record(qid="b", category=1, strict_score=0, trace=_s4_partial_trace())) is False


def test_accumulate_full_funnel() -> None:
    records = [
        _record(qid="p", category=4, strict_score=1, trace=_s4_full_trace()),
        _record(qid="s1", category=1, strict_score=0, trace=_s1_trace()),
        _record(qid="s2", category=1, strict_score=0, trace=_s2_trace()),
        _record(qid="s3", category=1, strict_score=0, trace=_s3_trace()),
        _record(qid="s4f", category=2, strict_score=0, trace=_s4_full_trace()),
        _record(qid="s4p", category=1, strict_score=0, trace=_s4_partial_trace()),
    ]
    acc = _accumulate(records)
    assert acc.total_questions == 6
    assert acc.total_passed == 1
    assert acc.total_failed == 5
    assert dict(acc.stage_counts) == {"S1": 1, "S2": 1, "S3": 1, "S4": 2}
    assert acc.s4_full == 1
    assert acc.s4_partial == 1
    assert acc.s3_dying_items == 1
    assert dict(acc.s3_eviction) == {"budget_exhausted": 1}
    assert acc.s3_rank1_deaths == 1
    assert acc.s3_top10_deaths == 1
    assert acc.s4_partial_dying_items == 1
    assert dict(acc.s4_partial_eviction) == {"lower_score": 1}


def test_abstention_and_containment() -> None:
    assert _is_abstention("That information is not supported by the memories.") is True
    assert _is_abstention("Caroline attended on May 7, 2023.") is False
    # Month-name / digit normalization for gold containment.
    assert _gold_contained("She went on May 7, 2023.", "7 May 2023") is True
    assert _gold_contained("She went in April.", "7 May 2023") is False


def test_build_funnel_report_baseline_mismatch(tmp_path: Path) -> None:
    report = {
        "benchmark_name": "LoCoMo",
        "conversations": [
            {
                "conversation_id": "conv-x",
                "results": [
                    {
                        "question": {
                            "question_id": "conv-x:q1",
                            "category": 1,
                            "question_text": "q",
                            "ground_truth": "g",
                            "evidence_turn_ids": [],
                        },
                        "prediction": "not supported",
                        "score_result": {"score": 0, "reasoning": "", "judge_model": "kimi"},
                        "trace": _s3_trace(),
                    }
                ],
            }
        ],
    }
    report_path = tmp_path / "locomo-report-20260101T000000Z.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = build_funnel_report([report_path])
    assert result["funnel"]["S3"] == 1
    # A synthetic mini-run does not reproduce the real 2026-06-25 counts.
    assert result["baseline_match"]["count_exact"] is False
    assert result["abstention"]["abstaining_failures"] == 1
    assert result["memory_error_counters"]["available"] is False


def test_iter_report_records_fails_fast_on_malformed_results() -> None:
    base_question = {
        "question_id": "conv-x:q1",
        "category": 1,
        "question_text": "q",
        "ground_truth": "g",
        "evidence_turn_ids": [],
    }
    base_score = {"score": 0, "reasoning": "", "judge_model": "kimi"}

    def report_with(question: dict[str, Any], score_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "conversations": [
                {
                    "conversation_id": "conv-x",
                    "results": [
                        {
                            "question": question,
                            "prediction": "p",
                            "score_result": score_result,
                            "trace": {},
                        }
                    ],
                }
            ]
        }

    no_qid = dict(base_question)
    del no_qid["question_id"]
    with pytest.raises(ValueError, match="no question_id"):
        list(iter_report_records(report_with(no_qid, base_score)))

    no_category = dict(base_question)
    del no_category["category"]
    with pytest.raises(ValueError, match="no question category"):
        list(iter_report_records(report_with(no_category, base_score)))

    with pytest.raises(ValueError, match="no stored score_result.score"):
        list(iter_report_records(report_with(base_question, {"reasoning": ""})))

    # score=0 is a valid stored verdict, not a missing field.
    records = list(iter_report_records(report_with(base_question, base_score)))
    assert records[0].strict_score == 0


def _s3_dedupe_into_selected_trace() -> dict[str, Any]:
    """Gold collapsed into a SELECTED duplicate carrier (CS-2.3 artifact shape)."""
    return {
        "evidence_memory_ids": ["m1"],
        "evidence_message_ids": ["msgA"],
        "selected_evidence_memory_ids": [],
        "critical_evidence_custody": {
            "items": [
                {
                    "memory_id": "m1",
                    "in_raw_candidates": True,
                    "scored": False,
                    "selected": False,
                    "survival_stage": "fusion_dedupe_into_selected",
                    "deduped_into": "m_rep",
                    "deduped_into_selected": True,
                    "source_message_ids": ["msgA"],
                }
            ]
        },
        "retrieval_custody": [
            {
                "candidate_id": "m1",
                "eviction_reason": "deduped_duplicate_carrier",
                "score_rank": None,
            }
        ],
    }


def test_dedupe_into_selected_counts_as_s4_survival() -> None:
    """CS-2.3: gold deduped into a selected carrier reached the context
    content-wise; the funnel must not read it as an S3 retrieval death."""
    record = _record(
        qid="q_dedupe",
        category=1,
        strict_score=0,
        trace=_s3_dedupe_into_selected_trace(),
    )
    assert _classify_stage(record) == "S4"
    assert _s4_is_full(record) is True


def test_dedupe_into_unselected_rep_stays_s3() -> None:
    trace = _s3_dedupe_into_selected_trace()
    item = trace["critical_evidence_custody"]["items"][0]
    item["deduped_into_selected"] = False
    item["survival_stage"] = "fusion_dedupe"
    record = _record(qid="q_dedupe", category=1, strict_score=0, trace=trace)
    assert _classify_stage(record) == "S3"


def test_pre_dedupe_artifacts_classify_identically() -> None:
    """Old artifacts carry no dedupe keys; classification must be unchanged."""
    record = _record(
        qid="q_old",
        category=1,
        strict_score=0,
        trace=_s3_trace(),
    )
    assert _classify_stage(record) == "S3"
    full_record = _record(
        qid="q_old_full",
        category=1,
        strict_score=0,
        trace=_s4_full_trace(),
    )
    assert _classify_stage(full_record) == "S4"
    assert _s4_is_full(full_record) is True
