"""Tests for failed-question custody reports."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.custody_report import build_failed_question_custody_report


def test_failed_question_custody_report_extracts_failure_context(tmp_path: Path) -> None:
    source_report = tmp_path / "report.json"
    source_payload = b'{"benchmark_name": "LoCoMo"}'
    source_report.write_bytes(source_payload)
    failed = QuestionResult(
        question=BenchmarkQuestion(
            question_text="What did I decide?",
            ground_truth="Use SQLite",
            category=1,
            evidence_turn_ids=["D1:1"],
            question_id="conv-test:q1",
        ),
        prediction="I do not know",
        score_result=ScoreResult(
            score=0,
            reasoning="Missing answer",
            judge_model="judge-model",
        ),
        memories_used=0,
        retrieval_time_ms=12.0,
        trace={
            "diagnosis_bucket": "retrieval_no_candidates",
            "sufficiency_diagnostic": "missing_raw_evidence",
            "evidence_turn_ids": ["D1:1"],
            "selected_memory_ids": [],
            "retrieval_custody": [
                {
                    "candidate_id": "mem_1",
                    "candidate_kind": "memory",
                    "channels": ["fts"],
                    "selected": False,
                }
            ],
        },
    )
    passed = failed.model_copy(
        update={
            "score_result": ScoreResult(
                score=1,
                reasoning="ok",
                judge_model="judge-model",
            )
        }
    )
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=0.5,
        category_breakdown={1: 0.5},
        conversations=[
            ConversationReport(
                conversation_id="conv-test",
                results=[failed, passed],
                accuracy=0.5,
                category_breakdown={1: 0.5},
            )
        ],
        total_questions=2,
        total_correct=1,
        ablation_config=None,
        timestamp="2026-04-26T00:00:00+00:00",
        model_info={},
        duration_seconds=1.0,
    )

    custody = build_failed_question_custody_report(
        report,
        source_report=str(source_report),
    )

    assert custody.total_failed_questions == 1
    assert custody.source_report_sha256 == hashlib.sha256(source_payload).hexdigest()
    assert custody.diagnosis_counts == {"retrieval_no_candidates": 1}
    assert custody.sufficiency_counts == {"missing_raw_evidence": 1}
    assert custody.memories_used == {
        "count": 1,
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    assert custody.retrieval_time_ms == {
        "count": 1,
        "mean": 12.0,
        "min": 12.0,
        "max": 12.0,
    }
    assert custody.retrieval_custody_summary == {
        "candidate_count": 1,
        "selected_count": 0,
        "channel_counts": {"fts": 1},
        "selected_channel_counts": {},
        "candidate_kind_counts": {"memory": 1},
        "composer_decision_counts": {},
        "filter_reason_counts": {},
    }
    assert custody.failures[0].memories_used == 0
    assert custody.failures[0].retrieval_time_ms == 12.0
    assert custody.failures[0].retrieval_custody[0]["candidate_id"] == "mem_1"
