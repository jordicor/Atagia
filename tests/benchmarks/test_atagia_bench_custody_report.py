"""Tests for Atagia-bench failed-question custody reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.atagia_bench.custody_report import (
    build_failed_question_custody_report,
    save_failed_question_custody_report,
)
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaQuestionResult


def test_atagia_bench_failed_question_custody_report_extracts_context(tmp_path: Path) -> None:
    source_report = tmp_path / "report.json"
    source_payload = b'{"benchmark_name": "atagia-bench"}'
    source_report.write_bytes(source_payload)
    failed = AtagiaQuestionResult(
        question_id="q1",
        question_text="What did I decide?",
        ground_truth="Use SQLite",
        prediction="I do not know",
        answer_type="fact",
        category_tags=["retrieval"],
        evidence_turn_ids=["turn_1"],
        grade=GradeResult(
            passed=False,
            score=0.0,
            reason="Missing answer",
            grader_name="exact_match",
        ),
        memories_used=0,
        retrieval_time_ms=12.0,
        conversation_id="cnv_1",
        persona_id="persona_1",
        trace={
            "diagnosis_bucket": "retrieval_no_candidates",
            "sufficiency_diagnostic": "missing_raw_evidence",
            "evidence_memory_ids": ["mem_evidence"],
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
            "question_id": "q2",
            "grade": GradeResult(
                passed=True,
                score=1.0,
                reason="ok",
                grader_name="exact_match",
            ),
        }
    )
    report = AtagiaBenchReport(
        timestamp="2026-04-26T00:00:00+00:00",
        run_duration_seconds=1.0,
        config={},
        personas_used=["persona_1"],
        total_questions=2,
        total_passed=1,
        pass_rate=0.5,
        avg_score=0.5,
        critical_error_count=0,
        per_question=[failed, passed],
        per_category=[],
    )

    expected_hash = hashlib.sha256(source_payload).hexdigest()
    custody = build_failed_question_custody_report(
        report,
        source_report=str(source_report),
    )
    output_path = save_failed_question_custody_report(
        custody,
        tmp_path / "atagia-bench-failed-custody.json",
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert custody.total_failed_questions == 1
    assert custody.source_report_sha256 == expected_hash
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
    assert custody.failures[0].persona_id == "persona_1"
    assert custody.failures[0].category_tags == ["retrieval"]
    assert custody.failures[0].memories_used == 0
    assert custody.failures[0].retrieval_time_ms == 12.0
    assert custody.failures[0].retrieval_custody[0]["candidate_id"] == "mem_1"
    assert persisted["total_failed_questions"] == 1
    assert persisted["source_report_sha256"] == expected_hash
    assert persisted["retrieval_time_ms"]["mean"] == 12.0
    assert persisted["failures"][0]["retrieval_time_ms"] == 12.0
