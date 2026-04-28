from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.atagia_bench.adapter import AtagiaBenchQuestion
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import (
    AtagiaBenchReport,
    AtagiaBenchRunner,
    AtagiaQuestionResult,
    CategoryStats,
    load_holdout_question_ids,
)


def test_load_holdout_question_ids_returns_sorted_unique_values(tmp_path: Path) -> None:
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(
        json.dumps({"question_ids": ["q2", "q1", "q2"]}),
        encoding="utf-8",
    )

    assert load_holdout_question_ids(holdout_path) == ["q1", "q2"]


def test_filter_questions_can_exclude_holdout_ids() -> None:
    questions = [
        AtagiaBenchQuestion(
            question_id="q1",
            question_text="Question one?",
            ground_truth="one",
            answer_type="fact",
            category_tags=["smoke"],
            evidence_turn_ids=[],
            grader="exact_match",
        ),
        AtagiaBenchQuestion(
            question_id="q2",
            question_text="Question two?",
            ground_truth="two",
            answer_type="fact",
            category_tags=["holdout"],
            evidence_turn_ids=[],
            grader="exact_match",
        ),
    ]

    filtered = AtagiaBenchRunner._filter_questions(
        questions,
        category_filter=None,
        question_filter=None,
        exclude_question_filter={"q2"},
    )

    assert [question.question_id for question in filtered] == ["q1"]


def test_save_run_manifest_records_dataset_hash_and_question_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "personas.json").write_text("[]", encoding="utf-8")
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "general_qa.json").write_text("{}", encoding="utf-8")
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps({"question_ids": ["q1"]}), encoding="utf-8")
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key=None,
        llm_model="static-model",
        judge_model="judge-model",
        manifests_dir=manifests_dir,
        data_dir=data_dir,
    )
    report = AtagiaBenchReport(
        timestamp="2026-04-26T00:00:00+00:00",
        run_duration_seconds=1.0,
        config={
            "provider": "static",
            "answer_model": "static-model",
            "judge_model": "judge-model",
            "benchmark_split": "holdout",
            "question_filter": ["q1"],
            "exclude_question_filter": None,
            "holdout_question_ids": ["q1"],
            "retrieval_custody_summary": {
                "candidate_count": 1,
                "selected_count": 1,
            },
        },
        personas_used=["mini"],
        total_questions=1,
        total_passed=1,
        pass_rate=1.0,
        avg_score=1.0,
        critical_error_count=0,
        per_question=[
            AtagiaQuestionResult(
                question_id="q1",
                question_text="Question?",
                ground_truth="Answer",
                prediction="Answer",
                answer_type="fact",
                grade=GradeResult(
                    passed=True,
                    score=1.0,
                    reason="ok",
                    grader_name="exact_match",
                ),
                memories_used=2,
                retrieval_time_ms=42.0,
                conversation_id="cnv_1",
                persona_id="mini",
            )
        ],
        per_category=[
            CategoryStats(category="smoke", count=1, pass_count=1, pass_rate=1.0, avg_score=1.0)
        ],
    )
    report_path = tmp_path / "atagia-bench-report-20260426T000000Z.json"
    report_payload = b'{"benchmark_name": "atagia-bench-v0"}'
    report_path.write_bytes(report_payload)
    custody_path = tmp_path / "atagia-bench-failed-custody.json"
    custody_payload = b'{"total_failed_questions": 0}'
    custody_path.write_bytes(custody_payload)
    taxonomy_path = tmp_path / "atagia-bench-failure-taxonomy.json"
    taxonomy_payload = b'{"total_failed_questions": 0}'
    taxonomy_path.write_bytes(taxonomy_payload)

    manifest_path = runner.save_run_manifest(
        report,
        report_path=report_path,
        holdout_path=holdout_path,
        custody_path=custody_path,
        taxonomy_path=taxonomy_path,
        failure_taxonomy_summary={"taxonomy_counts": {}},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_kind"] == "atagia_bench_run_manifest"
    assert manifest["dataset"]["sha256"]
    assert manifest["dataset"]["holdout_sha256"]
    assert manifest["report_sha256"] == hashlib.sha256(report_payload).hexdigest()
    assert manifest["custody_path"] == str(custody_path)
    assert manifest["custody_sha256"] == hashlib.sha256(custody_payload).hexdigest()
    assert manifest["taxonomy_path"] == str(taxonomy_path)
    assert manifest["taxonomy_sha256"] == hashlib.sha256(taxonomy_payload).hexdigest()
    assert manifest["failure_taxonomy_summary"] == {"taxonomy_counts": {}}
    assert manifest["manifests"]["sha256"]
    assert manifest["migrations"]["sha256"]
    assert manifest["migrations"]["latest_version"] == max(manifest["migrations"]["versions"])
    assert manifest["selection"] == {
        "benchmark_split": "holdout",
        "question_filter": ["q1"],
        "exclude_question_filter": None,
        "holdout_question_count": 1,
        "selected_question_count": 1,
    }
    assert manifest["diagnosis_bucket_counts"] == {"unknown": 1}
    assert manifest["sufficiency_diagnostic_counts"] == {"unknown": 1}
    assert manifest["retrieval_custody_summary"] == {
        "candidate_count": 1,
        "selected_count": 1,
    }
    assert manifest["result_summary"]["retrieval_time_ms"] == {
        "count": 1,
        "mean": 42.0,
        "min": 42.0,
        "max": 42.0,
    }
    assert manifest["result_summary"]["memories_used"] == {
        "count": 1,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
    }
    assert manifest["question_ids"] == ["q1"]
    assert manifest["benchmark_questions_persisted_as_messages"] is False
