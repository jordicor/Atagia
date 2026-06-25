from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.atagia_bench.adapter import AtagiaBenchAdapter
from benchmarks.atagia_bench.adapter import AtagiaBenchQuestion
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import (
    AtagiaBenchReport,
    AtagiaBenchRunner,
    AtagiaQuestionResult,
    CategoryStats,
    load_holdout_question_ids,
)
from atagia.services.run_counters import RunCounterAccumulator


def _write_minimal_atagia_bench_data(
    tmp_path: Path,
    *,
    question: dict[str, object],
) -> Path:
    data_dir = tmp_path / "atagia-bench-data"
    persona_dir = data_dir / "mini_persona"
    persona_dir.mkdir(parents=True)
    (data_dir / "personas.json").write_text(
        json.dumps(
            [
                {
                    "persona_id": "mini_persona",
                    "display_name": "Mini Persona",
                    "age": 42,
                    "occupation": "Tester",
                    "profile": "Minimal profile.",
                    "assistant_modes": ["general_qa"],
                    "conversation_count": 1,
                    "test_scenarios": ["source evidence validation"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (persona_dir / "conversations.json").write_text(
        json.dumps(
            [
                {
                    "conversation_id": "mini-conv-1",
                    "assistant_mode_id": "general_qa",
                    "timestamp_base": "2025-12-01T10:00:00",
                    "turns": [
                        {
                            "turn_id": "mini-t1",
                            "role": "user",
                            "text": "The notebook is red.",
                            "timestamp": "2025-12-01T10:00:00",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (persona_dir / "questions.json").write_text(
        json.dumps([question]),
        encoding="utf-8",
    )
    return data_dir


def test_atagia_bench_adapter_rejects_llm_judge_without_evidence(
    tmp_path: Path,
) -> None:
    data_dir = _write_minimal_atagia_bench_data(
        tmp_path,
        question={
            "question_id": "mini-q1",
            "question_text": "What color is the notebook?",
            "ground_truth": "red",
            "answer_type": "llm_judge",
            "category_tags": ["smoke"],
            "evidence_turn_ids": [],
            "grader": "llm_judge",
        },
    )

    with pytest.raises(ValueError, match="mini-q1.*no evidence_turn_ids"):
        AtagiaBenchAdapter(data_dir).load()


def test_atagia_bench_adapter_rejects_unresolved_evidence_turn_id(
    tmp_path: Path,
) -> None:
    data_dir = _write_minimal_atagia_bench_data(
        tmp_path,
        question={
            "question_id": "mini-q2",
            "question_text": "What color is the notebook?",
            "ground_truth": "red",
            "answer_type": "llm_judge",
            "category_tags": ["smoke"],
            "evidence_turn_ids": ["missing-turn"],
            "grader": "llm_judge",
        },
    )

    with pytest.raises(ValueError, match="mini-q2.*missing-turn"):
        AtagiaBenchAdapter(data_dir).load()


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


def test_save_run_manifest_records_dataset_hash_and_question_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ATAGIA_FACT_FACET_SURFACES_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_APPLICABILITY_GATE_MODE", "shadow")
    monkeypatch.setenv("ATAGIA_GRAPH_PROJECTION_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_RESPONSE_MODE", "smart_fast")
    monkeypatch.setenv("ATAGIA_ADAPTIVE_RETRIEVAL", "true")
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
        embedding_backend="sqlite_vec",
        answer_postcondition_guard_enabled=True,
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
        priority_failure_count=0,
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
    assert manifest["result_summary"]["priority_failure_count"] == 0
    assert manifest["retrieval_custody_summary"] == {
        "candidate_count": 1,
        "selected_count": 1,
    }
    assert manifest["activation_flags"] == {
        "fact_facet_surfaces_enabled": True,
        "fact_facet_retrieval_enabled": True,
        "applicability_gate_mode": "shadow",
        "answer_postcondition_guard_enabled": True,
        "embedding_backend": "sqlite_vec",
        "graph_projection_enabled": True,
        "response_mode": "smart_fast",
        "adaptive_retrieval": True,
    }
    assert manifest["run_counters"] == {"counts": {}, "labeled_counts": {}}
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


def test_run_manifest_records_non_empty_run_counters(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "personas.json").write_text("[]", encoding="utf-8")
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key=None,
        llm_model="static-model",
        judge_model="judge-model",
        manifests_dir=manifests_dir,
        data_dir=data_dir,
    )
    run_counters = RunCounterAccumulator()
    run_counters.increment("grounding_dropped_count")
    report = AtagiaBenchReport(
        timestamp="2026-04-26T00:00:00+00:00",
        run_duration_seconds=1.0,
        config={
            "provider": "static",
            "benchmark_split": "all",
            "question_filter": None,
            "exclude_question_filter": None,
            "holdout_question_ids": None,
            "run_counters": run_counters.snapshot(),
        },
        personas_used=[],
        total_questions=0,
        total_passed=0,
        pass_rate=0.0,
        avg_score=0.0,
        priority_failure_count=0,
        per_question=[],
        per_category=[],
    )
    report_path = tmp_path / "atagia-bench-report.json"
    report_path.write_text("{}", encoding="utf-8")

    manifest = runner.build_run_manifest(report, report_path=report_path)

    assert manifest["run_counters"] == {
        "counts": {"grounding_dropped_count": 1},
        "labeled_counts": {},
    }


def test_runner_role_specific_model_config_uses_base_as_role_fallback(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "personas.json").write_text("[]", encoding="utf-8")
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key=None,
        llm_model="gpt-5.5-instant",
        judge_model="gpt-5.4-mini",
        retrieval_model="gpt-5.4-nano",
        component_models={"need_detector_language": "gpt-5.4-mini"},
        manifests_dir=manifests_dir,
        data_dir=data_dir,
    )

    assert runner._model_config_summary() == {
        "model_mode": "role_specific",
        "base_model": "openai/gpt-5.5-instant",
        "forced_global_model": "",
        "ingest_model": "openai/gpt-5.5-instant",
        "retrieval_model": "openai/gpt-5.4-nano",
        "answer_model": "openai/gpt-5.5-instant",
        "component_models": {"need_detector_language": "openai/gpt-5.4-mini"},
        "answer_stance": "reactive",
        "answer_stance_prompt_variant": "baseline",
        "judge_model": "openai/gpt-5.4-mini",
    }
    assert runner._atagia_model_kwargs() == {
        "llm_ingest_model": "openai/gpt-5.5-instant",
        "llm_retrieval_model": "openai/gpt-5.4-nano",
        "llm_chat_model": "openai/gpt-5.5-instant",
        "llm_component_models": {"need_detector_language": "openai/gpt-5.4-mini"},
    }


def test_benchmark_db_helpers_resolve_directory_and_copy_sidecars(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_db = source_dir / "benchmark.db"
    source_db.write_bytes(b"db")
    (source_dir / "benchmark.db-wal").write_bytes(b"wal")
    (source_dir / "benchmark.db-shm").write_bytes(b"shm")

    resolved = AtagiaBenchRunner._resolve_reuse_db(source_dir)
    destination = tmp_path / "copy" / "benchmark.db"
    AtagiaBenchRunner._copy_sqlite_db(resolved, destination)

    assert resolved == source_db
    assert destination.read_bytes() == b"db"
    assert destination.with_name("benchmark.db-wal").read_bytes() == b"wal"
    assert destination.with_name("benchmark.db-shm").read_bytes() == b"shm"
