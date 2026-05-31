"""Tests for LoCoMo session-2 runner utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.base import (
    BenchmarkQuestion,
    BenchmarkReport,
    ConversationReport,
    QuestionResult,
    ScoreResult,
)
from benchmarks.locomo import full_runner as full_runner_module
from benchmarks.locomo.full_runner import (
    DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS,
    FullLoCoMoRunConfig,
    build_evaluate_command,
    build_ingest_command,
    run_full_locomo,
)
from benchmarks.locomo.mode_comparison import build_mode_comparison


def test_full_runner_builds_two_phase_commands_and_manifest(tmp_path: Path) -> None:
    config = FullLoCoMoRunConfig(
        data_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "out",
        db_dir=tmp_path / "dbs",
        provider="openai",
        answer_model="openai/chat-latest",
        ingest_model="openai/ingest",
        retrieval_model="openai/retrieval",
        judge_model="anthropic/judge",
        component_models=("extractor=openai/extractor",),
        conversations="conv-1,conv-2",
        parallel_conversations=10,
        parallel_questions=8,
        adaptive_parallel_questions=True,
        adaptive_parallel_min=2,
        adaptive_parallel_retries=3,
        ingest_mode="online_async",
        diff_against=str(tmp_path / "baseline.json"),
        dry_run=True,
    )

    ingest_command = build_ingest_command(config)
    evaluate_command = build_evaluate_command(config)
    manifest_path = run_full_locomo(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "--ingest-only" in ingest_command
    assert "--keep-db" in ingest_command
    assert ingest_command[ingest_command.index("--benchmark-db-dir") + 1] == str(config.db_dir)
    assert ingest_command[ingest_command.index("--parallel-conversations") + 1] == "10"
    assert "--evaluate-only" in evaluate_command
    assert evaluate_command[evaluate_command.index("--reuse-db-dir") + 1] == str(config.db_dir)
    assert evaluate_command[evaluate_command.index("--parallel-questions") + 1] == "8"
    assert "--adaptive-parallel-questions" in evaluate_command
    assert evaluate_command[evaluate_command.index("--adaptive-parallel-min") + 1] == "2"
    assert evaluate_command[evaluate_command.index("--adaptive-parallel-retries") + 1] == "3"
    assert "--diff-against" in evaluate_command
    assert manifest["manifest_kind"] == "locomo_full_run_manifest"
    assert manifest["dry_run"] is True
    assert [phase["phase"] for phase in manifest["phases"]] == ["ingest", "evaluate"]
    assert all(phase["returncode"] is None for phase in manifest["phases"])


def test_full_runner_defaults_to_batch_ingest_with_periodic_flush(tmp_path: Path) -> None:
    config = FullLoCoMoRunConfig(
        data_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "out",
        db_dir=tmp_path / "dbs",
        provider="openai",
        answer_model="openai/chat-latest",
        dry_run=True,
    )

    ingest_command = build_ingest_command(config)

    assert ingest_command[ingest_command.index("--ingest-mode") + 1] == "online_batch"
    assert ingest_command[ingest_command.index("--flush-every-turns") + 1] == str(
        DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS
    )


def test_full_runner_skips_evaluate_when_ingest_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched_commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], cwd: Path) -> None:
            del cwd
            launched_commands.append(command)
            self._returncode = 7 if "--ingest-only" in command else 0

        def wait(self) -> int:
            return self._returncode

        def poll(self) -> int:
            return self._returncode

        def terminate(self) -> None:
            self._returncode = -15

        def kill(self) -> None:
            self._returncode = -9

    monkeypatch.setattr(full_runner_module.subprocess, "Popen", FakeProcess)
    config = FullLoCoMoRunConfig(
        data_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "out",
        db_dir=tmp_path / "dbs",
        provider="openai",
        answer_model="openai/chat-latest",
    )

    with pytest.raises(SystemExit, match="ingest"):
        run_full_locomo(config)

    manifest = json.loads(
        (config.output_dir / "locomo-full-run-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(launched_commands) == 1
    assert "--ingest-only" in launched_commands[0]
    assert [phase["phase"] for phase in manifest["phases"]] == ["ingest", "evaluate"]
    assert manifest["phases"][0]["returncode"] == 7
    assert manifest["phases"][1]["skipped"] is True
    assert manifest["phases"][1]["skipped_reason"] == "ingest_failed"


def test_mode_comparison_reports_pairwise_question_deltas() -> None:
    before = _report(
        score=0,
        prediction="blue",
        diagnosis="retrieval_or_ranking_miss",
        total_correct=0,
    )
    after = _report(
        score=1,
        prediction="red",
        diagnosis="passed",
        total_correct=1,
    )

    comparison = build_mode_comparison({"bulk": before, "online_async": after})
    pairwise = comparison["pairwise"][0]

    assert comparison["reports"]["bulk"]["ingest_mode"] == "bulk"
    assert comparison["reports"]["online_async"]["ingest_mode"] == "online_async"
    assert pairwise["before"] == "bulk"
    assert pairwise["after"] == "online_async"
    assert pairwise["accuracy_delta"] == 1.0
    assert pairwise["correct_delta"] == 1
    assert pairwise["common_questions"] == 1
    assert pairwise["changed_questions"] == 1
    assert pairwise["improved_questions"] == 1
    assert pairwise["regressed_questions"] == 0
    assert pairwise["per_question_deltas"] == [
        {
            "conversation_id": "conv-1",
            "question_id": "conv-1:q1",
            "category": 1,
            "question": "What color notebooks?",
            "score_before": 0,
            "score_after": 1,
            "score_delta": 1,
            "diagnosis_before": "retrieval_or_ranking_miss",
            "diagnosis_after": "passed",
            "prediction_changed": True,
        }
    ]


def _report(
    *,
    score: int,
    prediction: str,
    diagnosis: str,
    total_correct: int,
) -> BenchmarkReport:
    question = BenchmarkQuestion(
        question_text="What color notebooks?",
        ground_truth="red",
        category=1,
        evidence_turn_ids=["D1:1"],
        question_id="conv-1:q1",
    )
    result = QuestionResult(
        question=question,
        prediction=prediction,
        score_result=ScoreResult(
            score=score,
            reasoning="ok" if score else "miss",
            judge_model="judge-model",
        ),
        memories_used=1,
        retrieval_time_ms=1.0,
        trace={"diagnosis_bucket": diagnosis},
    )
    mode = "online_async" if score else "bulk"
    return BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=float(score),
        category_breakdown={1: float(score)},
        conversations=[
            ConversationReport(
                conversation_id="conv-1",
                results=[result],
                accuracy=float(score),
                category_breakdown={1: float(score)},
            )
        ],
        total_questions=1,
        total_correct=total_correct,
        ablation_config=None,
        timestamp="2026-05-22T00:00:00+00:00",
        model_info={"ingest_mode": mode},
        duration_seconds=1.0,
    )
