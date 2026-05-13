from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from benchmarks.atagia_bench.adapter import AtagiaBenchPersonaData
from benchmarks.atagia_bench.graders import GradeResult
from benchmarks.atagia_bench.runner import AtagiaBenchRunner, AtagiaQuestionResult


class FakeParallelRunner(AtagiaBenchRunner):
    async def _run_persona(
        self,
        persona_data: AtagiaBenchPersonaData,
        *,
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        exclude_question_filter: set[str] | None,
        ablation: Any,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db: str | Path | None,
        evaluate_only: bool,
    ) -> tuple[list[AtagiaQuestionResult], dict[str, Any], dict[str, Any]]:
        persona_id = persona_data.persona.persona_id
        if persona_id == "persona_a":
            await asyncio.sleep(0.01)
        question_id = f"{persona_id}-q1"
        return [
            AtagiaQuestionResult(
                question_id=question_id,
                question_text="Question?",
                ground_truth="Answer",
                prediction="Answer",
                answer_type="fact",
                category_tags=["smoke"],
                grade=GradeResult(
                    passed=True,
                    score=1.0,
                    reason="ok",
                    grader_name="exact_match",
                ),
                memories_used=1,
                retrieval_time_ms=2.0,
                conversation_id=f"{persona_id}-conv",
                persona_id=persona_id,
                trace={"retrieval_custody": []},
            )
        ], {
            "total_calls": 1,
            "failed_calls": 0,
            "total_latency_ms": 10.0,
            "mean_latency_ms": 10.0,
            "token_totals": {"total_tokens": 3},
            "cost_totals": {"cost": 0.001},
            "model_call_counts": {"fake-model": 1},
            "by_purpose": {
                "fake": {
                    "calls": 1,
                    "failed_calls": 0,
                    "total_latency_ms": 10.0,
                    "mean_latency_ms": 10.0,
                    "token_totals": {"total_tokens": 3},
                    "cost_totals": {"cost": 0.001},
                }
            },
        }, {}


def test_parallel_personas_preserves_report_order_and_records_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "personas.json").write_text(
        json.dumps(
            [
                _persona_payload("persona_a", "Persona A"),
                _persona_payload("persona_b", "Persona B"),
            ]
        ),
        encoding="utf-8",
    )
    for persona_id in ("persona_a", "persona_b"):
        persona_dir = data_dir / persona_id
        persona_dir.mkdir()
        (persona_dir / "conversations.json").write_text("[]", encoding="utf-8")
        (persona_dir / "questions.json").write_text("[]", encoding="utf-8")

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    runner = FakeParallelRunner(
        llm_provider="openai",
        llm_api_key=None,
        llm_model="fake-model",
        judge_model="judge-model",
        manifests_dir=manifests_dir,
        data_dir=data_dir,
    )

    report = asyncio.run(runner.run(parallel_personas=2))

    assert report.personas_used == ["persona_a", "persona_b"]
    assert [result.persona_id for result in report.per_question] == [
        "persona_a",
        "persona_b",
    ]
    assert report.config["parallel_personas"] == 2
    assert report.config["llm_call_summary"]["total_calls"] == 2
    assert report.config["llm_call_summary"]["token_totals"] == {"total_tokens": 6.0}
    assert report.config["llm_call_summary"]["cost_totals"] == {"cost": 0.002}


def _persona_payload(persona_id: str, display_name: str) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "display_name": display_name,
        "age": 30,
        "occupation": "tester",
        "profile": "profile",
        "assistant_modes": ["general_qa"],
        "conversation_count": 0,
        "test_scenarios": [],
    }
