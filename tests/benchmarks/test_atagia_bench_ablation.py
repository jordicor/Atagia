"""Atagia-bench must APPLY the ablation it reports, and prove its configuration.

Both properties are structural: a report that serializes an ``ablation_config``
the engine never received, or omits the effective settings a run executed with,
describes a run that did not happen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atagia.models.schemas_replay import AblationConfig
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from tests.benchmarks.test_atagia_bench_manifest import (
    _write_minimal_atagia_bench_data,
)
from tests.benchmarks.test_locomo_benchmark import (
    BenchmarkProvider,
    _install_stub_client,
)

_QUESTION = {
    "question_id": "mini-q1",
    "question_text": "What color is the notebook?",
    "ground_truth": "red",
    "answer_type": "exact_match",
    "category_tags": ["factual"],
    "evidence_turn_ids": ["mini-t1"],
    "grader": "exact_match",
}


def _runner(tmp_path: Path) -> AtagiaBenchRunner:
    return AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        data_dir=_write_minimal_atagia_bench_data(tmp_path, question=_QUESTION),
    )


@pytest.mark.asyncio
async def test_ablation_reaches_the_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retrieval overrides an Atagia-bench run declares must be the ones the
    engine ran with.

    The assertion reads ``applied_override_retrieval_params`` off the retrieval
    trace, which the pipeline fills from the ablation it actually resolved, so
    it fails if the ablation is recorded in the report but dropped at the
    ``engine.chat`` call (which is exactly what it used to do: every switch
    except ``skip_belief_revision`` and ``skip_compaction`` was inert, including
    trusted-evaluation's privacy overrides).
    """
    _install_stub_client(monkeypatch, BenchmarkProvider())
    runner = _runner(tmp_path)
    ablation = AblationConfig(
        override_retrieval_params={
            "privacy_ceiling": 3,
            "allow_private_sensitivity": True,
            "max_context_items": 2,
        }
    )

    report = await runner.run(
        ablation=ablation,
        benchmark_db_dir=tmp_path / "dbs",
        allow_temp_benchmark_db_dir=True,
    )

    assert len(report.per_question) == 1
    retrieval_trace = report.per_question[0].trace["retrieval_trace"]
    assert retrieval_trace["applied_override_retrieval_params"] == {
        "privacy_ceiling": 3,
        "allow_private_sensitivity": True,
        "max_context_items": 2,
    }
    # The report keeps claiming what ran, and now it is true.
    assert report.config["ablation_config"]["override_retrieval_params"] == {
        "privacy_ceiling": 3,
        "allow_private_sensitivity": True,
        "max_context_items": 2,
    }


@pytest.mark.asyncio
async def test_privacy_enforcement_ablation_reaches_the_retrieval_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``privacy_enforcement`` has a single source of truth.

    The AblationConfig field drives both the retrieval pipeline (through the
    forwarded ablation) and the engine's separate request-authority arguments
    (projected from the same field), so the two can no longer disagree.
    """
    _install_stub_client(monkeypatch, BenchmarkProvider())
    runner = _runner(tmp_path)

    report = await runner.run(
        ablation=AblationConfig(privacy_enforcement="off"),
        benchmark_db_dir=tmp_path / "dbs",
        allow_temp_benchmark_db_dir=True,
    )

    retrieval_trace = report.per_question[0].trace["retrieval_trace"]
    assert retrieval_trace["privacy_enforcement"] == "off"


@pytest.mark.asyncio
async def test_report_and_manifest_carry_effective_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CS-1.3 applies to every run, and Atagia-bench is a run: the effective
    configuration rides in the report and surfaces in the run manifest, with the
    provider secret redacted."""
    sentinel_key = "SENTINEL_ATAGIA_BENCH_KEY_7Q"
    _install_stub_client(monkeypatch, BenchmarkProvider())
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key=sentinel_key,
        llm_model="answer-model",
        judge_model="judge-model",
        data_dir=_write_minimal_atagia_bench_data(tmp_path, question=_QUESTION),
    )

    report = await runner.run(
        benchmark_db_dir=tmp_path / "dbs",
        allow_temp_benchmark_db_dir=True,
    )

    effective = report.config["effective_settings"]
    assert set(effective) == {"settings", "resolved_policy"}
    assert effective["settings"]["openai_api_key"]["value"] == "<redacted:set>"
    assert effective["settings"]["openai_api_key"]["redacted"] is True
    assert effective["resolved_policy"]["general_qa"]["context_budget_tokens"][
        "provenance"
    ] == "manifest"

    report_path = tmp_path / "atagia-bench-report.json"
    report_path.write_text(json.dumps({"benchmark_name": "Atagia-bench"}))
    manifest = runner.build_run_manifest(report, report_path=report_path)

    assert manifest["effective_settings"] == effective
    assert sentinel_key not in json.dumps(manifest)
