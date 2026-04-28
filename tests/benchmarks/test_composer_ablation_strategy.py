"""Tests for composer-strategy benchmark ablation wiring."""

from __future__ import annotations

from benchmarks.atagia_bench.__main__ import _parse_ablation as parse_atagia_bench_ablation
from benchmarks.atagia_bench.adapter import AtagiaBenchDataset
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.locomo.__main__ import _parse_ablation as parse_locomo_ablation
from atagia.models.schemas_replay import AblationConfig


def test_composer_budgeted_marginal_benchmark_presets_parse() -> None:
    locomo = parse_locomo_ablation("composer_budgeted_marginal")
    atagia_bench = parse_atagia_bench_ablation("composer_budgeted_marginal")

    assert locomo == AblationConfig(composer_strategy="budgeted_marginal")
    assert atagia_bench == AblationConfig(composer_strategy="budgeted_marginal")


def test_atagia_bench_report_config_records_composer_strategy_ablation() -> None:
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
    )

    report = runner._build_report(
        [],
        dataset=AtagiaBenchDataset(),
        duration_seconds=0.0,
        persona_ids=[],
        category_tags=None,
        question_ids=None,
        exclude_question_ids=None,
        benchmark_split="all",
        holdout_question_ids=None,
        ablation=AblationConfig(composer_strategy="budgeted_marginal"),
        trusted_evaluation=False,
    )

    assert report.config["ablation_config"]["composer_strategy"] == "budgeted_marginal"

