"""Tests for composer-strategy benchmark ablation wiring."""

from __future__ import annotations

from benchmarks.atagia_bench.__main__ import (
    _benchmark_ablation as build_atagia_bench_ablation,
    _parse_ablation as parse_atagia_bench_ablation,
)
from benchmarks.atagia_bench.adapter import AtagiaBenchDataset
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.locomo.__main__ import _parse_ablation as parse_locomo_ablation
from benchmarks.locomo.__main__ import (
    _benchmark_ablation as build_locomo_ablation,
)
from atagia.models.schemas_replay import AblationConfig


def test_composer_budgeted_marginal_benchmark_presets_parse() -> None:
    locomo = parse_locomo_ablation("composer_budgeted_marginal")
    atagia_bench = parse_atagia_bench_ablation("composer_budgeted_marginal")

    assert locomo == AblationConfig(composer_strategy="budgeted_marginal")
    assert atagia_bench == AblationConfig(composer_strategy="budgeted_marginal")


def test_composer_evidence_obligation_benchmark_presets_parse() -> None:
    locomo = parse_locomo_ablation("composer_evidence_obligation")
    atagia_bench = parse_atagia_bench_ablation("composer_evidence_obligation")

    assert locomo == AblationConfig(enable_evidence_obligation_coverage=True)
    assert atagia_bench == AblationConfig(enable_evidence_obligation_coverage=True)


def test_applicability_gate_benchmark_presets_parse() -> None:
    locomo_shadow = parse_locomo_ablation("applicability_gate_shadow")
    atagia_shadow = parse_atagia_bench_ablation("applicability_gate_shadow")
    locomo_enforced = parse_locomo_ablation("applicability_gate_enforced")
    atagia_enforced = parse_atagia_bench_ablation("applicability_gate_enforced")

    assert locomo_shadow == AblationConfig(applicability_gate_mode="shadow")
    assert atagia_shadow == AblationConfig(applicability_gate_mode="shadow")
    assert locomo_enforced == AblationConfig(applicability_gate_mode="enforced")
    assert atagia_enforced == AblationConfig(applicability_gate_mode="enforced")


def test_context_envelope_benchmark_presets_parse() -> None:
    locomo_4k = parse_locomo_ablation("context_envelope_4k")
    atagia_bench_8k = parse_atagia_bench_ablation("context_envelope_8k")
    locomo = parse_locomo_ablation("context_envelope_32k")
    atagia_bench = parse_atagia_bench_ablation("context_envelope_32k")
    expected = AblationConfig(
        context_envelope_budget_tokens=32_768,
        override_retrieval_params={
            "rerank_top_k": 25,
            "final_context_items": 20,
        },
    )

    assert locomo_4k == AblationConfig(context_envelope_budget_tokens=4_096)
    assert atagia_bench_8k == AblationConfig(context_envelope_budget_tokens=8_192)
    assert locomo == expected
    assert atagia_bench == expected


def test_atagia_bench_privacy_enforcement_flag_builds_benchmark_ablation() -> None:
    ablation = build_atagia_bench_ablation("composer_budgeted_marginal", "off")

    assert ablation == AblationConfig(
        privacy_enforcement="off",
        composer_strategy="budgeted_marginal",
    )


def test_atagia_bench_privacy_enforcement_defaults_to_off() -> None:
    ablation = build_atagia_bench_ablation(None, None)

    assert ablation == AblationConfig(privacy_enforcement="off")


def test_locomo_privacy_enforcement_defaults_to_off() -> None:
    ablation = build_locomo_ablation(None, None)

    assert ablation == AblationConfig(privacy_enforcement="off")


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
