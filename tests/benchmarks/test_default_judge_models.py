"""Tests for benchmark CLI judge model defaults."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.atagia_bench import __main__ as atagia_bench_cli
from benchmarks.compaction_eval import __main__ as compaction_eval_cli
from benchmarks.locomo import __main__ as locomo_cli
from benchmarks.third_party import __main__ as third_party_cli


@pytest.mark.parametrize(
    "cli_module",
    [
        atagia_bench_cli,
        compaction_eval_cli,
        locomo_cli,
        third_party_cli,
    ],
)
def test_benchmark_default_judge_is_direct_anthropic_opus_4_7(cli_module) -> None:
    args = SimpleNamespace(provider="anthropic", judge_model=None)

    assert cli_module._resolve_judge_model(args) == "anthropic/claude-opus-4-7"


@pytest.mark.parametrize(
    "cli_module",
    [
        atagia_bench_cli,
        compaction_eval_cli,
        locomo_cli,
        third_party_cli,
    ],
)
def test_benchmark_explicit_judge_model_overrides_default(cli_module) -> None:
    args = SimpleNamespace(provider="anthropic", judge_model="openrouter/openai/gpt-5.5")

    assert cli_module._resolve_judge_model(args) == "openrouter/openai/gpt-5.5"


@pytest.mark.parametrize(
    "cli_module",
    [
        atagia_bench_cli,
        compaction_eval_cli,
        locomo_cli,
        third_party_cli,
    ],
)
def test_non_anthropic_benchmark_default_judge_stays_direct_anthropic(cli_module) -> None:
    args = SimpleNamespace(provider="openrouter", judge_model=None)

    assert cli_module._resolve_judge_model(args) == "anthropic/claude-opus-4-7"
