"""Tests for benchmark CLI judge model defaults."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.atagia_bench import __main__ as atagia_bench_cli
from benchmarks.locomo import __main__ as locomo_cli


PUBLIC_CLI_MODULES = [atagia_bench_cli, locomo_cli]
OPTIONAL_PRIVATE_CLI_MODULES = [
    "benchmarks.compaction_eval.__main__",
    "benchmarks.third_party.__main__",
]


@pytest.mark.parametrize(
    "cli_module",
    PUBLIC_CLI_MODULES,
)
def test_benchmark_default_judge_is_openrouter_luna_medium(cli_module) -> None:
    args = SimpleNamespace(provider="anthropic", judge_model=None)

    assert cli_module._resolve_judge_model(args) == "openrouter/openai/gpt-5.6-luna,medium"


@pytest.mark.parametrize(
    "cli_module",
    PUBLIC_CLI_MODULES,
)
def test_benchmark_explicit_judge_model_overrides_default(cli_module) -> None:
    args = SimpleNamespace(provider="anthropic", judge_model="openrouter/openai/gpt-5.5")

    assert cli_module._resolve_judge_model(args) == "openrouter/openai/gpt-5.5"


@pytest.mark.parametrize(
    "cli_module",
    PUBLIC_CLI_MODULES,
)
def test_non_openai_benchmark_default_judge_stays_openrouter_luna(cli_module) -> None:
    args = SimpleNamespace(provider="openrouter", judge_model=None)

    assert cli_module._resolve_judge_model(args) == "openrouter/openai/gpt-5.6-luna,medium"


@pytest.mark.parametrize("module_name", OPTIONAL_PRIVATE_CLI_MODULES)
def test_optional_private_benchmark_judge_defaults(module_name: str) -> None:
    cli_module = pytest.importorskip(
        module_name,
        reason="private benchmark harness is not present in this checkout",
    )

    assert cli_module._resolve_judge_model(
        SimpleNamespace(provider="anthropic", judge_model=None)
    ) == "openrouter/openai/gpt-5.6-luna,medium"
    assert cli_module._resolve_judge_model(
        SimpleNamespace(provider="anthropic", judge_model="openrouter/test-judge")
    ) == "openrouter/test-judge"
    assert cli_module._resolve_judge_model(
        SimpleNamespace(provider="openrouter", judge_model=None)
    ) == "openrouter/openai/gpt-5.6-luna,medium"
