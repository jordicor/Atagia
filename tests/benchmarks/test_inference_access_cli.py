"""Inference-access propagation for the two supported benchmark CLIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.atagia_bench.__main__ import _build_parser as atagia_bench_parser
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.llm_config import provider_api_key_kwargs
from benchmarks.locomo.__main__ import _build_parser as locomo_parser
from benchmarks.locomo.benchmark import LoCoMoBenchmark


def test_primary_benchmark_parsers_expose_consistent_inference_flags() -> None:
    common = [
        "--inference-access-mode",
        "local_only",
        "--local-llm-endpoints-file",
        "/tmp/local-endpoints.json",
        "--zero-cost-openrouter-profile",
        "dedicated_free_tier_no_byok",
    ]

    atagia_args = atagia_bench_parser().parse_args(["--provider", "local", *common])
    locomo_args = locomo_parser().parse_args(common)

    for args in (atagia_args, locomo_args):
        assert args.inference_access_mode == "local_only"
        assert args.local_llm_endpoints_file == "/tmp/local-endpoints.json"
        assert args.zero_cost_openrouter_profile == "dedicated_free_tier_no_byok"


def test_primary_runners_propagate_local_judge_into_startup_audit(
    tmp_path: Path,
) -> None:
    judge = "local/gpu_b/judge-b"
    common = {
        "llm_provider": "local",
        "llm_api_key": None,
        "llm_model": "local/gpu_a/chat-a",
        "judge_model": judge,
        "inference_access_mode": "local_only",
        "local_llm_endpoints_file": tmp_path / "endpoints.json",
    }
    atagia_runner = AtagiaBenchRunner(data_dir=tmp_path, **common)
    locomo_runner = LoCoMoBenchmark(data_path=tmp_path / "locomo.json", **common)

    assert atagia_runner._inference_access_kwargs()[
        "_inference_startup_completion_models"
    ] == {"atagia_bench.judge": judge}
    assert locomo_runner._inference_access_kwargs()[
        "_inference_startup_completion_models"
    ] == {"locomo.judge": judge}


def test_local_endpoint_credentials_are_catalog_owned() -> None:
    with pytest.raises(ValueError, match="api_key_env"):
        provider_api_key_kwargs("local", "not-a-provider-key")
