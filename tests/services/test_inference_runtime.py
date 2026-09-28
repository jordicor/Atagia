"""Focused public activation tests for inference access modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atagia import Atagia
from atagia.app import initialize_runtime
from atagia.core.config import Settings
from atagia.services.inference_routes import InferenceAccessMode, InferenceRouteError
from atagia.services.inference_runtime import (
    format_inference_access_diagnostics,
    prepare_inference_access,
    require_unrestricted_inference,
)
from atagia.services.llm_client import ConfigurationError
from atagia.services.providers import build_llm_client


def _write_catalog(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "endpoints": [
                    {
                        "id": "gpu_a",
                        "adapter": "openai_compatible",
                        "base_url": "http://192.168.50.20:11434/v1",
                        "chat_models": ["chat-a", "chat-b"],
                        "embedding_models": ["embed-a"],
                    },
                    {
                        "id": "gpu_b",
                        "adapter": "openai_compatible",
                        "base_url": "http://192.168.50.21:11435/v1",
                        "chat_models": ["judge-b"],
                        "embedding_models": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _local_settings(
    tmp_path: Path,
    *,
    mode: str,
    extra: dict[str, str] | None = None,
) -> Settings:
    catalog_path = _write_catalog(tmp_path / f"{mode}-endpoints.json")
    environ = {
        "ATAGIA_SQLITE_PATH": str(tmp_path / f"{mode}.db"),
        "ATAGIA_INFERENCE_ACCESS_MODE": mode,
        "ATAGIA_LOCAL_LLM_ENDPOINTS_FILE": str(catalog_path),
        "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "local/gpu_a/chat-a",
    }
    environ.update(extra or {})
    return Settings.from_env(environ)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
async def test_restricted_runtime_bootstraps_all_local_without_network(
    tmp_path: Path,
    mode: str,
) -> None:
    runtime = await initialize_runtime(_local_settings(tmp_path, mode=mode))

    try:
        assert runtime.inference_access.policy.mode is InferenceAccessMode(mode)
        assert set(runtime.llm_client._providers) == {"local:gpu_a", "local:gpu_b"}
    finally:
        await runtime.close()


@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
@pytest.mark.parametrize("component", [
    "APPLICABILITY_RELEVANCE", "NEED_DETECTOR_MEMORY", "NEED_DETECTOR_EXACT", "NEED_DETECTOR_SHAPE",
    "NEED_DETECTOR_LANGUAGE", "NEED_DETECTOR_NEEDS", "NEED_DETECTOR_FACETS", "NEED_DETECTOR_CALLBACK", "INTENT_CLASSIFIER",
])
async def test_restricted_startup_rejects_native_typesafe_even_with_free_output(
    tmp_path: Path, mode: str, component: str,
) -> None:
    settings = _local_settings(tmp_path, mode=mode, extra={
        "ATAGIA_TYPESAFE_API_KEY": "test-key",
        "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
        f"ATAGIA_LLM_MODEL__{component}": "typesafe/jev-latest",
    })

    def unexpected_client(**_kwargs: Any) -> Any:
        raise AssertionError("No HTTP client should be opened for denied routes")

    with pytest.raises(InferenceRouteError, match="typesafe/jev-latest"):
        await prepare_inference_access(settings, http_client_factory=unexpected_client)


@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
async def test_restricted_startup_audits_the_common_finite_decision_route(
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _local_settings(
        tmp_path,
        mode=mode,
        extra={
            "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "",
            "ATAGIA_LLM_INGEST_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_RETRIEVAL_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_CHAT_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
        },
    )

    with pytest.raises(InferenceRouteError, match="typesafe/jev-latest"):
        await prepare_inference_access(settings)


@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
async def test_restricted_startup_accepts_a_local_finite_decision_model(
    tmp_path: Path,
    mode: str,
) -> None:
    settings = _local_settings(
        tmp_path,
        mode=mode,
        extra={
            "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "",
            "ATAGIA_LLM_INGEST_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_RETRIEVAL_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_CHAT_MODEL": "local/gpu_a/chat-a",
            "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
            "ATAGIA_LLM_FINITE_DECISION_MODEL": "local/gpu_a/chat-b",
        },
    )

    prepared = await prepare_inference_access(settings)

    assert any(
        route.source == "finite_decision"
        and route.model_spec == "local/gpu_a/chat-b"
        for route in prepared.startup_routes
    )


@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
async def test_restricted_factory_does_not_construct_unused_typesafe_transport(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _local_settings(tmp_path, mode=mode, extra={"ATAGIA_TYPESAFE_API_KEY": "test-key"})

    def unexpected_provider(**_kwargs: Any) -> Any:
        raise AssertionError("Restricted startup must not construct a TypeSafe transport")

    monkeypatch.setattr("atagia.services.providers.TypeSafeProvider", unexpected_provider)
    prepared = await prepare_inference_access(settings)
    client = build_llm_client(settings, prepared_inference_access=prepared)
    try:
        assert "typesafe" not in client._providers
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_zero_cost_attests_an_external_free_judge_before_provider_setup(
    tmp_path: Path,
) -> None:
    settings = _local_settings(
        tmp_path,
        mode="zero_cost",
        extra={
            "ATAGIA_OPENROUTER_API_KEY": "free-key",
            "ATAGIA_ZERO_COST_OPENROUTER_PROFILE": "dedicated_free_tier_no_byok",
        },
    )
    clients: list[Any] = []

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"data": {"is_free_tier": True}}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.urls: list[str] = []
            self.closed = False
            clients.append(self)

        async def get(self, url: str, **_kwargs: Any) -> Response:
            self.urls.append(url)
            return Response()

        async def aclose(self) -> None:
            self.closed = True

    prepared = await prepare_inference_access(
        settings,
        http_client_factory=Client,
        additional_completion_models={
            "benchmark.judge": "openrouter/example/judge:free"
        },
    )

    assert prepared.policy.openrouter_zero_cost_attestation is not None
    assert clients[0].kwargs == {"trust_env": False, "follow_redirects": False}
    assert clients[0].urls == ["https://openrouter.ai/api/v1/key"]
    assert clients[0].closed is True


@pytest.mark.asyncio
async def test_zero_cost_all_local_needs_no_openrouter_profile_or_attestation(
    tmp_path: Path,
) -> None:
    def unexpected_client(**_kwargs: Any) -> Any:
        raise AssertionError("control-plane HTTP must not be opened for all-local use")

    prepared = await prepare_inference_access(
        _local_settings(tmp_path, mode="zero_cost"),
        http_client_factory=unexpected_client,
        additional_completion_models={"benchmark.judge": "local/gpu_b/judge-b"},
    )

    assert prepared.policy.openrouter_zero_cost_attestation is None


@pytest.mark.asyncio
async def test_zero_cost_rejects_other_forbidden_routes_before_attestation(
    tmp_path: Path,
) -> None:
    def unexpected_client(**_kwargs: Any) -> Any:
        raise AssertionError("attestation must wait until all other routes pass")

    with pytest.raises(InferenceRouteError, match="benchmark.paid"):
        await prepare_inference_access(
            _local_settings(tmp_path, mode="zero_cost"),
            http_client_factory=unexpected_client,
            additional_completion_models={
                "benchmark.free": "openrouter/example/judge:free",
                "benchmark.paid": "anthropic/paid-judge",
            },
        )


@pytest.mark.asyncio
async def test_zero_cost_rejects_external_free_embedding_before_attestation(
    tmp_path: Path,
) -> None:
    def unexpected_client(**_kwargs: Any) -> Any:
        raise AssertionError("external embeddings are not attestable")

    settings = _local_settings(
        tmp_path,
        mode="zero_cost",
        extra={
            "ATAGIA_EMBEDDING_BACKEND": "sqlite_vec",
            "ATAGIA_EMBEDDING_MODEL": "openrouter/example/embed:free",
        },
    )

    with pytest.raises(InferenceRouteError, match="embedding"):
        await prepare_inference_access(
            settings,
            http_client_factory=unexpected_client,
        )


@pytest.mark.asyncio
async def test_local_only_startup_requires_a_catalog(
    tmp_path: Path,
) -> None:
    without_catalog = Settings.from_env(
        {
            "ATAGIA_INFERENCE_ACCESS_MODE": "local_only",
            "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "local/gpu_a/chat-a",
        }
    )
    with pytest.raises(InferenceRouteError, match="requires.*ENDPOINTS_FILE"):
        await prepare_inference_access(without_catalog)


@pytest.mark.asyncio
async def test_unrestricted_local_route_still_requires_a_catalog() -> None:
    settings = Settings.from_env(
        {"ATAGIA_LLM_FORCED_GLOBAL_MODEL": "local/gpu_a/chat-a"}
    )

    with pytest.raises(InferenceRouteError, match="LOCAL_LLM_ENDPOINTS_FILE"):
        await prepare_inference_access(settings)
    with pytest.raises(ConfigurationError, match="prepared local endpoint catalog"):
        build_llm_client(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_only", "zero_cost"])
async def test_restricted_startup_rejects_paid_benchmark_judge(
    tmp_path: Path,
    mode: str,
) -> None:
    with pytest.raises(InferenceRouteError, match="benchmark.judge"):
        await prepare_inference_access(
            _local_settings(tmp_path, mode=mode),
            additional_completion_models={"benchmark.judge": "anthropic/paid-judge"},
        )


@pytest.mark.asyncio
async def test_programmatic_atagia_activation_reports_effective_provenance(
    tmp_path: Path,
) -> None:
    catalog_path = _write_catalog(tmp_path / "programmatic-catalog.json")
    engine = Atagia(
        db_path=tmp_path / "programmatic.db",
        inference_access_mode="local_only",
        local_llm_endpoints_file=catalog_path,
        llm_forced_global_model="local/gpu_a/chat-a",
        _inference_startup_completion_models={"benchmark.judge": "local/gpu_b/judge-b"},
    )

    await engine.setup()
    try:
        report = engine.effective_settings_report()["settings"]
        assert report["inference_access_mode"] == {
            "value": "local_only",
            "provenance": "engine_override",
            "redacted": False,
        }
        assert report["local_llm_endpoints_file"]["value"] == str(catalog_path)
        assert report["local_llm_endpoints_file"]["provenance"] == "engine_override"
    finally:
        await engine.close()


def test_restricted_factory_and_direct_tools_fail_closed_without_preparation(
    tmp_path: Path,
) -> None:
    settings = _local_settings(tmp_path, mode="local_only")

    with pytest.raises(ConfigurationError, match="startup preparation"):
        build_llm_client(settings)
    with pytest.raises(InferenceRouteError, match="outside Atagia's governed client"):
        require_unrestricted_inference(settings, entry_point="direct benchmark")


@pytest.mark.asyncio
async def test_direct_provider_probe_refuses_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import gate_trap_probe

    constructed = 0

    def provider_spy(**_kwargs: Any) -> Any:
        nonlocal constructed
        constructed += 1
        raise AssertionError("provider construction must not be reached")

    monkeypatch.setenv("ATAGIA_INFERENCE_ACCESS_MODE", "local_only")
    monkeypatch.setattr(gate_trap_probe, "OpenRouterProvider", provider_spy)

    with pytest.raises(InferenceRouteError, match="gate_trap_probe"):
        await gate_trap_probe.main()
    assert constructed == 0


@pytest.mark.asyncio
async def test_diagnostics_list_routes_without_credentials(tmp_path: Path) -> None:
    prepared = await prepare_inference_access(
        _local_settings(tmp_path, mode="local_only"),
        additional_completion_models={"benchmark.judge": "local/gpu_b/judge-b"},
    )

    diagnostics = format_inference_access_diagnostics(prepared)
    assert "mode                 : local_only" in diagnostics
    assert "gpu_a, gpu_b" in diagnostics
    assert "local/gpu_b/judge-b" in diagnostics
    assert "api_key" not in diagnostics
