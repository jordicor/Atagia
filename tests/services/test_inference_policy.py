"""Focused tests for internal inference-route primitives and local catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator

import httpx
import pytest
from openai import AsyncOpenAI

import atagia.services.llm_client as llm_client_module
from atagia.core.config import Settings, default_resource_path
from atagia.services.inference_policy import (
    InferenceAccessPolicy,
    audit_inference_startup_routes,
    audit_privacy_filter_startup_urls,
    local_provider_registry_key,
    resolve_internal_inference_route,
)
from atagia.services.inference_routes import (
    BaseUrlClass,
    InferenceAccessMode,
    InferenceCostClass,
    InferenceOperation,
    InferenceRouteError,
    parse_local_model_spec,
    resolve_local_inference_route,
)
from atagia.services.llm_client import (
    ConfigurationError,
    InferenceAccessDeniedError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMMessage,
    LLMPolicyBlockedError,
    LLMToolSpec,
    LLMProvider,
    LLMStreamEvent,
    RetryPolicy,
    TransientLLMError,
)
from atagia.services.local_endpoint_catalog import (
    LOCAL_ENDPOINT_API_KEY_PLACEHOLDER,
    load_local_endpoint_catalog,
    local_endpoint_api_key,
    parse_local_endpoint_catalog,
)
from atagia.services.model_resolution import parse_model_spec
from atagia.services.providers.openai import LocalEndpointProvider
from atagia.services.providers import build_llm_client
from atagia.services.openrouter_zero_cost import (
    OpenRouterZeroCostAttestation,
    OpenRouterZeroCostControlPlaneError,
    attest_openrouter_zero_cost,
)


class EndpointSpyProvider(LLMProvider):
    """Endpoint-scoped in-memory provider used to prove routing boundaries."""

    def __init__(
        self, endpoint_id: str, *, fail_first_completion: bool = False
    ) -> None:
        self.name = local_provider_registry_key(endpoint_id)
        self.endpoint_id = endpoint_id
        self.fail_first_completion = fail_first_completion
        self.completion_requests: list[LLMCompletionRequest] = []
        self.embedding_requests: list[LLMEmbeddingRequest] = []
        self.stream_requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.completion_requests.append(request)
        if self.fail_first_completion and len(self.completion_requests) == 1:
            raise TransientLLMError("temporary local endpoint failure")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=f"complete:{self.endpoint_id}",
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        self.embedding_requests.append(request)
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )

    async def stream(
        self, request: LLMCompletionRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.stream_requests.append(request)
        yield LLMStreamEvent(type="text", content=f"stream:{self.endpoint_id}")
        yield LLMStreamEvent(type="done", payload={"usage": {}})


class ExternalSpyProvider(LLMProvider):
    """Provider double that makes any forbidden call observable."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.completion_requests: list[LLMCompletionRequest] = []
        self.embedding_requests: list[LLMEmbeddingRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        self.completion_requests.append(request)
        return LLMCompletionResponse(
            provider=self.name, model=request.model, output_text="{}"
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        self.calls += 1
        self.embedding_requests.append(request)
        return LLMEmbeddingResponse(provider=self.name, model=request.model, vectors=[])

    async def stream(
        self, request: LLMCompletionRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls += 1
        yield LLMStreamEvent(type="done", payload={})


def _catalog_document() -> dict[str, object]:
    return {
        "version": 1,
        "endpoints": [
            {
                "id": "gpu_4090",
                "adapter": "openai_compatible",
                "base_url": "http://192.168.50.20:11434/v1",
                "api_key_env": "ATAGIA_LOCAL_GPU_4090_API_KEY",
                "chat_models": ["qwen3-coder:30b", "shared-model"],
                "embedding_models": ["nomic-embed-text"],
            },
            {
                "id": "gpu_3090",
                "adapter": "openai_compatible",
                "base_url": "http://192.168.50.21:11435/v1",
                "chat_models": ["Qwen/Qwen3-32B", "shared-model"],
                "embedding_models": [],
            },
        ],
    }


def test_catalog_resolves_multiple_endpoints_and_models_deterministically() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())

    completion = resolve_local_inference_route(
        catalog,
        "local/gpu_4090/qwen3-coder:30b,high",
        InferenceOperation.COMPLETION,
    )
    embedding = resolve_local_inference_route(
        catalog,
        "local/gpu_4090/nomic-embed-text",
        InferenceOperation.EMBEDDING,
    )

    assert completion.endpoint_id == "gpu_4090"
    assert completion.canonical_model_spec == "local/gpu_4090/qwen3-coder:30b,high"
    assert completion.operation is InferenceOperation.COMPLETION
    assert completion.provider_slug == "local"
    assert completion.base_url_class is BaseUrlClass.LOCAL
    assert completion.cost_class is InferenceCostClass.LOCAL
    assert embedding.endpoint_id == "gpu_4090"
    assert embedding.operation is InferenceOperation.EMBEDDING


def test_same_model_on_two_endpoint_ids_remains_endpoint_specific() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())

    route_4090 = resolve_local_inference_route(
        catalog,
        "local/gpu_4090/shared-model",
        InferenceOperation.COMPLETION,
    )
    route_3090 = resolve_local_inference_route(
        catalog,
        "local/gpu_3090/shared-model",
        InferenceOperation.COMPLETION,
    )

    assert route_4090.endpoint_id == "gpu_4090"
    assert route_3090.endpoint_id == "gpu_3090"
    assert route_4090.canonical_model_spec != route_3090.canonical_model_spec


@pytest.mark.parametrize(
    ("model_spec", "operation", "message"),
    [
        (
            "local/unknown/qwen3-coder:30b",
            InferenceOperation.COMPLETION,
            "unknown endpoint",
        ),
        (
            "local/gpu_4090/unknown",
            InferenceOperation.COMPLETION,
            "does not serve model",
        ),
        (
            "local/gpu_4090/nomic-embed-text",
            InferenceOperation.COMPLETION,
            "does not serve model",
        ),
        (
            "local/gpu_4090/qwen3-coder:30b",
            InferenceOperation.EMBEDDING,
            "does not serve model",
        ),
    ],
)
def test_route_resolution_rejects_unknown_and_wrong_operation_capabilities(
    model_spec: str,
    operation: InferenceOperation,
    message: str,
) -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())

    with pytest.raises(InferenceRouteError, match=message):
        resolve_local_inference_route(catalog, model_spec, operation)


def test_local_model_parser_preserves_served_model_identity_and_thinking() -> None:
    parsed = parse_local_model_spec("local/gpu_3090/Qwen/Qwen3-32B,high")

    assert parsed.endpoint_id == "gpu_3090"
    assert parsed.served_model_id == "Qwen/Qwen3-32B"
    assert parsed.thinking_level == "high"
    assert parsed.canonical_spec == "local/gpu_3090/Qwen/Qwen3-32B,high"


@pytest.mark.parametrize("endpoint_id", ["GPU_4090", "gpu_4090A"])
def test_local_endpoint_ids_must_be_canonical_lowercase(endpoint_id: str) -> None:
    document = _catalog_document()
    endpoints = document["endpoints"]
    assert isinstance(endpoints, list)
    assert isinstance(endpoints[0], dict)
    endpoints[0]["id"] = endpoint_id

    with pytest.raises(InferenceRouteError, match="lowercase ASCII"):
        parse_local_endpoint_catalog(document)
    with pytest.raises(InferenceRouteError, match="lowercase ASCII"):
        parse_local_model_spec(f"local/{endpoint_id}/qwen3-coder:30b")


def test_local_model_parser_is_public_after_atomic_activation() -> None:
    parsed = parse_model_spec("local/gpu_4090/qwen3-coder:30b")

    assert parsed.provider_slug == "local"
    assert parsed.canonical_spec == "local/gpu_4090/qwen3-coder:30b"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
        "https://10.1.2.3:443/v1",
        "http://192.168.1.99:11434/v1",
        "http://[fd00::10]:11434/v1",
    ],
)
def test_catalog_accepts_only_intended_local_address_families(base_url: str) -> None:
    document = _catalog_document()
    endpoint = document["endpoints"][0]
    assert isinstance(endpoint, dict)
    endpoint["base_url"] = base_url

    catalog = parse_local_endpoint_catalog(document)

    assert catalog.endpoints[0].base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "https://8.8.8.8:443/v1",
        "http://localhost:11434/v1",
        "http://169.254.10.1:11434/v1",
        "http://[fe80::10]:11434/v1",
        "http://local-model.example:11434/v1",
        "http://127.0.0.1:11434/v1?redirect=cloud",
        "not-a-url",
    ],
)
def test_catalog_rejects_public_and_nonliteral_local_urls(base_url: str) -> None:
    document = _catalog_document()
    endpoint = document["endpoints"][0]
    assert isinstance(endpoint, dict)
    endpoint["base_url"] = base_url

    with pytest.raises(InferenceRouteError):
        parse_local_endpoint_catalog(document)


def test_catalog_rejects_duplicate_endpoint_ids_and_models() -> None:
    duplicate_endpoints = _catalog_document()
    endpoints = duplicate_endpoints["endpoints"]
    assert isinstance(endpoints, list)
    assert isinstance(endpoints[1], dict)
    endpoints[1]["id"] = "gpu_4090"

    with pytest.raises(InferenceRouteError, match="duplicate endpoint IDs"):
        parse_local_endpoint_catalog(duplicate_endpoints)

    duplicate_models = _catalog_document()
    model_endpoint = duplicate_models["endpoints"]
    assert isinstance(model_endpoint, list)
    assert isinstance(model_endpoint[0], dict)
    model_endpoint[0]["chat_models"] = ["qwen3-coder:30b", "qwen3-coder:30b"]

    with pytest.raises(InferenceRouteError, match="duplicate model IDs"):
        parse_local_endpoint_catalog(duplicate_models)


@pytest.mark.parametrize("field_name", ["chat_models", "embedding_models"])
def test_catalog_rejects_model_ids_ambiguous_with_thinking_suffixes(
    field_name: str,
) -> None:
    ambiguous_models = _catalog_document()
    endpoints = ambiguous_models["endpoints"]
    assert isinstance(endpoints, list)
    assert isinstance(endpoints[0], dict)
    endpoints[0][field_name] = ["m,high"]

    with pytest.raises(InferenceRouteError, match="must not contain a comma"):
        parse_local_endpoint_catalog(ambiguous_models)


def test_catalog_loader_reads_json_file(tmp_path: Path) -> None:
    catalog_file = tmp_path / "local_llm_endpoints.json"
    catalog_file.write_text(json.dumps(_catalog_document()), encoding="utf-8")

    catalog = load_local_endpoint_catalog(catalog_file)

    assert [endpoint.endpoint_id for endpoint in catalog.endpoints] == [
        "gpu_4090",
        "gpu_3090",
    ]


def test_catalog_loader_requires_an_absolute_path() -> None:
    with pytest.raises(InferenceRouteError, match="must be absolute"):
        load_local_endpoint_catalog("local_llm_endpoints.json")


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        ("not-a-url", "must use http or https"),
        ("http://user:password@127.0.0.1:11434/v1", "must not include user info"),
        ("http://127.0.0.1:11434/v1?unexpected=value", "query or fragment"),
        ("http://127.0.0.1:11434/v1#unexpected", "query or fragment"),
        ("http://127.0.0.1:11434/v\n1", "control characters"),
    ],
)
def test_catalog_rejects_parser_sensitive_urls(base_url: str, message: str) -> None:
    document = _catalog_document()
    endpoint = document["endpoints"][0]
    assert isinstance(endpoint, dict)
    endpoint["base_url"] = base_url

    with pytest.raises(InferenceRouteError, match=message):
        parse_local_endpoint_catalog(document)


def test_optional_and_named_endpoint_credentials() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())

    assert local_endpoint_api_key(catalog.endpoints[1], environ={}) == (
        LOCAL_ENDPOINT_API_KEY_PLACEHOLDER
    )
    assert (
        local_endpoint_api_key(
            catalog.endpoints[0],
            environ={"ATAGIA_LOCAL_GPU_4090_API_KEY": "local-secret"},
        )
        == "local-secret"
    )

    with pytest.raises(InferenceRouteError, match="ATAGIA_LOCAL_GPU_4090_API_KEY"):
        local_endpoint_api_key(catalog.endpoints[0], environ={})


def test_inference_access_modes_are_internal_fixed_values() -> None:
    assert {mode.value for mode in InferenceAccessMode} == {
        "unrestricted",
        "local_only",
        "zero_cost",
    }


def _local_client(
    catalog: object,
    providers: list[LLMProvider],
    *,
    retry_policy: RetryPolicy | None = None,
    **kwargs: object,
) -> LLMClient[object]:
    return LLMClient(
        providers=providers,
        retry_policy=retry_policy,
        inference_access_policy=InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
        local_endpoint_catalog=catalog,
        **kwargs,
    )


def _completion_request(
    model: str,
    *,
    metadata: dict[str, object] | None = None,
) -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="user", content="test")],
        metadata=metadata or {},
    )


class FreeTierControlPlaneClient:
    """No-network control-plane double with observable transport settings."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.aclosed = False

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        return self.response

    async def aclose(self) -> None:
        self.aclosed = True


async def _free_tier_attestation(
    api_key: str = "router-test-key",
) -> tuple[
    OpenRouterZeroCostAttestation, FreeTierControlPlaneClient, dict[str, object]
]:
    client = FreeTierControlPlaneClient(
        httpx.Response(200, json={"data": {"is_free_tier": True}})
    )
    transport_kwargs: dict[str, object] = {}

    def factory(**kwargs: object) -> FreeTierControlPlaneClient:
        transport_kwargs.update(kwargs)
        return client

    attestation = await attest_openrouter_zero_cost(
        api_key=api_key,
        profile="dedicated_free_tier_no_byok",
        base_url=None,
        http_client_factory=factory,
    )
    return attestation, client, transport_kwargs


def _zero_cost_policy(
    attestation: OpenRouterZeroCostAttestation,
) -> InferenceAccessPolicy:
    return InferenceAccessPolicy(
        InferenceAccessMode.ZERO_COST,
        openrouter_zero_cost_attestation=attestation,
    )


@pytest.mark.asyncio
async def test_zero_cost_control_plane_attestation_uses_restricted_transport() -> None:
    attestation, client, transport_kwargs = await _free_tier_attestation()

    assert attestation.matches(api_key="router-test-key", base_url=None)
    assert transport_kwargs == {"trust_env": False, "follow_redirects": False}
    assert client.calls == [
        (
            "https://openrouter.ai/api/v1/key",
            {"headers": {"Authorization": "Bearer router-test-key"}},
        )
    ]
    assert client.aclosed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"data": {"is_free_tier": True}}),
        httpx.Response(204, json={"data": {"is_free_tier": True}}),
        httpx.Response(200, json={"data": {"is_free_tier": False}}),
        httpx.Response(200, json={"data": {"is_free_tier": 1}}),
        httpx.Response(200, json={"data": {}}),
    ],
)
async def test_zero_cost_control_plane_fails_closed_for_nonfree_or_ambiguous_key_state(
    response: httpx.Response,
) -> None:
    client = FreeTierControlPlaneClient(response)

    with pytest.raises(OpenRouterZeroCostControlPlaneError):
        await attest_openrouter_zero_cost(
            api_key="router-test-key",
            profile="dedicated_free_tier_no_byok",
            base_url=None,
            http_client_factory=lambda **_: client,
        )

    assert client.calls
    assert client.aclosed is True


@pytest.mark.asyncio
async def test_zero_cost_control_plane_rejects_custom_openrouter_origin_without_io() -> (
    None
):
    invoked = False

    def factory(**_: object) -> FreeTierControlPlaneClient:
        nonlocal invoked
        invoked = True
        return FreeTierControlPlaneClient(httpx.Response(200))

    with pytest.raises(OpenRouterZeroCostControlPlaneError, match="custom base URLs"):
        await attest_openrouter_zero_cost(
            api_key="router-test-key",
            profile="dedicated_free_tier_no_byok",
            base_url="https://gateway.example/api/v1",
            http_client_factory=factory,
        )

    assert invoked is False


@pytest.mark.asyncio
async def test_zero_cost_allows_attested_free_openrouter_routes_and_local_embeddings() -> (
    None
):
    attestation, _, _ = await _free_tier_attestation()
    catalog = parse_local_endpoint_catalog(_catalog_document())
    openrouter = ExternalSpyProvider("openrouter")
    local = EndpointSpyProvider("gpu_4090")
    client = LLMClient(
        providers=[openrouter, local],
        inference_access_policy=_zero_cost_policy(attestation),
        local_endpoint_catalog=catalog,
    )

    variant = await client.complete(
        _completion_request("openrouter/example/model:free")
    )
    router = await client.complete(_completion_request("openrouter/openrouter/free"))
    embedding = await client.embed(
        LLMEmbeddingRequest(
            model="local/gpu_4090/nomic-embed-text",
            input_texts=["test"],
        )
    )

    assert variant.provider == "openrouter"
    assert router.provider == "openrouter"
    assert [request.model for request in openrouter.completion_requests] == [
        "example/model:free",
        "openrouter/free",
    ]
    assert embedding.vectors[0].values == [0.1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    [
        "openrouter//example/model:free",
        "openrouter/example//model:free",
        "openrouter/example/model:free/",
        "openrouter/:/model:free",
        " openrouter/example/model:free",
        "openrouter/example /model:free",
    ],
)
async def test_zero_cost_rejects_noncanonical_free_route_spellings_before_io(
    model_spec: str,
) -> None:
    attestation, _, _ = await _free_tier_attestation()
    openrouter = ExternalSpyProvider("openrouter")
    client = LLMClient(
        providers=[openrouter],
        inference_access_policy=_zero_cost_policy(attestation),
    )

    with pytest.raises(InferenceAccessDeniedError, match="malformed|unknown-cost"):
        await client.complete(_completion_request(model_spec))

    assert openrouter.calls == 0


def test_only_an_exact_free_openrouter_spec_receives_candidate_cost_class() -> None:
    exact = resolve_internal_inference_route(
        "openrouter/example/model:free",
        InferenceOperation.COMPLETION,
        local_catalog=None,
    )
    normalized = resolve_internal_inference_route(
        "openrouter//example/model:free",
        InferenceOperation.COMPLETION,
        local_catalog=None,
    )

    assert exact.cost_class is InferenceCostClass.ZERO_COST_OPENROUTER_CANDIDATE
    assert normalized.cost_class is InferenceCostClass.METERED


@pytest.mark.asyncio
async def test_zero_cost_denies_unattested_paid_malformed_and_external_embedding_routes() -> (
    None
):
    catalog = parse_local_endpoint_catalog(_catalog_document())
    openrouter = ExternalSpyProvider("openrouter")
    openai = ExternalSpyProvider("openai")
    local = EndpointSpyProvider("gpu_4090")
    client = LLMClient(
        providers=[openrouter, openai, local],
        inference_access_policy=InferenceAccessPolicy(InferenceAccessMode.ZERO_COST),
        local_endpoint_catalog=catalog,
    )

    with pytest.raises(InferenceAccessDeniedError, match="attestation"):
        await client.complete(_completion_request("openrouter/example/model:free"))
    with pytest.raises(InferenceAccessDeniedError, match="ordinary or paid"):
        await client.complete(_completion_request("openrouter/example/model"))
    with pytest.raises(InferenceAccessDeniedError, match="malformed"):
        await client.complete(_completion_request("openrouter/free"))
    with pytest.raises(InferenceAccessDeniedError, match="external provider"):
        await client.complete(_completion_request("openai/gpt-5-mini"))
    with pytest.raises(InferenceAccessDeniedError, match="embeddings only"):
        await client.embed(
            LLMEmbeddingRequest(
                model="openrouter/example/model:free",
                input_texts=["test"],
            )
        )

    local_completion = await client.complete(
        _completion_request("local/gpu_4090/qwen3-coder:30b")
    )
    local_embedding = await client.embed(
        LLMEmbeddingRequest(
            model="local/gpu_4090/nomic-embed-text",
            input_texts=["test"],
        )
    )

    assert openrouter.calls == 0
    assert openai.calls == 0
    assert local_completion.output_text == "complete:gpu_4090"
    assert local_embedding.vectors[0].values == [0.1]


@pytest.mark.asyncio
async def test_zero_cost_rejects_caller_openrouter_routing_body_and_tools_before_io() -> (
    None
):
    attestation, _, _ = await _free_tier_attestation()
    openrouter = ExternalSpyProvider("openrouter")
    client = LLMClient(
        providers=[openrouter],
        inference_access_policy=_zero_cost_policy(attestation),
    )

    with pytest.raises(InferenceAccessDeniedError, match="provider_extra_body"):
        await client.complete(
            _completion_request(
                "openrouter/example/model:free",
                metadata={"provider_extra_body": {}},
            )
        )
    with pytest.raises(InferenceAccessDeniedError, match="tools"):
        await client.complete(
            LLMCompletionRequest(
                model="openrouter/example/model:free",
                messages=[LLMMessage(role="user", content="test")],
                tools=[LLMToolSpec(name="lookup")],
            )
        )

    assert openrouter.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_extra_body", [{}, None, False])
async def test_zero_cost_local_routes_reject_falsey_provider_body_presence_before_io(
    provider_extra_body: object,
) -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    local = EndpointSpyProvider("gpu_4090")
    client = LLMClient(
        providers=[local],
        inference_access_policy=InferenceAccessPolicy(InferenceAccessMode.ZERO_COST),
        local_endpoint_catalog=catalog,
    )
    metadata = {"provider_extra_body": provider_extra_body}

    with pytest.raises(InferenceAccessDeniedError, match="provider_extra_body"):
        await client.complete(
            _completion_request(
                "local/gpu_4090/qwen3-coder:30b",
                metadata=metadata,
            )
        )
    with pytest.raises(InferenceAccessDeniedError, match="provider_extra_body"):
        await client.embed(
            LLMEmbeddingRequest(
                model="local/gpu_4090/nomic-embed-text",
                input_texts=["test"],
                metadata=metadata,
            )
        )

    assert not local.completion_requests
    assert not local.embedding_requests


@pytest.mark.asyncio
async def test_zero_cost_paid_intimacy_fallback_is_denied_before_a_second_provider_call() -> (
    None
):
    class FreeRoutePolicyRefusal(ExternalSpyProvider):
        async def complete(
            self, request: LLMCompletionRequest
        ) -> LLMCompletionResponse:
            self.calls += 1
            self.completion_requests.append(request)
            raise LLMPolicyBlockedError("provider policy refusal")

    attestation, _, _ = await _free_tier_attestation()
    openrouter = FreeRoutePolicyRefusal("openrouter")
    client = LLMClient(
        providers=[openrouter],
        inference_access_policy=_zero_cost_policy(attestation),
        intimacy_fallback_models={"chat": "openrouter/example/ordinary"},
    )

    with pytest.raises(InferenceAccessDeniedError, match="ordinary or paid"):
        await client.complete(
            _completion_request(
                "openrouter/example/model:free",
                metadata={"purpose": "chat_reply"},
            )
        )

    assert openrouter.calls == 1


@pytest.mark.asyncio
async def test_local_only_routes_completion_stream_and_embeddings_by_endpoint() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    gpu_4090 = EndpointSpyProvider("gpu_4090")
    gpu_3090 = EndpointSpyProvider("gpu_3090")
    client = _local_client(catalog, [gpu_4090, gpu_3090])

    completion = await client.complete(
        _completion_request("local/gpu_4090/qwen3-coder:30b")
    )
    stream_events = [
        event
        async for event in client.stream(
            _completion_request("local/gpu_3090/Qwen/Qwen3-32B")
        )
    ]
    embeddings = await client.embed(
        LLMEmbeddingRequest(
            model="local/gpu_4090/nomic-embed-text",
            input_texts=["test"],
        )
    )

    assert completion.output_text == "complete:gpu_4090"
    assert [event.content for event in stream_events if event.type == "text"] == [
        "stream:gpu_3090"
    ]
    assert embeddings.vectors[0].values == [0.1]
    assert [request.model for request in gpu_4090.completion_requests] == [
        "qwen3-coder:30b"
    ]
    assert [request.model for request in gpu_3090.stream_requests] == ["Qwen/Qwen3-32B"]
    assert [request.model for request in gpu_4090.embedding_requests] == [
        "nomic-embed-text"
    ]
    assert not gpu_3090.completion_requests
    assert not gpu_3090.embedding_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_spec", "request_metadata", "expected_reasoning_effort"),
    [
        (
            "local/gpu_4090/qwen3-coder:30b",
            {"reasoning_effort": "high"},
            None,
        ),
        ("local/gpu_4090/qwen3-coder:30b,low", {}, "low"),
        ("local/gpu_4090/qwen3-coder:30b,none", {}, "none"),
        ("local/gpu_4090/qwen3-coder:30b,xhigh", {}, "xhigh"),
    ],
)
async def test_local_endpoint_wire_model_is_catalog_owned_for_completion_and_stream(
    model_spec: str,
    request_metadata: dict[str, object],
    expected_reasoning_effort: str | None,
) -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    endpoint = catalog.endpoints[0]
    serialized_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        serialized_bodies.append(body)
        if body["stream"]:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk",'
                    '"created":0,"model":"qwen3-coder:30b","choices":['
                    '{"index":0,"delta":{"content":"streamed"},"finish_reason":null}]}\n\n'
                    'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk",'
                    '"created":0,"model":"qwen3-coder:30b","choices":['
                    '{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-complete",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3-coder:30b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "complete"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = LocalEndpointProvider(endpoint, api_key="test")
    await provider.aclose()
    wire_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider._owned_http_clients = [wire_client]
    sdk_client = AsyncOpenAI(
        api_key="test",
        base_url=endpoint.base_url,
        http_client=wire_client,
        max_retries=0,
    )
    provider._client = sdk_client
    provider._embedding_client = sdk_client
    client = _local_client(catalog, [provider])
    request = _completion_request(model_spec, metadata=request_metadata)

    try:
        completion = await client.complete(request)
        events = [event async for event in client.stream(request)]

        assert completion.output_text == "complete"
        assert [event.content for event in events if event.type == "text"] == [
            "streamed"
        ]
        assert [body["model"] for body in serialized_bodies] == [
            "qwen3-coder:30b",
            "qwen3-coder:30b",
        ]
        assert [body.get("reasoning_effort") for body in serialized_bodies] == [
            expected_reasoning_effort,
            expected_reasoning_effort,
        ]

        forged_request = _completion_request(
            "local/gpu_4090/qwen3-coder:30b",
            metadata={"provider_extra_body": {"model": "forged-model"}},
        )
        with pytest.raises(InferenceAccessDeniedError, match="provider_extra_body"):
            await client.complete(forged_request)
        with pytest.raises(InferenceAccessDeniedError, match="provider_extra_body"):
            await anext(client.stream(forged_request))

        assert len(serialized_bodies) == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_local_only_denies_external_completion_stream_and_embedding_before_spies() -> (
    None
):
    catalog = parse_local_endpoint_catalog(_catalog_document())
    external = ExternalSpyProvider("openai")
    client = _local_client(catalog, [external])
    forged_metadata = {
        "atagia_local_endpoint_id": "gpu_4090",
        "atagia_inference_route": "local/gpu_4090/qwen3-coder:30b",
    }

    with pytest.raises(InferenceAccessDeniedError):
        await client.complete(
            _completion_request("openai/gpt-5-mini", metadata=forged_metadata)
        )
    with pytest.raises(InferenceAccessDeniedError):
        await anext(
            client.stream(
                _completion_request("openai/gpt-5-mini", metadata=forged_metadata)
            )
        )
    with pytest.raises(InferenceAccessDeniedError):
        await client.embed(
            LLMEmbeddingRequest(
                model="openai/text-embedding-3-small",
                input_texts=["test"],
                metadata=forged_metadata,
            )
        )

    assert external.calls == 0


@pytest.mark.asyncio
async def test_local_only_denial_precedes_guard_meter_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardSpy:
        def __init__(self) -> None:
            self.calls = 0

        def begin_call(self, **_: object) -> object:
            self.calls += 1
            raise AssertionError("the route authorization must run before the guard")

    catalog = parse_local_endpoint_catalog(_catalog_document())
    external = ExternalSpyProvider("openai")
    guard = GuardSpy()
    meter_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        llm_client_module,
        "record_call_on_active_meter",
        lambda **kwargs: meter_calls.append(kwargs),
    )
    client = _local_client(catalog, [external], llm_run_guard=guard)

    with pytest.raises(InferenceAccessDeniedError):
        await client.complete(_completion_request("openai/gpt-5-mini"))

    assert guard.calls == 0
    assert meter_calls == []
    assert external.calls == 0


@pytest.mark.asyncio
async def test_local_only_rejects_legacy_direct_provider_shortcut() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    external = ExternalSpyProvider("openai")
    client = LLMClient(
        provider_name="openai",
        providers=[external],
        inference_access_policy=InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
        local_endpoint_catalog=catalog,
    )

    with pytest.raises(InferenceAccessDeniedError):
        await client.complete(_completion_request("any-model"))

    assert external.calls == 0


@pytest.mark.asyncio
async def test_unrestricted_single_provider_preserves_unqualified_completion_and_embedding() -> (
    None
):
    provider = ExternalSpyProvider("openai")
    client = LLMClient(
        providers=[provider],
        allow_unqualified_single_provider_models=True,
    )

    completion = await client.complete(_completion_request("bare-chat-model"))
    embedding = await client.embed(
        LLMEmbeddingRequest(model="bare-embedding-model", input_texts=["test"])
    )

    assert completion.model == "bare-chat-model"
    assert embedding.model == "bare-embedding-model"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_restricted_single_provider_rejects_unqualified_models() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    provider = ExternalSpyProvider("openai")
    client = _local_client(
        catalog,
        [provider],
        allow_unqualified_single_provider_models=True,
    )

    with pytest.raises(ConfigurationError, match="provider/model"):
        await client.complete(_completion_request("bare-chat-model"))
    with pytest.raises(ConfigurationError, match="provider/model"):
        await client.embed(
            LLMEmbeddingRequest(model="bare-embedding-model", input_texts=["test"])
        )

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_local_only_denial_cannot_trigger_intimacy_or_structured_recovery() -> (
    None
):
    catalog = parse_local_endpoint_catalog(_catalog_document())
    primary = ExternalSpyProvider("openai")
    fallback = ExternalSpyProvider("anthropic")
    client = _local_client(
        catalog,
        [primary, fallback],
        intimacy_fallback_models={"chat": "anthropic/claude-test"},
        structured_output_rescue_enabled=True,
        structured_output_rescue_model="anthropic/claude-test",
    )

    with pytest.raises(InferenceAccessDeniedError):
        await client.complete(
            _completion_request(
                "openai/gpt-5-mini",
                metadata={"purpose": "chat_reply"},
            )
        )
    with pytest.raises(InferenceAccessDeniedError):
        await client.complete_structured(
            _completion_request("openai/gpt-5-mini"),
            dict[str, str],
        )

    assert primary.calls == 0
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_local_only_retry_reuses_the_same_admitted_endpoint() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    gpu_4090 = EndpointSpyProvider("gpu_4090", fail_first_completion=True)
    gpu_3090 = EndpointSpyProvider("gpu_3090")
    client = _local_client(
        catalog,
        [gpu_4090, gpu_3090],
        retry_policy=RetryPolicy(
            attempts=2,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_fraction=0.0,
        ),
    )

    completion = await client.complete(
        _completion_request("local/gpu_4090/shared-model")
    )

    assert completion.output_text == "complete:gpu_4090"
    assert [request.model for request in gpu_4090.completion_requests] == [
        "shared-model",
        "shared-model",
    ]
    assert not gpu_3090.completion_requests


def _startup_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "llm_forced_global_model": "local/gpu_4090/qwen3-coder:30b",
        "llm_ingest_model": None,
        "llm_retrieval_model": None,
        "llm_chat_model": None,
        "llm_component_models": {},
        "llm_intimacy_ingest_model": None,
        "llm_intimacy_retrieval_model": None,
        "llm_intimacy_component_models": {},
        "llm_structured_output_rescue_enabled": False,
        "llm_structured_output_rescue_model": None,
        "embedding_backend": "none",
        "embedding_model": None,
        "openai_proxy_upstream_model": None,
        "opf_privacy_filter_enabled": False,
        "opf_primary_url": "http://127.0.0.1:8008",
        "opf_fallback_url": "http://127.0.0.1:8008",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_startup_audit_covers_shadowed_overrides_and_rejects_external_route() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    settings = _startup_settings(
        llm_component_models={"chat": "openai/gpt-5-mini"},
    )

    with pytest.raises(InferenceRouteError, match="component.chat"):
        audit_inference_startup_routes(
            settings,
            InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
            local_catalog=catalog,
        )


def test_startup_audit_includes_intimacy_rescue_embedding_and_proxy_routes() -> None:
    catalog = parse_local_endpoint_catalog(_catalog_document())
    settings = _startup_settings(
        llm_intimacy_ingest_model="local/gpu_4090/qwen3-coder:30b",
        llm_structured_output_rescue_enabled=True,
        llm_structured_output_rescue_model="local/gpu_4090/qwen3-coder:30b",
        embedding_backend="sqlite_vec",
        embedding_model="local/gpu_4090/nomic-embed-text",
        openai_proxy_upstream_model="local/gpu_3090/Qwen/Qwen3-32B",
    )

    routes = audit_inference_startup_routes(
        settings,
        InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
        local_catalog=catalog,
    )

    assert {route.source for route in routes} >= {
        "forced_global",
        "intimacy_category.ingest",
        "structured_output_rescue",
        "embedding",
        "proxy_upstream",
        "proxy_chat_fallback",
    }


def test_enabled_external_opf_endpoint_is_denied_by_startup_audit() -> None:
    settings = _startup_settings(
        opf_privacy_filter_enabled=True,
        opf_primary_url="https://198.51.100.20:8008",
    )

    with pytest.raises(InferenceRouteError, match="opf_primary_url"):
        audit_privacy_filter_startup_urls(
            settings,
            InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
        )


@pytest.mark.asyncio
async def test_local_endpoint_provider_owns_a_restricted_client_per_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    catalog = parse_local_endpoint_catalog(_catalog_document())
    providers = [
        LocalEndpointProvider(endpoint, api_key="test", request_timeout_seconds=1.0)
        for endpoint in catalog.endpoints
    ]

    try:
        transports = [provider._owned_http_clients[0] for provider in providers]
        assert len({id(transport) for transport in transports}) == 2
        assert all(transport.follow_redirects is False for transport in transports)
        assert all(transport._trust_env is False for transport in transports)
    finally:
        for provider in providers:
            await provider.aclose()


@pytest.mark.asyncio
async def test_factory_registers_one_restricted_provider_per_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_LOCAL_GPU_4090_API_KEY", "test")
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.invalid",
        openrouter_app_name="Atagia test",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )
    catalog = parse_local_endpoint_catalog(_catalog_document())
    client = build_llm_client(
        settings,
        inference_access_policy=InferenceAccessPolicy(InferenceAccessMode.LOCAL_ONLY),
        local_endpoint_catalog=catalog,
    )

    try:
        assert set(client._providers) == {"local:gpu_4090", "local:gpu_3090"}
        assert all(
            isinstance(provider, LocalEndpointProvider)
            for provider in client._providers.values()
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_zero_cost_factory_registers_only_attested_official_openrouter_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_LOCAL_GPU_4090_API_KEY", "test")
    attestation, _, _ = await _free_tier_attestation("router-key")
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key="router-key",
        openrouter_site_url="https://atagia.invalid",
        openrouter_app_name="Atagia test",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )
    catalog = parse_local_endpoint_catalog(_catalog_document())
    client = build_llm_client(
        settings,
        inference_access_policy=_zero_cost_policy(attestation),
        local_endpoint_catalog=catalog,
    )

    try:
        assert set(client._providers) == {
            "local:gpu_4090",
            "local:gpu_3090",
            "openrouter",
        }
        router = client._provider("openrouter")
        assert router._owned_http_clients[0].follow_redirects is False
        assert router._owned_http_clients[0]._trust_env is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "stream", "complete_streamed"])
async def test_zero_cost_factory_drops_attestation_for_a_different_inference_key(
    operation: str,
) -> None:
    attestation, _, _ = await _free_tier_attestation("other-key")
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key="router-key",
        openrouter_site_url="https://atagia.invalid",
        openrouter_app_name="Atagia test",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )
    client = build_llm_client(
        settings,
        inference_access_policy=_zero_cost_policy(attestation),
    )

    try:
        assert "openrouter" not in client._providers
        with pytest.raises(InferenceAccessDeniedError, match="attestation"):
            request = _completion_request("openrouter/example/model:free")
            if operation == "stream":
                await anext(client.stream(request))
            else:
                await getattr(client, operation)(request)
    finally:
        await client.aclose()
