"""Concrete LLM provider implementations and factory helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from atagia.core.config import Settings
from atagia.services.inference_policy import InferenceAccessPolicy
from atagia.services.inference_runtime import (
    PreparedInferenceAccess,
    configured_inference_access_mode,
)
from atagia.services.inference_routes import InferenceAccessMode
from atagia.services.local_endpoint_catalog import (
    LocalEndpointCatalog,
    local_endpoint_api_key,
)
from atagia.services.llm_client import ConfigurationError, LLMClient, RetryPolicy
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.llm_run_guard import LLMRunGuard, runtime_llm_run_guard_config
from atagia.services.model_resolution import (
    ModelResolutionError,
    PROVIDER_SLUG_TO_NAME,
    parse_embedding_model_spec,
    required_provider_slugs,
    resolve_intimacy_fallback_models,
    validate_required_provider_keys,
)
from atagia.services.openrouter_zero_cost import OpenRouterZeroCostAttestation
from atagia.services.providers.anthropic import AnthropicProvider
from atagia.services.providers.gemini import GeminiProvider
from atagia.services.providers.kimi import KimiProvider
from atagia.services.providers.minimax import MiniMaxProvider
from atagia.services.providers.openai import LocalEndpointProvider, OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider
from atagia.services.providers.typesafe import TypeSafeProvider


def build_llm_client(
    settings: Settings,
    *,
    retry_policy: RetryPolicy | None = None,
    inference_access_policy: InferenceAccessPolicy | None = None,
    local_endpoint_catalog: LocalEndpointCatalog | None = None,
    zero_cost_openrouter_authorization: OpenRouterZeroCostAttestation | None = None,
    prepared_inference_access: PreparedInferenceAccess | None = None,
) -> LLMClient[Any]:
    """Build an LLM client from settings and optional internal route injection."""
    configured_mode = configured_inference_access_mode(settings)
    if configured_mode is not InferenceAccessMode.UNRESTRICTED:
        if prepared_inference_access is None:
            raise ConfigurationError(
                "Restricted inference settings require async startup preparation "
                "before build_llm_client()."
            )
        if any(
            value is not None
            for value in (
                inference_access_policy,
                local_endpoint_catalog,
                zero_cost_openrouter_authorization,
            )
        ):
            raise ConfigurationError(
                "Do not combine prepared public inference access with internal "
                "policy injection."
            )
        if prepared_inference_access.policy.mode is not configured_mode:
            raise ConfigurationError(
                "Prepared inference access mode does not match Settings."
            )

    if prepared_inference_access is not None:
        policy = prepared_inference_access.policy
        local_endpoint_catalog = prepared_inference_access.local_catalog
    else:
        policy = inference_access_policy or InferenceAccessPolicy()
    if (
        zero_cost_openrouter_authorization is not None
        and policy.openrouter_zero_cost_attestation is not None
        and zero_cost_openrouter_authorization
        != policy.openrouter_zero_cost_attestation
    ):
        raise ConfigurationError(
            "Conflicting zero-cost OpenRouter authorizations were supplied."
        )
    authorization = (
        zero_cost_openrouter_authorization or policy.openrouter_zero_cost_attestation
    )
    if policy.mode is InferenceAccessMode.ZERO_COST:
        if authorization is not None and authorization.matches(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        ):
            policy = replace(policy, openrouter_zero_cost_attestation=authorization)
        else:
            # A stale, different-key, or different-origin attestation cannot
            # remain attached to the runtime policy just because local routes
            # are still allowed to start.
            policy = replace(policy, openrouter_zero_cost_attestation=None)
    restricted = policy.restricted
    if not restricted:
        try:
            if (
                "local" in required_provider_slugs(settings)
                and local_endpoint_catalog is None
            ):
                raise ConfigurationError(
                    "Local model specifications require a prepared local endpoint "
                    "catalog before build_llm_client()."
                )
            validate_required_provider_keys(settings)
        except ModelResolutionError as exc:
            raise ConfigurationError(str(exc)) from exc

    providers = [
        LocalEndpointProvider(
            endpoint,
            api_key=local_endpoint_api_key(endpoint),
            request_timeout_seconds=settings.llm_request_timeout_seconds,
        )
        for endpoint in (
            local_endpoint_catalog.endpoints if local_endpoint_catalog else ()
        )
    ]
    if not restricted and settings.typesafe_api_key:
        providers.append(
            TypeSafeProvider(
                api_key=settings.typesafe_api_key,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )
    if not restricted and settings.anthropic_api_key:
        providers.append(
            AnthropicProvider(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                request_timeout_seconds=settings.anthropic_request_timeout_seconds,
            )
        )
    if not restricted and settings.openai_api_key:
        providers.append(
            OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                embedding_base_url=settings.openai_embedding_base_url,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )
    if not restricted and settings.kimi_api_key:
        providers.append(
            KimiProvider(
                api_key=settings.kimi_api_key,
                base_url=settings.kimi_base_url,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )
    if not restricted and settings.minimax_api_key:
        providers.append(
            MiniMaxProvider(
                api_key=settings.minimax_api_key,
                base_url=settings.minimax_base_url,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )
    if not restricted and settings.openrouter_api_key:
        providers.append(
            OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                base_url=settings.openrouter_base_url,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )
    if (
        policy.mode is InferenceAccessMode.ZERO_COST
        and policy.openrouter_zero_cost_attestation is not None
        and settings.openrouter_api_key
    ):
        providers.append(
            OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                base_url=settings.openrouter_base_url,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
                zero_cost=True,
            )
        )
    if not restricted and settings.google_api_key:
        providers.append(
            GeminiProvider(
                api_key=settings.google_api_key,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
        )

    diagnostic_recorder = None
    if settings.diagnostic_capture_enabled:
        from atagia.diagnostics.recorder import DiagnosticRecorder

        diagnostic_recorder = DiagnosticRecorder(
            settings.diagnostic_capture_dir,
            max_blob_bytes=settings.diagnostic_capture_max_blob_bytes,
            max_session_bytes=settings.diagnostic_capture_max_session_bytes,
        )
    client = LLMClient(
        providers=providers,
        retry_policy=retry_policy,
        intimacy_fallback_models=resolve_intimacy_fallback_models(settings),
        intimacy_proactive_routing_enabled=(
            settings.llm_intimacy_proactive_routing_enabled
        ),
        structured_output_retry_attempts=settings.llm_structured_output_retry_attempts,
        structured_output_rescue_enabled=settings.llm_structured_output_rescue_enabled,
        structured_output_rescue_model=settings.llm_structured_output_rescue_model,
        technical_recovery_config=LLMTechnicalRecoveryConfig.from_settings(settings),
        llm_run_guard=LLMRunGuard(runtime_llm_run_guard_config(settings)),
        max_concurrent_requests_per_provider=(
            settings.llm_max_concurrent_requests_per_provider
        ),
        inference_access_policy=policy,
        local_endpoint_catalog=local_endpoint_catalog,
        diagnostic_recorder=diagnostic_recorder,
    )

    for provider_slug in (
        sorted(required_provider_slugs(settings).difference({"local"}))
        if not restricted
        else ()
    ):
        provider_name = PROVIDER_SLUG_TO_NAME[provider_slug]
        try:
            client._provider(provider_name)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"No configured provider adapter for resolved provider {provider_slug!r}"
            ) from exc

    if settings.embedding_backend != "none" and not restricted:
        try:
            embedding = parse_embedding_model_spec(settings.embedding_model)
            embedding_provider = (
                None
                if embedding.provider_slug == "local"
                else client._provider(embedding.provider_name)
            )
        except ModelResolutionError as exc:
            raise ConfigurationError(str(exc)) from exc
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"No configured credentials for embedding provider {embedding.provider_slug!r}"
            ) from exc
        if (
            embedding_provider is not None
            and not embedding_provider.supports_embeddings
        ):
            raise ConfigurationError(
                f"Provider {embedding.provider_slug!r} does not support embeddings"
            )

    return client


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "KimiProvider",
    "LocalEndpointProvider",
    "MiniMaxProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "TypeSafeProvider",
    "build_llm_client",
]
