"""Concrete LLM provider implementations and factory helpers."""

from __future__ import annotations

from typing import Any

from atagia.core.config import Settings
from atagia.services.llm_client import ConfigurationError, LLMClient, RetryPolicy
from atagia.services.model_resolution import (
    ModelResolutionError,
    PROVIDER_SLUG_TO_NAME,
    parse_embedding_model_spec,
    required_provider_slugs,
    resolve_intimacy_fallback_models,
    validate_required_provider_keys,
)
from atagia.services.providers.anthropic import AnthropicProvider
from atagia.services.providers.gemini import GeminiProvider
from atagia.services.providers.openai import OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider


def build_llm_client(
    settings: Settings,
    *,
    retry_policy: RetryPolicy | None = None,
) -> LLMClient[Any]:
    """Build an LLM client from provider-qualified model settings."""
    try:
        validate_required_provider_keys(settings)
    except ModelResolutionError as exc:
        raise ConfigurationError(str(exc)) from exc

    providers = []
    if settings.anthropic_api_key:
        providers.append(
            AnthropicProvider(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
            )
        )
    if settings.openai_api_key:
        providers.append(
            OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        )
    if settings.openrouter_api_key:
        providers.append(
            OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                base_url=settings.openrouter_base_url,
            )
        )
    if settings.google_api_key:
        providers.append(GeminiProvider(api_key=settings.google_api_key))

    client = LLMClient(
        providers=providers,
        retry_policy=retry_policy,
        intimacy_fallback_models=resolve_intimacy_fallback_models(settings),
        intimacy_proactive_routing_enabled=(
            settings.llm_intimacy_proactive_routing_enabled
        ),
    )

    for provider_slug in sorted(required_provider_slugs(settings)):
        provider_name = PROVIDER_SLUG_TO_NAME[provider_slug]
        try:
            client._provider(provider_name)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"No configured provider adapter for resolved provider {provider_slug!r}"
            ) from exc

    if settings.embedding_backend != "none":
        try:
            embedding = parse_embedding_model_spec(settings.embedding_model)
            embedding_provider = client._provider(embedding.provider_name)
        except ModelResolutionError as exc:
            raise ConfigurationError(str(exc)) from exc
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"No configured credentials for embedding provider {embedding.provider_slug!r}"
            ) from exc
        if not embedding_provider.supports_embeddings:
            raise ConfigurationError(
                f"Provider {embedding.provider_slug!r} does not support embeddings"
            )

    return client


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "build_llm_client",
]
