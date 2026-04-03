"""Concrete LLM provider implementations and factory helpers."""

from __future__ import annotations

from typing import Any

from atagia.core.config import Settings
from atagia.services.llm_client import ConfigurationError, LLMClient, RetryPolicy
from atagia.services.providers.anthropic import AnthropicProvider
from atagia.services.providers.openai import OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider


def build_llm_client(
    settings: Settings,
    *,
    retry_policy: RetryPolicy | None = None,
) -> LLMClient[Any]:
    """Build an LLM client from environment-backed settings."""
    providers = []

    embedding_provider_name = settings.llm_provider
    anthropic_key = settings.llm_api_key
    if anthropic_key:
        providers.append(
            AnthropicProvider(
                api_key=anthropic_key,
                base_url=settings.llm_base_url if settings.llm_provider == "anthropic" else None,
            )
        )

    openai_key = settings.openai_api_key or (
        settings.llm_api_key if settings.llm_provider == "openai" else None
    )
    if openai_key:
        providers.append(
            OpenAIProvider(
                api_key=openai_key,
                base_url=settings.llm_base_url if settings.llm_provider == "openai" else None,
            )
        )
        if settings.llm_provider == "anthropic":
            embedding_provider_name = "openai"

    openrouter_key = settings.openrouter_api_key or (
        settings.llm_api_key if settings.llm_provider == "openrouter" else None
    )
    if openrouter_key:
        providers.append(
            OpenRouterProvider(
                api_key=openrouter_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                base_url=settings.llm_base_url if settings.llm_provider == "openrouter" else None,
            )
        )

    client = LLMClient(
        provider_name=settings.llm_provider,
        providers=providers,
        embedding_provider_name=embedding_provider_name,
        retry_policy=retry_policy,
    )

    try:
        client._provider()
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"No configured credentials for provider {settings.llm_provider!r}"
        ) from exc

    return client


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "build_llm_client",
]
