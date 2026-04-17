"""Concrete LLM provider implementations and factory helpers."""

from __future__ import annotations

from typing import Any

from atagia.core.config import Settings
from atagia.services.llm_client import ConfigurationError, LLMClient, RetryPolicy
from atagia.services.providers.anthropic import AnthropicProvider
from atagia.services.providers.openai import OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider


def _resolve_embedding_provider_name(
    settings: Settings,
    *,
    openai_key: str | None,
) -> str:
    explicit_provider = settings.embedding_provider_name
    if explicit_provider is not None and explicit_provider.strip():
        return explicit_provider.strip().lower()
    if settings.llm_provider in {"openai", "openrouter"}:
        return settings.llm_provider
    if settings.llm_provider == "anthropic":
        if openai_key:
            return "openai"
        return "anthropic"
    return settings.llm_provider


def build_llm_client(
    settings: Settings,
    *,
    retry_policy: RetryPolicy | None = None,
) -> LLMClient[Any]:
    """Build an LLM client from environment-backed settings."""
    providers = []
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

    embedding_provider_name = _resolve_embedding_provider_name(
        settings,
        openai_key=openai_key,
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

    if settings.embedding_backend != "none":
        if settings.llm_provider == "anthropic" and settings.embedding_provider_name is None and not openai_key:
            raise ConfigurationError(
                "Embeddings require ATAGIA_OPENAI_API_KEY when ATAGIA_LLM_PROVIDER='anthropic'. "
                "OpenRouter is not auto-selected; set ATAGIA_EMBEDDING_PROVIDER explicitly to override."
            )
        try:
            embedding_provider = client.embedding_provider
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"No configured credentials for embedding provider {client.embedding_provider_name!r}"
            ) from exc
        if not embedding_provider.supports_embeddings:
            raise ConfigurationError(
                f"Provider {client.embedding_provider_name!r} does not support embeddings"
            )

    return client


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "build_llm_client",
]
