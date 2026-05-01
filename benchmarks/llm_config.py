"""Shared LLM configuration helpers for benchmark CLIs."""

from __future__ import annotations

from atagia.services.model_resolution import PROVIDER_NAME_TO_SLUG


def provider_api_key_kwargs(provider: str | None, api_key: str | None) -> dict[str, str]:
    """Return Atagia constructor kwargs for a provider-specific API key."""
    if not api_key:
        return {}
    provider_slug = PROVIDER_NAME_TO_SLUG.get((provider or "").strip().lower())
    if provider_slug is None:
        raise ValueError("A valid --provider is required when --api-key is set")
    key_names = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "google": "google_api_key",
        "openrouter": "openrouter_api_key",
    }
    return {key_names[provider_slug]: api_key}
