"""Build a real, provider-backed inner provider for LIVE card 2 runs.

Routing matches production: the harness registers a real provider whose ``name``
matches the provider slug the compactor model resolves to (``gemini``,
``minimax``, ``openrouter``, ...), and the engine's ``LLMClient`` routes by
model-id prefix. We build exactly one provider -- the one the chosen model needs
-- so a missing unrelated key never blocks the run.
"""

from __future__ import annotations

import os

from atagia.services.llm_client import LLMProvider
from atagia.services.model_resolution import (
    PROVIDER_SLUG_TO_NAME,
    parse_model_spec,
)
from atagia.services.providers.anthropic import AnthropicProvider
from atagia.services.providers.gemini import GeminiProvider
from atagia.services.providers.minimax import MiniMaxProvider
from atagia.services.providers.openai import OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider



def _first_environment_value(*names: str) -> str | None:
    """Return the first configured provider credential from process env."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def build_live_provider(model: str) -> LLMProvider:
    """Build the real provider the compactor ``model`` routes to.

    Raises ``SystemExit`` with a clear message if the required key is missing,
    so a live run fails fast instead of producing misleading metrics.
    """
    parsed = parse_model_spec(model)
    provider_name = PROVIDER_SLUG_TO_NAME[parsed.provider_slug]

    if provider_name == "openrouter":
        api_key = _first_environment_value(
            "ATAGIA_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        )
        if not api_key:
            raise SystemExit(
                "Live run needs OPENROUTER_API_KEY (or ATAGIA_OPENROUTER_API_KEY) "
                f"for model {model!r}."
            )
        return OpenRouterProvider(
            api_key=api_key,
            site_url="https://atagia.org",
            app_name="Atagia card2 summary harness",
        )
    if provider_name == "gemini":
        api_key = _first_environment_value(
            "ATAGIA_GOOGLE_API_KEY",
            "GEMINI_KEY",
            "GOOGLE_API_KEY",
        )
        if not api_key:
            raise SystemExit(
                "Live run needs GOOGLE/GEMINI key (ATAGIA_GOOGLE_API_KEY or "
                f"GEMINI_KEY) for model {model!r}."
            )
        return GeminiProvider(api_key=api_key)
    if provider_name == "minimax":
        api_key = _first_environment_value(
            "ATAGIA_MINIMAX_API_KEY",
            "MINIMAX_API_KEY",
            "MINIMAX_KEY",
        )
        if not api_key:
            raise SystemExit(
                "Live run needs MINIMAX key (ATAGIA_MINIMAX_API_KEY or "
                f"MINIMAX_API_KEY) for model {model!r}."
            )
        return MiniMaxProvider(api_key=api_key)
    if provider_name == "anthropic":
        api_key = _first_environment_value(
            "ATAGIA_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
            "CLAUDE_KEY",
        )
        if not api_key:
            raise SystemExit(
                "Live run needs ANTHROPIC key for model "
                f"{model!r}."
            )
        return AnthropicProvider(api_key=api_key)
    if provider_name == "openai":
        api_key = _first_environment_value(
            "ATAGIA_OPENAI_API_KEY",
            "OPENAI_KEY",
            "OPENAI_API_KEY",
        )
        if not api_key:
            raise SystemExit(
                "Live run needs OPENAI key for model "
                f"{model!r}."
            )
        return OpenAIProvider(api_key=api_key)
    raise SystemExit(f"Unsupported provider {provider_name!r} for model {model!r}.")
