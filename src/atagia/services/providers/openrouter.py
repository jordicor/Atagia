"""OpenRouter provider adapter."""

from __future__ import annotations

from openai import AsyncOpenAI

from atagia.services.providers.openai import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter uses the OpenAI-compatible SDK surface."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        *,
        site_url: str,
        app_name: str,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
            client=client,
        )

