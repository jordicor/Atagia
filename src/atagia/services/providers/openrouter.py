"""OpenRouter provider adapter."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from atagia.services.llm_client import LLMCompletionRequest
from atagia.services.providers.openai import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter uses the OpenAI-compatible SDK surface."""

    name = "openrouter"
    supports_native_structured_output = False

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

    def _completion_response_format(
        self,
        request: LLMCompletionRequest,
    ) -> dict[str, Any] | None:
        return None
