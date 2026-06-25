"""Kimi provider adapter."""

from __future__ import annotations

from openai import AsyncOpenAI

from atagia.services.llm_client import LLMCompletionRequest
from atagia.services.providers.openai import OpenAICompatibleProvider, _response_format


class KimiProvider(OpenAICompatibleProvider):
    """Kimi/Moonshot exposes an OpenAI-compatible chat completions API."""

    name = "kimi"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/v1",
            request_timeout_seconds=request_timeout_seconds,
            client=client,
        )

    def _completion_response_format(
        self,
        request: LLMCompletionRequest,
    ) -> dict[str, object] | None:
        return _response_format(request.response_schema, preserve_nullability=True)
