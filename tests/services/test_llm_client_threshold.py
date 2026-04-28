"""Tests for the LLMClient `max_output_tokens` threshold policy.

The policy is enforced once at the entry point of `LLMClient.complete`, so
both direct `complete` calls and `complete_structured` (which delegates to
`complete`) inherit the cleaned value.
"""

from __future__ import annotations

import pytest

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMMessage,
    LLMProvider,
)


class RecordingProvider(LLMProvider):
    """Captures every request that reaches the provider layer."""

    name = "stub"

    def __init__(self, payload: str = "ok") -> None:
        self.payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.payload,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )


def _request(max_output_tokens: int | None) -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="model-test",
        messages=[LLMMessage(role="user", content="hello")],
        max_output_tokens=max_output_tokens,
    )


@pytest.mark.asyncio
async def test_llm_client_complete_applies_threshold() -> None:
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    await client.complete(_request(max_output_tokens=256))

    assert len(provider.requests) == 1
    assert provider.requests[0].max_output_tokens is None


@pytest.mark.asyncio
async def test_llm_client_complete_drops_value_at_floor() -> None:
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    await client.complete(_request(max_output_tokens=512))

    assert provider.requests[0].max_output_tokens is None


@pytest.mark.asyncio
async def test_llm_client_complete_passes_high_value_through() -> None:
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    await client.complete(_request(max_output_tokens=2048))

    assert provider.requests[0].max_output_tokens == 2048


@pytest.mark.asyncio
async def test_llm_client_complete_passes_none_through() -> None:
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    await client.complete(_request(max_output_tokens=None))

    assert provider.requests[0].max_output_tokens is None


@pytest.mark.asyncio
async def test_llm_client_complete_structured_applies_threshold() -> None:
    provider = RecordingProvider(payload='{"label":"ok","score":7}')
    client = LLMClient(provider_name="stub", providers=[provider])

    from pydantic import BaseModel

    class StructuredPayload(BaseModel):
        label: str
        score: int

    await client.complete_structured(
        _request(max_output_tokens=256),
        StructuredPayload,
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].max_output_tokens is None


@pytest.mark.asyncio
async def test_llm_client_complete_structured_passes_high_value_through() -> None:
    provider = RecordingProvider(payload='{"label":"ok","score":7}')
    client = LLMClient(provider_name="stub", providers=[provider])

    from pydantic import BaseModel

    class StructuredPayload(BaseModel):
        label: str
        score: int

    await client.complete_structured(
        _request(max_output_tokens=2048),
        StructuredPayload,
    )

    assert provider.requests[0].max_output_tokens == 2048


@pytest.mark.asyncio
async def test_llm_client_stream_applies_threshold() -> None:
    """`stream()` must enforce the same sub-floor policy as `complete()`."""
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    events = [event async for event in client.stream(_request(max_output_tokens=256))]

    assert events  # provider yields at least one event so the path actually ran
    assert len(provider.requests) == 1
    assert provider.requests[0].max_output_tokens is None


@pytest.mark.asyncio
async def test_llm_client_stream_passes_high_value_through() -> None:
    provider = RecordingProvider()
    client = LLMClient(provider_name="stub", providers=[provider])

    [event async for event in client.stream(_request(max_output_tokens=2048))]

    assert provider.requests[0].max_output_tokens == 2048
