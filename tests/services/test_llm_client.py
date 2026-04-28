"""Tests for the provider-agnostic LLM client."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from atagia.models.schemas_memory import ExtractionResult
from atagia.services.llm_client import (
    ConfigurationError,
    LLMClient,
    LLMError,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMMessage,
    LLMProvider,
    LLMStreamEvent,
    OutputLimitExceededError,
    RetryPolicy,
    StructuredOutputError,
    TransientLLMError,
)


class FlakyProvider(LLMProvider):
    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        if self.calls < 3:
            raise TransientLLMError("temporary failure")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="final answer",
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1, 0.2])],
        )


class JsonProvider(LLMProvider):
    name = "json"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.payload,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class GrammarFallbackProvider(LLMProvider):
    name = "grammar-fallback"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_response_schema: list[bool] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        self.seen_response_schema.append(request.response_schema is not None)
        if request.response_schema is not None:
            raise LLMError("The compiled grammar is too large, which would cause performance issues.")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text='{"label":"ok","score":9}',
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class SchemaFallbackProvider(LLMProvider):
    name = "schema-fallback"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        if request.response_schema is not None:
            raise LLMError("For 'object' type, 'additionalProperties' must be explicitly set to false")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text='{"label":"ok","score":8}',
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class StructuredPayload(BaseModel):
    label: str
    score: int


class StreamingRetryProvider(LLMProvider):
    name = "stream-retry"

    def __init__(self) -> None:
        self.stream_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(provider=self.name, model=request.model)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )

    async def stream(self, request: LLMCompletionRequest):
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise TransientLLMError("setup failed")
        yield LLMStreamEvent(type="text", content="stream ok")
        yield LLMStreamEvent(type="done", payload={"usage": {}})


class MidStreamFailureProvider(LLMProvider):
    name = "mid-stream"

    def __init__(self) -> None:
        self.stream_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(provider=self.name, model=request.model)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )

    async def stream(self, request: LLMCompletionRequest):
        self.stream_calls += 1
        yield LLMStreamEvent(type="text", content="first")
        raise TransientLLMError("stream interrupted")


class RecordingProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="model-test",
        messages=[LLMMessage(role="user", content="hello")],
    )


@pytest.mark.asyncio
async def test_complete_retries_transient_errors() -> None:
    provider = FlakyProvider()
    client = LLMClient(
        provider_name="stub",
        providers=[provider],
        retry_policy=RetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0),
    )

    response = await client.complete(_request())

    assert response.output_text == "final answer"
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_complete_structured_parses_json_payload() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider('{"label":"ok","score":7}')])

    payload = await client.complete_structured(_request(), StructuredPayload)

    assert payload == StructuredPayload(label="ok", score=7)


@pytest.mark.asyncio
async def test_complete_structured_rejects_invalid_json() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider("not-json")])

    with pytest.raises(StructuredOutputError):
        await client.complete_structured(_request(), StructuredPayload)


@pytest.mark.asyncio
async def test_complete_structured_extracts_json_from_fenced_or_prefixed_text() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[
            JsonProvider('Here is the payload:\n```json\n{"label":"ok","score":7}\n```')
        ],
    )

    payload = await client.complete_structured(_request(), StructuredPayload)

    assert payload == StructuredPayload(label="ok", score=7)


@pytest.mark.asyncio
async def test_complete_structured_extracts_generic_fence_and_repairs_json() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[
            JsonProvider(
                'Here is the payload:\n```\n{"label":"Use {braces}","score":7,}\n```\nDone.'
            )
        ],
    )

    payload = await client.complete_structured(_request(), StructuredPayload)

    assert payload == StructuredPayload(label="Use {braces}", score=7)


@pytest.mark.asyncio
async def test_complete_structured_rejects_truncated_json_with_details() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[JsonProvider('{"label":"ok","score":')],
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.complete_structured(_request(), StructuredPayload)

    assert "Response was not valid JSON" in exc_info.value.details[0]
    assert any("truncated" in detail for detail in exc_info.value.details)


@pytest.mark.asyncio
async def test_complete_structured_falls_back_when_provider_rejects_large_grammar() -> None:
    provider = GrammarFallbackProvider()
    client = LLMClient(provider_name="grammar-fallback", providers=[provider])

    payload = await client.complete_structured(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert payload == StructuredPayload(label="ok", score=9)
    assert provider.calls == 2
    assert provider.seen_response_schema == [True, False]


@pytest.mark.asyncio
async def test_complete_structured_falls_back_when_provider_rejects_schema_contract() -> None:
    provider = SchemaFallbackProvider()
    client = LLMClient(provider_name="schema-fallback", providers=[provider])

    payload = await client.complete_structured(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert payload == StructuredPayload(label="ok", score=8)
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_complete_structured_normalizes_legacy_extraction_payload_shapes() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[
            JsonProvider(
                '{"extraction_metadata":{"nothing_durable":false},'
                '"durable_memories":[{"item_id":"mem_1","canonical_text":"User likes tea."}],'
                '"rationale":"legacy output"}'
            )
        ],
    )

    payload = await client.complete_structured(_request(), ExtractionResult)

    assert payload.nothing_durable is False
    assert len(payload.evidences) == 1
    assert payload.evidences[0].canonical_text == "User likes tea."
    assert payload.evidences[0].scope == "conversation"


@pytest.mark.asyncio
async def test_complete_structured_normalizes_legacy_belief_claims_and_downgrades_invalid_ones() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[
            JsonProvider(
                '{"extraction_metadata":{"nothing_durable":false},'
                '"durable_memory_items":['
                '{"item_id":"belief_1","memory_type":"belief","canonical_text":"User likes tea.","claim_key":null,"claim_value":null},'
                '{"item_id":"belief_2","memory_type":"belief","canonical_text":"Notifications are enabled.","claim_key":"preferences.notifications.enabled","claim_value":true}'
                ']}'
            )
        ],
    )

    payload = await client.complete_structured(_request(), ExtractionResult)

    assert len(payload.evidences) == 1
    assert payload.evidences[0].canonical_text == "User likes tea."
    assert len(payload.beliefs) == 1
    assert payload.beliefs[0].claim_key == "preferences.notifications.enabled"
    assert payload.beliefs[0].claim_value == "true"
    assert payload.evidences[0].source_kind == "extracted"
    assert payload.evidences[0].confidence == 0.5


@pytest.mark.asyncio
async def test_complete_structured_defaults_legacy_item_confidence_when_missing() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[
            JsonProvider(
                '{"extraction_metadata":{"nothing_durable":false},'
                '"durable_memory_items":[{"item_id":"sig_1","memory_type":"contract_signal","canonical_text":"Prefer concise replies.","dimension_name":"response_style","value_json":{"label":"concise"}}]}'
            )
        ],
    )

    payload = await client.complete_structured(_request(), ExtractionResult)

    assert len(payload.contract_signals) == 1
    assert payload.contract_signals[0].confidence == 0.5
    assert payload.contract_signals[0].canonical_text == "Prefer concise replies."


@pytest.mark.asyncio
async def test_complete_structured_still_fails_fast_for_non_extraction_validation_errors() -> None:
    provider = JsonProvider('{"label":"ok"}')
    client = LLMClient(provider_name="json", providers=[provider])

    with pytest.raises(StructuredOutputError):
        await client.complete_structured(_request(), StructuredPayload)

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_complete_structured_exposes_sanitized_validation_details() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider('{"label":"ok"}')])

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.complete_structured(_request(), StructuredPayload)

    assert exc_info.value.details == ("$.score: Field required",)


@pytest.mark.asyncio
async def test_missing_provider_raises_configuration_error() -> None:
    client = LLMClient(provider_name="missing", providers=[])

    with pytest.raises(ConfigurationError):
        await client.complete(_request())


@pytest.mark.asyncio
async def test_complete_routes_provider_qualified_model_and_applies_openai_profile() -> None:
    provider = RecordingProvider("openai")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(update={"model": "openai/gpt-5-mini,high"})
    )

    assert provider.requests[0].model == "gpt-5-mini"
    assert provider.requests[0].metadata["reasoning_effort"] == "high"
    assert provider.requests[0].metadata["atagia_canonical_model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_complete_applies_gemini_profile_without_exposing_thinking() -> None:
    provider = RecordingProvider("gemini")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={"model": "google/gemini-3.1-flash-lite-preview"}
        )
    )

    assert provider.requests[0].model == "gemini-3.1-flash-lite-preview"
    assert provider.requests[0].include_thinking is False
    assert provider.requests[0].metadata["gemini_thinking_level"] == "MINIMAL"


@pytest.mark.asyncio
async def test_complete_applies_openrouter_reasoning_body_after_resolution() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash,medium",
                "metadata": {"provider_extra_body": {"temperature": 0.1}},
            }
        )
    )

    assert provider.requests[0].model == "deepseek/deepseek-v4-flash"
    assert provider.requests[0].metadata["provider_extra_body"] == {
        "reasoning": {"effort": "medium"},
        "temperature": 0.1,
    }


@pytest.mark.asyncio
async def test_complete_applies_openrouter_flashlite_profile_after_resolution() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={"model": "openrouter/google/gemini-3.1-flash-lite-preview"}
        )
    )

    assert provider.requests[0].model == "google/gemini-3.1-flash-lite-preview"
    assert provider.requests[0].metadata["provider_extra_body"] == {
        "reasoning": {"effort": "minimal"}
    }


@pytest.mark.asyncio
async def test_complete_rejects_unqualified_model_without_explicit_test_escape() -> None:
    provider = RecordingProvider("openai")
    client = LLMClient(providers=[provider])

    with pytest.raises(ConfigurationError, match="provider/model"):
        await client.complete(_request())


@pytest.mark.asyncio
async def test_stream_retries_before_first_event() -> None:
    provider = StreamingRetryProvider()
    client = LLMClient(
        provider_name="stream-retry",
        providers=[provider],
        retry_policy=RetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0),
    )

    events = [event async for event in client.stream(_request())]

    assert [event.type for event in events] == ["text", "done"]
    assert events[0].content == "stream ok"
    assert provider.stream_calls == 2


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_partial_output() -> None:
    provider = MidStreamFailureProvider()
    client = LLMClient(
        provider_name="mid-stream",
        providers=[provider],
        retry_policy=RetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0),
    )

    stream = client.stream(_request())
    first_event = await anext(stream)

    assert first_event.type == "text"
    assert first_event.content == "first"
    with pytest.raises(TransientLLMError):
        await anext(stream)
    assert provider.stream_calls == 1


def test_output_limit_exceeded_error_is_subclass_of_llm_error() -> None:
    error = OutputLimitExceededError("hit max output tokens")

    assert isinstance(error, LLMError)
    assert isinstance(error, RuntimeError)
    assert not isinstance(error, TransientLLMError)
    assert not isinstance(error, StructuredOutputError)
    assert str(error) == "hit max output tokens"
