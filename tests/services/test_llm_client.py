"""Tests for the provider-agnostic LLM client."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from atagia.models.schemas_memory import ExtractionResult, IntimacyBoundary
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
    LLMPolicyBlockedError,
    LLMProvider,
    LLMRequestError,
    LLMStreamEvent,
    OutputLimitExceededError,
    RetryPolicy,
    StructuredOutputError,
    TransientLLMError,
)
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.run_counters import (
    RunCounterAccumulator,
    use_run_counter_accumulator,
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


class SequentialJsonProvider(LLMProvider):
    def __init__(self, name: str, payloads: list[str]) -> None:
        self.name = name
        self.payloads = list(payloads)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        output_text = self.payloads.pop(0) if self.payloads else "{}"
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
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
            raise LLMRequestError(
                "400 invalid_request_error: schema grammar rejected",
                status_code=400,
            )
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
            raise LLMRequestError(
                "400 invalid_request_error: additionalProperties must be false",
                status_code=400,
            )
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


class PromptStructuredProvider(LLMProvider):
    name = "prompt-structured"
    supports_native_structured_output = False

    def __init__(self) -> None:
        self.calls = 0
        self.seen_response_schema: list[bool] = []
        self.seen_messages: list[list[LLMMessage]] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        self.seen_response_schema.append(request.response_schema is not None)
        self.seen_messages.append(request.messages)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text='{"label":"ok","score":10}',
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class VendorAwareStructuredProvider(PromptStructuredProvider):
    name = "openrouter"

    def supports_native_structured_output_for(self, request: LLMCompletionRequest) -> bool:
        return request.model.startswith(("anthropic/", "google/", "openai/", "x-ai/"))


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


class PostDoneLimitProvider(LLMProvider):
    name = "post-done-limit"

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
        yield LLMStreamEvent(type="text", content="partial")
        yield LLMStreamEvent(type="done", payload={"usage": {"completion_tokens": 2}})
        raise OutputLimitExceededError("hit max output tokens")


class PostDoneLimitThenSuccessProvider(PostDoneLimitProvider):
    name = "post-done-limit-then-success"

    async def stream(self, request: LLMCompletionRequest):
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield LLMStreamEvent(type="text", content="partial")
            yield LLMStreamEvent(type="done", payload={"usage": {"completion_tokens": 2}})
            raise OutputLimitExceededError(
                "hit max output tokens",
                provider=self.name,
                finish_reason="length",
                partial_output_chars=7,
            )
        yield LLMStreamEvent(type="text", content="observer recovered")
        yield LLMStreamEvent(type="done", payload={"usage": {}})


class OutputLimitThenSuccessProvider(LLMProvider):
    name = "output-limit-then-success"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise OutputLimitExceededError(
                "hit max output tokens",
                provider=self.name,
                finish_reason="length",
                max_output_tokens=request.max_output_tokens,
                partial_output_chars=4096,
                partial_output_excerpt="partial",
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="recovered",
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )


class RunawayThenSuccessProvider(LLMProvider):
    name = "runaway-then-success"

    def __init__(self) -> None:
        self.stream_calls = 0
        self.requests: list[LLMCompletionRequest] = []

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
        self.requests.append(request)
        if self.stream_calls == 1:
            for _ in range(12):
                yield LLMStreamEvent(type="text", content="alpha beta gamma " * 4)
            yield LLMStreamEvent(type="done", payload={"usage": {}})
            return
        yield LLMStreamEvent(type="text", content="stream recovered")
        yield LLMStreamEvent(type="done", payload={"usage": {}})


class ThinkingAndContentProvider(LLMProvider):
    name = "thinking-and-content"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(provider=self.name, model=request.model)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )

    async def stream(self, request: LLMCompletionRequest):
        yield LLMStreamEvent(type="thinking", content="hidden chain hidden chain ")
        yield LLMStreamEvent(type="text", content="visible one ")
        yield LLMStreamEvent(type="thinking", content="more hidden reasoning ")
        yield LLMStreamEvent(type="text", content="visible two")
        yield LLMStreamEvent(type="done", payload={"usage": {}})


class ClosingStreamProvider(LLMProvider):
    name = "closing-stream"

    def __init__(self) -> None:
        self.closed = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(provider=self.name, model=request.model)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )

    async def stream(self, request: LLMCompletionRequest):
        try:
            yield LLMStreamEvent(type="text", content="partial")
            yield LLMStreamEvent(type="text", content="unreachable")
        finally:
            self.closed = True


class AbortObserver:
    async def on_text(
        self,
        _chunk: str,
        _accumulated_text: str,
        _request: LLMCompletionRequest,
    ) -> None:
        raise RuntimeError("observer abort")


class NoopObserver:
    async def on_text(
        self,
        _chunk: str,
        _accumulated_text: str,
        _request: LLMCompletionRequest,
    ) -> None:
        return None


class RecordingObserver:
    def __init__(self) -> None:
        self.accumulated_texts: list[str] = []

    async def on_text(
        self,
        _chunk: str,
        accumulated_text: str,
        _request: LLMCompletionRequest,
    ) -> None:
        self.accumulated_texts.append(accumulated_text)


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


class PolicyBlockingProvider(RecordingProvider):
    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        raise LLMPolicyBlockedError("provider blocked the response (finish_reason=content_filter)")


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
async def test_complete_uses_intimacy_fallback_for_policy_block() -> None:
    primary = PolicyBlockingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
    )

    response = await client.complete(
        LLMCompletionRequest(
            model="openai/gpt-5-mini",
            messages=[LLMMessage(role="user", content="hello")],
            metadata={"purpose": "memory_extraction"},
        )
    )

    assert response.output_text == "ok"
    assert primary.requests[0].model == "gpt-5-mini"
    assert fallback.requests[0].model == "z-ai/glm-4.6"
    assert fallback.requests[0].metadata["atagia_intimacy_fallback_used"] is True
    assert fallback.requests[0].metadata["atagia_intimacy_primary_model"] == "openai/gpt-5-mini"
    assert fallback.requests[0].metadata["atagia_component_id"] == "extractor"


@pytest.mark.asyncio
async def test_complete_proactively_uses_intimacy_model_for_known_context() -> None:
    primary = RecordingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
        intimacy_proactive_routing_enabled=True,
    )

    response = await client.complete(
        LLMCompletionRequest(
            model="openai/gpt-5-mini",
            messages=[LLMMessage(role="user", content="hello")],
            metadata={
                "purpose": "memory_extraction",
                "source_intimacy_boundary": "romantic_private",
            },
        )
    )

    assert response.output_text == "ok"
    assert primary.requests == []
    assert fallback.requests[0].model == "z-ai/glm-4.6"
    assert fallback.requests[0].metadata["atagia_intimacy_proactive_route"] is True
    assert fallback.requests[0].metadata["atagia_intimacy_primary_model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_complete_does_not_proactively_route_without_setting() -> None:
    primary = RecordingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
    )

    await client.complete(
        LLMCompletionRequest(
            model="openai/gpt-5-mini",
            messages=[LLMMessage(role="user", content="hello")],
            metadata={
                "purpose": "memory_extraction",
                "source_intimacy_boundary": "romantic_private",
            },
        )
    )

    assert primary.requests[0].model == "gpt-5-mini"
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_complete_does_not_use_intimacy_fallback_for_non_policy_error() -> None:
    class FailingProvider(RecordingProvider):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            raise LLMError("ordinary provider failure")

    primary = FailingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
    )

    with pytest.raises(LLMError, match="ordinary provider failure"):
        await client.complete(
            LLMCompletionRequest(
                model="openai/gpt-5-mini",
                messages=[LLMMessage(role="user", content="hello")],
                metadata={"purpose": "memory_extraction"},
            )
        )

    assert len(primary.requests) == 1
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_complete_does_not_use_intimacy_fallback_for_generic_refusal_word() -> None:
    class FailingProvider(RecordingProvider):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            raise LLMError("ordinary refusal counter update failed")

    primary = FailingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
    )

    with pytest.raises(LLMError, match="ordinary refusal counter"):
        await client.complete(
            LLMCompletionRequest(
                model="openai/gpt-5-mini",
                messages=[LLMMessage(role="user", content="hello")],
                metadata={"purpose": "memory_extraction"},
            )
        )

    assert len(primary.requests) == 1
    assert fallback.requests == []


@pytest.mark.asyncio
async def test_stream_uses_intimacy_fallback_for_pre_output_policy_block() -> None:
    primary = PolicyBlockingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
    )

    events = [
        event
        async for event in client.stream(
            LLMCompletionRequest(
                model="openai/gpt-5-mini",
                messages=[LLMMessage(role="user", content="hello")],
                metadata={"purpose": "memory_extraction"},
            )
        )
    ]

    assert [event.type for event in events] == ["text", "done"]
    assert events[0].content == "ok"
    assert primary.requests[0].model == "gpt-5-mini"
    assert fallback.requests[0].model == "z-ai/glm-4.6"
    assert fallback.requests[0].metadata["atagia_intimacy_fallback_used"] is True


@pytest.mark.asyncio
async def test_complete_streamed_proactively_routes_known_intimacy_enum_boundary() -> None:
    primary = RecordingProvider("openai")
    fallback = RecordingProvider("openrouter")
    client = LLMClient(
        providers=[primary, fallback],
        intimacy_fallback_models={"extractor": "openrouter/z-ai/glm-4.6"},
        intimacy_proactive_routing_enabled=True,
    )

    response = await client.complete_streamed(
        LLMCompletionRequest(
            model="openai/gpt-5-mini",
            messages=[LLMMessage(role="user", content="hello")],
            metadata={
                "purpose": "memory_extraction",
                "source_intimacy_boundary": IntimacyBoundary.ROMANTIC_PRIVATE,
            },
        )
    )

    assert response.output_text == "ok"
    assert primary.requests == []
    assert fallback.requests[0].model == "z-ai/glm-4.6"
    assert fallback.requests[0].metadata["atagia_intimacy_proactive_route"] is True


@pytest.mark.asyncio
async def test_complete_structured_parses_json_payload() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider('{"label":"ok","score":7}')])

    payload = await client.complete_structured(_request(), StructuredPayload)

    assert payload == StructuredPayload(label="ok", score=7)


@pytest.mark.asyncio
async def test_complete_structured_with_response_returns_raw_response() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider('{"label":"ok","score":7}')])

    result = await client.complete_structured_with_response(_request(), StructuredPayload)

    assert result.value == StructuredPayload(label="ok", score=7)
    assert result.response.output_text == '{"label":"ok","score":7}'
    assert result.used_schema_fallback is False


@pytest.mark.asyncio
async def test_complete_structured_rejects_invalid_json() -> None:
    client = LLMClient(provider_name="json", providers=[JsonProvider("not-json")])

    with pytest.raises(StructuredOutputError) as exc_info:
        await client.complete_structured(_request(), StructuredPayload)
    assert exc_info.value.output_text == "not-json"


@pytest.mark.asyncio
async def test_complete_structured_extracts_json_from_fenced_or_prefixed_text() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[JsonProvider('Here is the payload:\n```json\n{"label":"ok","score":7}\n```')],
    )

    payload = await client.complete_structured(_request(), StructuredPayload)

    assert payload == StructuredPayload(label="ok", score=7)


@pytest.mark.asyncio
async def test_complete_structured_extracts_generic_fence_and_repairs_json() -> None:
    client = LLMClient(
        provider_name="json",
        providers=[JsonProvider('Here is the payload:\n```\n{"label":"Use {braces}","score":7,}\n```\nDone.')],
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


class SchemaDrop4xxThenJsonProvider(LLMProvider):
    """Raises a typed 4xx on the native-schema call, then returns JSON on fallback."""

    name = "schema-drop-4xx"

    def __init__(self, status_code: int = 400) -> None:
        self.status_code = status_code
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.response_schema is not None:
            raise LLMRequestError(
                "400 INVALID_ARGUMENT: schema rejected",
                status_code=self.status_code,
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text='{"label":"ok","score":12}',
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class NonRequestErrorProvider(LLMProvider):
    """Raises a non-typed LLMError on the native-schema call (no schema-drop retry)."""

    name = "non-request-error"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        raise LLMError("400 invalid_request_error: schema is too complex")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )


class StreamingSchemaDrop4xxThenJsonProvider(LLMProvider):
    """Streaming analogue: raises a typed 4xx before emitting any chunk on the
    native-schema attempt, then streams JSON on the prompt-JSON fallback."""

    name = "streaming-schema-drop-4xx"

    def __init__(self, status_code: int = 400) -> None:
        self.status_code = status_code
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise NotImplementedError

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
        )

    async def stream(self, request: LLMCompletionRequest):
        self.requests.append(request)
        if request.response_schema is not None:
            raise LLMRequestError(
                "400 INVALID_ARGUMENT: schema rejected",
                status_code=self.status_code,
            )
        yield LLMStreamEvent(type="text", content='{"label":"ok","score":12}')
        yield LLMStreamEvent(type="done", payload={"usage": {}})


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
async def test_complete_structured_falls_back_via_prompt_json_on_typed_4xx() -> None:
    provider = SchemaDrop4xxThenJsonProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    result = await client.complete_structured_with_response(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="ok", score=12)
    assert result.used_schema_fallback is True
    # Exactly one native-schema attempt + one prompt-JSON retry.
    assert len(provider.requests) == 2
    assert provider.requests[0].response_schema is not None
    fallback_request = provider.requests[1]
    assert fallback_request.response_schema is None
    # The F0.1 compact spec is embedded in the appended instruction.
    instruction = fallback_request.messages[-1].content
    assert "Return exactly one raw JSON" in instruction
    assert "The JSON object must follow this structure" in instruction


@pytest.mark.asyncio
async def test_complete_structured_streamed_falls_back_via_prompt_json_on_typed_4xx() -> None:
    provider = StreamingSchemaDrop4xxThenJsonProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    payload = await client.complete_structured_streamed(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert payload == StructuredPayload(label="ok", score=12)
    # Exactly one native-schema stream attempt + one prompt-JSON retry.
    assert len(provider.requests) == 2
    assert provider.requests[0].response_schema is not None
    fallback_request = provider.requests[1]
    assert fallback_request.response_schema is None
    # The F0.1 compact spec is embedded in the appended instruction.
    instruction = fallback_request.messages[-1].content
    assert "Return exactly one raw JSON" in instruction
    assert "The JSON object must follow this structure" in instruction
    assert "label (string)" in instruction
    assert "score (integer)" in instruction
    # The fallback reason is recorded in the trace metadata.
    assert fallback_request.metadata["atagia_structured_output_schema_drop_fallback"] is True
    assert fallback_request.metadata["atagia_structured_output_schema_drop_status_code"] == 400
    assert (
        fallback_request.metadata["atagia_structured_output_schema_drop_reason"]
        == "client_request_error_4xx"
    )


@pytest.mark.asyncio
async def test_complete_structured_does_not_drop_schema_for_non_request_error() -> None:
    provider = NonRequestErrorProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        structured_output_retry_attempts=0,
    )

    with pytest.raises(LLMError, match="schema is too complex"):
        await client.complete_structured(
            _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
            StructuredPayload,
        )

    # No schema-drop retry: the single native-schema attempt raised straight through.
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_complete_structured_prompts_json_for_provider_without_native_schema() -> None:
    provider = PromptStructuredProvider()
    client = LLMClient(provider_name="prompt-structured", providers=[provider])

    result = await client.complete_structured_with_response(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="ok", score=10)
    assert result.used_schema_fallback is True
    assert provider.calls == 1
    assert provider.seen_response_schema == [False]
    assert "Return exactly one raw JSON" in provider.seen_messages[0][-1].content


@pytest.mark.asyncio
async def test_complete_structured_uses_model_aware_native_schema_support() -> None:
    provider = VendorAwareStructuredProvider()
    client = LLMClient(providers=[provider])

    result = await client.complete_structured_with_response(
        _request().model_copy(
            update={
                "model": "openrouter/openai/gpt-chat-latest",
                "response_schema": StructuredPayload.model_json_schema(),
            }
        ),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="ok", score=10)
    assert result.used_schema_fallback is False
    assert provider.seen_response_schema == [True]
    assert provider.seen_messages[0][-1].content == "hello"


@pytest.mark.asyncio
async def test_complete_structured_falls_back_for_model_without_native_schema_support() -> None:
    provider = VendorAwareStructuredProvider()
    client = LLMClient(providers=[provider])

    result = await client.complete_structured_with_response(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "response_schema": StructuredPayload.model_json_schema(),
            }
        ),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="ok", score=10)
    assert result.used_schema_fallback is True
    assert provider.seen_response_schema == [False]
    assert "Return exactly one raw JSON" in provider.seen_messages[0][-1].content


@pytest.mark.asyncio
async def test_complete_structured_retries_same_model_before_rescue() -> None:
    primary = SequentialJsonProvider(
        "openrouter",
        ["not-json", '{"label":"ok","score":7}'],
    )
    rescue = SequentialJsonProvider("anthropic", ['{"label":"rescue","score":10}'])
    client = LLMClient(
        providers=[primary, rescue],
        structured_output_retry_attempts=1,
        structured_output_rescue_enabled=True,
        structured_output_rescue_model="anthropic/claude-opus-4-7",
    )

    result = await client.complete_structured_with_response(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "metadata": {"purpose": "need_detection"},
                "response_schema": StructuredPayload.model_json_schema(),
            }
        ),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="ok", score=7)
    assert result.used_structured_output_retry is True
    assert result.used_structured_output_rescue is False
    assert result.response.raw_response["atagia_structured_output_repair"] == {
        "kind": "retry",
        "primary_model": "openrouter/deepseek/deepseek-v4-flash",
        "repair_model": "openrouter/deepseek/deepseek-v4-flash",
        "retry_attempts": 1,
    }
    assert [request.model for request in primary.requests] == [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash",
    ]
    assert not rescue.requests
    assert primary.requests[1].metadata["atagia_structured_output_retry"] is True
    assert primary.requests[1].metadata["atagia_structured_output_retry_attempt"] == 1
    assert "Validation errors" in primary.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_complete_structured_uses_rescue_after_same_model_retry_fails() -> None:
    primary = SequentialJsonProvider(
        "openrouter",
        ["not-json", '{"label":"missing score"}'],
    )
    rescue = SequentialJsonProvider("anthropic", ['{"label":"rescued","score":11}'])
    client = LLMClient(
        providers=[primary, rescue],
        structured_output_retry_attempts=1,
        structured_output_rescue_enabled=True,
        structured_output_rescue_model="anthropic/claude-opus-4-7,high",
    )

    result = await client.complete_structured_with_response(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "metadata": {"purpose": "need_detection"},
            }
        ),
        StructuredPayload,
    )

    assert result.value == StructuredPayload(label="rescued", score=11)
    assert result.used_structured_output_retry is True
    assert result.used_structured_output_rescue is True
    assert result.response.raw_response["atagia_structured_output_repair"] == {
        "kind": "rescue",
        "primary_model": "openrouter/deepseek/deepseek-v4-flash",
        "repair_model": "anthropic/claude-opus-4-7,high",
        "retry_attempts": 1,
    }
    assert len(primary.requests) == 2
    assert len(rescue.requests) == 1
    assert rescue.requests[0].model == "claude-opus-4-7"
    assert rescue.requests[0].response_schema is not None
    assert rescue.requests[0].metadata["atagia_structured_output_rescue"] is True
    assert (
        rescue.requests[0].metadata["atagia_structured_output_rescue_original_model"]
        == "openrouter/deepseek/deepseek-v4-flash"
    )
    assert rescue.requests[0].metadata["anthropic_output_effort"] == "high"
    assert "configured rescue model" in rescue.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_complete_structured_can_disable_same_model_retry() -> None:
    primary = SequentialJsonProvider(
        "openrouter",
        ["not-json", '{"label":"ok","score":7}'],
    )
    client = LLMClient(
        providers=[primary],
        structured_output_retry_attempts=0,
    )

    with pytest.raises(StructuredOutputError):
        await client.complete_structured(
            _request().model_copy(update={"model": "openrouter/deepseek/deepseek-v4-flash"}),
            StructuredPayload,
        )

    assert len(primary.requests) == 1


@pytest.mark.parametrize("status_code", [400, 404, 422, 499])
def test_should_retry_without_schema_for_client_request_4xx(status_code: int) -> None:
    request = _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()})

    assert LLMClient._should_retry_without_schema(
        LLMRequestError("client error", status_code=status_code),
        request,
    )


def test_should_not_retry_without_schema_for_non_request_error() -> None:
    request = _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()})

    assert not LLMClient._should_retry_without_schema(
        LLMError("400 invalid_request_error: schema is too complex"),
        request,
    )


def test_should_not_retry_without_schema_for_5xx_request_error() -> None:
    request = _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()})

    assert not LLMClient._should_retry_without_schema(
        LLMRequestError("server error", status_code=500),
        request,
    )


def test_should_not_retry_without_schema_when_schema_already_dropped() -> None:
    request = _request().model_copy(update={"response_schema": None})

    assert not LLMClient._should_retry_without_schema(
        LLMRequestError("client error", status_code=400),
        request,
    )


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
                "]}"
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
async def test_complete_structured_retries_non_extraction_validation_errors_once() -> None:
    provider = JsonProvider('{"label":"ok"}')
    client = LLMClient(provider_name="json", providers=[provider])

    with pytest.raises(StructuredOutputError):
        await client.complete_structured(_request(), StructuredPayload)

    assert provider.calls == 2


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

    await client.complete(_request().model_copy(update={"model": "openai/gpt-5-mini,high"}))

    assert provider.requests[0].model == "gpt-5-mini"
    assert provider.requests[0].metadata["reasoning_effort"] == "high"
    assert provider.requests[0].metadata["atagia_canonical_model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_complete_applies_gemini_profile_without_exposing_thinking() -> None:
    provider = RecordingProvider("gemini")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "google/gemini-3.1-flash-lite",
                "temperature": 0.0,
            }
        )
    )

    assert provider.requests[0].model == "gemini-3.1-flash-lite"
    assert provider.requests[0].include_thinking is False
    assert provider.requests[0].temperature == 1.0
    assert provider.requests[0].metadata["gemini_thinking_level"] == "MINIMAL"
    assert provider.requests[0].metadata["atagia_effective_temperature"] == 1.0


@pytest.mark.asyncio
async def test_complete_applies_purpose_temperature_when_request_omits_temperature() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "metadata": {"purpose": "chat_reply"},
            }
        )
    )

    assert provider.requests[0].temperature == 1.0
    assert provider.requests[0].metadata["atagia_temperature_source"] == "purpose_default"
    assert provider.requests[0].metadata["atagia_temperature_reason"] == "chat answer generation"


@pytest.mark.asyncio
async def test_complete_applies_low_mechanical_temperature_for_verifier_purpose() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "metadata": {"purpose": "answer_postcondition_verification"},
            }
        )
    )

    assert provider.requests[0].temperature == 0.2
    assert provider.requests[0].metadata["atagia_temperature_reason"] == "mechanical verifier/classifier"


@pytest.mark.asyncio
async def test_complete_floors_explicit_zero_temperature() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "temperature": 0.0,
            }
        )
    )

    assert provider.requests[0].temperature == 0.1
    assert provider.requests[0].metadata["atagia_requested_temperature"] == 0.0
    assert (
        provider.requests[0].metadata["atagia_temperature_source"]
        == "request+minimum_floor"
    )


@pytest.mark.asyncio
async def test_complete_applies_gemini_purpose_temperature_then_model_floor() -> None:
    provider = RecordingProvider("gemini")
    client = LLMClient(providers=[provider])

    await client.complete(
        _request().model_copy(
            update={
                "model": "google/gemini-3.1-flash-lite",
                "metadata": {"purpose": "memory_extraction"},
            }
        )
    )

    assert provider.requests[0].temperature == 1.0
    assert (
        provider.requests[0].metadata["atagia_temperature_source"]
        == "purpose_default+model_floor"
    )
    assert (
        provider.requests[0].metadata["atagia_temperature_reason"]
        == "google/gemini-3.1-flash-lite"
    )


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
            update={
                "model": "openrouter/google/gemini-3.1-flash-lite",
                "temperature": 0.0,
            }
        )
    )

    assert provider.requests[0].model == "google/gemini-3.1-flash-lite"
    assert provider.requests[0].temperature == 1.0
    assert (
        provider.requests[0].metadata["atagia_temperature_source"]
        == "request+minimum_floor+model_floor"
    )
    assert provider.requests[0].metadata["provider_extra_body"] == {"reasoning": {"effort": "minimal"}}


@pytest.mark.asyncio
async def test_complete_applies_local_qwen_sampling_profile() -> None:
    provider = RecordingProvider("openai")
    client = LLMClient(providers=[provider])

    await client.complete(_request().model_copy(update={"model": "openai/qwen3-coder:30b"}))

    assert provider.requests[0].model == "qwen3-coder:30b"
    assert provider.requests[0].temperature == 0.7
    assert (
        provider.requests[0].metadata["atagia_temperature_source"]
        == "model_profile_default"
    )
    assert provider.requests[0].metadata["provider_extra_body"] == {
        "top_p": 0.8,
        "top_k": 20,
    }


@pytest.mark.asyncio
async def test_complete_applies_anthropic_opus_47_effort_profile() -> None:
    provider = RecordingProvider("anthropic")
    client = LLMClient(providers=[provider])

    await client.complete(_request().model_copy(update={"model": "anthropic/claude-opus-4-7,high"}))

    assert provider.requests[0].model == "claude-opus-4-7"
    assert provider.requests[0].metadata["anthropic_thinking_adaptive"] is True
    assert provider.requests[0].metadata["anthropic_output_effort"] == "high"


@pytest.mark.asyncio
async def test_complete_applies_openrouter_gpt55_reasoning_profile() -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(_request().model_copy(update={"model": "openrouter/openai/gpt-5.5,high"}))

    assert provider.requests[0].model == "openai/gpt-5.5"
    assert provider.requests[0].metadata["provider_extra_body"] == {"reasoning": {"effort": "high"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
        "openrouter/mistralai/mistral-small-3.2-24b-instruct",
    ],
)
async def test_complete_applies_openrouter_native_structured_profile(
    model: str,
) -> None:
    provider = RecordingProvider("openrouter")
    client = LLMClient(providers=[provider])

    await client.complete(_request().model_copy(update={"model": model}))

    assert provider.requests[0].metadata["openrouter_native_structured_output"] is True
    if model.startswith("openrouter/qwen/"):
        assert provider.requests[0].temperature == 0.7
        assert provider.requests[0].metadata["provider_extra_body"] == {
            "top_p": 0.8,
            "top_k": 20,
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


@pytest.mark.asyncio
async def test_complete_streamed_propagates_error_after_done_event() -> None:
    provider = PostDoneLimitProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    with pytest.raises(OutputLimitExceededError, match="max output tokens"):
        await client.complete_streamed(_request())
    assert provider.stream_calls == 2


@pytest.mark.asyncio
async def test_complete_recovers_once_from_output_limit_by_default() -> None:
    provider = OutputLimitThenSuccessProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    response = await client.complete(_request())

    assert response.output_text == "recovered"
    assert len(provider.requests) == 2
    retry_request = provider.requests[1]
    assert retry_request.metadata["atagia_technical_recovery_retry"] is True
    assert retry_request.metadata["atagia_technical_recovery_retry_attempt"] == 1
    assert "output limit" in retry_request.messages[-1].content
    assert response.raw_response["atagia_technical_recovery"]["operation"] == "completion"
    assert response.raw_response["atagia_technical_recovery"]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_complete_can_disable_technical_recovery_for_raw_provider_checks() -> None:
    provider = OutputLimitThenSuccessProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
    )

    with pytest.raises(OutputLimitExceededError, match="max output tokens"):
        await client.complete(_request())
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_complete_streamed_recovers_from_mechanical_runaway_when_enabled() -> None:
    provider = RunawayThenSuccessProvider()
    run_counters = RunCounterAccumulator()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        technical_recovery_config=LLMTechnicalRecoveryConfig(
            enabled=True,
            output_limit_retry_attempts=1,
            runaway_watchdog_enabled=True,
            runaway_min_elapsed_seconds=0.0,
            runaway_min_output_tokens=8,
            runaway_check_interval_tokens=4,
            runaway_max_checks=3,
            runaway_hard_abort_min_output_tokens=24,
            runaway_min_repeat_count=2,
            runaway_min_repeat_ratio_tokens=0.05,
        ),
    )

    with use_run_counter_accumulator(run_counters):
        response = await client.complete_streamed(_request())

    assert response.output_text == "stream recovered"
    assert provider.stream_calls == 2
    assert provider.requests[1].metadata["atagia_technical_recovery_retry"] is True
    recovery = response.raw_response["atagia_technical_recovery"]
    assert recovery["operation"] == "streamed_completion"
    assert recovery["finish_reason"] == "technical_runaway_watchdog"
    assert run_counters.snapshot()["labeled_counts"][
        "mechanical_runaway_abort_count"
    ] == {"layer=generic_runaway_observer|mode=soft_check": 1}


@pytest.mark.asyncio
async def test_complete_streamed_observer_sees_only_content_not_thinking() -> None:
    provider = ThinkingAndContentProvider()
    observer = RecordingObserver()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
    )

    response = await client.complete_streamed(_request(), observer=observer)

    assert response.output_text == "visible one visible two"
    assert response.thinking == "hidden chain hidden chain more hidden reasoning "
    assert observer.accumulated_texts == ["visible one ", "visible one visible two"]


@pytest.mark.asyncio
async def test_complete_streamed_retries_output_limit_with_external_observer() -> None:
    provider = PostDoneLimitThenSuccessProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    response = await client.complete_streamed(_request(), observer=NoopObserver())

    assert response.output_text == "observer recovered"
    assert provider.stream_calls == 2
    recovery = response.raw_response["atagia_technical_recovery"]
    assert recovery["operation"] == "streamed_completion"
    assert recovery["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_complete_streamed_closes_stream_on_observer_abort() -> None:
    provider = ClosingStreamProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    with pytest.raises(RuntimeError, match="observer abort"):
        await client.complete_streamed(_request(), observer=AbortObserver())

    assert provider.closed is True


@pytest.mark.asyncio
async def test_complete_structured_streamed_parses_json_payload() -> None:
    provider = JsonProvider('{"label":"ok","score":9}')
    client = LLMClient(provider_name="json", providers=[provider])

    payload = await client.complete_structured_streamed(_request(), StructuredPayload)

    assert payload.label == "ok"
    assert payload.score == 9


@pytest.mark.asyncio
async def test_complete_structured_streamed_preserves_schema_fallback() -> None:
    provider = GrammarFallbackProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])

    payload = await client.complete_structured_streamed(
        _request().model_copy(update={"response_schema": StructuredPayload.model_json_schema()}),
        StructuredPayload,
    )

    assert payload.label == "ok"
    assert payload.score == 9
    assert provider.seen_response_schema == [True, False]


def test_output_limit_exceeded_error_is_subclass_of_llm_error() -> None:
    error = OutputLimitExceededError("hit max output tokens")

    assert isinstance(error, LLMError)
    assert isinstance(error, RuntimeError)
    assert not isinstance(error, TransientLLMError)
    assert not isinstance(error, StructuredOutputError)
    assert str(error) == "hit max output tokens"


class AlwaysTransientProvider(LLMProvider):
    name = "always-transient"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        raise TransientLLMError("temporary failure")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1])],
        )


def _purpose_request(purpose: str) -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="model-test",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={"purpose": purpose},
    )


@pytest.mark.asyncio
async def test_interactive_purpose_uses_short_retry_policy() -> None:
    provider = AlwaysTransientProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=RetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0),
    )

    with pytest.raises(TransientLLMError):
        await client.complete(_purpose_request("need_detection"))

    # Interactive purpose caps at 2 attempts regardless of the base policy.
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_non_interactive_purpose_keeps_injected_retry_policy() -> None:
    provider = AlwaysTransientProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=RetryPolicy(attempts=4, base_delay_seconds=0, max_delay_seconds=0),
    )

    with pytest.raises(TransientLLMError):
        await client.complete(_purpose_request("memory_extraction"))

    # Non-interactive purpose uses the injected base policy (4 attempts).
    assert provider.calls == 4


@pytest.mark.asyncio
async def test_unset_purpose_keeps_injected_retry_policy() -> None:
    provider = AlwaysTransientProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=RetryPolicy(attempts=4, base_delay_seconds=0, max_delay_seconds=0),
    )

    with pytest.raises(TransientLLMError):
        await client.complete(_request())

    assert provider.calls == 4


def test_default_interactive_retry_policy_values() -> None:
    from atagia.services.llm_client import _DEFAULT_INTERACTIVE_RETRY_POLICY

    assert _DEFAULT_INTERACTIVE_RETRY_POLICY.attempts == 2
    assert _DEFAULT_INTERACTIVE_RETRY_POLICY.max_delay_seconds == 1.5


def test_retry_policy_for_resolves_interactive_and_default() -> None:
    base = RetryPolicy(attempts=4, base_delay_seconds=0, max_delay_seconds=0)
    interactive = RetryPolicy(attempts=2, max_delay_seconds=1.5)
    client = LLMClient(
        provider_name="stub",
        providers=[RecordingProvider("stub")],
        retry_policy=base,
        interactive_retry_policy=interactive,
    )

    assert client._retry_policy_for(_purpose_request("applicability_scoring")) is interactive
    assert client._retry_policy_for(_purpose_request("context_cache_signal_detection")) is interactive
    assert client._retry_policy_for(_purpose_request("coverage_expansion")) is interactive
    assert client._retry_policy_for(_purpose_request("memory_extraction")) is base
    assert client._retry_policy_for(_request()) is base
