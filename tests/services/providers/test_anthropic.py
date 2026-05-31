"""Tests for the Anthropic provider adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMError,
    LLMMessage,
    LLMRequestError,
    LLMToolSpec,
    OutputLimitExceededError,
    TransientLLMError,
)
from atagia.models.schemas_memory import ExtractionResult
from atagia.services.providers.anthropic import AnthropicProvider


class FakeAnthropicMessages:
    def __init__(self, completion_response=None, stream_manager=None, error=None) -> None:
        self.completion_response = completion_response
        self.stream_manager = stream_manager
        self.error = error
        self.create_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.completion_response

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream_manager


class FakeAnthropicClient:
    def __init__(self, messages: FakeAnthropicMessages) -> None:
        self.messages = messages


class FakeStreamManager:
    def __init__(self, events, final_message) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="claude-opus-4-6",
        messages=[
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Summarize this."),
        ],
        max_output_tokens=8192,
        include_thinking=True,
        metadata={"anthropic_prompt_cache": True, "thinking_budget_tokens": -1},
        tools=[LLMToolSpec(name="lookup", description="Lookup data", input_schema={"type": "object"})],
        response_schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )


def test_anthropic_uses_prompt_fallback_for_large_native_schema() -> None:
    provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages()),
    )
    request = _request().model_copy(
        update={"response_schema": ExtractionResult.model_json_schema()}
    )

    assert provider.supports_native_structured_output_for(request) is False


def test_anthropic_keeps_native_schema_for_small_schema() -> None:
    provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages()),
    )

    assert provider.supports_native_structured_output_for(_request()) is True


def _schema_just_over_cap_unstripped_under_when_stripped() -> dict:
    """Build a schema that exceeds the cap unstripped but fits once nullability is stripped.

    Nullable optional fields emit ``anyOf: [{...}, {"type": "null"}]`` plus a
    ``default: null``; ``strip_json_schema_nullability`` collapses those, so the
    sent payload is materially smaller than the raw schema. The gate must decide
    on the sent (stripped) payload, not the raw one.
    """
    from atagia.services.llm_schema import strip_json_schema_nullability
    from atagia.services.providers.anthropic import (
        _ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS,
        _sanitize_schema,
    )

    cap = _ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS
    for field_count in range(40, 400):
        properties = {
            f"field_{index:03d}": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": f"Field {index:03d}",
            }
            for index in range(field_count)
        }
        schema = {"type": "object", "properties": properties}
        unstripped_len = len(
            json.dumps(_sanitize_schema(schema), separators=(",", ":"), sort_keys=True)
        )
        stripped_len = len(
            json.dumps(
                _sanitize_schema(strip_json_schema_nullability(schema)),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if stripped_len <= cap < unstripped_len:
            return schema
    raise AssertionError("could not craft a schema straddling the native-schema cap")


def test_anthropic_gate_decides_on_sent_payload_not_raw_schema() -> None:
    provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages()),
    )
    schema = _schema_just_over_cap_unstripped_under_when_stripped()
    request = _request().model_copy(update={"response_schema": schema})

    # The raw schema is over the cap; the stripped payload (what is sent) is under it.
    assert provider.supports_native_structured_output_for(request) is True


def test_anthropic_gate_matches_sent_output_config_payload() -> None:
    from atagia.services.providers.anthropic import (
        _ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS,
        _output_config,
    )

    provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages()),
    )
    schema = _schema_just_over_cap_unstripped_under_when_stripped()
    request = _request().model_copy(update={"response_schema": schema})

    gate = provider.supports_native_structured_output_for(request)
    sent_schema = _output_config(request)["format"]["schema"]
    sent_len = len(json.dumps(sent_schema, separators=(",", ":"), sort_keys=True))

    assert gate is True
    assert sent_len <= _ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS


@pytest.mark.asyncio
async def test_anthropic_stream_uses_output_config_with_format_and_effort() -> None:
    final_message = SimpleNamespace(stop_reason="end_turn", usage=None)
    stream_manager = FakeStreamManager(
        events=[SimpleNamespace(type="text", text="ok")],
        final_message=final_message,
    )
    messages = FakeAnthropicMessages(stream_manager=stream_manager)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))
    request = _request().model_copy(
        update={"metadata": {"anthropic_output_effort": "high"}}
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == ["text", "done"]
    stream_call = messages.stream_calls[0]
    assert "output_format" not in stream_call
    output_config = stream_call["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["effort"] == "high"


def test_anthropic_provider_passes_timeout_to_sdk(monkeypatch) -> None:
    captured: dict = {}

    def fake_anthropic_client(**kwargs):
        captured.update(kwargs)
        return FakeAnthropicClient(FakeAnthropicMessages())

    monkeypatch.setattr(
        "atagia.services.providers.anthropic.AsyncAnthropic",
        fake_anthropic_client,
    )

    AnthropicProvider(api_key="test", request_timeout_seconds=45.5)

    assert captured["timeout"] == 45.5


async def _collect_stream_events_and_error(provider: AnthropicProvider):
    received: list = []
    raised: Exception | None = None
    iterator = provider.stream(_request()).__aiter__()
    while True:
        try:
            received.append(await iterator.__anext__())
        except StopAsyncIteration:
            break
        except Exception as exc:
            raised = exc
            break
    return received, raised


@pytest.mark.asyncio
async def test_anthropic_complete_maps_blocks_and_request_shape() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[
            SimpleNamespace(type="thinking", thinking="internal"),
            SimpleNamespace(type="text", text="final answer"),
            SimpleNamespace(type="tool_use", id="tool_1", name="lookup", input={"q": "x"}),
        ],
        usage=SimpleNamespace(model_dump=lambda exclude_none=True: {"input_tokens": 11, "output_tokens": 7}),
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    completion = await provider.complete(_request())

    assert completion.output_text == "final answer"
    assert completion.thinking == "internal"
    assert completion.tool_calls == [{"id": "tool_1", "type": "tool_use", "name": "lookup", "input": {"q": "x"}}]
    create_call = messages.create_calls[0]
    assert create_call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert create_call["thinking"] == {"type": "adaptive"}
    assert create_call["tools"][0]["name"] == "lookup"
    assert create_call["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_anthropic_complete_falls_back_to_request_model_when_response_model_is_null() -> None:
    response = SimpleNamespace(
        model=None,
        content=[SimpleNamespace(type="text", text="final answer")],
        usage=None,
        model_dump=lambda: {"id": "msg_1", "model": None},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    completion = await provider.complete(_request())

    assert completion.model == "claude-opus-4-6"
    assert completion.output_text == "final answer"


@pytest.mark.asyncio
async def test_anthropic_complete_omits_temperature_for_opus_4_7() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-7",
        content=[SimpleNamespace(type="text", text="judged")],
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))
    request = _request().model_copy(
        update={
            "model": "claude-opus-4-7",
            "temperature": 0.0,
        }
    )

    completion = await provider.complete(request)

    assert completion.output_text == "judged"
    assert "temperature" not in messages.create_calls[0]


@pytest.mark.asyncio
async def test_anthropic_complete_omits_temperature_for_opus_4_8() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-8",
        content=[SimpleNamespace(type="text", text="judged")],
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))
    request = _request().model_copy(
        update={
            "model": "anthropic/claude-opus-4-8",
            "temperature": 0.0,
        }
    )

    completion = await provider.complete(request)

    assert completion.output_text == "judged"
    assert "temperature" not in messages.create_calls[0]


@pytest.mark.asyncio
async def test_anthropic_complete_sends_adaptive_effort_for_opus_4_7() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-7",
        content=[SimpleNamespace(type="text", text="judged")],
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))
    request = _request().model_copy(
        update={
            "model": "claude-opus-4-7",
            "metadata": {
                "anthropic_thinking_adaptive": True,
                "anthropic_output_effort": "xhigh",
            },
        }
    )

    completion = await provider.complete(request)

    assert completion.output_text == "judged"
    create_call = messages.create_calls[0]
    assert create_call["thinking"] == {"type": "adaptive"}
    assert create_call["output_config"]["effort"] == "xhigh"


@pytest.mark.asyncio
async def test_anthropic_stream_omits_temperature_for_opus_4_7() -> None:
    final_message = SimpleNamespace(stop_reason="end_turn", usage=None)
    stream_manager = FakeStreamManager(
        events=[SimpleNamespace(type="text", text="judged")],
        final_message=final_message,
    )
    messages = FakeAnthropicMessages(stream_manager=stream_manager)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))
    request = _request().model_copy(
        update={
            "model": "anthropic/claude-opus-4-7",
            "temperature": 0.0,
        }
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == ["text", "done"]
    assert "temperature" not in messages.stream_calls[0]


@pytest.mark.asyncio
async def test_anthropic_complete_raises_non_transient_on_max_tokens() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[SimpleNamespace(type="text", text="partial")],
        stop_reason="max_tokens",
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    with pytest.raises(OutputLimitExceededError, match="max output tokens") as exc_info:
        await provider.complete(_request())

    assert isinstance(exc_info.value, LLMError)
    assert not isinstance(exc_info.value, TransientLLMError)


@pytest.mark.asyncio
async def test_anthropic_complete_raises_transient_on_pause_turn() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[SimpleNamespace(type="text", text="partial")],
        stop_reason="pause_turn",
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    with pytest.raises(TransientLLMError, match="pause_turn"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_anthropic_complete_raises_on_refusal() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[],
        stop_reason="refusal",
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    with pytest.raises(LLMError, match="blocked") as exc_info:
        await provider.complete(_request())

    assert not isinstance(exc_info.value, TransientLLMError)


@pytest.mark.asyncio
async def test_anthropic_complete_raises_transient_on_empty_end_turn() -> None:
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {"id": "msg_1"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    with pytest.raises(TransientLLMError, match="end_turn"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_anthropic_stream_maps_text_thinking_and_tool_calls() -> None:
    final_message = SimpleNamespace(
        usage=SimpleNamespace(model_dump=lambda exclude_none=True: {"input_tokens": 3, "output_tokens": 5})
    )
    stream_manager = FakeStreamManager(
        events=[
            SimpleNamespace(type="thinking", thinking="step"),
            SimpleNamespace(type="text", text="answer"),
            SimpleNamespace(
                type="content_block_stop",
                content_block=SimpleNamespace(type="tool_use", id="tool_1", name="lookup", input={"q": "x"}),
            ),
        ],
        final_message=final_message,
    )
    messages = FakeAnthropicMessages(stream_manager=stream_manager)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == ["thinking", "text", "tool_call", "done"]
    assert events[0].content == "step"
    assert events[1].content == "answer"
    assert events[2].payload["name"] == "lookup"
    assert events[3].payload["usage"] == {"input_tokens": 3, "output_tokens": 5}
    stream_call = messages.stream_calls[0]
    assert "output_format" not in stream_call
    assert stream_call["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_anthropic_stream_buffers_sdk_tool_use_events() -> None:
    final_message = SimpleNamespace(
        stop_reason="tool_use",
        usage=SimpleNamespace(model_dump=lambda exclude_none=True: {"input_tokens": 3, "output_tokens": 5}),
    )
    stream_manager = FakeStreamManager(
        events=[
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use",
                    id="tool_1",
                    name="lookup",
                    input={},
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta",
                    partial_json='{"q":"x"}',
                ),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
        ],
        final_message=final_message,
    )
    messages = FakeAnthropicMessages(stream_manager=stream_manager)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == ["tool_call", "done"]
    assert events[0].payload == {
        "id": "tool_1",
        "type": "tool_use",
        "name": "lookup",
        "input": {"q": "x"},
    }
    assert events[1].payload["usage"] == {"input_tokens": 3, "output_tokens": 5}


@pytest.mark.asyncio
async def test_anthropic_stream_emits_done_then_raises_on_max_tokens() -> None:
    final_message = SimpleNamespace(
        stop_reason="max_tokens",
        usage=SimpleNamespace(
            model_dump=lambda exclude_none=True: {"input_tokens": 3, "output_tokens": 5}
        ),
    )
    stream_manager = FakeStreamManager(
        events=[SimpleNamespace(type="text", text="partial")],
        final_message=final_message,
    )
    messages = FakeAnthropicMessages(stream_manager=stream_manager)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    received, raised = await _collect_stream_events_and_error(provider)

    assert [event.type for event in received] == ["text", "done"]
    assert received[0].content == "partial"
    assert received[1].payload["usage"] == {"input_tokens": 3, "output_tokens": 5}
    assert isinstance(raised, OutputLimitExceededError)
    assert isinstance(raised, LLMError)
    assert not isinstance(raised, TransientLLMError)
    assert "max output tokens" in str(raised)


@pytest.mark.asyncio
async def test_anthropic_sanitizes_schema_and_filters_non_numeric_usage() -> None:
    request = _request().model_copy(
        update={
            "max_output_tokens": None,
            "response_schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "note": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "nested": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                            "ratio": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "exclusiveMaximum": 1,
                            }
                        },
                    },
                },
            },
        }
    )
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            model_dump=lambda exclude_none=True: {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_creation": {"ephemeral": 1},
                "inference_geo": "global",
                "service_tier": "standard",
            }
        ),
        model_dump=lambda: {"id": "msg_2"},
    )
    final_message = SimpleNamespace(
        usage=SimpleNamespace(
            model_dump=lambda exclude_none=True: {
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation": {"ephemeral": 1},
                "service_tier": "standard",
            }
        )
    )
    stream_manager = FakeStreamManager(events=[], final_message=final_message)
    messages = FakeAnthropicMessages(
        completion_response=response,
        stream_manager=stream_manager,
    )
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    completion = await provider.complete(request)
    events = [event async for event in provider.stream(request)]

    completion_call = messages.create_calls[0]
    stream_call = messages.stream_calls[0]
    completion_schema = completion_call["output_config"]["format"]["schema"]
    assert "output_format" not in stream_call
    stream_schema = stream_call["output_config"]["format"]["schema"]
    assert completion_call["max_tokens"] == 8192
    assert stream_call["max_tokens"] == 8192
    assert completion.usage == {"input_tokens": 11, "output_tokens": 7}
    assert events[-1].payload["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert completion_schema["additionalProperties"] is False
    assert completion_schema["properties"]["nested"]["additionalProperties"] is False
    assert completion_schema["properties"]["note"] == {"type": "string"}
    assert "minimum" not in completion_schema["properties"]["score"]
    assert "maximum" not in completion_schema["properties"]["score"]
    assert "minItems" not in completion_schema["properties"]["nested"]["properties"]["tags"]
    assert "maxItems" not in completion_schema["properties"]["nested"]["properties"]["tags"]
    assert "exclusiveMinimum" not in completion_schema["properties"]["nested"]["properties"]["ratio"]
    assert "exclusiveMaximum" not in completion_schema["properties"]["nested"]["properties"]["ratio"]
    assert stream_schema == completion_schema


@pytest.mark.asyncio
async def test_anthropic_adds_additional_properties_when_omitted() -> None:
    """Pydantic models with `extra="ignore"` emit schemas without
    `additionalProperties`. Anthropic rejects those, so the sanitizer must
    inject `additionalProperties: false` for every object that has properties.
    """
    request = _request().model_copy(
        update={
            "max_output_tokens": None,
            "response_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {
                            "flag": {"type": "boolean"},
                        },
                    },
                },
            },
        }
    )
    response = SimpleNamespace(
        model="claude-opus-4-6",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=None,
        model_dump=lambda: {"id": "msg_3"},
    )
    messages = FakeAnthropicMessages(completion_response=response)
    provider = AnthropicProvider(api_key="test", client=FakeAnthropicClient(messages))

    await provider.complete(request)

    schema = messages.create_calls[0]["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nested"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_anthropic_maps_retryable_and_permanent_errors() -> None:
    request = _request()
    transient_error = anthropic.APIConnectionError(
        message="boom",
        request=httpx.Request("POST", "https://example.com"),
    )
    permanent_error = anthropic.BadRequestError(
        "bad request",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")),
        body={},
    )

    transient_provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages(error=transient_error)),
    )
    permanent_provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages(error=permanent_error)),
    )

    with pytest.raises(TransientLLMError):
        await transient_provider.complete(request)
    with pytest.raises(LLMRequestError) as exc_info:
        await permanent_provider.complete(request)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value, LLMError)


@pytest.mark.asyncio
async def test_anthropic_maps_non_transient_4xx_status_error_to_request_error() -> None:
    request = _request()
    not_found_error = anthropic.APIStatusError(
        "not found",
        response=httpx.Response(404, request=httpx.Request("POST", "https://example.com")),
        body={},
    )
    provider = AnthropicProvider(
        api_key="test",
        client=FakeAnthropicClient(FakeAnthropicMessages(error=not_found_error)),
    )

    with pytest.raises(LLMRequestError) as exc_info:
        await provider.complete(request)
    assert exc_info.value.status_code == 404
    assert not isinstance(exc_info.value, TransientLLMError)
