"""Tests for the Anthropic provider adapter."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMError,
    LLMMessage,
    LLMToolSpec,
    OutputLimitExceededError,
    TransientLLMError,
)
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
        max_output_tokens=512,
        include_thinking=True,
        metadata={"anthropic_prompt_cache": True, "thinking_budget_tokens": -1},
        tools=[LLMToolSpec(name="lookup", description="Lookup data", input_schema={"type": "object"})],
        response_schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )


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
    assert messages.stream_calls[0]["output_format"]["type"] == "json_schema"


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
    stream_schema = stream_call["output_format"]["schema"]
    assert completion_call["max_tokens"] == 8192
    assert stream_call["max_tokens"] == 8192
    assert completion.usage == {"input_tokens": 11, "output_tokens": 7}
    assert events[-1].payload["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert completion_schema["additionalProperties"] is False
    assert completion_schema["properties"]["nested"]["additionalProperties"] is False
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
    with pytest.raises(LLMError):
        await permanent_provider.complete(request)
