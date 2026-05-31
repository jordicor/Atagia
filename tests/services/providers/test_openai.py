"""Tests for the OpenAI provider adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMEmbeddingRequest,
    LLMError,
    LLMMessage,
    LLMRequestError,
    LLMToolSpec,
    OutputLimitExceededError,
    TransientLLMError,
)
from atagia.services.providers.openai import OpenAICompatibleProvider, OpenAIProvider
from atagia.services.providers.openrouter import OpenRouterProvider


class FakeChatCompletions:
    def __init__(self, create_result=None, error=None) -> None:
        self.create_result = create_result
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.create_result


class FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class FakeRaisingStream:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        raise self.error
        yield

    async def aclose(self):
        self.closed = True


class FakeEmbeddings:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions, embeddings: FakeEmbeddings) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.embeddings = embeddings


def _request(model: str = "gpt-5-mini") -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="system", content="You are helpful."), LLMMessage(role="user", content="Hello")],
        max_output_tokens=8192,
        temperature=0.2,
        response_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "notes": {
                    "type": "object",
                    "default": {},
                    "properties": {
                        "summary": {"type": "string"},
                    },
                },
                "status": {"$ref": "#/$defs/Status", "default": "ok"},
            },
            "$defs": {"Status": {"type": "string", "enum": ["ok", "warning"]}},
        },
        tools=[LLMToolSpec(name="lookup", description="Lookup data", input_schema={"type": "object"})],
        metadata={"user_id": "usr_1"},
    )


def _api_error(message: str) -> openai.APIError:
    return openai.APIError(
        message,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body=None,
    )


async def _collect_stream_events_and_error(provider: OpenAIProvider):
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
async def test_openai_complete_maps_response_and_uses_structured_output() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello world",
                    tool_calls=[
                        SimpleNamespace(
                            id="tool_1",
                            type="function",
                            function=SimpleNamespace(name="lookup", arguments='{"q":"x"}'),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(model_dump=lambda exclude_none=True: {"prompt_tokens": 11, "completion_tokens": 7}),
        model_dump=lambda: {"id": "cmpl_1"},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))

    completion = await provider.complete(_request())

    assert completion.output_text == "hello world"
    assert completion.tool_calls[0]["name"] == "lookup"
    call = completions.calls[0]
    assert call["max_completion_tokens"] == 8192
    assert "max_tokens" not in call
    assert call["response_format"]["type"] == "json_schema"
    schema = call["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["label", "notes", "status"]
    assert schema["properties"]["notes"]["additionalProperties"] is False
    assert schema["properties"]["notes"]["required"] == ["summary"]
    assert "default" not in schema["properties"]["notes"]
    assert "default" not in schema["properties"]["status"]
    assert call["tools"][0]["function"]["name"] == "lookup"
    assert call["user"] == "usr_1"


@pytest.mark.asyncio
async def test_openai_preserves_nullable_optionals_for_strict_structured_output() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"ok"}'))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))
    request = _request().model_copy(
        update={
            "response_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "note": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
            },
        }
    )

    await provider.complete(request)

    response_format = completions.calls[0]["response_format"]
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["label", "note"]
    assert schema["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


@pytest.mark.asyncio
async def test_openai_compatible_provider_strips_nullable_by_default() -> None:
    response = SimpleNamespace(
        model="local-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"ok"}'))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAICompatibleProvider(
        api_key="test",
        client=FakeOpenAIClient(completions, FakeEmbeddings()),
    )
    request = _request(model="local-model").model_copy(
        update={
            "response_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "note": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
            },
        }
    )

    await provider.complete(request)

    response_format = completions.calls[0]["response_format"]
    assert response_format["json_schema"]["strict"] is False
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["note"] == {"type": "string"}


@pytest.mark.asyncio
async def test_openai_complete_preserves_native_tool_call_history() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))
    request = _request().model_copy(
        update={
            "messages": [
                LLMMessage(role="user", content="Use the lookup tool."),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_lookup",
                            "type": "tool_use",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"query\":\"atagia\"}",
                            },
                        }
                    ],
                ),
                LLMMessage(
                    role="tool",
                    content="{\"result\":\"ok\"}",
                    name="call_lookup",
                ),
            ],
            "response_schema": None,
        }
    )

    await provider.complete(request)

    messages = completions.calls[0]["messages"]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_lookup",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{\"query\":\"atagia\"}",
            },
        }
    ]
    assert messages[2]["tool_call_id"] == "call_lookup"


@pytest.mark.asyncio
async def test_openai_complete_forwards_provider_extra_body_from_metadata() -> None:
    """Metadata['provider_extra_body'] is forwarded as `extra_body` to the SDK,
    so OpenRouter callers can set the rich `reasoning` object and other
    provider-specific fields not exposed by the OpenAI SDK directly.
    """
    response = SimpleNamespace(
        model="deepseek/deepseek-v4-flash",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))

    request = _request(model="deepseek/deepseek-v4-flash").model_copy(
        update={
            "metadata": {
                "provider_extra_body": {"reasoning": {"effort": "none"}},
            },
        }
    )
    await provider.complete(request)

    call = completions.calls[0]
    assert call["extra_body"] == {"reasoning": {"effort": "none"}}


@pytest.mark.asyncio
async def test_openrouter_openai_reasoning_model_omits_temperature() -> None:
    response = SimpleNamespace(
        model="openai/gpt-5.5",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://example.test",
        app_name="Atagia",
        client=FakeOpenAIClient(completions, FakeEmbeddings()),
    )
    request = _request(model="openai/gpt-5.5").model_copy(
        update={
            "metadata": {"provider_extra_body": {"reasoning": {"effort": "high"}}},
        }
    )

    await provider.complete(request)

    call = completions.calls[0]
    assert "temperature" not in call
    assert call["max_tokens"] == 8192
    assert "max_completion_tokens" not in call
    assert call["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"require_parameters": True},
    }


@pytest.mark.asyncio
async def test_openai_chat_latest_uses_completion_tokens_and_omits_temperature() -> None:
    response = SimpleNamespace(
        model="chat-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))

    await provider.complete(_request(model="chat-latest"))

    call = completions.calls[0]
    assert "temperature" not in call
    assert call["max_completion_tokens"] == 8192
    assert "max_tokens" not in call


@pytest.mark.asyncio
async def test_openai_complete_falls_back_to_request_model_when_response_model_is_null() -> None:
    response = SimpleNamespace(
        model=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
        usage=None,
        model_dump=lambda: {"id": "cmpl_1", "model": None},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))

    completion = await provider.complete(_request(model="qwen/qwen3.6-plus"))

    assert completion.model == "qwen/qwen3.6-plus"
    assert completion.output_text == "hello world"


@pytest.mark.asyncio
async def test_openai_complete_maps_reasoning_content_alias() -> None:
    response = SimpleNamespace(
        model="qwen/qwen3.6-plus",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello world",
                    reasoning_content="short hidden reasoning",
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "cmpl_1"},
    )
    completions = FakeChatCompletions(create_result=response)
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(completions, FakeEmbeddings()))

    completion = await provider.complete(_request(model="qwen/qwen3.6-plus"))

    assert completion.thinking == "short hidden reasoning"


@pytest.mark.asyncio
async def test_openai_complete_raises_non_transient_on_length_finish_reason() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="partial"),
                finish_reason="length",
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "cmpl_1"},
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(response), FakeEmbeddings()),
    )

    with pytest.raises(OutputLimitExceededError, match="max output tokens") as exc_info:
        await provider.complete(_request())

    assert isinstance(exc_info.value, LLMError)
    assert not isinstance(exc_info.value, TransientLLMError)
    assert exc_info.value.finish_reason == "length"
    assert exc_info.value.partial_output_chars == len("partial")
    assert exc_info.value.partial_output_excerpt == "partial"


@pytest.mark.asyncio
async def test_openai_complete_raises_on_content_filter_finish_reason() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="content_filter",
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "cmpl_1"},
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(response), FakeEmbeddings()),
    )

    with pytest.raises(LLMError, match="blocked") as exc_info:
        await provider.complete(_request())

    assert not isinstance(exc_info.value, TransientLLMError)


@pytest.mark.asyncio
async def test_openai_compatible_complete_raises_transient_on_error_finish_reason() -> None:
    response = SimpleNamespace(
        model="gpt-5-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="error",
                error=SimpleNamespace(message="upstream failed"),
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "cmpl_1"},
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(response), FakeEmbeddings()),
    )

    with pytest.raises(TransientLLMError, match="upstream failed"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_openai_stream_emits_done_then_raises_on_length_finish_reason() -> None:
    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="partial", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage={"prompt_tokens": 4, "completion_tokens": 2},
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="length",
                )
            ],
        ),
    ]
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(
            FakeChatCompletions(FakeStream(chunks)),
            FakeEmbeddings(),
        ),
    )

    received, raised = await _collect_stream_events_and_error(provider)

    assert [event.type for event in received] == ["text", "done"]
    assert received[0].content == "partial"
    assert received[1].payload["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
    }
    assert isinstance(raised, OutputLimitExceededError)
    assert isinstance(raised, LLMError)
    assert not isinstance(raised, TransientLLMError)
    assert "max output tokens" in str(raised)
    assert raised.finish_reason == "length"
    assert raised.partial_output_chars == len("partial")
    assert raised.partial_output_excerpt == "partial"


@pytest.mark.asyncio
async def test_openai_stream_closes_underlying_stream_on_cancel() -> None:
    stream = FakeStream(
        [
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial", tool_calls=None),
                        finish_reason=None,
                    )
                ],
            )
        ]
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(stream), FakeEmbeddings()),
    )

    iterator = provider.stream(_request()).__aiter__()
    first = await anext(iterator)
    await iterator.aclose()

    assert first.type == "text"
    assert stream.closed is True


@pytest.mark.asyncio
async def test_openai_embed_maps_vectors() -> None:
    embeddings = FakeEmbeddings(
        result=SimpleNamespace(
            model="text-embedding-3-small",
            data=[
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            ],
            model_dump=lambda: {"id": "emb_1"},
        )
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(), embeddings),
    )

    response = await provider.embed(
        LLMEmbeddingRequest(
            model="text-embedding-3-small",
            input_texts=["a", "b"],
            dimensions=1536,
            metadata={"user_id": "usr_1"},
        )
    )

    assert [vector.values for vector in response.vectors] == [[0.1, 0.2], [0.3, 0.4]]
    assert embeddings.calls[0]["dimensions"] == 1536
    assert embeddings.calls[0]["user"] == "usr_1"


@pytest.mark.asyncio
async def test_openai_embed_can_use_separate_embedding_base_url(monkeypatch) -> None:
    created_clients: list[SimpleNamespace] = []

    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str | None,
            default_headers: dict[str, str] | None,
            max_retries: int,
        ) -> None:
            response = SimpleNamespace(
                model="gpt-5-mini",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
                model_dump=lambda: {},
            )
            embedding_response = SimpleNamespace(
                model="qwen3-embedding:4b",
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                model_dump=lambda: {},
            )
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers
            self.max_retries = max_retries
            self.chat = SimpleNamespace(
                completions=FakeChatCompletions(create_result=response)
            )
            self.embeddings = FakeEmbeddings(result=embedding_response)
            created_clients.append(self)

    monkeypatch.setattr(
        "atagia.services.providers.openai.AsyncOpenAI",
        FakeAsyncOpenAI,
    )
    provider = OpenAIProvider(
        api_key="ollama",
        base_url="http://4090.test/v1",
        embedding_base_url="http://4080.test/v1",
    )

    await provider.complete(_request(model="qwen3-coder:30b"))
    await provider.embed(
        LLMEmbeddingRequest(
            model="qwen3-embedding:4b",
            input_texts=["hello"],
            dimensions=1536,
        )
    )

    assert [client.base_url for client in created_clients] == [
        "http://4090.test/v1",
        "http://4080.test/v1",
    ]
    assert len(created_clients[0].chat.completions.calls) == 1
    assert created_clients[0].embeddings.calls == []
    assert len(created_clients[1].embeddings.calls) == 1
    assert created_clients[1].chat.completions.calls == []


@pytest.mark.asyncio
async def test_openai_embed_falls_back_to_request_model_when_response_model_is_null() -> None:
    embeddings = FakeEmbeddings(
        result=SimpleNamespace(
            model=None,
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            model_dump=lambda: {"id": "emb_1", "model": None},
        )
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(), embeddings),
    )

    response = await provider.embed(
        LLMEmbeddingRequest(
            model="openrouter-embedding-model",
            input_texts=["a"],
            dimensions=None,
            metadata={},
        )
    )

    assert response.model == "openrouter-embedding-model"


@pytest.mark.asyncio
async def test_openai_maps_retryable_and_permanent_errors() -> None:
    transient_error = openai.APIConnectionError(
        message="boom",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    rate_limit_error = openai.APIStatusError(
        "rate limited",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )
    permanent_error = openai.BadRequestError(
        "bad request",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body={},
    )

    transient_provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(error=transient_error), FakeEmbeddings()),
    )
    permanent_provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(error=permanent_error), FakeEmbeddings()),
    )
    rate_limit_provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(error=rate_limit_error), FakeEmbeddings()),
    )

    with pytest.raises(TransientLLMError):
        await transient_provider.complete(_request())
    with pytest.raises(TransientLLMError):
        await rate_limit_provider.complete(_request())
    with pytest.raises(LLMRequestError) as exc_info:
        await permanent_provider.complete(_request())
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value, LLMError)


@pytest.mark.asyncio
async def test_openai_maps_non_transient_4xx_status_error_to_request_error() -> None:
    not_found_error = openai.APIStatusError(
        "not found",
        response=httpx.Response(
            404,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(error=not_found_error), FakeEmbeddings()),
    )

    with pytest.raises(LLMRequestError) as exc_info:
        await provider.complete(_request())
    assert exc_info.value.status_code == 404
    assert not isinstance(exc_info.value, TransientLLMError)


@pytest.mark.asyncio
async def test_openrouter_maps_bad_request_to_request_error() -> None:
    bad_request = openai.BadRequestError(
        "bad request",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        ),
        body={},
    )
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://example.test",
        app_name="Atagia",
        client=FakeOpenAIClient(FakeChatCompletions(error=bad_request), FakeEmbeddings()),
    )

    with pytest.raises(LLMRequestError) as exc_info:
        await provider.complete(_request(model="deepseek/deepseek-v4-flash"))
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_openai_maps_injected_sse_api_error_as_transient() -> None:
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(
            FakeChatCompletions(
                error=_api_error("JSON error injected into SSE stream")
            ),
            FakeEmbeddings(),
        ),
    )

    with pytest.raises(TransientLLMError, match="JSON error injected"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_openai_stream_maps_injected_sse_api_error_as_transient() -> None:
    stream = FakeRaisingStream(_api_error("JSON error injected into SSE stream"))
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(FakeChatCompletions(stream), FakeEmbeddings()),
    )

    received, raised = await _collect_stream_events_and_error(provider)

    assert received == []
    assert isinstance(raised, TransientLLMError)
    assert "JSON error injected" in str(raised)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_openai_maps_non_json_provider_response_as_transient() -> None:
    provider = OpenAIProvider(
        api_key="test",
        client=FakeOpenAIClient(
            FakeChatCompletions(
                error=json.JSONDecodeError("Expecting value", "<html>gateway error</html>", 0)
            ),
            FakeEmbeddings(),
        ),
    )

    with pytest.raises(TransientLLMError):
        await provider.complete(_request())
