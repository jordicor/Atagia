"""Tests for the OpenRouter provider adapter and provider factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atagia.core.config import Settings
from atagia.services.llm_client import (
    ConfigurationError,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMMessage,
    LLMProvider,
    TransientLLMError,
)
from atagia.services.providers import build_llm_client
from atagia.services.providers.openrouter import OpenRouterProvider


class FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for chunk in self._chunks:
            yield chunk


class FakeChatCompletions:
    def __init__(self, stream_result) -> None:
        self.stream_result = stream_result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream_result


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.embeddings = SimpleNamespace(create=None)


@pytest.mark.asyncio
async def test_openrouter_stream_maps_text_tool_call_and_done() -> None:
    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hello ", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(name="lookup", arguments='{"q":"hel'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 4},
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                type="function",
                                function=SimpleNamespace(name=None, arguments='lo"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]
    completions = FakeChatCompletions(FakeStream(chunks))
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model="openai/gpt-4.1-mini",
        messages=[LLMMessage(role="user", content="hello")],
        max_output_tokens=128,
    )
    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == ["text", "tool_call", "done"]
    assert events[0].content == "hello "
    assert events[1].payload["arguments"] == '{"q":"hello"}'
    assert events[2].payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    assert completions.calls[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_openrouter_complete_raises_transient_on_error_finish_reason() -> None:
    response = SimpleNamespace(
        model="anthropic/claude-sonnet-4.6",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="error",
                error={"message": "upstream provider failed"},
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model="anthropic/claude-sonnet-4.6",
        messages=[LLMMessage(role="user", content="hello")],
        max_output_tokens=128,
    )

    with pytest.raises(TransientLLMError, match="upstream provider failed"):
        await provider.complete(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-flash-lite",
        "openai/gpt-chat-latest",
        "x-ai/grok-4-fast",
    ],
)
async def test_openrouter_complete_uses_response_format_for_major_structured_vendors(
    model: str,
) -> None:
    response = SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":0.8}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="user", content="Return JSON.")],
        max_output_tokens=128,
        response_schema={
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
    )

    completion = await provider.complete(request)

    assert completion.output_text == '{"score":0.8}'
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    if model.startswith("google/"):
        assert call["response_format"]["json_schema"]["strict"] is False
        schema = call["response_format"]["json_schema"]["schema"]
        assert "required" not in schema
        assert "minimum" not in schema["properties"]["score"]
        assert "maximum" not in schema["properties"]["score"]
    else:
        assert call["response_format"]["json_schema"]["strict"] is True
        assert call["response_format"]["json_schema"]["schema"]["required"] == ["score"]
    assert call["extra_body"]["provider"]["require_parameters"] is True


@pytest.mark.asyncio
async def test_openrouter_strips_nullable_for_google_structured_output() -> None:
    response = SimpleNamespace(
        model="google/gemini-3.1-flash-lite",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"ok"}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )
    request = LLMCompletionRequest(
        model="google/gemini-3.1-flash-lite",
        messages=[LLMMessage(role="user", content="Return JSON.")],
        max_output_tokens=128,
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"$ref": "#/$defs/Status"},
                "label": {"type": "string"},
                "note": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
            },
            "$defs": {"Status": {"type": "string", "enum": ["ok", "warning"]}},
        },
    )

    await provider.complete(request)

    response_format = completions.calls[0]["response_format"]
    assert response_format["json_schema"]["strict"] is False
    schema = response_format["json_schema"]["schema"]
    assert "$defs" not in schema
    assert "additionalProperties" not in schema
    assert schema["properties"]["status"] == {"type": "string", "enum": ["ok", "warning"]}
    assert schema["properties"]["note"] == {"type": "string"}


@pytest.mark.asyncio
async def test_openrouter_preserves_nullable_for_openai_vendor() -> None:
    response = SimpleNamespace(
        model="openai/gpt-chat-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"ok"}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )
    request = LLMCompletionRequest(
        model="openai/gpt-chat-latest",
        messages=[LLMMessage(role="user", content="Return JSON.")],
        max_output_tokens=128,
        response_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "note": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
            },
        },
    )

    await provider.complete(request)

    response_format = completions.calls[0]["response_format"]
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


@pytest.mark.asyncio
async def test_openrouter_complete_preserves_provider_body_when_requiring_schema_support() -> None:
    response = SimpleNamespace(
        model="openai/gpt-chat-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":0.8}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model="openai/gpt-chat-latest",
        messages=[LLMMessage(role="user", content="Return JSON.")],
        response_schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
        },
        metadata={
            "provider_extra_body": {
                "provider": {"sort": "latency", "require_parameters": False},
                "reasoning": {"effort": "low"},
            }
        },
    )

    await provider.complete(request)

    assert completions.calls[0]["extra_body"] == {
        "provider": {"sort": "latency", "require_parameters": True},
        "reasoning": {"effort": "low"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "qwen/qwen3-coder-30b-a3b-instruct",
        "mistralai/mistral-small-3.2-24b-instruct",
    ],
)
async def test_openrouter_profile_flag_uses_native_response_format(
    model: str,
) -> None:
    response = SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":0.8}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="user", content="Return JSON.")],
        max_output_tokens=128,
        response_schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
        },
        metadata={"openrouter_native_structured_output": True},
    )

    await provider.complete(request)

    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["extra_body"]["provider"]["require_parameters"] is True


@pytest.mark.asyncio
async def test_openrouter_openai_chat_latest_omits_unsupported_temperature() -> None:
    response = SimpleNamespace(
        model="openai/gpt-chat-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":0.8}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model="openai/gpt-chat-latest",
        messages=[LLMMessage(role="user", content="Return JSON.")],
        temperature=0.0,
        max_output_tokens=8192,
        response_schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
        },
    )

    await provider.complete(request)

    call = completions.calls[0]
    assert "temperature" not in call
    assert call["max_tokens"] == 8192
    assert call["response_format"]["type"] == "json_schema"
    assert call["extra_body"]["provider"]["require_parameters"] is True


@pytest.mark.asyncio
async def test_openrouter_complete_omits_response_format_for_flexible_vendor() -> None:
    response = SimpleNamespace(
        model="deepseek/deepseek-v4-flash",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score":0.8}'))],
        usage=None,
        model_dump=lambda: {"id": "gen_1"},
    )
    completions = FakeChatCompletions(response)
    provider = OpenRouterProvider(
        api_key="test",
        site_url="https://atagia.org",
        app_name="Atagia",
        client=FakeOpenAIClient(completions),
    )

    request = LLMCompletionRequest(
        model="deepseek/deepseek-v4-flash",
        messages=[LLMMessage(role="user", content="Return JSON.")],
        max_output_tokens=128,
        response_schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
        },
    )

    completion = await provider.complete(request)

    assert completion.output_text == '{"score":0.8}'
    assert "response_format" not in completions.calls[0]
    assert "extra_body" not in completions.calls[0]


def test_openrouter_default_headers_and_factory_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("atagia.services.providers.openai.AsyncOpenAI", FakeAsyncOpenAI)

    provider = OpenRouterProvider(
        api_key="router-key",
        site_url="https://atagia.org",
        app_name="Atagia",
    )

    assert provider.name == "openrouter"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://atagia.org",
        "X-Title": "Atagia",
    }

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key="router-key",
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="openrouter/deepseek/deepseek-v4-flash",
    )
    client = build_llm_client(settings)
    assert client.provider_name is None
    assert client._provider("openrouter").name == "openrouter"


@pytest.mark.asyncio
async def test_build_llm_client_routes_embeddings_to_openai_for_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnthropicProvider(LLMProvider):
        name = "anthropic"

        def __init__(self, *args, **kwargs) -> None:
            self.embed_calls = 0

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            self.embed_calls += 1
            raise AssertionError("Anthropic provider should not handle embeddings in this configuration")

    class FakeOpenAIProvider(LLMProvider):
        name = "openai"

        def __init__(self, *args, **kwargs) -> None:
            self.embed_calls = 0

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            self.embed_calls += 1
            return LLMEmbeddingResponse(
                provider=self.name,
                model=request.model,
                vectors=[LLMEmbeddingVector(index=0, values=[0.1, 0.2])],
            )

    monkeypatch.setattr("atagia.services.providers.AnthropicProvider", FakeAnthropicProvider)
    monkeypatch.setattr("atagia.services.providers.OpenAIProvider", FakeOpenAIProvider)

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        anthropic_api_key="anthropic-key",
        openai_api_key="openai-key",
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="anthropic/claude-sonnet-4-6",
        embedding_backend="sqlite_vec",
        embedding_model="openai/text-embedding-3-small",
    )

    client = build_llm_client(settings)
    embedding = await client.embed(
        LLMEmbeddingRequest(model="openai/text-embedding-3-small", input_texts=["hello"])
    )

    assert client.provider_name is None
    assert embedding.provider == "openai"


def test_build_llm_client_requires_openai_for_anthropic_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnthropicProvider(LLMProvider):
        name = "anthropic"
        supports_embeddings = False

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings should fail at startup before use")

    class FakeOpenRouterProvider(LLMProvider):
        name = "openrouter"

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            return LLMEmbeddingResponse(provider=self.name, model=request.model, vectors=[])

    monkeypatch.setattr("atagia.services.providers.AnthropicProvider", FakeAnthropicProvider)
    monkeypatch.setattr("atagia.services.providers.OpenRouterProvider", FakeOpenRouterProvider)

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        anthropic_api_key="anthropic-key",
        openai_api_key=None,
        openrouter_api_key="router-key",
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="anthropic/claude-sonnet-4-6",
        embedding_backend="sqlite_vec",
        embedding_model="openai/text-embedding-3-small",
    )

    with pytest.raises(ConfigurationError, match="openai"):
        build_llm_client(settings)


@pytest.mark.asyncio
async def test_build_llm_client_routes_embeddings_by_provider_qualified_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnthropicProvider(LLMProvider):
        name = "anthropic"
        supports_embeddings = False

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Anthropic provider should not handle embeddings in this configuration")

    class FakeOpenRouterProvider(LLMProvider):
        name = "openrouter"

        def __init__(self, *args, **kwargs) -> None:
            self.embed_calls = 0

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            self.embed_calls += 1
            return LLMEmbeddingResponse(
                provider=self.name,
                model=request.model,
                vectors=[LLMEmbeddingVector(index=0, values=[0.3, 0.4])],
            )

    monkeypatch.setattr("atagia.services.providers.AnthropicProvider", FakeAnthropicProvider)
    monkeypatch.setattr("atagia.services.providers.OpenRouterProvider", FakeOpenRouterProvider)

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        anthropic_api_key="anthropic-key",
        openai_api_key=None,
        openrouter_api_key="router-key",
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="anthropic/claude-sonnet-4-6",
        embedding_backend="sqlite_vec",
        embedding_model="openrouter/openai/text-embedding-3-small",
    )

    client = build_llm_client(settings)
    embedding = await client.embed(
        LLMEmbeddingRequest(
            model="openrouter/openai/text-embedding-3-small",
            input_texts=["hello"],
        )
    )

    assert client.provider_name is None
    assert embedding.provider == "openrouter"
