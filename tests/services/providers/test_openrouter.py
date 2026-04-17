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
        llm_provider="openrouter",
        llm_api_key=None,
        openai_api_key=None,
        openrouter_api_key="router-key",
        llm_base_url=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_extraction_model=None,
        llm_scoring_model=None,
        llm_classifier_model=None,
        llm_chat_model=None,
        embedding_provider_name=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )
    client = build_llm_client(settings)
    assert client.provider_name == "openrouter"


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
        llm_provider="anthropic",
        llm_api_key="anthropic-key",
        openai_api_key="openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_extraction_model=None,
        llm_scoring_model=None,
        llm_classifier_model=None,
        llm_chat_model=None,
        embedding_provider_name=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        embedding_backend="sqlite_vec",
        embedding_model="text-embedding-3-small",
    )

    client = build_llm_client(settings)
    embedding = await client.embed(
        LLMEmbeddingRequest(model="text-embedding-3-small", input_texts=["hello"])
    )

    assert client.provider_name == "anthropic"
    assert client.embedding_provider_name == "openai"
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
        llm_provider="anthropic",
        llm_api_key="anthropic-key",
        openai_api_key=None,
        openrouter_api_key="router-key",
        llm_base_url=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_extraction_model=None,
        llm_scoring_model=None,
        llm_classifier_model=None,
        llm_chat_model=None,
        embedding_provider_name=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        embedding_backend="sqlite_vec",
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(ConfigurationError, match="OpenRouter is not auto-selected"):
        build_llm_client(settings)


@pytest.mark.asyncio
async def test_build_llm_client_respects_explicit_embedding_provider_override(
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
        llm_provider="anthropic",
        llm_api_key="anthropic-key",
        openai_api_key=None,
        openrouter_api_key="router-key",
        llm_base_url=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_extraction_model=None,
        llm_scoring_model=None,
        llm_classifier_model=None,
        llm_chat_model=None,
        embedding_provider_name="openrouter",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        embedding_backend="sqlite_vec",
        embedding_model="text-embedding-3-small",
    )

    client = build_llm_client(settings)
    embedding = await client.embed(
        LLMEmbeddingRequest(model="text-embedding-3-small", input_texts=["hello"])
    )

    assert client.provider_name == "anthropic"
    assert client.embedding_provider_name == "openrouter"
    assert embedding.provider == "openrouter"
