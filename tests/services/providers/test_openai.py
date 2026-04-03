"""Tests for the OpenAI provider adapter."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMEmbeddingRequest,
    LLMError,
    LLMMessage,
    LLMToolSpec,
    TransientLLMError,
)
from atagia.services.providers.openai import OpenAIProvider


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
        max_output_tokens=256,
        temperature=0.2,
        response_schema={"type": "object", "properties": {"label": {"type": "string"}}},
        tools=[LLMToolSpec(name="lookup", description="Lookup data", input_schema={"type": "object"})],
        metadata={"user_id": "usr_1"},
    )


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
    assert call["max_completion_tokens"] == 256
    assert "max_tokens" not in call
    assert call["response_format"]["type"] == "json_schema"
    assert call["tools"][0]["function"]["name"] == "lookup"
    assert call["user"] == "usr_1"


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
            metadata={"user_id": "usr_1"},
        )
    )

    assert [vector.values for vector in response.vectors] == [[0.1, 0.2], [0.3, 0.4]]
    assert embeddings.calls[0]["user"] == "usr_1"


@pytest.mark.asyncio
async def test_openai_maps_retryable_and_permanent_errors() -> None:
    transient_error = openai.APIConnectionError(
        message="boom",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
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

    with pytest.raises(TransientLLMError):
        await transient_provider.complete(_request())
    with pytest.raises(LLMError):
        await permanent_provider.complete(_request())

