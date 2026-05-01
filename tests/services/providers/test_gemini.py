"""Tests for the native Gemini provider adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from atagia.core.config import Settings
from atagia.services.llm_client import (
    ConfigurationError,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMError,
    LLMMessage,
    LLMPolicyBlockedError,
    LLMProvider,
    LLMToolSpec,
    OutputLimitExceededError,
    TransientLLMError,
)
from atagia.services.providers import build_llm_client
from atagia.services.providers.gemini import (
    GeminiProvider,
    _sanitize_schema_for_gemini,
)


class FakeAsyncStream:
    def __init__(self, chunks, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    def __aiter__(self):
        return self._iterator()

    async def _iterator(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self):
        self.closed = True


class FakeGeminiModels:
    def __init__(
        self,
        *,
        completion_response=None,
        embedding_response=None,
        stream_response=None,
        error: Exception | None = None,
    ) -> None:
        self.completion_response = completion_response
        self.embedding_response = embedding_response
        self.stream_response = stream_response
        self.error = error
        self.generate_content_calls: list[dict] = []
        self.generate_content_stream_calls: list[dict] = []
        self.embed_content_calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.generate_content_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.completion_response

    async def generate_content_stream(self, **kwargs):
        self.generate_content_stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream_response

    async def embed_content(self, **kwargs):
        self.embed_content_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.embedding_response


class FakeGeminiClient:
    def __init__(self, models: FakeGeminiModels) -> None:
        self.aio = SimpleNamespace(models=models)


def _usage(**values):
    return SimpleNamespace(model_dump=lambda **_kwargs: values)


def _part(*, text: str | None = None, thought: bool | None = None, function_call=None):
    return SimpleNamespace(text=text, thought=thought, function_call=function_call)


def _response(
    *,
    parts,
    usage=None,
    model_version: str | None = "gemini-3-flash-preview",
    finish_reason: str | None = None,
):
    finish = SimpleNamespace(name=finish_reason) if finish_reason is not None else None
    return SimpleNamespace(
        model_version=model_version,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=finish,
            )
        ],
        usage_metadata=usage,
        model_dump=lambda **_kwargs: {"response_id": "resp_1"},
    )


def _request(**overrides) -> LLMCompletionRequest:
    payload = {
        "model": "gemini-3-flash-preview",
        "messages": [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
        ],
        "temperature": 0.2,
        "max_output_tokens": 256,
    }
    payload.update(overrides)
    return LLMCompletionRequest(**payload)


async def _collect_stream_events_and_error(provider: GeminiProvider):
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
async def test_gemini_complete_maps_text_and_request_shape() -> None:
    response = _response(
        parts=[
            _part(text="hidden", thought=True),
            _part(text="hello world", thought=False),
        ],
        usage=_usage(
            prompt_token_count=11,
            response_token_count=7,
            total_token_count=18,
            traffic_type="on_demand",
        ),
    )
    models = FakeGeminiModels(completion_response=response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    completion = await provider.complete(_request(include_thinking=True))

    assert completion.output_text == "hello world"
    assert completion.thinking == "hidden"
    assert completion.usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    call = models.generate_content_calls[0]
    assert call["model"] == "gemini-3-flash-preview"
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "Hello"
    assert call["config"].system_instruction == "You are helpful."
    assert call["config"].temperature == 0.2
    assert call["config"].max_output_tokens == 256
    assert call["config"].thinking_config.include_thoughts is True
    from google.genai import types as genai_types
    assert call["config"].thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH


@pytest.mark.asyncio
async def test_gemini_complete_omits_thinking_config_when_not_opted_in() -> None:
    """Provider profiles, not the raw adapter, decide hidden thinking level."""
    response = _response(
        parts=[_part(text="ok", thought=False)],
        usage=_usage(prompt_token_count=5, response_token_count=1, total_token_count=6),
    )
    models = FakeGeminiModels(completion_response=response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    await provider.complete(_request())

    config = models.generate_content_calls[0]["config"]
    assert config.thinking_config is None


@pytest.mark.asyncio
async def test_gemini_complete_respects_metadata_thinking_level_override() -> None:
    """Callers can override thinking_level via metadata['gemini_thinking_level']."""
    from google.genai import types as genai_types

    response = _response(parts=[_part(text="ok")], usage=None)
    models = FakeGeminiModels(completion_response=response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    request = _request().model_copy(
        update={"metadata": {"gemini_thinking_level": "low"}}
    )
    await provider.complete(request)

    config = models.generate_content_calls[0]["config"]
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW


@pytest.mark.asyncio
async def test_gemini_complete_rejects_unknown_thinking_level() -> None:
    response = _response(parts=[_part(text="ok")], usage=None)
    models = FakeGeminiModels(completion_response=response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    request = _request().model_copy(
        update={"metadata": {"gemini_thinking_level": "ridiculous"}}
    )
    with pytest.raises(LLMError, match="Unknown gemini_thinking_level"):
        await provider.complete(request)


@pytest.mark.asyncio
async def test_gemini_complete_uses_structured_output_and_sanitizes_schema() -> None:
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"$ref": "#/$defs/Status"},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "note": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        },
        "$defs": {"Status": {"type": "string", "enum": ["ok", "warning"]}},
    }
    models = FakeGeminiModels(
        completion_response=_response(parts=[_part(text='{"status":"ok"}')])
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    await provider.complete(_request(response_schema=response_schema))

    config = models.generate_content_calls[0]["config"]
    schema = config.response_schema
    assert config.response_mime_type == "application/json"
    assert "additionalProperties" not in schema
    assert "$defs" not in schema
    assert schema["properties"]["status"] == {"type": "string", "enum": ["ok", "warning"]}
    assert "minimum" not in schema["properties"]["score"]
    assert "maximum" not in schema["properties"]["score"]
    assert schema["properties"]["note"] == {"type": "string", "nullable": True}


@pytest.mark.asyncio
async def test_gemini_complete_maps_tools_and_function_calls() -> None:
    function_call = SimpleNamespace(id=None, name="lookup", args={"query": "atagia"})
    models = FakeGeminiModels(
        completion_response=_response(parts=[_part(function_call=function_call)])
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    completion = await provider.complete(
        _request(
            tools=[
                LLMToolSpec(
                    name="lookup",
                    description="Lookup data",
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ]
        )
    )

    tool_call = completion.tool_calls[0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["name"] == "lookup"
    assert tool_call["input"] == {"query": "atagia"}
    config = models.generate_content_calls[0]["config"]
    assert config.automatic_function_calling.disable is True
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "lookup"
    assert "additionalProperties" not in declaration.parameters_json_schema


@pytest.mark.asyncio
async def test_gemini_complete_raises_non_transient_on_max_tokens() -> None:
    models = FakeGeminiModels(
        completion_response=_response(
            parts=[_part(text="partial")],
            finish_reason="MAX_TOKENS",
        )
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(OutputLimitExceededError, match="max output tokens") as exc_info:
        await provider.complete(_request())

    assert isinstance(exc_info.value, LLMError)
    assert not isinstance(exc_info.value, TransientLLMError)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["OTHER", "MALFORMED_FUNCTION_CALL"])
async def test_gemini_complete_raises_transient_on_retryable_finish_reason(
    finish_reason: str,
) -> None:
    models = FakeGeminiModels(
        completion_response=_response(
            parts=[_part(text="partial")],
            finish_reason=finish_reason,
        )
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(TransientLLMError, match=finish_reason):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_gemini_complete_allows_stop_with_output_text() -> None:
    models = FakeGeminiModels(
        completion_response=_response(
            parts=[_part(text="ok")],
            finish_reason="STOP",
        )
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    completion = await provider.complete(_request())

    assert completion.output_text == "ok"


@pytest.mark.asyncio
async def test_gemini_complete_raises_transient_on_stop_without_content() -> None:
    models = FakeGeminiModels(
        completion_response=_response(parts=[], finish_reason="STOP")
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(TransientLLMError, match="STOP"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_gemini_complete_allows_stop_with_tool_calls_without_text() -> None:
    function_call = SimpleNamespace(id="fc_1", name="lookup", args={"q": "x"})
    models = FakeGeminiModels(
        completion_response=_response(
            parts=[_part(function_call=function_call)],
            finish_reason="STOP",
        )
    )
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    completion = await provider.complete(_request())

    assert completion.output_text == ""
    assert completion.tool_calls == [
        {
            "id": "fc_1",
            "type": "function",
            "name": "lookup",
            "input": {"q": "x"},
        }
    ]


@pytest.mark.asyncio
async def test_gemini_stream_maps_text_tool_call_and_done() -> None:
    function_call = SimpleNamespace(id="fc_1", name="lookup", args={"q": "x"})
    stream = FakeAsyncStream(
        [
            _response(parts=[_part(text="hel")], usage=None),
            _response(parts=[_part(text="lo")], usage=None),
            _response(
                parts=[_part(function_call=function_call)],
                usage=_usage(prompt_token_count=3, response_token_count=2),
            ),
        ]
    )
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == ["text", "text", "tool_call", "done"]
    assert events[0].content == "hel"
    assert events[1].content == "lo"
    assert events[2].payload == {
        "id": "fc_1",
        "type": "function",
        "name": "lookup",
        "input": {"q": "x"},
    }
    assert events[3].payload["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert models.generate_content_stream_calls[0]["config"].system_instruction == "You are helpful."


@pytest.mark.asyncio
async def test_gemini_stream_propagates_errors_after_partial_output() -> None:
    stream_error = RuntimeError("stream broke")
    stream = FakeAsyncStream([_response(parts=[_part(text="partial")])], error=stream_error)
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(RuntimeError, match="stream broke"):
        [event async for event in provider.stream(_request())]


@pytest.mark.asyncio
async def test_gemini_stream_emits_done_then_raises_on_max_tokens() -> None:
    stream = FakeAsyncStream(
        [
            _response(
                parts=[_part(text="partial")],
                usage=_usage(prompt_token_count=4, response_token_count=2),
                finish_reason="MAX_TOKENS",
            )
        ]
    )
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    received, raised = await _collect_stream_events_and_error(provider)

    assert [event.type for event in received] == ["text", "done"]
    assert received[0].content == "partial"
    assert received[1].payload["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert isinstance(raised, OutputLimitExceededError)
    assert isinstance(raised, LLMError)
    assert not isinstance(raised, TransientLLMError)
    assert "max output tokens" in str(raised)


@pytest.mark.asyncio
async def test_gemini_stream_emits_done_then_raises_on_transient_finish_reason() -> None:
    stream = FakeAsyncStream(
        [
            _response(parts=[_part(text="partial")]),
            _response(
                parts=[],
                usage=_usage(prompt_token_count=4, response_token_count=2),
                finish_reason="OTHER",
            ),
        ]
    )
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    received, raised = await _collect_stream_events_and_error(provider)

    assert [event.type for event in received] == ["text", "done"]
    assert received[0].content == "partial"
    assert received[1].payload["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert isinstance(raised, TransientLLMError)
    assert "OTHER" in str(raised)


@pytest.mark.asyncio
async def test_gemini_embed_maps_vectors_and_dimension_config() -> None:
    embedding_response = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[0.1, 0.2]),
            SimpleNamespace(values=[0.3, 0.4]),
        ],
        model_dump=lambda **_kwargs: {"metadata": "ok"},
    )
    models = FakeGeminiModels(embedding_response=embedding_response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    response = await provider.embed(
        LLMEmbeddingRequest(
            model="gemini-embedding-2",
            input_texts=["a", "b"],
            dimensions=768,
        )
    )

    assert [vector.values for vector in response.vectors] == [[0.1, 0.2], [0.3, 0.4]]
    call = models.embed_content_calls[0]
    assert call["model"] == "gemini-embedding-2"
    assert call["contents"] == ["a", "b"]
    assert call["config"].output_dimensionality == 768


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_factory", "expected_error"),
    [
        (lambda: httpx.ConnectError("connection failed"), TransientLLMError),
        (lambda: httpx.TimeoutException("timeout"), TransientLLMError),
        (
            lambda: genai_errors.ClientError(
                429,
                {"error": {"code": 429, "message": "rate limit", "status": "RESOURCE_EXHAUSTED"}},
            ),
            TransientLLMError,
        ),
        (
            lambda: genai_errors.ServerError(
                500,
                {"error": {"code": 500, "message": "server", "status": "INTERNAL"}},
            ),
            TransientLLMError,
        ),
        (
            lambda: genai_errors.ClientError(
                503,
                {"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}},
            ),
            TransientLLMError,
        ),
        (
            lambda: genai_errors.ClientError(
                400,
                {"error": {"code": 400, "message": "bad schema", "status": "INVALID_ARGUMENT"}},
            ),
            LLMError,
        ),
        (
            lambda: json.JSONDecodeError("Expecting value", "<html>bad gateway</html>", 0),
            TransientLLMError,
        ),
        (lambda: RuntimeError("surprise"), LLMError),
    ],
)
async def test_gemini_maps_errors(error_factory, expected_error) -> None:
    models = FakeGeminiModels(error=error_factory())
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(expected_error):
        await provider.complete(_request())


def test_gemini_schema_sanitizer_handles_nested_refs_arrays_and_nullable() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "profile": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "age": {
                        "anyOf": [
                            {"type": "integer", "minimum": 0, "maximum": 140},
                            {"type": "null"},
                        ]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Tag"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
            },
            "alias": {"type": ["string", "null"]},
        },
        "$defs": {"Tag": {"type": "string", "enum": ["a", "b"]}},
    }

    sanitized = _sanitize_schema_for_gemini(schema)

    assert "additionalProperties" not in sanitized
    assert "$defs" not in sanitized
    profile = sanitized["properties"]["profile"]
    assert "additionalProperties" not in profile
    assert profile["properties"]["age"] == {"type": "integer", "nullable": True}
    assert "minItems" not in profile["properties"]["tags"]
    assert "maxItems" not in profile["properties"]["tags"]
    assert profile["properties"]["tags"]["items"] == {
        "type": "string",
        "enum": ["a", "b"],
    }
    assert sanitized["properties"]["alias"] == {"type": "string", "nullable": True}


def test_gemini_schema_sanitizer_rejects_unresolved_ref() -> None:
    with pytest.raises(LLMError, match="could not be resolved"):
        _sanitize_schema_for_gemini({"$ref": "#/$defs/Missing", "$defs": {}})


def test_settings_loads_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAGIA_GOOGLE_API_KEY", "google-key")

    settings = Settings.from_env()

    assert settings.google_api_key == "google-key"


@pytest.mark.asyncio
async def test_build_llm_client_registers_gemini_and_uses_gemini_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeGeminiProvider(LLMProvider):
        name = "gemini"
        supports_embedding_dimensions = True

        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            return LLMEmbeddingResponse(
                provider=self.name,
                model=request.model,
                vectors=[LLMEmbeddingVector(index=0, values=[0.1, 0.2])],
            )

    monkeypatch.setattr("atagia.services.providers.GeminiProvider", FakeGeminiProvider)

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        google_api_key="google-key",
        llm_forced_global_model="google/gemini-3.1-flash-lite-preview",
        embedding_backend="sqlite_vec",
        embedding_model="google/gemini-embedding-2",
    )

    client = build_llm_client(settings)
    embedding = await client.embed(
        LLMEmbeddingRequest(model="google/gemini-embedding-2", input_texts=["hello"])
    )

    assert captured["api_key"] == "google-key"
    assert client.provider_name is None
    assert embedding.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_complete_raises_on_response_safety_block() -> None:
    blocked_response = SimpleNamespace(
        model_version="gemini-3-flash-preview",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[]),
                finish_reason=SimpleNamespace(name="SAFETY"),
            )
        ],
        prompt_feedback=None,
        usage_metadata=None,
        model_dump=lambda **_kwargs: {},
    )
    models = FakeGeminiModels(completion_response=blocked_response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(LLMError, match=r"response:SAFETY"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_gemini_complete_raises_on_prompt_block() -> None:
    blocked_response = SimpleNamespace(
        model_version="gemini-3-flash-preview",
        candidates=[],
        prompt_feedback=SimpleNamespace(
            block_reason=SimpleNamespace(name="PROHIBITED_CONTENT")
        ),
        usage_metadata=None,
        model_dump=lambda **_kwargs: {},
    )
    models = FakeGeminiModels(completion_response=blocked_response)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    with pytest.raises(LLMError, match=r"prompt:PROHIBITED_CONTENT"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_gemini_stream_emits_done_before_propagating_error() -> None:
    stream_error = RuntimeError("stream broke mid-flight")
    stream = FakeAsyncStream(
        [
            _response(
                parts=[_part(text="partial")],
                usage=_usage(prompt_token_count=4, response_token_count=2),
            )
        ],
        error=stream_error,
    )
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

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

    assert [event.type for event in received] == ["text", "done"]
    assert received[1].payload["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert isinstance(raised, RuntimeError)
    assert "stream broke" in str(raised)


@pytest.mark.asyncio
async def test_gemini_stream_raises_policy_blocked_error_for_safety_block() -> None:
    stream = FakeAsyncStream(
        [
            _response(
                parts=[],
                finish_reason="SAFETY",
            )
        ]
    )
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    iterator = provider.stream(_request()).__aiter__()
    done = await iterator.__anext__()

    assert done.type == "done"
    with pytest.raises(LLMPolicyBlockedError, match=r"response:SAFETY"):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_gemini_stream_closes_underlying_stream_on_cancel() -> None:
    stream = FakeAsyncStream([_response(parts=[_part(text="partial")])])
    models = FakeGeminiModels(stream_response=stream)
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(models))

    iterator = provider.stream(_request()).__aiter__()
    first = await anext(iterator)
    await iterator.aclose()

    assert first.type == "text"
    assert stream.closed is True


def test_build_llm_client_requires_gemini_credentials() -> None:
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="google/gemini-3.1-flash-lite-preview",
    )

    with pytest.raises(ConfigurationError, match="Missing API key"):
        build_llm_client(settings)
