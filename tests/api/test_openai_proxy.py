from __future__ import annotations

from pathlib import Path
import json
import re

import httpx
import pytest

from atagia.app import create_app
from atagia.core.config import Settings
from atagia.services.llm_client import (
    LLMError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMStreamEvent,
    TransientLLMError,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


class ProxyProvider(LLMProvider):
    name = "proxy-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.raise_after_first_stream_event = False
        self.raise_before_first_stream_event = False
        self.raise_after_non_output_stream_event = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "needs": [],
                        "temporal_range": None,
                        "sub_queries": ["remember proxy"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "remember proxy",
                                "fts_phrase": "remember proxy",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
        if purpose == "applicability_scoring":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "scores": [
                            {"score_key": score_key, "llm_applicability": 0.5}
                            for _memory_id, score_key in candidate_keys
                        ],
                    }
                ),
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": False,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "chat_reply" and request.tools:
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                tool_calls=[
                    {
                        "id": "call_lookup",
                        "type": "function",
                        "name": "lookup",
                        "arguments": json.dumps({"query": "atagia"}),
                    }
                ],
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Proxy reply.",
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def stream(self, request: LLMCompletionRequest):
        self.requests.append(request)
        assert request.metadata.get("purpose") == "chat_reply"
        if self.raise_before_first_stream_event:
            raise LLMError("preflight failed")
        if self.raise_after_non_output_stream_event:
            yield LLMStreamEvent(type="done", payload={})
            raise LLMError("blocked after metadata")
        if request.tools:
            yield LLMStreamEvent(
                type="tool_call",
                payload={
                    "id": "call_stream_lookup",
                    "type": "function",
                    "name": "lookup",
                    "arguments": json.dumps({"query": "stream"}),
                },
            )
            yield LLMStreamEvent(type="done", payload={})
            return
        yield LLMStreamEvent(type="text", content="Proxy ")
        if self.raise_after_first_stream_event:
            raise TransientLLMError("stream failed")
        yield LLMStreamEvent(type="text", content="stream.")
        yield LLMStreamEvent(type="done", payload={})

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in proxy tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-openai-proxy.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="proxy-reply-model",
        llm_forced_global_model="openai/proxy-reply-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
        allow_insecure_http=False,
        small_corpus_token_threshold_ratio=0.0,
    )


@pytest.mark.asyncio
async def test_openai_proxy_models_and_non_streaming_completion(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            models = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer service-key"},
            )
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-User-Persona-Id": "persona_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Character-Id": "char_proxy",
                    "X-Atagia-Conversation-Id": "cnv_proxy",
                    "X-Atagia-Mode": "general_qa",
                    "X-Atagia-Incognito": "false",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "system", "content": "Base system"},
                        {"role": "user", "content": "Remember the proxy path."},
                    ],
                },
            )

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "atagia-memory-proxy"
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Proxy reply."
    chat_requests = [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests
    assert chat_requests[-1].metadata["user_persona_id"] == "persona_proxy"
    assert chat_requests[-1].metadata["platform_id"] == "proxy_desktop"
    assert chat_requests[-1].metadata["character_id"] == "char_proxy"
    assert chat_requests[-1].metadata["mode"] == "general_qa"
    assert chat_requests[-1].metadata["incognito"] is False
    assert "[ATAGIA MEMORY CONTEXT - INTERNAL]" in chat_requests[-1].messages[0].content
    assert chat_requests[-1].messages[-1].content == "Remember the proxy path."


@pytest.mark.asyncio
async def test_openai_proxy_requires_explicit_conversation_id(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy_a",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-Conversation-Id" in response.json()["error"]["message"]
    assert not [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_requires_explicit_platform_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Conversation-Id": "cnv_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-Platform-Id" in response.json()["error"]["message"]
    assert not [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_reads_redesign_metadata_identity(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello metadata"}],
                    "metadata": {
                        "atagia_conversation_id": "cnv_proxy_metadata",
                        "atagia_user_persona_id": "persona_meta",
                        "atagia_platform_id": "platform_meta",
                        "atagia_character_id": "char_meta",
                        "atagia_mode": "general_qa",
                        "atagia_incognito": True,
                    },
                },
            )

    assert response.status_code == 200
    chat_requests = [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests[-1].metadata["conversation_id"] == "cnv_proxy_metadata"
    assert chat_requests[-1].metadata["user_persona_id"] == "persona_meta"
    assert chat_requests[-1].metadata["platform_id"] == "platform_meta"
    assert chat_requests[-1].metadata["character_id"] == "char_meta"
    assert chat_requests[-1].metadata["mode"] == "general_qa"
    assert chat_requests[-1].metadata["incognito"] is True
    assert chat_requests[-1].metadata["cross_chat_memory"] is False


@pytest.mark.asyncio
async def test_openai_proxy_streaming_completion(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Stream with memory."},
                    ],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert "chat.completion.chunk" in text
    assert "Proxy " in text
    assert "stream." in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_openai_proxy_rejects_unknown_model_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    error = response.json()["error"]
    assert "Unknown model" in error["message"]
    assert error["param"] == "model"
    assert error["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_openai_proxy_maps_tool_requests_and_tool_results(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_tools",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Use the lookup tool."},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_previous",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{\"query\":\"previous\"}",
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_previous",
                            "content": "{\"result\":\"ok\"}",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Lookup memory",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "lookup"
    chat_request = [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ][-1]
    assert chat_request.tools[0].name == "lookup"
    assert chat_request.metadata["openai_tool_choice"] == "auto"
    assert chat_request.messages[-2].tool_calls[0]["id"] == "call_previous"
    assert chat_request.messages[-1].role == "tool"
    assert chat_request.messages[-1].name == "call_previous"
    assert chat_request.messages[-1].content == "{\"result\":\"ok\"}"


@pytest.mark.asyncio
async def test_openai_proxy_honors_tool_choice_none_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_tools_none",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Do not call tools."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "none",
                },
            )

    assert response.status_code == 200
    assert "tool_calls" not in response.json()["choices"][0]["message"]
    chat_request = [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ][-1]
    assert chat_request.tools == []
    assert chat_request.metadata["openai_tool_choice"] == "none"


@pytest.mark.asyncio
async def test_openai_proxy_streams_tool_calls(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_tools",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Use lookup."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert '"tool_calls"' in text
    assert "call_stream_lookup" in text
    assert '"finish_reason": "tool_calls"' in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_streaming_openai_proxy_preflight_errors_become_503(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_before_first_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_preflight",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["message"] == "LLM service unavailable"
    assert error["code"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_streaming_openai_proxy_non_output_preflight_errors_become_503(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_non_output_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_preflight_metadata",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_streaming_openai_proxy_emits_sse_error_after_partial_failure(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_first_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_error",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert "Proxy " in text
    assert "atagia_upstream_stream_error" in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_openai_proxy_response_persistence_fails_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_record_response(*args, **kwargs):
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.OpenAIProxyService._record_response",
        fail_record_response,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_fail_open",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Proxy reply."


@pytest.mark.asyncio
async def test_openai_proxy_conversation_id_collision_returns_404(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_a",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "shared_conversation_id",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            second = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_b",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "shared_conversation_id",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert first.status_code == 200
    assert second.status_code == 404
    assert second.json()["error"]["message"] == "Conversation not found for user"


@pytest.mark.asyncio
async def test_openai_proxy_memory_context_failure_fails_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_context(*args, **kwargs):
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_context_fail_open",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Proxy reply."
    chat_requests = [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests
    assert "[ATAGIA MEMORY CONTEXT - INTERNAL]" not in chat_requests[-1].messages[0].content


@pytest.mark.asyncio
async def test_openai_proxy_unexpected_context_value_error_fails_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_context(*args, **kwargs):
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_value_error_fail_open",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Proxy reply."


@pytest.mark.asyncio
async def test_openai_proxy_workspace_mismatch_is_not_fail_open(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_proxy",
        "X-Atagia-Platform-Id": "proxy_desktop",
    }
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await client.post(
                "/v1/workspaces",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "workspace_id": "wrk_proxy_a",
                    "name": "Workspace A",
                    "metadata": {},
                },
            )
            await client.post(
                "/v1/workspaces",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "workspace_id": "wrk_proxy_b",
                    "name": "Workspace B",
                    "metadata": {},
                },
            )
            await client.post(
                "/v1/conversations",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "conversation_id": "cnv_proxy_workspace",
                    "assistant_mode_id": "general_qa",
                    "workspace_id": "wrk_proxy_a",
                    "platform_id": "proxy_desktop",
                    "title": None,
                    "metadata": {},
                },
            )
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    **headers,
                    "X-Atagia-Conversation-Id": "cnv_proxy_workspace",
                    "X-Atagia-Workspace-Id": "wrk_proxy_b",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 409
    assert (
        response.json()["error"]["message"]
        == "Requested workspace does not match the existing conversation workspace"
    )
    assert not [
        request for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_validation_errors_use_openai_shape(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                },
            )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "validation_error"
    assert error["param"] == "messages"


@pytest.mark.asyncio
async def test_openai_proxy_requires_resolved_user_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer service-key"},
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-User-Id" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_openai_proxy_requires_resolved_user_id(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer service-key"},
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-User-Id" in response.json()["error"]["message"]
