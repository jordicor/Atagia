"""Plan-fidelity ASGI coverage for OpenAI proxy limits and telemetry."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from atagia.app import create_app
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMStreamEvent,
)
from atagia.services.openai_proxy_service import (
    OpenAIProxyService,
    ProxyContextAttempt,
)
from tests.api.test_openai_proxy import _settings as proxy_settings
from tests.turn_telemetry_support import TurnCallMeterMixin


_REJECTED_STORE_TABLES = (
    "users",
    "conversations",
    "messages",
    "artifacts",
    "artifact_blobs",
    "proxy_turn_runs",
    "proxy_message_id_claims",
    "worker_job_runs",
)


class _CaptureLLMClient(TurnCallMeterMixin):
    def __init__(
        self,
        *,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = "stop",
        output_text: str = "Boundary reply.",
    ) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.usage = dict(usage) if usage is not None else None
        self.finish_reason = finish_reason
        self.output_text = output_text

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider="boundary-fixture",
            model=request.model,
            output_text=self.output_text,
            usage=dict(self.usage) if self.usage is not None else {},
            finish_reason=self.finish_reason,
        )

    async def stream(
        self,
        request: LLMCompletionRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        yield LLMStreamEvent(type="text", content=self.output_text)
        payload: dict[str, Any] = {}
        if self.usage is not None:
            payload["usage"] = dict(self.usage)
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        yield LLMStreamEvent(type="done", payload=payload)


@pytest.fixture(autouse=True)
def _avoid_unrelated_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(*_args: Any, **_kwargs: Any) -> ProxyContextAttempt:
        # These tests never exercise retrieval, so the attempt cost nothing.
        return ProxyContextAttempt(context=None, elapsed_ms=0.0)

    monkeypatch.setattr(
        OpenAIProxyService,
        "_context_for_turn_fail_open",
        no_context,
    )


def _headers(suffix: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_proxy_plan_fidelity",
        "X-Atagia-Conversation-Id": f"cnv_proxy_{suffix}",
        "X-Atagia-Platform-Id": "proxy-plan-tests",
        "X-Atagia-Message-Id": f"msg_request_{suffix}",
        "X-Atagia-Response-Message-Id": f"msg_response_{suffix}",
    }


def _body(
    *,
    stream: bool,
    content: Any = "Boundary request.",
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "atagia-memory-proxy",
        "stream": stream,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }
    if metadata is not None:
        body["metadata"] = metadata
    return body


async def _stored_counts(runtime: Any) -> dict[str, int]:
    connection = await runtime.open_connection()
    try:
        counts: dict[str, int] = {}
        for table in _REJECTED_STORE_TABLES:
            cursor = await connection.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            await cursor.close()
            counts[table] = int(row[0])
        return counts
    finally:
        await connection.close()


def _sse_payloads(response: httpx.Response) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for event in response.text.split("\n\n"):
        if not event.startswith("data: "):
            continue
        data = event.removeprefix("data: ")
        if data == "[DONE]":
            continue
        payload = json.loads(data)
        assert isinstance(payload, dict)
        payloads.append(payload)
    return payloads


def _text_block(value: str = "ok") -> dict[str, Any]:
    return {"type": "text", "text": value}


def _image_block(value: str = "https://example.invalid/image.png") -> dict[str, Any]:
    return {"type": "input_image", "image_url": {"url": value}}


def _file_block(encoded: str) -> dict[str, Any]:
    return {
        "type": "input_file",
        "file": {"filename": "fixture.bin", "content_base64": encoded},
    }


def _nested_content(depth: int = 140) -> object:
    content: object = "ok"
    for _ in range(depth):
        content = {"content": content}
    return content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("client_limits", "expected_limit"),
    [
        pytest.param({"max_tokens": 1}, 1, id="max-tokens-one"),
        pytest.param(
            {"max_completion_tokens": 1},
            1,
            id="max-completion-tokens-one",
        ),
        pytest.param({"max_tokens": 7}, 7, id="at-server-cap"),
        pytest.param({"max_completion_tokens": 19}, 7, id="above-server-cap"),
        pytest.param(
            {"max_tokens": 6, "max_completion_tokens": 2},
            2,
            id="both-fields-completion-smaller",
        ),
        pytest.param(
            {"max_tokens": 2, "max_completion_tokens": 6},
            2,
            id="both-fields-max-smaller",
        ),
    ],
)
async def test_external_token_ceiling_reaches_actual_chat_request(
    tmp_path: Path,
    stream: bool,
    client_limits: dict[str, int],
    expected_limit: int,
) -> None:
    settings = replace(
        proxy_settings(tmp_path),
        openai_proxy_max_output_tokens=7,
    )
    app = create_app(settings)
    llm_client = _CaptureLLMClient(finish_reason="length")
    suffix = f"tokens_{stream}_{expected_limit}_{len(client_limits)}"

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(suffix),
                json=_body(stream=stream, **client_limits),
            )

    assert response.status_code == 200
    assert len(llm_client.requests) == 1
    upstream = llm_client.requests[0]
    assert upstream.metadata["purpose"] == "chat_reply"
    assert upstream.max_output_tokens == expected_limit
    assert upstream.external_answer is True
    if stream:
        finish_chunks = [
            payload
            for payload in _sse_payloads(response)
            if payload.get("choices")
            and payload["choices"][0].get("finish_reason") is not None
        ]
        assert finish_chunks[-1]["choices"][0]["finish_reason"] == "length"
    else:
        assert response.json()["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("invalid_value", [0, -1, "invalid"])
async def test_invalid_external_token_ceiling_is_422_before_provider_or_writes(
    tmp_path: Path,
    stream: bool,
    field: str,
    invalid_value: int | str,
) -> None:
    app = create_app(proxy_settings(tmp_path))
    llm_client = _CaptureLLMClient()

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"invalid_{stream}_{field}_{invalid_value}"),
                json=_body(stream=stream, **{field: invalid_value}),
            )
        counts = await _stored_counts(app.state.runtime)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["param"] == field
    assert llm_client.requests == []
    assert set(counts.values()) == {0}


_BUDGET_SETTINGS = {
    "request_max_message_text_bytes": 8,
    "request_max_attachments": 2,
    "request_max_attachment_decoded_bytes": 6,
    "request_max_attachments_decoded_bytes": 9,
    "request_max_metadata_bytes": 12,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("content", "metadata"),
    [
        pytest.param("1234567", None, id="text-below"),
        pytest.param("12345678", None, id="text-at"),
        pytest.param(
            [_text_block(), _image_block()],
            None,
            id="attachment-count-below",
        ),
        pytest.param(
            [_text_block(), _image_block(), _image_block("https://example.invalid/2")],
            None,
            id="attachment-count-at",
        ),
        pytest.param("ok", {"k": "123"}, id="metadata-below"),
        pytest.param("ok", {"k": "1234"}, id="metadata-at"),
        pytest.param(
            [_text_block(), _file_block("MTIzNDU=")],
            None,
            id="decoded-blob-below",
        ),
        pytest.param(
            [_text_block(), _file_block("MTIzNDU2")],
            None,
            id="decoded-blob-at",
        ),
        pytest.param(
            [_text_block(), _file_block("MTIzNA=="), _file_block("MTIzNA==")],
            None,
            id="decoded-aggregate-below",
        ),
        pytest.param(
            [_text_block(), _file_block("MTIzNA=="), _file_block("MTIzNDU=")],
            None,
            id="decoded-aggregate-at",
        ),
    ],
)
async def test_openai_proxy_accepts_at_and_below_every_field_budget(
    tmp_path: Path,
    stream: bool,
    content: Any,
    metadata: dict[str, Any] | None,
) -> None:
    app = create_app(replace(proxy_settings(tmp_path), **_BUDGET_SETTINGS))
    llm_client = _CaptureLLMClient()

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"accepted_{stream}"),
                json=_body(stream=stream, content=content, metadata=metadata),
            )

    assert response.status_code == 200
    assert len(llm_client.requests) == 1
    assert llm_client.requests[0].metadata["purpose"] == "chat_reply"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("content", "metadata", "status_code", "error_code", "error_param"),
    [
        pytest.param(
            "123456789",
            None,
            413,
            "request_too_large",
            "messages.0.content",
            id="text-above",
        ),
        pytest.param(
            {"type": "text", "text": "123456789"},
            None,
            413,
            "request_too_large",
            "messages.0.content",
            id="single-text-block-above",
        ),
        pytest.param(
            {"content": "123456789"},
            None,
            413,
            "request_too_large",
            "messages.0.content",
            id="nested-content-text-above",
        ),
        pytest.param(
            ["1234", "5678"],
            None,
            413,
            "request_too_large",
            "messages.0.content",
            id="raw-string-list-above",
        ),
        pytest.param(
            {"type": "input_file", "data": "MTIzNDU2Nw=="},
            None,
            413,
            "request_too_large",
            "messages.0.content.decoded_bytes",
            id="single-file-block-decoded-above",
        ),
        pytest.param(
            {"type": "input_file", "data": "", "metadata": {"key": "12345"}},
            None,
            413,
            "request_too_large",
            "messages.0.content.metadata",
            id="attachment-metadata-above",
        ),
        pytest.param(
            {
                "multi_ai": True,
                "responses": [
                    {
                        "model": "m",
                        "content": _file_block("MTIzNDU2Nw=="),
                    }
                ],
            },
            None,
            413,
            "request_too_large",
            "messages.0.content.responses.0.content.decoded_bytes",
            id="multi-ai-file-block-decoded-above",
        ),
        pytest.param(
            _nested_content(),
            None,
            422,
            "invalid_request_structure",
            "messages.0.content",
            id="content-nesting-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block(),
                _image_block("https://example.invalid/2"),
                _image_block("https://example.invalid/3"),
            ],
            None,
            413,
            "request_too_large",
            "attachments",
            id="many-tiny-attachments",
        ),
        pytest.param(
            "ok",
            {"k": "12345"},
            413,
            "request_too_large",
            "metadata",
            id="metadata-above",
        ),
        pytest.param(
            [_text_block(), _file_block("MTIzNDU2Nw==")],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="decoded-blob-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block("DATA:image/png;base64,MTIzNDU2Nw=="),
            ],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="uppercase-data-url-decoded-blob-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block("DaTa:image/png;BASE64,MTIzNDU2Nw=="),
            ],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="mixed-case-data-url-decoded-blob-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block(" data:image/png;base64,MTIzNDU2Nw=="),
            ],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="leading-space-data-url-decoded-blob-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block("\tdata:image/png;base64,MTIzNDU2Nw=="),
            ],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="leading-tab-data-url-decoded-blob-above",
        ),
        pytest.param(
            [
                _text_block(),
                _image_block("data:İİİİ;base64,MTIzNDU2Nw=="),
            ],
            None,
            413,
            "request_too_large",
            "messages.0.content.1.decoded_bytes",
            id="unicode-expanding-preamble-decoded-blob-above",
        ),
        pytest.param(
            [_text_block(), _file_block("MTIzNDU="), _file_block("MTIzNDU=")],
            None,
            413,
            "request_too_large",
            "attachments.decoded_bytes",
            id="decoded-aggregate-above",
        ),
        pytest.param(
            [_text_block(), _file_block("AA?=")],
            None,
            422,
            "invalid_request_structure",
            "messages.0.content.1.file.content_base64",
            id="invalid-base64",
        ),
        pytest.param(
            [_text_block(), _image_block("data:image/png,not-base64")],
            None,
            422,
            "invalid_request_structure",
            "messages.0.content.1.image_url.url",
            id="invalid-data-url",
        ),
    ],
)
async def test_openai_proxy_rejects_field_budget_without_any_partial_effect(
    tmp_path: Path,
    stream: bool,
    content: Any,
    metadata: dict[str, Any] | None,
    status_code: int,
    error_code: str,
    error_param: str,
) -> None:
    app = create_app(replace(proxy_settings(tmp_path), **_BUDGET_SETTINGS))
    llm_client = _CaptureLLMClient()

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"rejected_{stream}_{error_param}"),
                json=_body(stream=stream, content=content, metadata=metadata),
            )
        counts = await _stored_counts(app.state.runtime)

    assert response.status_code == status_code
    error = response.json()["error"]
    assert error["code"] == error_code
    assert error["param"] == error_param
    assert llm_client.requests == []
    assert set(counts.values()) == {0}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_openai_proxy_body_limit_has_openai_413_and_no_partial_effect(
    tmp_path: Path,
    chunked: bool,
) -> None:
    app = create_app(
        replace(
            proxy_settings(tmp_path),
            request_max_body_bytes=128,
        )
    )
    llm_client = _CaptureLLMClient()
    encoded = json.dumps(
        _body(stream=False, content="x" * 256),
        separators=(",", ":"),
    ).encode("utf-8")

    async def chunks() -> AsyncIterator[bytes]:
        yield encoded[:64]
        yield encoded[64:]

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    **_headers(f"body_{chunked}"),
                    "Content-Type": "application/json",
                },
                content=chunks() if chunked else encoded,
            )
        counts = await _stored_counts(app.state.runtime)

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "request_body_too_large"
    assert error["type"] == "invalid_request_error"
    assert llm_client.requests == []
    assert set(counts.values()) == {0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "usage", "provider_reason", "expected_reason"),
    [
        pytest.param(
            "openai",
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            "stop",
            "stop",
            id="openai",
        ),
        pytest.param(
            "anthropic",
            {"input_tokens": 7, "output_tokens": 2},
            "length",
            "length",
            id="anthropic",
        ),
        pytest.param(
            "gemini",
            {"input_tokens": 5, "output_tokens": 1},
            "length",
            "length",
            id="gemini",
        ),
    ],
)
async def test_real_done_usage_and_reason_match_non_stream_semantics(
    tmp_path: Path,
    provider: str,
    usage: dict[str, int],
    provider_reason: str,
    expected_reason: str,
) -> None:
    app = create_app(proxy_settings(tmp_path))
    llm_client = _CaptureLLMClient(
        usage=usage,
        finish_reason=provider_reason,
        output_text=f"{provider} completion",
    )

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            non_stream = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"{provider}_nonstream"),
                json=_body(stream=False),
            )
            stream_with_usage = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"{provider}_stream_usage"),
                json=_body(
                    stream=True,
                    stream_options={"include_usage": True},
                ),
            )
            stream_without_usage = await client.post(
                "/v1/chat/completions",
                headers=_headers(f"{provider}_stream_no_usage"),
                json=_body(
                    stream=True,
                    stream_options={"include_usage": False},
                ),
            )

    assert non_stream.status_code == 200
    non_stream_payload = non_stream.json()
    assert non_stream_payload["choices"][0]["message"]["content"] == (
        f"{provider} completion"
    )
    assert non_stream_payload["choices"][0]["finish_reason"] == expected_reason
    assert non_stream_payload["usage"] == usage

    assert stream_with_usage.status_code == 200
    with_usage_payloads = _sse_payloads(stream_with_usage)
    content = "".join(
        str(payload["choices"][0]["delta"].get("content") or "")
        for payload in with_usage_payloads
        if payload.get("choices")
    )
    assert content == non_stream_payload["choices"][0]["message"]["content"]
    finish_chunks = [
        payload
        for payload in with_usage_payloads
        if payload.get("choices")
        and payload["choices"][0].get("finish_reason") is not None
    ]
    assert [chunk["choices"][0]["finish_reason"] for chunk in finish_chunks] == [
        expected_reason
    ]
    usage_chunks = [payload for payload in with_usage_payloads if "usage" in payload]
    assert usage_chunks == [
        {
            "id": usage_chunks[0]["id"],
            "object": "chat.completion.chunk",
            "created": usage_chunks[0]["created"],
            "model": "atagia-memory-proxy",
            "choices": [],
            "usage": usage,
        }
    ]
    assert with_usage_payloads[-1] == usage_chunks[0]

    assert stream_without_usage.status_code == 200
    without_usage_payloads = _sse_payloads(stream_without_usage)
    assert all("usage" not in payload for payload in without_usage_payloads)
    finish_chunks = [
        payload
        for payload in without_usage_payloads
        if payload.get("choices")
        and payload["choices"][0].get("finish_reason") is not None
    ]
    assert [chunk["choices"][0]["finish_reason"] for chunk in finish_chunks] == [
        expected_reason
    ]
    assert len(llm_client.requests) == 3
    assert all(
        request.metadata["purpose"] == "chat_reply" for request in llm_client.requests
    )


@pytest.mark.asyncio
async def test_missing_provider_usage_is_not_fabricated_in_any_proxy_shape(
    tmp_path: Path,
) -> None:
    app = create_app(proxy_settings(tmp_path))
    llm_client = _CaptureLLMClient(usage=None, finish_reason="end_turn")

    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = llm_client
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            non_stream = await client.post(
                "/v1/chat/completions",
                headers=_headers("missing_usage_nonstream"),
                json=_body(stream=False),
            )
            stream = await client.post(
                "/v1/chat/completions",
                headers=_headers("missing_usage_stream"),
                json=_body(
                    stream=True,
                    stream_options={"include_usage": True},
                ),
            )

    assert non_stream.status_code == 200
    assert "usage" not in non_stream.json()
    assert non_stream.json()["choices"][0]["finish_reason"] == "stop"
    payloads = _sse_payloads(stream)
    assert all("usage" not in payload for payload in payloads)
    finish_chunks = [
        payload
        for payload in payloads
        if payload.get("choices")
        and payload["choices"][0].get("finish_reason") is not None
    ]
    assert [chunk["choices"][0]["finish_reason"] for chunk in finish_chunks] == ["stop"]
