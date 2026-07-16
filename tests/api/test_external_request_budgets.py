"""ASGI coverage for encoded-body and decoded attachment budgets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
import pytest

from atagia.api.request_body_limit import RequestBodyLimitMiddleware
from atagia.app import create_app
from atagia.core.config import Settings

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "sqlite_path": str(tmp_path / "request-budgets.db"),
        "migrations_path": str(MIGRATIONS_DIR),
        "manifests_path": str(MANIFESTS_DIR),
        "storage_backend": "inprocess",
        "redis_url": "redis://localhost:6379/0",
        "openai_api_key": "test-openai-key",
        "openrouter_api_key": None,
        "openrouter_site_url": "http://localhost",
        "openrouter_app_name": "Atagia",
        "llm_chat_model": "openai/test-model",
        "llm_forced_global_model": "openai/test-model",
        "service_mode": False,
        "service_api_key": None,
        "admin_api_key": None,
        "workers_enabled": False,
        "debug": False,
        "allow_insecure_http": True,
        "request_max_body_bytes": 1_048_576,
        "request_max_message_text_bytes": 8,
        "request_max_attachments": 2,
        "request_max_attachment_decoded_bytes": 6,
        "request_max_attachments_decoded_bytes": 9,
        "request_max_metadata_bytes": 12,
    }
    values.update(overrides)
    return Settings(**values)


def _ingest_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": "usr_budget",
        "role": "user",
        "text": "12345678",
        "mode": "coding_debug",
    }
    payload.update(overrides)
    return payload


def _stored_counts(client: TestClient) -> tuple[int, int, int]:
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        counts = []
        for table in ("messages", "artifacts", "artifact_blobs"):
            cursor = client.portal.call(
                connection.execute,
                f"SELECT COUNT(*) FROM {table}",
            )
            row = client.portal.call(cursor.fetchone)
            client.portal.call(cursor.close)
            counts.append(int(row[0]))
        return counts[0], counts[1], counts[2]
    finally:
        client.portal.call(connection.close)


async def _await_with_event_loop_ticker(
    request: Any,
    *,
    interval_seconds: float = 0.01,
) -> tuple[httpx.Response, float]:
    gaps: list[float] = []
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def ticker() -> None:
        previous = loop.time()
        while True:
            await asyncio.sleep(interval_seconds)
            current = loop.time()
            gaps.append(current - previous)
            previous = current
            if stop.is_set():
                return

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(interval_seconds * 2)
    try:
        response = await request
    finally:
        stop.set()
        await ticker_task
    return response, max(gaps, default=0.0)


def test_direct_api_accepts_exact_decoded_boundaries(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations/cnv_budget/messages",
            json=_ingest_payload(
                attachments=[
                    {"kind": "base64", "content_base64": "MTIzNA=="},
                    {"kind": "base64", "content_base64": "MTIzNDU="},
                ]
            ),
        )

        assert response.status_code == 200
        assert _stored_counts(client) == (1, 2, 0)


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (_ingest_payload(text="123456789"), 413),
        (
            _ingest_payload(
                attachments=[
                    {"kind": "pasted_text", "content_text": "a"},
                    {"kind": "pasted_text", "content_text": "b"},
                    {"kind": "pasted_text", "content_text": "c"},
                ]
            ),
            413,
        ),
        (
            _ingest_payload(
                attachments=[{"kind": "base64", "content_base64": "MTIzNDU2Nw=="}]
            ),
            413,
        ),
        (
            _ingest_payload(
                attachments=[
                    {"kind": "base64", "content_base64": "MTIzNDU="},
                    {"kind": "base64", "content_base64": "MTIzNDU="},
                ]
            ),
            413,
        ),
        (
            _ingest_payload(
                attachments=[
                    {"kind": "base64", "content_base64": "AA?="},
                ]
            ),
            422,
        ),
        (
            _ingest_payload(
                attachments=[
                    {
                        "kind": "pasted_text",
                        "content_text": "a",
                        "metadata": {"secret": "large"},
                    }
                ]
            ),
            413,
        ),
    ],
)
def test_direct_api_rejects_budget_or_structure_without_partial_rows(
    tmp_path: Path,
    payload: dict[str, Any],
    expected_status: int,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations/cnv_budget/messages",
            json=payload,
        )

        assert response.status_code == expected_status
        assert _stored_counts(client) == (0, 0, 0)


def test_content_length_is_rejected_before_route_materialization(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path, request_max_body_bytes=64))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations/cnv_budget/messages",
            json=_ingest_payload(),
        )

        assert response.status_code == 413
        assert "64 bytes" in response.json()["detail"]
        assert _stored_counts(client) == (0, 0, 0)


@pytest.mark.asyncio
async def test_chunked_body_without_content_length_is_rejected_at_the_cap() -> None:
    inner_calls = 0

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal inner_calls
        inner_calls += 1
        while (await receive()).get("more_body", False):
            pass
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestBodyLimitMiddleware(inner_app, max_body_bytes=5)
    messages = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "path": "/v1/direct", "headers": []},
        receive,
        send,
    )

    assert inner_calls == 1
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_body_exactly_at_the_limit_reaches_the_application() -> None:
    received_body = b""

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        del scope
        nonlocal received_body
        message = await receive()
        received_body += message.get("body", b"")
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestBodyLimitMiddleware(inner_app, max_body_bytes=5)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"12345", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "path": "/v1/direct",
            "headers": [(b"content-length", b"5")],
        },
        receive,
        send,
    )

    assert received_body == b"12345"
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
async def test_concurrent_oversized_requests_are_rejected_without_writes(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/conversations/cnv_budget/messages",
                        json=_ingest_payload(text="x" * 9),
                    )
                    for _ in range(20)
                ]
            )

        connection = await app.state.runtime.open_connection()
        try:
            cursor = await connection.execute("SELECT COUNT(*) FROM messages")
            message_count = int((await cursor.fetchone())[0])
            await cursor.close()
        finally:
            await connection.close()

    assert {response.status_code for response in responses} == {413}
    assert message_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("direct", "proxy"))
async def test_large_rejected_attachment_keeps_the_event_loop_responsive(
    tmp_path: Path,
    surface: str,
) -> None:
    encoded_attachment = "A" * (16 * 1024 * 1024)
    settings = _settings(
        tmp_path,
        request_max_body_bytes=20 * 1024 * 1024,
        request_max_attachment_decoded_bytes=1024 * 1024,
        request_max_attachments_decoded_bytes=1024 * 1024,
    )
    app = create_app(settings)
    if surface == "direct":
        path = "/v1/conversations/cnv_budget/messages"
        headers = {"Content-Type": "application/json"}
        body = _ingest_payload(
            text="ok",
            attachments=[{"kind": "base64", "content_base64": encoded_attachment}],
        )
    else:
        path = "/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "X-Atagia-User-Id": "usr_budget",
            "X-Atagia-Conversation-Id": "cnv_budget",
            "X-Atagia-Platform-Id": "budget-tests",
        }
        body = {
            "model": settings.openai_proxy_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "data": encoded_attachment}],
                }
            ],
        }
    encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response, max_ticker_gap = await _await_with_event_loop_ticker(
                client.post(path, headers=headers, content=encoded_body)
            )
        connection = await app.state.runtime.open_connection()
        try:
            cursor = await connection.execute("SELECT COUNT(*) FROM messages")
            message_count = int((await cursor.fetchone())[0])
            await cursor.close()
        finally:
            await connection.close()

    assert response.status_code == 413, response.text
    assert message_count == 0
    assert max_ticker_gap < 0.2, (
        f"{surface} budget rejection blocked the event loop for {max_ticker_gap:.3f}s"
    )
