"""Tests for the OpenAI Privacy Filter sidecar client."""

from __future__ import annotations

import httpx
import pytest

from atagia.services.privacy_filter_client import (
    OpenAIPrivacyFilterClient,
    PrivacyFilterUnavailable,
)


def _client(transport: httpx.MockTransport) -> OpenAIPrivacyFilterClient:
    return OpenAIPrivacyFilterClient(
        primary_url="http://primary.test",
        fallback_url="http://fallback.test",
        timeout_seconds=1.0,
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_detect_uses_primary_endpoint_and_strips_raw_span_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://primary.test/detect"
        return httpx.Response(
            200,
            json={
                "spans": [
                    {"label": "private_address", "start": 7, "end": 11, "text": "3847"}
                ]
            },
        )

    client = _client(httpx.MockTransport(handler))

    detection = await client.detect("code is 3847")

    assert detection.endpoint_used == "http://primary.test"
    assert detection.span_count == 1
    assert detection.labels == ["private_address"]
    audit = detection.spans[0].to_audit_dict()
    assert audit["label"] == "private_address"
    assert "text" not in audit
    assert audit["text_sha256"]


@pytest.mark.asyncio
async def test_detect_falls_back_when_primary_fails() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "primary.test":
            return httpx.Response(503)
        return httpx.Response(200, json={"spans": []})

    client = _client(httpx.MockTransport(handler))

    detection = await client.detect("safe text")

    assert calls == ["http://primary.test/detect", "http://fallback.test/detect"]
    assert detection.endpoint_used == "http://fallback.test"
    assert detection.span_count == 0


@pytest.mark.asyncio
async def test_detect_raises_when_both_endpoints_fail() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(503)

    client = _client(httpx.MockTransport(handler))

    with pytest.raises(PrivacyFilterUnavailable):
        await client.detect("text")
    assert calls == ["http://primary.test/detect", "http://fallback.test/detect"]
    assert client.attempted_endpoints == ("http://primary.test", "http://fallback.test")
