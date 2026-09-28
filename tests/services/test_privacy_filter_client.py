"""Tests for the OpenAI Privacy Filter sidecar client."""

from __future__ import annotations

import httpx
import pytest

from atagia.core.config import Settings
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


def test_from_settings_activates_restricted_opf_transport() -> None:
    settings = Settings.from_env(
        {
            "ATAGIA_INFERENCE_ACCESS_MODE": "zero_cost",
            "ATAGIA_OPF_PRIVACY_FILTER_ENABLED": "true",
            "ATAGIA_OPF_PRIMARY_URL": "http://127.0.0.1:8008",
            "ATAGIA_OPF_FALLBACK_URL": "http://192.168.50.22:8008",
        }
    )

    client = OpenAIPrivacyFilterClient.from_settings(settings)

    assert client._restricted_transport is True


@pytest.mark.asyncio
async def test_detect_uses_primary_endpoint_and_strips_raw_span_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://primary.test/detect"
        return httpx.Response(
            200,
            json={
                "spans": [
                    {"label": "private_address", "start": 7, "end": 11, "text": "8642"}
                ]
            },
        )

    client = _client(httpx.MockTransport(handler))

    detection = await client.detect("code is 8642")

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


@pytest.mark.asyncio
async def test_restricted_opf_transport_ignores_proxy_environment_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"spans": []}

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

        async def __aenter__(self) -> "RecordingClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def request(self, *args: object, **kwargs: object) -> Response:
            return Response()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(
        "atagia.services.privacy_filter_client.httpx.AsyncClient",
        RecordingClient,
    )
    client = OpenAIPrivacyFilterClient(
        primary_url="http://127.0.0.1:8008",
        fallback_url="http://127.0.0.1:8008",
        timeout_seconds=1.0,
        restricted_transport=True,
    )

    detection = await client.detect("test")

    assert detection.span_count == 0
    assert created == [
        {
            "timeout": 1.0,
            "trust_env": False,
            "follow_redirects": False,
        }
    ]
