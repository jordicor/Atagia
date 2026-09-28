"""Proxy boundary matrices for memory scope and prompt authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from fastapi import FastAPI
from fastapi.testclient import TestClient
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
from tests.api.test_openai_proxy import _settings
from tests.turn_telemetry_support import TurnCallMeterMixin


class _ProxyLLMClient(TurnCallMeterMixin):
    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            provider="proxy-control-test",
            model=request.model,
            output_text="ok",
        )

    async def stream(self, request: LLMCompletionRequest):
        yield LLMStreamEvent(type="text", content="ok")
        yield LLMStreamEvent(type="done", payload={})


_Headers: TypeAlias = dict[str, str] | list[tuple[str, str]]


def _app(tmp_path: Path) -> FastAPI:
    return create_app(_settings(tmp_path))


def _post(
    tmp_path: Path,
    *,
    headers: _Headers,
    body: dict[str, Any],
):
    app = _app(tmp_path)
    with TestClient(app) as client:
        app.state.runtime.llm_client = _ProxyLLMClient()
        return client.post("/v1/chat/completions", headers=headers, json=body)


def _headers(**extra: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_proxy_controls",
        "X-Atagia-Conversation-Id": "cnv_proxy_controls",
        "X-Atagia-Platform-Id": "proxy-tests",
        **extra,
    }


def _body(*, stream: bool, **extra: Any) -> dict[str, Any]:
    return {
        "model": "atagia-memory-proxy",
        "stream": stream,
        "messages": [{"role": "user", "content": "Check memory scope."}],
        **extra,
    }


def _capture_retrieval_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Any, Any]]:
    calls: list[tuple[Any, Any]] = []

    async def capture_context(
        _self: Any,
        identity: Any,
        _input_projection: Any,
        *,
        message_metadata: dict[str, Any],
        prompt_authority_context: Any = None,
    ) -> ProxyContextAttempt:
        assert message_metadata
        calls.append((identity, prompt_authority_context))
        return ProxyContextAttempt(context=None, elapsed_ms=0.0)

    monkeypatch.setattr(
        OpenAIProxyService,
        "_context_for_turn_fail_open",
        capture_context,
    )
    return calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("incognito", [None, False, True])
@pytest.mark.parametrize("cross_chat_memory", [None, False, True, "invalid"])
def test_proxy_memory_scope_matrix_reaches_retrieval_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    incognito: bool | None,
    cross_chat_memory: bool | str | None,
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)
    extra: dict[str, Any] = {}
    if incognito is not None:
        extra["incognito"] = incognito
    if cross_chat_memory is not None:
        extra["cross_chat_memory"] = cross_chat_memory

    response = _post(
        tmp_path,
        headers=_headers(),
        body=_body(stream=stream, **extra),
    )

    if cross_chat_memory == "invalid":
        assert response.status_code == 400
        assert response.json()["error"]["message"] == (
            "Invalid boolean for cross_chat_memory from typed.cross_chat_memory"
        )
        assert calls == []
        return

    assert response.status_code == 200
    assert len(calls) == 1
    identity, authority = calls[0]
    assert identity.incognito is incognito
    assert identity.cross_chat_memory is (
        False
        if incognito is True
        else True
        if cross_chat_memory is None
        else cross_chat_memory
    )
    assert authority.privacy_enforcement == "enforce"
    assert authority.normalized_privilege_level == "standard"
    assert authority.authenticated_user_is_atagia_master is False
    assert authority.authority_source == "ordinary_http_boundary:service_api_key"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("body_extra", "metadata", "header_extra", "expected"),
    [
        (
            {"incognito": False},
            {"atagia_incognito": "false"},
            {"X-Atagia-Incognito": "0"},
            (False, True),
        ),
        (
            {"cross_chat_memory": False},
            {"atagia_cross_chat_memory": "off"},
            {"X-Atagia-Cross-Chat-Memory": "no"},
            (None, False),
        ),
        (
            {"incognito": True, "cross_chat_memory": True},
            {"incognito": "yes", "cross_chat_memory": "on"},
            {
                "X-Atagia-Incognito": "1",
                "X-Atagia-Cross-Chat-Memory": "true",
            },
            (True, False),
        ),
    ],
)
def test_proxy_consistent_cross_source_claims_reach_retrieval_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    body_extra: dict[str, object],
    metadata: dict[str, object],
    header_extra: dict[str, str],
    expected: tuple[bool | None, bool],
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)

    response = _post(
        tmp_path,
        headers=_headers(**header_extra),
        body=_body(stream=stream, metadata=metadata, **body_extra),
    )

    assert response.status_code == 200
    identity, _authority = calls[0]
    assert (identity.incognito, identity.cross_chat_memory) == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("setting", "body_extra", "metadata", "header_extra"),
    [
        ("incognito", {"incognito": True}, {"incognito": False}, {}),
        (
            "incognito",
            {"incognito": True},
            {},
            {"X-Atagia-Incognito": "false"},
        ),
        (
            "incognito",
            {},
            {"atagia_incognito": True},
            {"X-Atagia-Incognito": "false"},
        ),
        (
            "cross_chat_memory",
            {"cross_chat_memory": False},
            {"cross_chat_memory": True},
            {},
        ),
        (
            "cross_chat_memory",
            {"cross_chat_memory": False},
            {},
            {"X-Atagia-Cross-Chat-Memory": "true"},
        ),
        (
            "cross_chat_memory",
            {},
            {"atagia_cross_chat_memory": False},
            {"X-Atagia-Cross-Chat-Memory": "true"},
        ),
    ],
)
def test_proxy_cross_source_contradictions_return_stable_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    setting: str,
    body_extra: dict[str, object],
    metadata: dict[str, object],
    header_extra: dict[str, str],
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)

    response = _post(
        tmp_path,
        headers=_headers(**header_extra),
        body=_body(stream=stream, metadata=metadata, **body_extra),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Conflicting {setting} values across request sources"
    )
    assert calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("metadata", "header_extra", "message"),
    [
        (
            {"incognito": "maybe"},
            {},
            "Invalid boolean for incognito from metadata.incognito",
        ),
        (
            {},
            {"X-Atagia-Cross-Chat-Memory": "sometimes"},
            (
                "Invalid boolean for cross_chat_memory "
                "from header.X-Atagia-Cross-Chat-Memory"
            ),
        ),
    ],
)
def test_proxy_invalid_cross_source_booleans_return_stable_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    metadata: dict[str, object],
    header_extra: dict[str, str],
    message: str,
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)

    response = _post(
        tmp_path,
        headers=_headers(**header_extra),
        body=_body(stream=stream, metadata=metadata),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == message
    assert calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("claim_source", "claim", "value"),
    [
        ("body", "privacy_enforcement", "off"),
        ("body", "authenticated_user_is_atagia_master", True),
        ("metadata", "atagia_privacy_enforcement", "off"),
        ("metadata", "authenticated_user_privilege_level", "atagia_master"),
        ("header", "X-Atagia-Privacy-Enforcement", "off"),
        ("header", "X-Atagia-Authenticated-Atagia-Master", "true"),
    ],
)
def test_proxy_rejects_remote_authority_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    claim_source: str,
    claim: str,
    value: object,
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)
    body_extra: dict[str, Any] = {}
    header_extra: dict[str, str] = {}
    if claim_source == "body":
        body_extra[claim] = value
    elif claim_source == "metadata":
        body_extra["metadata"] = {claim: value}
    else:
        header_extra[claim] = str(value)

    response = _post(
        tmp_path,
        headers=_headers(**header_extra),
        body=_body(stream=stream, **body_extra),
    )

    assert response.status_code == 400
    assert "Remote authority claim" in response.json()["error"]["message"]
    assert calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "header_name",
    ["X-Atagia-Incognito", "X-Atagia-Cross-Chat-Memory"],
)
def test_proxy_rejects_contradictory_duplicate_scope_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    header_name: str,
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)
    headers = list(_headers().items())
    headers.extend([(header_name, "false"), (header_name, "true")])

    response = _post(
        tmp_path,
        headers=headers,
        body=_body(stream=stream),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Duplicate {header_name} header is not allowed"
    )
    assert calls == []


@pytest.mark.parametrize(
    "header_name",
    [
        "Authorization",
        "X-Atagia-User-Id",
        "X-Atagia-Conversation-Id",
        "X-Atagia-Assistant-Mode",
        "X-Atagia-Mode",
        "X-Atagia-Workspace-Id",
        "X-Atagia-User-Persona-Id",
        "X-Atagia-Platform-Id",
        "X-Atagia-Character-Id",
        "X-Atagia-Active-Presence-Id",
        "X-Atagia-Mind-Id",
        "X-Atagia-Mind-Topology",
        "X-Atagia-Embodiment-Id",
        "X-Atagia-Realm-Id",
        "X-Atagia-Space-Id",
        "X-Atagia-Message-Id",
        "X-Atagia-Source-Seq",
        "X-Atagia-Response-Message-Id",
        "X-Atagia-Response-Source-Seq",
        "X-Atagia-Ingest-Origin",
        "X-Atagia-Confirmation-Strategy",
        "X-Atagia-Memory-Privacy-Mode",
        "X-Atagia-Response-Mode",
        "X-Atagia-Adaptive-Retrieval",
    ],
)
def test_proxy_rejects_every_duplicate_singleton_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
) -> None:
    calls = _capture_retrieval_identity(monkeypatch)
    headers = [item for item in _headers().items() if item[0] != header_name]
    first = "Bearer service-key" if header_name == "Authorization" else "first"
    second = "Bearer other-key" if header_name == "Authorization" else "second"
    headers.extend([(header_name, first), (header_name, second)])

    response = _post(
        tmp_path,
        headers=headers,
        body=_body(stream=False),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Duplicate {header_name} header is not allowed"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("header_name", "values"),
    [
        ("Authorization", ("Bearer service-key", "Bearer other-key")),
        ("X-Atagia-User-Id", ("usr_proxy_controls", "usr_other")),
    ],
)
def test_proxy_models_rejects_duplicate_authentication_headers(
    tmp_path: Path,
    header_name: str,
    values: tuple[str, str],
) -> None:
    headers = [item for item in _headers().items() if item[0] != header_name]
    headers.extend([(header_name, values[0]), (header_name, values[1])])

    with TestClient(_app(tmp_path)) as client:
        response = client.get("/v1/models", headers=headers)

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Duplicate {header_name} header is not allowed"
    )
