"""Regression tests for ordinary HTTP prompt-authority boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from atagia.api.routes_chat import router as chat_router
from atagia.services.chat_service import ChatService
from atagia.services.errors import ConversationNotFoundError
from atagia.services.sidecar_service import SidecarService


_SERVICE_HEADERS = {
    "Authorization": "Bearer service-key",
    "X-Atagia-User-Id": "usr_security",
}
_ROUTES = (
    (
        "/v1/chat/cnv_security/reply",
        ChatService,
        "chat_reply",
        {
            "user_id": "usr_security",
            "message_text": "Check authority.",
            "metadata": {},
            "platform_id": "web",
        },
    ),
    (
        "/v1/conversations/cnv_security/context",
        SidecarService,
        "get_context",
        {
            "user_id": "usr_security",
            "message_text": "Check authority.",
            "platform_id": "web",
        },
    ),
    (
        "/v1/conversations/cnv_security/messages",
        SidecarService,
        "ingest_message",
        {
            "user_id": "usr_security",
            "role": "user",
            "text": "Check authority.",
            "platform_id": "web",
        },
    ),
    (
        "/v1/conversations/cnv_security/responses",
        SidecarService,
        "add_response",
        {
            "user_id": "usr_security",
            "text": "Check authority.",
            "platform_id": "web",
        },
    ),
)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(chat_router)
    app.state.runtime = SimpleNamespace(
        settings=SimpleNamespace(
            service_mode=True,
            service_api_key="service-key",
            admin_api_key="admin-key",
        )
    )
    return app


def _capture_downstream(
    monkeypatch: pytest.MonkeyPatch,
    service_class: type[Any],
    method_name: str,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def capture(_self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise ConversationNotFoundError("captured after route boundary")

    monkeypatch.setattr(service_class, method_name, capture)
    return calls


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_ordinary_http_routes_pass_only_server_derived_standard_authority(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(path, headers=_SERVICE_HEADERS, json=body)

    assert response.status_code == 404
    assert len(calls) == 1
    authority = calls[0]["prompt_authority_context"]
    assert authority.privacy_enforcement == "enforce"
    assert authority.normalized_privilege_level == "standard"
    assert authority.authenticated_user_is_atagia_master is False
    assert authority.user_id == "usr_security"
    assert authority.authority_source == "ordinary_http_boundary:service_api_key"
    assert "privacy_enforcement" not in calls[0]
    assert "authenticated_user_privilege_level" not in calls[0]
    assert "authenticated_user_is_atagia_master" not in calls[0]


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("privacy_enforcement", "off"),
        ("authenticated_user_privilege_level", "atagia_master"),
        ("authenticated_user_is_atagia_master", True),
    ],
)
@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_ordinary_http_routes_reject_body_authority_claims(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
    claim: str,
    value: object,
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers=_SERVICE_HEADERS,
        json={**body, claim: value},
    )

    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    "header",
    [
        "X-Atagia-Privacy-Enforcement",
        "X-Atagia-Authenticated-User-Privilege-Level",
        "X-Atagia-Authenticated-User-Is-Atagia-Master",
    ],
)
@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_ordinary_http_routes_reject_header_authority_claims(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
    header: str,
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers={**_SERVICE_HEADERS, header: "off"},
        json=body,
    )

    assert response.status_code == 400
    assert calls == []


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("privacy_enforcement", "off"),
        ("authenticated_user_privilege_level", "atagia_master"),
        ("authenticated_user_is_atagia_master", True),
        ("atagia_privacy_enforcement", "off"),
        ("atagia_authenticated_atagia_master", True),
    ],
)
def test_chat_metadata_rejects_authority_claims(
    monkeypatch: pytest.MonkeyPatch,
    claim: str,
    value: object,
) -> None:
    calls = _capture_downstream(monkeypatch, ChatService, "chat_reply")
    path, _service, _method, body = _ROUTES[0]

    response = TestClient(_app()).post(
        path,
        headers=_SERVICE_HEADERS,
        json={**body, "metadata": {claim: value}},
    )

    assert response.status_code == 400
    assert calls == []


def test_admin_key_is_not_accepted_as_ordinary_master_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_downstream(monkeypatch, ChatService, "chat_reply")
    path, _service, _method, body = _ROUTES[0]

    response = TestClient(_app()).post(
        path,
        headers={
            "Authorization": "Bearer admin-key",
            "X-Atagia-User-Id": "usr_security",
        },
        json=body,
    )

    assert response.status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES[:3],
)
def test_direct_routes_apply_incognito_strictest_wins_before_service(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers=_SERVICE_HEADERS,
        json={**body, "incognito": True, "cross_chat_memory": True},
    )

    assert response.status_code == 404
    assert calls[0]["incognito"] is True
    assert calls[0]["cross_chat_memory"] is False


@pytest.mark.parametrize(
    ("scope_headers", "expected_incognito", "expected_cross_chat"),
    [
        (
            {
                "X-Atagia-Incognito": "true",
                "X-Atagia-Cross-Chat-Memory": "true",
            },
            True,
            False,
        ),
        (
            {
                "X-Atagia-Incognito": "false",
                "X-Atagia-Cross-Chat-Memory": "false",
            },
            False,
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_direct_routes_apply_header_memory_scope_before_service(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
    scope_headers: dict[str, str],
    expected_incognito: bool,
    expected_cross_chat: bool,
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers={**_SERVICE_HEADERS, **scope_headers},
        json=body,
    )

    assert response.status_code == 404
    assert calls[0]["incognito"] is expected_incognito
    if "cross_chat_memory" in calls[0]:
        assert calls[0]["cross_chat_memory"] is expected_cross_chat


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_direct_routes_reject_invalid_typed_incognito_with_400(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers=_SERVICE_HEADERS,
        json={**body, "incognito": "sometimes"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid boolean for incognito from typed.incognito"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_direct_routes_reject_invalid_header_scope_with_400(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers={
            **_SERVICE_HEADERS,
            "X-Atagia-Cross-Chat-Memory": "sometimes",
        },
        json=body,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid boolean for cross_chat_memory from header.X-Atagia-Cross-Chat-Memory"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
def test_direct_routes_reject_typed_header_scope_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)

    response = TestClient(_app()).post(
        path,
        headers={**_SERVICE_HEADERS, "X-Atagia-Incognito": "false"},
        json={**body, "incognito": True},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Conflicting incognito values across request sources"
    )
    assert calls == []


def test_chat_rejects_contradictory_typed_and_metadata_scope_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_downstream(monkeypatch, ChatService, "chat_reply")
    path, _service, _method, body = _ROUTES[0]

    response = TestClient(_app()).post(
        path,
        headers=_SERVICE_HEADERS,
        json={
            **body,
            "cross_chat_memory": False,
            "metadata": {"atagia_cross_chat_memory": True},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Conflicting cross_chat_memory values across request sources"
    )
    assert calls == []


def test_chat_rejects_contradictory_metadata_and_header_scope_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_downstream(monkeypatch, ChatService, "chat_reply")
    path, _service, _method, body = _ROUTES[0]

    response = TestClient(_app()).post(
        path,
        headers={
            **_SERVICE_HEADERS,
            "X-Atagia-Cross-Chat-Memory": "true",
        },
        json={
            **body,
            "metadata": {"atagia_cross_chat_memory": False},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Conflicting cross_chat_memory values across request sources"
    )
    assert calls == []


def test_openapi_does_not_advertise_ordinary_authority_claims() -> None:
    schema_text = str(TestClient(_app()).get("/openapi.json").json())

    assert "privacy_enforcement" not in schema_text
    assert "authenticated_user_privilege_level" not in schema_text
    assert "authenticated_user_is_atagia_master" not in schema_text


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
@pytest.mark.parametrize(
    "header_name",
    ["X-Atagia-Incognito", "X-Atagia-Cross-Chat-Memory"],
)
def test_direct_routes_reject_contradictory_duplicate_scope_headers(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
    header_name: str,
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)
    headers = list(_SERVICE_HEADERS.items())
    headers.extend([(header_name, "false"), (header_name, "true")])

    response = TestClient(_app()).post(path, headers=headers, json=body)

    setting = header_name.removeprefix("X-Atagia-").lower().replace("-", "_")
    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"Conflicting {setting} values across request sources"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("path", "service_class", "method_name", "body"),
    _ROUTES,
)
@pytest.mark.parametrize(
    ("header_name", "values"),
    [
        ("Authorization", ("Bearer service-key", "Bearer different-key")),
        ("X-Atagia-User-Id", ("usr_security", "usr_other")),
    ],
)
def test_direct_routes_reject_duplicate_authentication_headers(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    service_class: type[Any],
    method_name: str,
    body: dict[str, Any],
    header_name: str,
    values: tuple[str, str],
) -> None:
    calls = _capture_downstream(monkeypatch, service_class, method_name)
    headers = [item for item in _SERVICE_HEADERS.items() if item[0] != header_name]
    headers.extend([(header_name, values[0]), (header_name, values[1])])

    response = TestClient(_app()).post(path, headers=headers, json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"Duplicate {header_name} header is not allowed"
    )
    assert calls == []
