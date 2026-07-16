"""End-to-end path-ID transport coverage for clients, importers, and admin routes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import httpx
import pytest

from atagia.app import create_app
from atagia.api.path_ids import TransportIdRoute
from atagia.api.routes_activity import router as activity_router
from atagia.api.routes_admin import router as admin_router
from atagia.api.routes_chat import router as chat_router
from atagia.api.routes_memory import router as memory_router
from atagia.api.routes_verbatim_pins import router as verbatim_pins_router
from atagia.client import HttpAtagiaClient
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    VerbatimPinTargetKind,
)
from atagia.transport_ids import encode_path_id

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = ROOT / "src" / "atagia" / "resources" / "manifests"
PATH_ID_CASES = (
    "ordinary-safe_id:1",
    "id/with/slashes",
    "literal%2Fencoding",
    "id with spaces",
    "unicode-日本語-ñ",
    "__atagia_b64_reserved-prefix",
    "__atagia_b64_aWQvZG91YmxlLWRlY29kZS1ndWFyZA",
)


def _settings(tmp_path: Path, *, service_mode: bool) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "transport-ids.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_forced_global_model="openai/test-model",
        service_mode=service_mode,
        service_api_key="service-key" if service_mode else None,
        admin_api_key="admin-key" if service_mode else None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


def test_every_application_dynamic_route_uses_central_transport_decoding(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path, service_mode=True))
    assert app.routes
    dynamic_routes = [
        route
        for router in (
            activity_router,
            admin_router,
            chat_router,
            memory_router,
            verbatim_pins_router,
        )
        for route in router.routes
        if isinstance(route, APIRoute) and "{" in route.path
    ]

    assert dynamic_routes
    assert all(isinstance(route, TransportIdRoute) for route in dynamic_routes)
    assert {
        "atagia.api.routes_activity",
        "atagia.api.routes_admin",
        "atagia.api.routes_chat",
        "atagia.api.routes_memory",
        "atagia.api.routes_verbatim_pins",
    } <= {route.endpoint.__module__ for route in dynamic_routes}


def test_dynamic_route_families_round_trip_unsafe_ids_once_at_the_asgi_boundary(
    tmp_path: Path,
) -> None:
    raw_user_id = "__atagia_b64_aWQvZG91YmxlLWRlY29kZS1ndWFyZA"
    raw_conversation_id = "__atagia_b64_reserved-prefix"
    raw_memory_id = "memory/id with slash"
    raw_pin_id = "pin-unicode-日本語-ñ"
    platform_id = "transport-matrix"
    app = create_app(_settings(tmp_path, service_mode=True))
    service_headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": raw_user_id,
        "X-Atagia-Platform-Id": platform_id,
    }
    admin_headers = {"Authorization": "Bearer admin-key"}

    with TestClient(app) as client:
        runtime = client.app.state.runtime
        connection = client.portal.call(runtime.open_connection)
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            pins = VerbatimPinRepository(connection, runtime.clock)
            client.portal.call(users.create_user, raw_user_id)
            client.portal.call(
                lambda: conversations.create_conversation(
                    raw_conversation_id,
                    raw_user_id,
                    None,
                    "personal_assistant",
                    "Transport route matrix",
                    platform_id=platform_id,
                )
            )
            source = client.portal.call(
                lambda: messages.create_message(
                    "msg_transport_matrix",
                    raw_conversation_id,
                    "user",
                    1,
                    "Transport source",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    memory_id=raw_memory_id,
                    user_id=raw_user_id,
                    conversation_id=raw_conversation_id,
                    assistant_mode_id="personal_assistant",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.USER,
                    canonical_text="Transport memory",
                    payload={"source_message_ids": [str(source["id"])]},
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_category=MemoryCategory.UNKNOWN,
                    status=MemoryStatus.ACTIVE,
                    platform_id=platform_id,
                )
            )
            client.portal.call(
                lambda: pins.create_verbatim_pin(
                    pin_id=raw_pin_id,
                    user_id=raw_user_id,
                    scope=MemoryScope.CONVERSATION,
                    target_kind=VerbatimPinTargetKind.MESSAGE,
                    target_id=str(source["id"]),
                    conversation_id=raw_conversation_id,
                    assistant_mode_id="personal_assistant",
                    canonical_text="Transport pin",
                    index_text="Transport pin",
                    privacy_level=0,
                    created_by=raw_user_id,
                    platform_id=platform_id,
                )
            )
        finally:
            client.portal.call(connection.close)

        activity = client.get(
            f"/v1/users/{encode_path_id(raw_user_id)}/activity/conversations",
            params={
                "conversation_id": raw_conversation_id,
                "platform_id": platform_id,
                "refresh": "false",
            },
            headers=service_headers,
        )
        chat = client.post(
            f"/v1/conversations/{encode_path_id(raw_conversation_id)}/incognito",
            json={
                "user_id": raw_user_id,
                "platform_id": platform_id,
                "incognito": False,
            },
            headers=service_headers,
        )
        memory = client.get(
            f"/v1/memory/objects/{encode_path_id(raw_memory_id)}",
            params={
                "user_id": raw_user_id,
                "conversation_id": raw_conversation_id,
                "platform_id": platform_id,
            },
            headers=service_headers,
        )
        pin = client.get(
            f"/v1/verbatim-pins/{encode_path_id(raw_pin_id)}",
            params={
                "user_id": raw_user_id,
                "conversation_id": raw_conversation_id,
                "platform_id": platform_id,
            },
            headers=service_headers,
        )
        admin = client.get(
            f"/v1/admin/consequence-chains/{encode_path_id(raw_user_id)}",
            headers=admin_headers,
        )
        malformed = client.get(
            "/v1/admin/consequence-chains/__atagia_b64_A",
            headers=admin_headers,
        )

    assert activity.status_code == 200, activity.text
    assert activity.json()["user_id"] == raw_user_id
    assert chat.status_code == 200, chat.text
    assert chat.json()["id"] == raw_conversation_id
    assert memory.status_code == 200, memory.text
    assert memory.json()["id"] == raw_memory_id
    assert pin.status_code == 200, pin.text
    assert pin.json()["id"] == raw_pin_id
    assert admin.status_code == 200, admin.text
    assert malformed.status_code == 400
    assert malformed.json()["detail"] == "Invalid Atagia transport id"


@pytest.mark.asyncio
async def test_http_admin_client_round_trips_path_ids_exactly_once(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path, service_mode=True))
    async with app.router.lifespan_context(app):
        connection = await app.state.runtime.open_connection()
        try:
            users = UserRepository(connection, app.state.runtime.clock)
            conversations = ConversationRepository(connection, app.state.runtime.clock)
            messages = MessageRepository(connection, app.state.runtime.clock)
            memories = MemoryObjectRepository(connection, app.state.runtime.clock)
            for index, raw_id in enumerate(PATH_ID_CASES):
                user_id = f"user::{raw_id}"
                conversation_id = f"cnv_transport_{index}"
                await users.create_user(user_id)
                await conversations.create_conversation(
                    conversation_id,
                    user_id,
                    None,
                    "personal_assistant",
                    "Transport test",
                    platform_id="transport-tests",
                )
                source_message = await messages.create_message(
                    f"msg_transport_source_{index}",
                    conversation_id,
                    "user",
                    1,
                    f"Review source {index}",
                )
                for action in ("archive", "delete"):
                    await memories.create_memory_object(
                        memory_id=f"{action}::{raw_id}",
                        user_id=user_id,
                        conversation_id=conversation_id,
                        assistant_mode_id="personal_assistant",
                        object_type=MemoryObjectType.EVIDENCE,
                        scope=MemoryScope.USER,
                        canonical_text=f"Review item {index} {action}",
                        payload={
                            "ingest_origin": "backfill",
                            "source_message_ids": [str(source_message["id"])],
                        },
                        source_kind=MemorySourceKind.EXTRACTED,
                        confidence=0.9,
                        privacy_level=0,
                        memory_category=MemoryCategory.UNKNOWN,
                        status=MemoryStatus.REVIEW_REQUIRED,
                        platform_id="transport-tests",
                    )
        finally:
            await connection.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                admin_api_key="admin-key",
                http_client=http_client,
            )
            for raw_id in PATH_ID_CASES:
                user_id = f"user::{raw_id}"
                archived = await client.archive_review_required_memory(
                    user_id,
                    f"archive::{raw_id}",
                )
                deleted = await client.delete_review_required_memory(
                    user_id,
                    f"delete::{raw_id}",
                )
                assert archived.memory_id == f"archive::{raw_id}"
                assert deleted.memory_id == f"delete::{raw_id}"

        connection = await app.state.runtime.open_connection()
        try:
            for raw_id in PATH_ID_CASES:
                user_id = f"user::{raw_id}"
                archived = await MemoryObjectRepository(
                    connection,
                    app.state.runtime.clock,
                ).get_memory_object(f"archive::{raw_id}", user_id)
                deleted = await MemoryObjectRepository(
                    connection,
                    app.state.runtime.clock,
                ).get_memory_object(f"delete::{raw_id}", user_id)
                assert archived is not None
                assert archived["status"] == MemoryStatus.ARCHIVED.value
                assert deleted is None
        finally:
            await connection.close()


def test_importer_round_trips_conversation_path_ids_into_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_importer_module("atagia_importers_transport")
    app = create_app(_settings(tmp_path, service_mode=True))
    with TestClient(app) as asgi_client:

        def route_urlopen(request: Any, *, timeout: float) -> Any:
            del timeout
            response = asgi_client.request(
                request.get_method(),
                urlsplit(request.full_url).path,
                content=request.data,
                headers=dict(request.header_items()),
            )
            response.raise_for_status()
            return _UrlopenResponse(response.content)

        monkeypatch.setattr(module, "urlopen", route_urlopen)
        importer = module.AtagiaImportClient(
            base_url="http://testserver",
            api_key="service-key",
        )
        for index, conversation_id in enumerate(PATH_ID_CASES):
            importer.ingest_message(
                user_id="usr_import_transport",
                conversation_id=conversation_id,
                role="user",
                text=f"Imported transport row {index}",
                platform_id="importer-tests",
                message_id=f"msg_transport_{index}",
                source_seq=index + 1,
            )

        connection = asgi_client.portal.call(
            asgi_client.app.state.runtime.open_connection
        )
        try:
            cursor = asgi_client.portal.call(
                connection.execute,
                """
                SELECT id
                FROM conversations
                WHERE user_id = ?
                ORDER BY id
                """,
                ("usr_import_transport",),
            )
            stored_conversation_ids = {
                str(row[0]) for row in asgi_client.portal.call(cursor.fetchall)
            }
            asgi_client.portal.call(cursor.close)
            cursor = asgi_client.portal.call(
                connection.execute,
                """
                SELECT m.conversation_id, m.text
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.user_id = ?
                """,
                ("usr_import_transport",),
            )
            stored_messages = {
                (str(row[0]), str(row[1]))
                for row in asgi_client.portal.call(cursor.fetchall)
            }
            asgi_client.portal.call(cursor.close)
        finally:
            asgi_client.portal.call(connection.close)

    assert stored_conversation_ids == set(PATH_ID_CASES)
    assert stored_messages == {
        (conversation_id, f"Imported transport row {index}")
        for index, conversation_id in enumerate(PATH_ID_CASES)
    }


class _UrlopenResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> "_UrlopenResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._content


def _load_importer_module(name: str) -> ModuleType:
    path = ROOT / "integrations" / "importers" / "atagia_importers.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module
