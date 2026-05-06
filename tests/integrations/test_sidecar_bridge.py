from __future__ import annotations

from typing import Any

import httpx
import pytest

from atagia.integrations.sidecar_bridge import (
    SidecarBridge,
    SidecarBridgeConfig,
    SidecarBridgeError,
)
from atagia.models.schemas_jobs import WorkerControlMode


class FakeAtagiaClient:
    def __init__(self) -> None:
        self.created_users: list[str] = []
        self.created_conversations: list[dict[str, Any]] = []
        self.context_calls: list[dict[str, Any]] = []
        self.ingest_calls: list[dict[str, Any]] = []
        self.response_calls: list[dict[str, Any]] = []
        self.flush_calls: list[dict[str, Any]] = []
        self.worker_control_calls: list[dict[str, Any]] = []
        self.closed = False
        self.fail_context = False
        self.fail_ingest = False
        self.fail_response = False
        self.context_result = {"system_prompt": "Memory context"}
        self.worker_control_state = {"mode": "active"}

    async def create_user(self, user_id: str) -> None:
        self.created_users.append(user_id)

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str:
        self.created_conversations.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "mode": mode,
                "incognito": incognito,
            }
        )
        return conversation_id or "generated_conversation"

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> object:
        if self.fail_context:
            raise RuntimeError("context failure")
        self.context_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": message,
                "mode": mode,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "occurred_at": occurred_at,
                "attachments": attachments,
                "message_id": message_id,
                "source_seq": source_seq,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
            }
        )
        return self.context_result

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        text: str,
        mode: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        if self.fail_ingest:
            raise RuntimeError("ingest failure")
        self.ingest_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "role": role,
                "text": text,
                "mode": mode,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "occurred_at": occurred_at,
                "attachments": attachments,
                "message_id": message_id,
                "source_seq": source_seq,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
            }
        )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        if self.fail_response:
            raise RuntimeError("response failure")
        self.response_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "text": text,
                "occurred_at": occurred_at,
                "message_id": message_id,
                "source_seq": source_seq,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "mode": mode,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
            }
        )

    async def list_pending_memory_confirmations(self, user_id: str, **filters: Any) -> object:
        return {"items": [], "user_id": user_id, "filters": filters}

    async def confirm_pending_memory(self, user_id: str, memory_id: str) -> object:
        return {"ok": True, "user_id": user_id, "memory_id": memory_id}

    async def decline_pending_memory(self, user_id: str, memory_id: str) -> object:
        return {"ok": True, "user_id": user_id, "memory_id": memory_id}

    async def list_review_required_memories(self, **filters: Any) -> object:
        return {"items": [], "filters": filters}

    async def archive_review_required_memory(self, user_id: str, memory_id: str) -> object:
        return {"ok": True, "user_id": user_id, "memory_id": memory_id}

    async def delete_review_required_memory(self, user_id: str, memory_id: str) -> object:
        return {"ok": True, "user_id": user_id, "memory_id": memory_id}

    async def close(self) -> None:
        self.closed = True

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        self.flush_calls.append(
            {
                "timeout_seconds": timeout_seconds,
                "user_id": user_id,
            }
        )
        return True

    async def get_worker_control(self) -> dict[str, Any]:
        return self.worker_control_state

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        self.worker_control_calls.append(
            {
                "mode": WorkerControlMode(mode).value,
                "reason": reason,
                "timeout_seconds": timeout_seconds,
            }
        )
        self.worker_control_state = {"mode": WorkerControlMode(mode).value}
        return self.worker_control_state


class FakeFactory:
    def __init__(self, client: FakeAtagiaClient | None = None) -> None:
        self.client = client or FakeAtagiaClient()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> FakeAtagiaClient:
        self.calls.append(kwargs)
        return self.client


def test_config_from_env_parses_sidecar_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_TRANSPORT", "http")
    monkeypatch.setenv("ATAGIA_DB_PATH", "/tmp/atagia.db")
    monkeypatch.setenv("ATAGIA_BASE_URL", "http://localhost:8100")
    monkeypatch.setenv("ATAGIA_SERVICE_API_KEY", "service-key")
    monkeypatch.setenv("ATAGIA_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("ATAGIA_MODE", "companion")
    monkeypatch.setenv("ATAGIA_USER_PERSONA_ID", "persona-1")
    monkeypatch.setenv("ATAGIA_PLATFORM_ID", "platform-1")
    monkeypatch.setenv("ATAGIA_CHARACTER_ID", "character-1")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILE", "normal")
    monkeypatch.setenv("ATAGIA_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("ATAGIA_INCOGNITO", "true")

    config = SidecarBridgeConfig.from_env()

    assert config.enabled is True
    assert config.transport == "http"
    assert config.db_path == "/tmp/atagia.db"
    assert config.base_url == "http://localhost:8100"
    assert config.api_key == "service-key"
    assert config.admin_api_key == "admin-key"
    assert config.mode == "companion"
    assert config.user_persona_id == "persona-1"
    assert config.platform_id == "platform-1"
    assert config.character_id == "character-1"
    assert config.operational_profile == "normal"
    assert config.timeout_seconds == 12.5
    assert config.incognito is True


@pytest.mark.asyncio
async def test_bridge_maps_ids_and_context_options_to_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(
            enabled=True,
            transport="local",
            db_path="/tmp/atagia.db",
            mode="personal_assistant",
            platform_id="platform-1",
            character_id="character-1",
            timeout_seconds=7.0,
            operational_profile="low_power",
            operational_signals={"battery": "low"},
        ),
        client_factory=factory,
    )

    conversation_id = await bridge.ensure_user_and_conversation(129, 1892)
    context = await bridge.get_context_for_turn(
        129,
        1892,
        "remember this",
        occurred_at="2026-04-16T04:00:00+00:00",
        attachments=[{"kind": "document", "name": "brief.pdf"}],
    )

    assert conversation_id == "1892"
    assert context is factory.client.context_result
    assert factory.calls == [
        {
            "transport": "local",
            "db_path": "/tmp/atagia.db",
            "base_url": None,
            "api_key": None,
            "timeout": 7.0,
        }
    ]
    assert factory.client.created_users == ["129"]
    assert factory.client.created_conversations == [
            {
                "user_id": "129",
                "conversation_id": "1892",
                "user_persona_id": None,
                "platform_id": "platform-1",
                "character_id": "character-1",
                "mode": "personal_assistant",
                "incognito": False,
            }
    ]
    assert factory.client.context_calls == [
        {
            "user_id": "129",
            "conversation_id": "1892",
            "message": "remember this",
            "mode": "personal_assistant",
            "user_persona_id": None,
            "platform_id": "platform-1",
            "character_id": "character-1",
            "occurred_at": "2026-04-16T04:00:00+00:00",
            "attachments": [{"kind": "document", "name": "brief.pdf"}],
            "message_id": None,
            "source_seq": None,
            "operational_profile": "low_power",
            "operational_signals": {"battery": "low"},
            "incognito": False,
            "ingest_origin": None,
            "confirmation_strategy": None,
            "memory_privacy_mode": None,
        }
    ]


@pytest.mark.asyncio
async def test_bridge_accepts_aurvek_platform_id() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, transport="local", platform_id="aurvek"),
        client_factory=factory,
    )

    context = await bridge.get_context_for_turn(
        "aurvek:user:1",
        "aurvek:conv:2",
        "hello",
        character_id="prompt:3",
        message_id="aurvek:msg:4",
    )

    assert context is factory.client.context_result
    assert factory.client.context_calls == [
        {
            "user_id": "aurvek:user:1",
            "conversation_id": "aurvek:conv:2",
            "message": "hello",
            "mode": "personal_assistant",
            "user_persona_id": None,
            "platform_id": "aurvek",
            "character_id": "prompt:3",
            "occurred_at": None,
            "attachments": None,
            "message_id": "aurvek:msg:4",
            "source_seq": None,
            "operational_profile": None,
            "operational_signals": {},
            "incognito": False,
            "ingest_origin": None,
            "confirmation_strategy": None,
            "memory_privacy_mode": None,
        }
    ]


@pytest.mark.asyncio
async def test_bridge_persists_ingest_and_response() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, transport="local", platform_id="platform-1"),
        client_factory=factory,
    )

    ingested = await bridge.ingest_message(
        129,
        1892,
        "user",
        "hello",
        message_id="aurvek:msg:1",
        source_seq=10,
    )
    recorded = await bridge.record_assistant_response(
        129,
        1892,
        "got it",
        message_id="aurvek:msg:2",
        source_seq=11,
    )

    assert ingested is True
    assert recorded is True
    assert factory.client.ingest_calls[0]["text"] == "hello"
    assert factory.client.ingest_calls[0]["message_id"] == "aurvek:msg:1"
    assert factory.client.ingest_calls[0]["source_seq"] == 10
    assert factory.client.ingest_calls[0]["platform_id"] == "platform-1"
    assert factory.client.ingest_calls[0]["incognito"] is False
    assert factory.client.response_calls[0]["text"] == "got it"
    assert factory.client.response_calls[0]["message_id"] == "aurvek:msg:2"
    assert factory.client.response_calls[0]["source_seq"] == 11


@pytest.mark.asyncio
async def test_bridge_passes_backfill_ingest_origin() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, transport="local", platform_id="aurvek"),
        client_factory=factory,
    )

    ingested = await bridge.ingest_message(
        "aurvek:user:1",
        "aurvek:conv:2",
        "user",
        "historical message",
        message_id="aurvek:msg:9",
        ingest_origin="backfill",
    )

    assert ingested is True
    assert factory.client.ingest_calls[0]["ingest_origin"] == "backfill"
    assert factory.client.ingest_calls[0]["confirmation_strategy"] is None
    assert factory.client.ingest_calls[0]["memory_privacy_mode"] is None


@pytest.mark.asyncio
async def test_bridge_passes_memory_privacy_mode() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, transport="local", platform_id="aurvek"),
        client_factory=factory,
    )

    context = await bridge.get_context_for_turn(
        "aurvek:user:1",
        "aurvek:conv:2",
        "remember this",
        memory_privacy_mode="trusted_private",
    )
    recorded = await bridge.record_assistant_response(
        "aurvek:user:1",
        "aurvek:conv:2",
        "done",
        memory_privacy_mode="trusted_private",
    )

    assert context is factory.client.context_result
    assert recorded is True
    assert factory.client.context_calls[0]["memory_privacy_mode"] == "trusted_private"
    assert factory.client.response_calls[0]["memory_privacy_mode"] == "trusted_private"


@pytest.mark.asyncio
async def test_bridge_fails_open_when_context_or_response_fails() -> None:
    client = FakeAtagiaClient()
    client.fail_context = True
    client.fail_response = True
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, platform_id="platform-1"),
        client_factory=FakeFactory(client),
    )

    assert await bridge.get_context_for_turn(129, 1892, "hello") is None
    assert isinstance(bridge.last_error, SidecarBridgeError)
    assert bridge.last_error.operation == "get_context_for_turn"
    assert await bridge.record_assistant_response(129, 1892, "response") is False
    assert bridge.last_error is not None
    assert bridge.last_error.operation == "record_assistant_response"


@pytest.mark.asyncio
async def test_bridge_exposes_structured_http_error_for_sync_admin() -> None:
    client = FakeAtagiaClient()
    response = httpx.Response(
        status_code=409,
        json={"detail": "source_seq already exists"},
        request=httpx.Request("POST", "http://atagia.test/v1/conversations/c/messages"),
    )
    client.fail_ingest = True

    async def failing_ingest(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.HTTPStatusError(
            "409 conflict",
            request=response.request,
            response=response,
        )

    client.ingest_message = failing_ingest  # type: ignore[method-assign]
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, platform_id="platform-1"),
        client_factory=FakeFactory(client),
    )

    assert await bridge.ingest_message(129, 1892, "user", "hello") is False
    assert bridge.last_error == SidecarBridgeError(
        operation="ingest_message",
        error_type="HTTPStatusError",
        message="source_seq already exists",
        status_code=409,
        details="source_seq already exists",
    )


@pytest.mark.asyncio
async def test_bridge_close_closes_initialized_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, platform_id="platform-1"),
        client_factory=factory,
    )

    assert await bridge.get_context_for_turn(129, 1892, "hello") is not None

    await bridge.close()

    assert factory.client.closed is True


@pytest.mark.asyncio
async def test_bridge_flush_delegates_to_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, platform_id="platform-1", timeout_seconds=8.0),
        client_factory=factory,
    )

    assert await bridge.flush(user_id="aurvek:user:1") is True

    assert factory.client.flush_calls == [
        {
            "timeout_seconds": 8.0,
            "user_id": "aurvek:user:1",
        }
    ]


@pytest.mark.asyncio
async def test_bridge_worker_control_delegates_to_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(
            enabled=True,
            transport="http",
            base_url="http://atagia.test",
            api_key="service-key",
            admin_api_key="admin-key",
            platform_id="platform-1",
            timeout_seconds=8.0,
        ),
        client_factory=factory,
    )

    paused = await bridge.pause_new_jobs(reason="backup")
    hard_paused = await bridge.hard_pause(reason="restore")
    resumed = await bridge.resume_processing(reason="done")

    assert paused == {"mode": "pause_new_jobs"}
    assert hard_paused == {"mode": "hard_pause"}
    assert resumed == {"mode": "active"}
    assert factory.calls == [
        {
            "transport": "http",
            "db_path": None,
            "base_url": "http://atagia.test",
            "api_key": "service-key",
            "timeout": 8.0,
            "admin_api_key": "admin-key",
        }
    ]
    assert factory.client.worker_control_calls == [
        {
            "mode": "pause_new_jobs",
            "reason": "backup",
            "timeout_seconds": 8.0,
        },
        {
            "mode": "hard_pause",
            "reason": "restore",
            "timeout_seconds": 8.0,
        },
        {
            "mode": "active",
            "reason": "done",
            "timeout_seconds": 8.0,
        },
    ]
