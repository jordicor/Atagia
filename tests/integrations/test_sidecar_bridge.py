from __future__ import annotations

from typing import Any

import pytest

from atagia.integrations.sidecar_bridge import SidecarBridge, SidecarBridgeConfig


class FakeAtagiaClient:
    def __init__(self) -> None:
        self.created_users: list[str] = []
        self.created_conversations: list[dict[str, Any]] = []
        self.context_calls: list[dict[str, Any]] = []
        self.ingest_calls: list[dict[str, Any]] = []
        self.response_calls: list[dict[str, Any]] = []
        self.closed = False
        self.fail_context = False
        self.fail_response = False
        self.context_result = {"system_prompt": "Memory context"}

    async def create_user(self, user_id: str) -> None:
        self.created_users.append(user_id)

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
    ) -> str:
        self.created_conversations.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "assistant_mode_id": assistant_mode_id,
            }
        )
        return conversation_id or "generated_conversation"

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> object:
        if self.fail_context:
            raise RuntimeError("context failure")
        self.context_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": message,
                "mode": mode,
                "workspace_id": workspace_id,
                "occurred_at": occurred_at,
                "attachments": attachments,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
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
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        self.ingest_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "role": role,
                "text": text,
                "mode": mode,
                "workspace_id": workspace_id,
                "occurred_at": occurred_at,
                "attachments": attachments,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
            }
        )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        if self.fail_response:
            raise RuntimeError("response failure")
        self.response_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "text": text,
                "occurred_at": occurred_at,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
            }
        )

    async def close(self) -> None:
        self.closed = True


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
    monkeypatch.setenv("ATAGIA_ASSISTANT_MODE", "companion")
    monkeypatch.setenv("ATAGIA_WORKSPACE_ID", "workspace-1")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILE", "normal")
    monkeypatch.setenv("ATAGIA_TIMEOUT_SECONDS", "12.5")

    config = SidecarBridgeConfig.from_env()

    assert config.enabled is True
    assert config.transport == "http"
    assert config.db_path == "/tmp/atagia.db"
    assert config.base_url == "http://localhost:8100"
    assert config.api_key == "service-key"
    assert config.assistant_mode == "companion"
    assert config.workspace_id == "workspace-1"
    assert config.operational_profile == "normal"
    assert config.timeout_seconds == 12.5


@pytest.mark.asyncio
async def test_bridge_maps_ids_and_context_options_to_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(
            enabled=True,
            transport="local",
            db_path="/tmp/atagia.db",
            assistant_mode="personal_assistant",
            workspace_id="workspace-1",
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
            "workspace_id": "workspace-1",
            "assistant_mode_id": "personal_assistant",
        }
    ]
    assert factory.client.context_calls == [
        {
            "user_id": "129",
            "conversation_id": "1892",
            "message": "remember this",
            "mode": "personal_assistant",
            "workspace_id": "workspace-1",
            "occurred_at": "2026-04-16T04:00:00+00:00",
            "attachments": [{"kind": "document", "name": "brief.pdf"}],
            "operational_profile": "low_power",
            "operational_signals": {"battery": "low"},
        }
    ]


@pytest.mark.asyncio
async def test_bridge_persists_ingest_and_response() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True, transport="local"),
        client_factory=factory,
    )

    ingested = await bridge.ingest_message(129, 1892, "user", "hello")
    recorded = await bridge.record_assistant_response(129, 1892, "got it")

    assert ingested is True
    assert recorded is True
    assert factory.client.ingest_calls[0]["text"] == "hello"
    assert factory.client.response_calls[0]["text"] == "got it"


@pytest.mark.asyncio
async def test_bridge_fails_open_when_context_or_response_fails() -> None:
    client = FakeAtagiaClient()
    client.fail_context = True
    client.fail_response = True
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True),
        client_factory=FakeFactory(client),
    )

    assert await bridge.get_context_for_turn(129, 1892, "hello") is None
    assert await bridge.record_assistant_response(129, 1892, "response") is False


@pytest.mark.asyncio
async def test_bridge_close_closes_initialized_client() -> None:
    factory = FakeFactory()
    bridge = SidecarBridge(
        SidecarBridgeConfig(enabled=True),
        client_factory=factory,
    )

    assert await bridge.get_context_for_turn(129, 1892, "hello") is not None

    await bridge.close()

    assert factory.client.closed is True
