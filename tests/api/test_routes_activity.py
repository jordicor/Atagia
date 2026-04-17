"""API tests for activity ranking and warm-up routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class NoopProvider(LLMProvider):
    name = "noop-activity-api-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM should not be called in activity API tests: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings should not be called in activity API tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-activity-api.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="test-model",
        llm_scoring_model="test-model",
        llm_classifier_model="test-model",
        llm_chat_model="test-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_activity_routes_respect_service_user_claims_and_warmup_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NoopProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    app = create_app(_settings(tmp_path))

    headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_1",
    }
    wrong_headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_2",
    }

    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            workspaces = WorkspaceRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(workspaces.create_workspace, "wrk_1", "usr_1", "Workspace", {"timezone": "UTC"})
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                "wrk_1",
                "coding_debug",
                "Activity Chat",
            )
            client.portal.call(
                messages.create_message,
                "msg_1",
                "cnv_1",
                "user",
                1,
                "Hello",
                2,
                {},
                "2026-03-10T11:00:00+00:00",
            )
            client.portal.call(
                messages.create_message,
                "msg_2",
                "cnv_1",
                "assistant",
                2,
                "Hi",
                2,
                {},
                "2026-03-10T11:01:00+00:00",
            )

        listing = client.get(
            "/v1/users/usr_1/activity/conversations",
            params={"limit": 5},
            headers=headers,
        )
        assert listing.status_code == 200
        assert listing.json()["conversation_count"] == 1
        assert listing.json()["conversations"][0]["conversation_id"] == "cnv_1"

        forbidden = client.get(
            "/v1/users/usr_1/activity/conversations",
            params={"limit": 5},
            headers=wrong_headers,
        )
        assert forbidden.status_code == 403

        invalid_limit = client.get(
            "/v1/users/usr_1/activity/conversations",
            params={"limit": -1},
            headers=headers,
        )
        assert invalid_limit.status_code == 422

        warmup = client.post(
            "/v1/conversations/cnv_1/warmup",
            json={"max_messages": 2},
            headers=headers,
        )
        assert warmup.status_code == 200
        assert warmup.json()["recent_window_key"] == "usr_1:cnv_1"
        assert warmup.json()["recent_message_count"] == 2
