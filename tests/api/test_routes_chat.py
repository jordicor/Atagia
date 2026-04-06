"""API tests for chat and creation routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import MemoryObjectRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.services.llm_client import (
    ConfigurationError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "queue-provider"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("No queued LLM output left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in Step 11 API tests")


def _settings(
    tmp_path: Path,
    *,
    service_mode: bool = False,
    allow_insecure_http: bool | None = None,
) -> Settings:
    if allow_insecure_http is None:
        allow_insecure_http = not service_mode
    return Settings(
        sqlite_path=str(tmp_path / "atagia-api.db"),
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
        llm_extraction_model="chat-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=service_mode,
        service_api_key="service-key" if service_mode else None,
        admin_api_key="admin-key" if service_mode else None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=allow_insecure_http,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_create_conversation_creates_and_returns_row(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {"source": "api"},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["user_id"] == "usr_1"
        assert payload["assistant_mode_id"] == "coding_debug"
        assert payload["title"] == "Debug Chat"
        assert payload["metadata_json"] == {"source": "api"}


def test_create_app_rejects_insecure_http_by_default(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="ATAGIA_SERVICE_MODE=true with API keys or ATAGIA_ALLOW_INSECURE_HTTP=true",
    ):
        create_app(_settings(tmp_path, service_mode=False, allow_insecure_http=False))


def test_chat_reply_runs_full_flow_and_returns_response(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = QueueProvider(
        [
            json.dumps([]),
            "Check the retry guard first.",
        ]
    )
    with TestClient(app) as client:
        client.app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        ).json()

        response = client.post(
            f"/v1/chat/{conversation['id']}/reply",
            json={
                "user_id": "usr_1",
                "message_text": "Please help me debug this retry loop.",
                "include_thinking": False,
                "metadata": {"channel": "api"},
                "debug": True,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["conversation_id"] == conversation["id"]
        assert payload["reply_text"] == "Check the retry guard first."
        assert payload["retrieval_event_id"] is not None
        assert payload["debug"]["cold_start"] is True
        assert len(provider.requests) == 2
        assert provider.requests[1].model == "reply-test-model"


def test_chat_reply_includes_current_user_state_in_system_prompt(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = QueueProvider(
        [
            json.dumps([]),
            "State-aware reply.",
        ]
    )
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        ).json()
        with _connection(client) as connection:
            memories = MemoryObjectRepository(connection, runtime.clock)
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id=conversation["id"],
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.STATE_SNAPSHOT,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="User is focused on websocket retries",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    payload={"focus_topic": "websocket retries", "urgency": "high"},
                    memory_id="mem_state_1",
                )
            )

        response = client.post(
            f"/v1/chat/{conversation['id']}/reply",
            json={
                "user_id": "usr_1",
                "message_text": "Please help me debug this retry loop.",
                "include_thinking": False,
                "metadata": {"channel": "api"},
                "debug": False,
            },
        )

        assert response.status_code == 200
        system_prompt = provider.requests[1].messages[0].content
        assert "[Current User State]" in system_prompt
        assert "focus_topic: websocket retries" in system_prompt
        assert "urgency: high" in system_prompt


def test_chat_reply_prevents_cross_user_conversation_access_in_service_mode(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, service_mode=True))
    provider = QueueProvider([json.dumps([]), "irrelevant"])
    create_headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_1",
    }
    with TestClient(app) as client:
        client.app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        conversation = client.post(
            "/v1/conversations",
            headers=create_headers,
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Owned by user one",
                "metadata": {},
            },
        ).json()

        response = client.post(
            f"/v1/chat/{conversation['id']}/reply",
            headers={
                "Authorization": "Bearer service-key",
                "X-Atagia-User-Id": "usr_2",
            },
            json={
                "user_id": "usr_2",
                "message_text": "Try to access another user's conversation.",
                "include_thinking": False,
                "metadata": {},
                "debug": False,
            },
        )

        assert response.status_code == 404


def test_chat_reply_includes_workspace_rollup_in_system_prompt(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = QueueProvider(
        [
            json.dumps([]),
            "Workspace-aware reply.",
        ]
    )
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 3, 9, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        workspace = client.post(
            "/v1/workspaces",
            json={
                "user_id": "usr_1",
                "name": "Debug Workspace",
                "metadata": {},
            },
        ).json()
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": workspace["id"],
                "title": "Debug Chat",
                "metadata": {},
            },
        ).json()
        with _connection(client) as connection:
            summaries = SummaryRepository(connection, runtime.clock)
            client.portal.call(
                summaries.create_summary,
                "usr_1",
                {
                    "id": "sum_workspace_1",
                    "conversation_id": None,
                    "workspace_id": workspace["id"],
                    "source_message_start_seq": None,
                    "source_message_end_seq": None,
                    "summary_kind": "workspace_rollup",
                    "summary_text": "The workspace prefers patch-first debugging.",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "score-test-model",
                    "created_at": "2026-04-03T09:00:00+00:00",
                },
            )

        response = client.post(
            f"/v1/chat/{conversation['id']}/reply",
            json={
                "user_id": "usr_1",
                "message_text": "Please help me debug this retry loop.",
                "include_thinking": False,
                "metadata": {},
                "debug": False,
            },
        )

        assert response.status_code == 200
        system_prompt = provider.requests[1].messages[0].content
        assert "[Workspace Context]" in system_prompt
        assert "The workspace prefers patch-first debugging." in system_prompt


def test_service_mode_requires_x_atagia_user_id_header(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, service_mode=True))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations",
            headers={"Authorization": "Bearer service-key"},
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "X-Atagia-User-Id header is required in service mode"


def test_create_app_rejects_equal_service_and_admin_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path, service_mode=True)
    bad_settings = Settings(
        sqlite_path=settings.sqlite_path,
        migrations_path=settings.migrations_path,
        manifests_path=settings.manifests_path,
        storage_backend=settings.storage_backend,
        redis_url=settings.redis_url,
        llm_provider=settings.llm_provider,
        llm_api_key=settings.llm_api_key,
        openai_api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        llm_base_url=settings.llm_base_url,
        openrouter_site_url=settings.openrouter_site_url,
        openrouter_app_name=settings.openrouter_app_name,
        llm_extraction_model=settings.llm_extraction_model,
        llm_scoring_model=settings.llm_scoring_model,
        llm_classifier_model=settings.llm_classifier_model,
        llm_chat_model=settings.llm_chat_model,
        service_mode=True,
        service_api_key="same-key",
        admin_api_key="same-key",
        workers_enabled=settings.workers_enabled,
        debug=settings.debug,
    )

    with pytest.raises(ConfigurationError, match="ATAGIA_ADMIN_API_KEY must differ"):
        create_app(bad_settings)
