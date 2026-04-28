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
        output_text = self.outputs.pop(0)
        if request.metadata.get("purpose") == "need_detection":
            payload = json.loads(output_text)
            if isinstance(payload, list):
                output_text = json.dumps(
                    {
                        "needs": payload,
                        "temporal_range": None,
                        "sub_queries": ["retry loop"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "retry loop",
                                "fts_phrase": "retry loop",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
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
        llm_chat_model="openai/reply-test-model",
        llm_ingest_model="openai/chat-test-model",
        llm_retrieval_model="openai/score-test-model",
        llm_component_models={"intent_classifier": "openai/classify-test-model"},
        service_mode=service_mode,
        service_api_key="service-key" if service_mode else None,
        admin_api_key="admin-key" if service_mode else None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=allow_insecure_http,
        small_corpus_token_threshold_ratio=0.0,
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


def test_create_conversation_reuses_existing_when_scope_matches(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        first = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {"source": "api"},
            },
        )
        second = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Ignored Recreate Title",
                "metadata": {"source": "second-call"},
            },
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["id"] == "cnv_fixed"
        assert second.json()["title"] == "Debug Chat"
        assert second.json()["metadata_json"] == {"source": "api"}


def test_create_conversation_rejects_existing_mode_mismatch(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        )
        response = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "personal_assistant",
                "workspace_id": None,
                "title": "Different Mode",
                "metadata": {},
            },
        )

        assert created.status_code == 200
        assert response.status_code == 409
        assert "assistant mode" in response.json()["detail"]


def test_create_conversation_rejects_existing_workspace_mismatch(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        workspace_a = client.post(
            "/v1/workspaces",
            json={
                "user_id": "usr_1",
                "workspace_id": "wrk_a",
                "name": "Workspace A",
                "metadata": {},
            },
        )
        workspace_b = client.post(
            "/v1/workspaces",
            json={
                "user_id": "usr_1",
                "workspace_id": "wrk_b",
                "name": "Workspace B",
                "metadata": {},
            },
        )
        created = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "coding_debug",
                "workspace_id": "wrk_a",
                "title": "Debug Chat",
                "metadata": {},
            },
        )
        response = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_fixed",
                "assistant_mode_id": "coding_debug",
                "workspace_id": "wrk_b",
                "title": "Different Workspace",
                "metadata": {},
            },
        )

        assert workspace_a.status_code == 200
        assert workspace_b.status_code == 200
        assert created.status_code == 200
        assert response.status_code == 409
        assert "workspace" in response.json()["detail"]


def test_create_conversation_handles_cross_user_id_collision(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_shared",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        )
        response = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_2",
                "conversation_id": "cnv_shared",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Other User Chat",
                "metadata": {},
            },
        )

        assert created.status_code == 200
        assert response.status_code == 404
        assert response.json()["detail"] == "Conversation not found for user"


def test_create_app_rejects_insecure_http_by_default(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="ATAGIA_SERVICE_MODE=true with API keys or ATAGIA_ALLOW_INSECURE_HTTP=true",
    ):
        create_app(_settings(tmp_path, service_mode=False, allow_insecure_http=False))


def test_sidecar_context_unknown_operational_profile_returns_404(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
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
            f"/v1/conversations/{conversation['id']}/context",
            json={
                "user_id": "usr_1",
                "message_text": "Please help me debug this retry loop.",
                "operational_profile": "space_station",
            },
        )

        assert response.status_code == 404
        assert "Unknown operational profile" in response.json()["detail"]


def test_sidecar_context_high_risk_operational_profile_requires_opt_in(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
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
            f"/v1/conversations/{conversation['id']}/context",
            json={
                "user_id": "usr_1",
                "message_text": "Please help me debug this retry loop.",
                "operational_profile": "emergency",
            },
        )

        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"]


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
        assert provider.requests[1].model == "openai/reply-test-model"


def test_chat_reply_accepts_attachments_and_persists_artifacts_without_raw_base64_leak(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = QueueProvider(
        [
            json.dumps([]),
            "Attachment-aware reply.",
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
                "message_text": "Please review the attachment.",
                "attachments": [
                    {
                        "kind": "base64",
                        "content_base64": "SGVsbG8sIHdvcmxkIQ==",
                        "mime_type": "text/plain",
                        "filename": "notes.txt",
                        "title": "Notes",
                        "source_ref": "upload-1",
                        "privacy_level": 1,
                        "preserve_verbatim": True,
                        "skip_raw_by_default": True,
                        "requires_explicit_request": True,
                        "metadata": {"source": "api"},
                    }
                ],
                "include_thinking": False,
                "metadata": {"channel": "api"},
                "debug": True,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["reply_text"] == "Attachment-aware reply."
        assert len(provider.requests) == 2
        prompt_text = provider.requests[1].messages[1].content
        assert "[Attachments omitted]" in prompt_text
        assert "artifact_id=" in prompt_text
        assert "SGVsbG8sIHdvcmxkIQ==" not in prompt_text

        with _connection(client) as connection:
            cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT id, artifact_type, source_kind, status, message_id, preserve_verbatim, skip_raw_by_default, requires_explicit_request, metadata_json FROM artifacts WHERE user_id = ?",
                    ("usr_1",),
                )
            )
            artifact_rows = client.portal.call(cursor.fetchall)
            assert len(artifact_rows) == 1
            artifact_row = artifact_rows[0]
            assert artifact_row["artifact_type"] == "base64"
            assert artifact_row["source_kind"] == "base64"
            assert artifact_row["status"] == "ready"
            assert artifact_row["message_id"] is not None
            assert artifact_row["preserve_verbatim"] == 1
            assert artifact_row["skip_raw_by_default"] == 1
            assert artifact_row["requires_explicit_request"] == 1
            artifact_metadata = json.loads(artifact_row["metadata_json"])
            assert artifact_metadata["source"] == "api"
            assert artifact_metadata["relevance_state"] == "active_work_material"
            assert artifact_metadata["relevance_source"] == "attachment_ingest"

            link_cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT COUNT(*) FROM artifact_links WHERE artifact_id = ? AND message_id = ?",
                    (artifact_row["id"], payload["request_message_id"]),
                )
            )
            link_count = client.portal.call(link_cursor.fetchone)[0]
            assert link_count == 1

            chunk_cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT text FROM artifact_chunks WHERE artifact_id = ? ORDER BY chunk_index ASC",
                    (artifact_row["id"],),
                )
            )
            chunk_rows = client.portal.call(chunk_cursor.fetchall)
            assert chunk_rows
            assert all("SGVsbG8sIHdvcmxkIQ==" not in row["text"] for row in chunk_rows)

            message_cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT text, metadata_json FROM messages WHERE id = ?",
                    (payload["request_message_id"],),
                )
            )
            message_row = client.portal.call(message_cursor.fetchone)
            assert "SGVsbG8sIHdvcmxkIQ==" not in message_row["text"]
            metadata_json = json.loads(message_row["metadata_json"])
            assert metadata_json["attachment_count"] == 1
            assert metadata_json["attachment_artifact_ids"] == [artifact_row["id"]]
            assert metadata_json["attachments"][0]["relevance_state"] == "active_work_material"
            assert metadata_json["attachments"][0]["relevance_source"] == "attachment_ingest"
    return None


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
        llm_ingest_model=settings.llm_ingest_model,
        llm_retrieval_model=settings.llm_retrieval_model,
        llm_component_models=settings.llm_component_models,
        service_mode=True,
        service_api_key="same-key",
        admin_api_key="same-key",
        workers_enabled=settings.workers_enabled,
        debug=settings.debug,
    )

    with pytest.raises(ConfigurationError, match="ATAGIA_ADMIN_API_KEY must differ"):
        create_app(bad_settings)
