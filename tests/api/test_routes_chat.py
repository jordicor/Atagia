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
from atagia.core.consent_repository import PendingMemoryConfirmationRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.summary_repository import SummaryRepository
from atagia.models.schemas_memory import (
    ConversationStatus,
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    SummaryViewKind,
)
from atagia.services.lifecycle_service import ConversationLifecycleService
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
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
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


def test_create_conversation_accepts_redesign_identity_fields(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_identity",
                "platform_id": "web_desktop",
                "user_persona_id": "persona_a",
                "character_id": "char_debugger",
                "mode": "coding_debug",
                "incognito": True,
                "title": "Debug Chat",
                "metadata": {"source": "api"},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["assistant_mode_id"] == "coding_debug"
        assert payload["mode"] == "coding_debug"
        assert payload["platform_id"] == "web_desktop"
        assert payload["user_persona_id"] == "persona_a"
        assert payload["character_id"] == "char_debugger"
        assert payload["incognito"] == 1
        assert payload["isolated_mode"] == 1

        mismatch = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_identity",
                "platform_id": "web_desktop",
                "mode": "coding_debug",
            },
        )
        assert mismatch.status_code == 404


def test_memory_preferences_routes_round_trip(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        create_user = client.post("/v1/users", json={"user_id": "usr_prefs"})
        defaults = client.get("/v1/users/usr_prefs/memory-preferences")
        updated = client.put(
            "/v1/users/usr_prefs/memory-preferences",
            json={
                "remember_across_chats": False,
                "memory_privacy_mode": "trusted_private",
            },
        )

        assert create_user.status_code == 200
        assert defaults.status_code == 200
        assert defaults.json() == {
            "user_id": "usr_prefs",
            "remember_across_chats": True,
            "remember_across_devices": True,
            "memory_privacy_mode": "balanced",
        }
        assert updated.status_code == 200
        assert updated.json() == {
            "user_id": "usr_prefs",
            "remember_across_chats": False,
            "remember_across_devices": True,
            "memory_privacy_mode": "trusted_private",
        }


def test_incognito_route_toggles_and_hides_broad_conversation_memories(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_incognito",
                "conversation_id": "cnv_incognito",
                "platform_id": "web",
                "mode": "coding_debug",
                "title": "Private turn",
                "metadata": {},
            },
        )
        assert conversation.status_code == 200
        peer_conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_incognito",
                "conversation_id": "cnv_incognito_peer",
                "platform_id": "web",
                "mode": "coding_debug",
                "title": "Peer chat",
                "metadata": {},
            },
        )
        assert peer_conversation.status_code == 200
        with _connection(client) as connection:
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)
            client.portal.call(
                messages.create_message,
                "msg_incognito_source",
                "cnv_incognito",
                "user",
                1,
                "Remember this broad note before incognito.",
                1,
                {},
            )
            client.portal.call(
                messages.create_message,
                "msg_incognito_peer",
                "cnv_incognito_peer",
                "user",
                1,
                "Can another chat see it?",
                1,
                {},
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_incognito",
                    conversation_id=None,
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.USER,
                    canonical_text="Broad memory from a later incognito chat",
                    payload={"source_message_ids": ["msg_incognito_source"]},
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_incognito_broad",
                    platform_id="web",
                )
            )
            client.portal.call(
                events.create_event,
                {
                    "id": "evt_incognito_peer",
                    "user_id": "usr_incognito",
                    "conversation_id": "cnv_incognito_peer",
                    "request_message_id": "msg_incognito_peer",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {},
                    "selected_memory_ids_json": ["mem_incognito_broad"],
                    "context_view_json": {"event_id": "evt_incognito_peer"},
                    "outcome_json": {},
                },
            )

        response = client.post(
            "/v1/conversations/cnv_incognito/incognito",
            json={"user_id": "usr_incognito", "platform_id": "web", "incognito": True},
        )

        assert response.status_code == 200
        assert response.json()["incognito"] == 1
        assert response.json()["isolated_mode"] == 1
        with _connection(client) as connection:
            rows = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT status FROM memory_objects WHERE id = ?",
                    ("mem_incognito_broad",),
                )
            )
            assert rows[0]["status"] == "review_required"
            event_rows = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM retrieval_events WHERE user_id = ? ORDER BY id",
                    ("usr_incognito",),
                )
            )
            assert [row["id"] for row in event_rows] == []


def test_save_from_incognito_returns_review_manifest_without_writes(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_incognito_save",
                "conversation_id": "cnv_incognito_save",
                "platform_id": "web",
                "mode": "coding_debug",
                "incognito": True,
                "title": "Private turn",
                "metadata": {},
            },
        )
        assert conversation.status_code == 200
        with _connection(client) as connection:
            messages = MessageRepository(connection, runtime.clock)
            client.portal.call(
                lambda: messages.create_message(
                    message_id="msg_incognito_user",
                    conversation_id="cnv_incognito_save",
                    role="user",
                    seq=None,
                    text="Please keep this incognito detail reviewable.",
                    token_count=None,
                    metadata={},
                    occurred_at="2026-04-16T04:00:00+00:00",
                )
            )
            client.portal.call(
                lambda: messages.create_message(
                    message_id="msg_incognito_assistant",
                    conversation_id="cnv_incognito_save",
                    role="assistant",
                    seq=None,
                    text="Understood.",
                    token_count=None,
                    metadata={},
                    occurred_at="2026-04-16T04:00:05+00:00",
                )
            )

        response = client.post(
            "/v1/conversations/cnv_incognito_save/save-from-incognito",
            json={"user_id": "usr_incognito_save", "platform_id": "web"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "review_required"
        assert payload["review_policy"] == "non_incognito"
        assert payload["source_message_count"] == 2
        assert payload["suggested_memory_count"] == 0
        assert payload["writes_performed"] is False
        assert [message["message_id"] for message in payload["source_messages"]] == [
            "msg_incognito_user",
            "msg_incognito_assistant",
        ]
        with _connection(client) as connection:
            rows = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT COUNT(*) AS count FROM memory_objects WHERE user_id = ?",
                    ("usr_incognito_save",),
                )
            )
            assert rows[0]["count"] == 0


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


def test_create_conversation_reuses_existing_conversation_with_new_mode(tmp_path: Path) -> None:
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
        assert response.status_code == 200
        assert response.json()["id"] == "cnv_fixed"
        assert response.json()["assistant_mode_id"] == "coding_debug"


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
                "platform_id": "web",
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
                "platform_id": "web",
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


def test_sidecar_ingest_message_id_is_idempotent_and_conflicts_on_change(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "platform_id": "aurvek",
                "mode": "coding_debug",
                "title": "Debug Chat",
                "metadata": {},
            },
        ).json()

        first = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:1",
                "source_seq": 7,
                "role": "user",
                "text": "Please remember the retry guard.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )
        duplicate = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:1",
                "source_seq": 7,
                "role": "user",
                "text": "Please remember the retry guard.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )
        conflict = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:1",
                "source_seq": 7,
                "role": "user",
                "text": "Changed content.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )

        assert first.status_code == 200
        assert first.json() == {
            "ok": True,
            "message_id": "aurvek:msg:1",
            "seq": 7,
            "source_seq": 7,
            "idempotent_replay": False,
        }
        assert duplicate.status_code == 200
        assert duplicate.json()["idempotent_replay"] is True
        assert conflict.status_code == 409
        assert "different role or text" in conflict.json()["detail"]

        with _connection(client) as connection:
            cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE id = ?",
                    ("aurvek:msg:1",),
                )
            )
            assert client.portal.call(cursor.fetchone)[0] == 1


def test_pending_memory_confirmation_routes_list_and_confirm(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "platform_id": "aurvek",
                "mode": "personal_assistant",
                "title": "Memory Chat",
                "metadata": {},
            },
        ).json()

        with _connection(client) as connection:
            memories = MemoryObjectRepository(connection, client.app.state.runtime.clock)
            confirmations = PendingMemoryConfirmationRepository(
                connection,
                client.app.state.runtime.clock,
            )
            pending = client.portal.call(
                lambda: memories.create_memory_object(
                    memory_id="mem_pending_route",
                    user_id="usr_1",
                    conversation_id=conversation["id"],
                    assistant_mode_id="personal_assistant",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.USER,
                    canonical_text="Banking card PIN: 4512",
                    index_text="bank card PIN",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.97,
                    privacy_level=3,
                    memory_category=MemoryCategory.PIN_OR_PASSWORD,
                    preserve_verbatim=True,
                    status=MemoryStatus.PENDING_USER_CONFIRMATION,
                    platform_id="aurvek",
                    commit=False,
                )
            )
            client.portal.call(
                lambda: confirmations.create_marker(
                    user_id="usr_1",
                    conversation_id=conversation["id"],
                    memory_id=str(pending["id"]),
                    category=MemoryCategory.PIN_OR_PASSWORD,
                    created_at=str(pending["created_at"]),
                    platform_id="aurvek",
                    intended_scope=MemoryScope.USER,
                    intended_sensitivity=MemorySensitivity.SECRET,
                    policy_snapshot={"source": "test"},
                    policy_proven=True,
                    commit=False,
                )
            )
            client.portal.call(connection.commit)

        listed = client.get(
            "/v1/users/usr_1/memory-confirmations",
            params={"platform_id": "aurvek"},
        )
        confirmed = client.post(
            "/v1/users/usr_1/memory-confirmations/mem_pending_route/confirm",
        )

        assert listed.status_code == 200
        assert listed.json()["items"][0]["memory_id"] == "mem_pending_route"
        assert listed.json()["items"][0]["label"] == "bank card PIN"
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == MemoryStatus.ACTIVE.value


def test_sidecar_ingest_source_seq_preserves_backfill_order(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        conversation = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "platform_id": "aurvek",
                "mode": "coding_debug",
                "title": "Debug Chat",
                "metadata": {},
            },
        ).json()

        later = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:3",
                "source_seq": 3,
                "role": "user",
                "text": "Later message.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )
        retried_old = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:2",
                "source_seq": 2,
                "role": "assistant",
                "text": "Retried old assistant message.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )
        conflict = client.post(
            f"/v1/conversations/{conversation['id']}/messages",
            json={
                "user_id": "usr_1",
                "message_id": "aurvek:msg:other",
                "source_seq": 2,
                "role": "user",
                "text": "Different message at same source seq.",
                "platform_id": "aurvek",
                "mode": "coding_debug",
            },
        )

        assert later.status_code == 200
        assert later.json()["seq"] == 3
        assert retried_old.status_code == 200
        assert retried_old.json()["seq"] == 2
        assert conflict.status_code == 409
        assert "source_seq already exists" in conflict.json()["detail"]

        with _connection(client) as connection:
            cursor = client.portal.call(
                lambda: connection.execute(
                    "SELECT id, seq FROM messages ORDER BY seq ASC"
                )
            )
            rows = client.portal.call(cursor.fetchall)
            assert [(row["id"], row["seq"]) for row in rows] == [
                ("aurvek:msg:2", 2),
                ("aurvek:msg:3", 3),
            ]


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
                        "intimacy_boundary": "romantic_private",
                        "intimacy_boundary_confidence": 0.8,
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
                    "SELECT id, artifact_type, source_kind, status, message_id, privacy_level, intimacy_boundary, intimacy_boundary_confidence, preserve_verbatim, skip_raw_by_default, requires_explicit_request, metadata_json FROM artifacts WHERE user_id = ?",
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
            assert artifact_row["privacy_level"] == 2
            assert artifact_row["intimacy_boundary"] == "romantic_private"
            assert artifact_row["intimacy_boundary_confidence"] == 0.8
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
                    "SELECT text, intimacy_boundary FROM artifact_chunks WHERE artifact_id = ? ORDER BY chunk_index ASC",
                    (artifact_row["id"],),
                )
            )
            chunk_rows = client.portal.call(chunk_cursor.fetchall)
            assert chunk_rows
            assert all("SGVsbG8sIHdvcmxkIQ==" not in row["text"] for row in chunk_rows)
            assert {row["intimacy_boundary"] for row in chunk_rows} == {"romantic_private"}

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
            assert metadata_json["attachments"][0]["intimacy_boundary"] == "romantic_private"
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
                "platform_id": "web",
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
                "platform_id": "web",
                "include_thinking": False,
                "metadata": {},
                "debug": False,
            },
        )

        assert response.status_code == 404


def test_chat_reply_uses_character_rollup_and_hides_legacy_workspace_rollup(tmp_path: Path) -> None:
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
            client.portal.call(
                summaries.create_summary,
                "usr_1",
                {
                    "id": "sum_character_1",
                    "conversation_id": None,
                    "workspace_id": workspace["id"],
                    "platform_id": "default",
                    "character_id": workspace["id"],
                    "source_message_start_seq": None,
                    "source_message_end_seq": None,
                    "summary_kind": "character_rollup",
                    "summary_text": "The character prefers patch-first debugging.",
                    "source_object_ids_json": [],
                    "sensitivity": "public",
                    "scope_canonical": "character",
                    "maya_score": 1.5,
                    "model": "score-test-model",
                    "created_at": "2026-04-03T09:01:00+00:00",
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
                "character_id": workspace["id"],
            },
        )

        assert response.status_code == 200
        system_prompt = provider.requests[1].messages[0].content
        assert "[Workspace Context]" in system_prompt
        assert "The character prefers patch-first debugging." in system_prompt
        assert "The workspace prefers patch-first debugging." not in system_prompt


def test_conversation_lifecycle_routes_close_delete_and_erasure(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))

        create_temp = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_life",
                "conversation_id": "cnv_temp",
                "assistant_mode_id": "coding_debug",
                "platform_id": "web",
                "temporary": True,
                "purge_on_close": True,
            },
        )
        assert create_temp.status_code == 200
        assert create_temp.json()["temporary"] == 1

        create_isolated = client.post(
            "/v1/conversations",
            json={
                "user_id": "usr_life",
                "conversation_id": "cnv_isolated",
                "assistant_mode_id": "coding_debug",
                "platform_id": "web",
                "cross_chat_memory": False,
            },
        )
        assert create_isolated.status_code == 200
        assert create_isolated.json()["isolated_mode"] == 1

        missing_purge_confirmation = client.post(
            "/v1/conversations/cnv_temp/close",
            json={"user_id": "usr_life", "platform_id": "web"},
        )
        assert missing_purge_confirmation.status_code == 400

        close_temp = client.post(
            "/v1/conversations/cnv_temp/close",
            json={
                "user_id": "usr_life",
                "platform_id": "web",
                "confirmation": "PURGE_ON_CLOSE",
            },
        )
        assert close_temp.status_code == 200
        assert close_temp.json()["conversation_id"] == "cnv_temp"

        with _connection(client) as connection:
            conversations = ConversationRepository(connection, runtime.clock)
            pending_temp = client.portal.call(conversations.get_conversation, "cnv_temp", "usr_life")
            assert pending_temp is not None
            assert pending_temp["status"] == ConversationStatus.PENDING_DELETION.value
            purged = client.portal.call(
                ConversationLifecycleService(runtime).purge_pending_deleted_conversations,
                connection,
            )
            assert purged == 1
            assert client.portal.call(conversations.get_conversation, "cnv_temp", "usr_life") is None

            users = UserRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            summaries = SummaryRepository(connection, runtime.clock)
            client.portal.call(users.create_user, "usr_delete")
            client.portal.call(
                lambda: conversations.create_conversation(
                    "cnv_delete",
                    "usr_delete",
                    None,
                    "coding_debug",
                    "Delete",
                    platform_id="web",
                )
            )
            client.portal.call(messages.create_message, "msg_delete", "cnv_delete", "user", 1, "Delete me", 1, {})
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_delete",
                    conversation_id="cnv_delete",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Conversation-scoped memory.",
                    payload={"source_message_ids": ["msg_delete"]},
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_delete",
                )
            )
            client.portal.call(
                summaries.create_summary,
                "usr_delete",
                {
                    "id": "sum_child",
                    "conversation_id": "cnv_delete",
                    "summary_kind": SummaryViewKind.CONVERSATION_CHUNK,
                    "hierarchy_level": 0,
                    "summary_text": "Child summary.",
                    "source_object_ids_json": ["mem_delete"],
                    "maya_score": 1.0,
                    "model": "test",
                    "created_at": runtime.clock.now().isoformat(),
                },
            )
            client.portal.call(
                lambda: memories.upsert_summary_mirror(
                    user_id="usr_delete",
                    summary_view_id="sum_child",
                    summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
                    hierarchy_level=0,
                    summary_text="Child summary.",
                    source_object_ids=["mem_delete"],
                    created_at=runtime.clock.now().isoformat(),
                    conversation_id="cnv_delete",
                    assistant_mode_id="coding_debug",
                    scope=MemoryScope.CONVERSATION,
                )
            )
            client.portal.call(
                summaries.create_summary,
                "usr_delete",
                {
                    "id": "sum_parent",
                    "conversation_id": "cnv_delete",
                    "summary_kind": SummaryViewKind.EPISODE,
                    "hierarchy_level": 1,
                    "summary_text": "Parent summary.",
                    "source_object_ids_json": ["sum_mem_sum_child"],
                    "maya_score": 1.0,
                    "model": "test",
                    "created_at": runtime.clock.now().isoformat(),
                },
            )
            client.portal.call(
                lambda: memories.upsert_summary_mirror(
                    user_id="usr_delete",
                    summary_view_id="sum_parent",
                    summary_kind=SummaryViewKind.EPISODE,
                    hierarchy_level=1,
                    summary_text="Parent summary.",
                    source_object_ids=["sum_mem_sum_child"],
                    created_at=runtime.clock.now().isoformat(),
                    conversation_id="cnv_delete",
                    assistant_mode_id="coding_debug",
                    scope=MemoryScope.CONVERSATION,
                )
            )
            client.portal.call(
                connection.execute,
                """
                INSERT INTO verbatim_pins(
                    id,
                    user_id,
                    conversation_id,
                    assistant_mode_id,
                    scope,
                    target_kind,
                    target_id,
                    canonical_text,
                    index_text,
                    created_by,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 'chat', 'message', ?, ?, ?, 'test', ?, ?)
                """,
                (
                    "pin_delete",
                    "usr_delete",
                    "cnv_delete",
                    "coding_debug",
                    "msg_delete",
                    "Pinned text.",
                    "Pinned text.",
                    runtime.clock.now().isoformat(),
                    runtime.clock.now().isoformat(),
                ),
            )
            client.portal.call(connection.commit)

        missing_confirmation = client.post(
            "/v1/conversations/cnv_delete/delete",
            json={"user_id": "usr_delete", "platform_id": "web", "confirmation": "WRONG"},
        )
        assert missing_confirmation.status_code == 400

        delete_response = client.post(
            "/v1/conversations/cnv_delete/delete",
            json={
                "user_id": "usr_delete",
                "platform_id": "web",
                "confirmation": "DELETE_CONVERSATION",
            },
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted_memories"] >= 1

        with _connection(client) as connection:
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            pending_delete = client.portal.call(conversations.get_conversation, "cnv_delete", "usr_delete")
            assert pending_delete is not None
            assert pending_delete["status"] == ConversationStatus.PENDING_DELETION.value
            deleted_memory = client.portal.call(memories.get_memory_object, "mem_delete", "usr_delete")
            assert deleted_memory is not None
            assert deleted_memory["status"] == MemoryStatus.DELETED.value
            child_mirror = client.portal.call(memories.get_memory_object, "sum_mem_sum_child", "usr_delete")
            parent_mirror = client.portal.call(memories.get_memory_object, "sum_mem_sum_parent", "usr_delete")
            assert child_mirror is None
            assert parent_mirror is None
            summary_rows = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM summary_views WHERE id IN ('sum_child', 'sum_parent')",
                )
            )
            assert summary_rows == []
            pins = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM verbatim_pins WHERE id = 'pin_delete'",
                )
            )
            assert pins == []
            tombstones = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT scope_summary FROM deletion_tombstones WHERE id = ?",
                    (delete_response.json()["tombstone_id"],),
                )
            )
            tombstone_scope = json.loads(tombstones[0]["scope_summary"])
            assert "conversation_id" not in tombstone_scope
            assert "user_id" not in tombstone_scope
            purged = client.portal.call(
                ConversationLifecycleService(runtime).purge_pending_deleted_conversations,
                connection,
            )
            assert purged == 1
            assert client.portal.call(conversations.get_conversation, "cnv_delete", "usr_delete") is None
            assert client.portal.call(memories.get_memory_object, "mem_delete", "usr_delete") is None

        with _connection(client) as connection:
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            client.portal.call(
                conversations.create_conversation,
                "cnv_audit",
                "usr_life",
                None,
                "coding_debug",
                "Audit",
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_life",
                    conversation_id="cnv_audit",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Audit memory.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_audit",
                )
            )
            client.portal.call(
                connection.executemany,
                """
                INSERT INTO admin_audit_log(
                    id,
                    admin_user_id,
                    action,
                    target_type,
                    target_id,
                    metadata_json,
                    created_at
                )
                VALUES (?, 'admin_1', 'inspect', ?, ?, ?, ?)
                """,
                [
                    ("aud_conversation", "conversation", "cnv_audit", "{}", runtime.clock.now().isoformat()),
                    ("aud_memory", "memory_object", "mem_audit", "{}", runtime.clock.now().isoformat()),
                    (
                        "aud_metadata",
                        "other",
                        "external",
                        '{"user_id":"usr_life"}',
                        runtime.clock.now().isoformat(),
                    ),
                ],
            )
            client.portal.call(connection.commit)

        erase_response = client.post(
            "/v1/users/usr_life/erase",
            json={"user_id": "usr_life", "confirmation": "ERASE_ALL_DATA"},
        )
        assert erase_response.status_code == 200

        with _connection(client) as connection:
            audit_rows = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM admin_audit_log WHERE id LIKE 'aud_%' ORDER BY id",
                )
            )
            assert audit_rows == []

        recreate_response = client.post(
            "/v1/users",
            json={"user_id": "usr_life"},
        )
        assert recreate_response.status_code == 410


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


def test_service_mode_requires_platform_id_on_conversation_boundary(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, service_mode=True))
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations",
            headers={
                "Authorization": "Bearer service-key",
                "X-Atagia-User-Id": "usr_1",
            },
            json={
                "user_id": "usr_1",
                "assistant_mode_id": "coding_debug",
                "workspace_id": None,
                "title": "Debug Chat",
                "metadata": {},
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "platform_id is required in service mode"


def test_create_app_rejects_equal_service_and_admin_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path, service_mode=True)
    bad_settings = Settings(
        sqlite_path=settings.sqlite_path,
        migrations_path=settings.migrations_path,
        manifests_path=settings.manifests_path,
        storage_backend=settings.storage_backend,
        redis_url=settings.redis_url,
        openai_api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_site_url=settings.openrouter_site_url,
        openrouter_app_name=settings.openrouter_app_name,
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
