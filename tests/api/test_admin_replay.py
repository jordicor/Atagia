"""API tests for admin replay, grounding, and export routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
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
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class ReplayAdminProvider(LLMProvider):
    name = "admin-replay-tests"

    def __init__(self, score_map: dict[str, float]) -> None:
        self.score_map = dict(score_map)

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "needs": [],
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
                ),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            payload = [
                {"memory_id": memory_id, "llm_applicability": self.score_map.get(memory_id, 0.5)}
                for memory_id in memory_ids
            ]
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text=json.dumps(payload))
        if purpose == "export_anonymization_rewrite":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "entities": [
                            {
                                "placeholder": "[person_001]",
                                "readable_label": "Person 1",
                                "source_forms": ["Maria"],
                            }
                        ],
                        "messages": [
                            {
                                "message_id": "msg_1",
                                "strict_content": "Please help [person_001] debug this retry loop.",
                                "readable_content": "Please help Person 1 debug this retry loop.",
                            },
                            {
                                "message_id": "msg_2",
                                "strict_content": "Try the previous workaround for [person_001].",
                                "readable_content": "Try the previous workaround for Person 1.",
                            },
                            {
                                "message_id": "msg_3",
                                "strict_content": "[person_001] still sees the retry loop fail.",
                                "readable_content": "Person 1 still sees the retry loop fail.",
                            },
                            {
                                "message_id": "msg_4",
                                "strict_content": "Let's narrow the scope for [person_001].",
                                "readable_content": "Let's narrow the scope for Person 1.",
                            },
                        ],
                    }
                ),
            )
        if purpose == "export_anonymization_verify":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "approved": True,
                        "remaining_identifiers": [],
                        "unsafe_descriptive_clues": [],
                        "reasoning": "Safe export projection.",
                    }
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in admin replay tests")


def _settings(tmp_path: Path, *, allow_admin_export_anonymization: bool = False) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-admin-replay.db"),
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
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
        allow_admin_export_anonymization=allow_admin_export_anonymization,
        small_corpus_token_threshold_ratio=0.0,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_admin_replay_routes_require_admin_key(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.post("/v1/admin/replay/event/ret_1", json={"user_id": "usr_1"}).status_code == 401
        assert client.post("/v1/admin/replay/conversation/cnv_1", json={"user_id": "usr_1"}).status_code == 401
        assert client.post("/v1/admin/grounding/ret_1", json={"user_id": "usr_1"}).status_code == 401
        assert client.post("/v1/admin/export/conversation/cnv_1", json={"user_id": "usr_1"}).status_code == 401


def test_admin_replay_grounding_and_export_routes_work(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ReplayAdminProvider({"mem_1": 0.3, "mem_2": 0.95})
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 5, 16, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(provider_name=provider.name, providers=[provider])
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            client.portal.call(messages.create_message, "msg_1", "cnv_1", "user", 1, "Please help me debug this retry loop.", 7, {})
            client.portal.call(messages.create_message, "msg_2", "cnv_1", "assistant", 2, "Try the previous workaround.", 5, {})
            client.portal.call(messages.create_message, "msg_3", "cnv_1", "user", 3, "The retry loop still fails.", 6, {})
            client.portal.call(messages.create_message, "msg_4", "cnv_1", "assistant", 4, "Let's narrow the scope.", 5, {})
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="retry loop direct fix",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_1",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="retry loop previous workaround with extra context",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_2",
                )
            )
            client.portal.call(
                events.create_event,
                {
                    "id": "ret_1",
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_1",
                    "response_message_id": "msg_2",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["retry loop"]},
                    "selected_memory_ids_json": ["mem_2"],
                    "context_view_json": {
                        "selected_memory_ids": ["mem_2"],
                        "contract_block": "",
                        "workspace_block": "",
                        "memory_block": "memories",
                        "state_block": "",
                        "total_tokens_estimate": 80,
                    },
                    "outcome_json": {
                        "scored_candidates": [
                            {"memory_id": "mem_1", "final_score": 0.3},
                            {"memory_id": "mem_2", "final_score": 0.95},
                        ]
                    },
                    "created_at": "2026-04-05T16:00:00+00:00",
                },
            )
            client.portal.call(
                events.create_event,
                {
                    "id": "ret_2",
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_3",
                    "response_message_id": "msg_4",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["retry loop fails"]},
                    "selected_memory_ids_json": ["mem_2"],
                    "context_view_json": {
                        "selected_memory_ids": ["mem_2"],
                        "contract_block": "",
                        "workspace_block": "",
                        "memory_block": "memories",
                        "state_block": "",
                        "total_tokens_estimate": 90,
                    },
                    "outcome_json": {
                        "scored_candidates": [
                            {"memory_id": "mem_1", "final_score": 0.2},
                            {"memory_id": "mem_2", "final_score": 0.9},
                        ]
                    },
                    "created_at": "2026-04-05T16:10:00+00:00",
                },
            )

            replay_event_response = client.post(
                "/v1/admin/replay/event/ret_1",
                json={"user_id": "usr_1"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert replay_event_response.status_code == 200
            assert replay_event_response.json()["original_event_id"] == "ret_1"

            replay_conversation_response = client.post(
                "/v1/admin/replay/conversation/cnv_1",
                json={"user_id": "usr_1"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert replay_conversation_response.status_code == 200
            assert len(replay_conversation_response.json()) == 2

            grounding_response = client.post(
                "/v1/admin/grounding/ret_1",
                json={"user_id": "usr_1"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert grounding_response.status_code == 200
            assert grounding_response.json()["items"][0]["memory_id"] == "mem_2"

            export_response = client.post(
                "/v1/admin/export/conversation/cnv_1",
                json={"user_id": "usr_1", "include_retrieval_traces": True},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert export_response.status_code == 200
            assert export_response.json()["conversation_id"] == "cnv_1"
            assert export_response.json()["export_kind"] == "raw_replay"
            assert export_response.json()["replay_compatible"] is True
            assert len(export_response.json()["retrieval_traces"]) == 2

            audit_entries = client.portal.call(AdminAuditRepository(connection, runtime.clock).list_entries)
            assert [entry["action"] for entry in audit_entries] == [
                "replay_event",
                "replay_conversation",
                "grounding_analysis",
                "export_conversation",
            ]
            assert [entry["target_type"] for entry in audit_entries] == [
                "retrieval_event",
                "conversation",
                "retrieval_event",
                "conversation",
            ]
            assert [entry["target_id"] for entry in audit_entries] == ["ret_1", "cnv_1", "ret_1", "cnv_1"]
            assert all(entry["admin_user_id"] == "admin_api_key" for entry in audit_entries)
            assert [entry["metadata_json"] for entry in audit_entries] == [
                {"user_id": "usr_1"},
                {"user_id": "usr_1"},
                {"user_id": "usr_1"},
                {"user_id": "usr_1", "anonymization_mode": "raw"},
            ]


def test_admin_export_route_supports_anonymized_projection_and_rejects_traces(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, allow_admin_export_anonymization=True))
    provider = ReplayAdminProvider({"mem_1": 0.3, "mem_2": 0.95})
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 5, 16, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(provider_name=provider.name, providers=[provider])
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            workspaces = WorkspaceRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(workspaces.create_workspace, "ws_ops", "usr_1", "Ops")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                "ws_ops",
                "coding_debug",
                "Chat",
            )
            client.portal.call(
                messages.create_message,
                "msg_1",
                "cnv_1",
                "user",
                1,
                "Maria needs help with this retry loop.",
                7,
                {},
            )
            client.portal.call(
                messages.create_message,
                "msg_2",
                "cnv_1",
                "assistant",
                2,
                "Let's help Maria debug it.",
                5,
                {},
            )
            client.portal.call(
                messages.create_message,
                "msg_3",
                "cnv_1",
                "user",
                3,
                "Maria still sees the retry loop fail.",
                6,
                {},
            )
            client.portal.call(
                messages.create_message,
                "msg_4",
                "cnv_1",
                "assistant",
                4,
                "Let's narrow the scope for Maria.",
                5,
                {},
            )

            rejected = client.post(
                "/v1/admin/export/conversation/cnv_1",
                json={
                    "user_id": "usr_1",
                    "include_retrieval_traces": True,
                    "anonymization_mode": "strict",
                },
                headers={"Authorization": "Bearer admin-key"},
            )
            assert rejected.status_code == 422

            response = client.post(
                "/v1/admin/export/conversation/cnv_1",
                json={
                    "user_id": "usr_1",
                    "include_retrieval_traces": False,
                    "anonymization_mode": "strict",
                },
                headers={"Authorization": "Bearer admin-key"},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["export_kind"] == "anonymized_projection"
            assert payload["replay_compatible"] is False
            assert re.fullmatch(r"anon_user_0001_[0-9a-f]{8}", payload["user_id"])
            assert re.fullmatch(r"anon_conversation_0001_[0-9a-f]{8}", payload["conversation_id"])
            assert re.fullmatch(r"anon_workspace_0001_[0-9a-f]{8}", payload["workspace_id"])
            assert re.fullmatch(r"anon_message_0001_[0-9a-f]{8}", payload["messages"][0]["message_id"])
            assert payload["messages"][0]["content"] == "Please help [person_001] debug this retry loop."
            assert payload["messages"][0]["created_at"] is None
            assert payload["messages"][0]["occurred_at"] is None
            assert payload["retrieval_traces"] is None
            assert payload["anonymization"]["mode"] == "strict"
