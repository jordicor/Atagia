"""API tests for memory routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-memory.db"),
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
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_memory_routes_support_feedback_lookup_and_contract_view(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 1, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(users.create_user, "usr_2")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            client.portal.call(messages.create_message, "msg_1", "cnv_1", "user", 1, "Need help", 2, {})
            client.portal.call(messages.create_message, "msg_2", "cnv_1", "assistant", 2, "Try this", 2, {})
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="User is debugging retries.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_1",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_2",
                    conversation_id=None,
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.GLOBAL_USER,
                    canonical_text="Other user's memory.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    memory_id="mem_2",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Different same-user memory.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    memory_id="mem_3",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.STATE_SNAPSHOT,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="User is focused on retries.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    payload={"focus_topic": "retries", "urgency": "high"},
                    memory_id="mem_state",
                )
            )
            event = client.portal.call(
                events.create_event,
                {
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_1",
                    "response_message_id": "msg_2",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["retry"]},
                    "selected_memory_ids_json": ["mem_1"],
                    "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
                    "outcome_json": {},
                },
            )

            feedback_response = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_1",
                    "feedback_type": "useful",
                    "score": 0.9,
                    "metadata": {"source": "api"},
                },
            )
            assert feedback_response.status_code == 200
            assert feedback_response.json()["memory_id"] == "mem_1"

            wrong_feedback = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_2",
                    "feedback_type": "irrelevant",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert wrong_feedback.status_code == 404

            mismatched_feedback = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_3",
                    "feedback_type": "irrelevant",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert mismatched_feedback.status_code == 409

            invalid_feedback_type = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_1",
                    "feedback_type": "not-valid",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert invalid_feedback_type.status_code == 422

            memory_response = client.get("/v1/memory/objects/mem_1", params={"user_id": "usr_1"})
            assert memory_response.status_code == 200
            assert memory_response.json()["canonical_text"] == "User is debugging retries."

            missing_memory = client.get("/v1/memory/objects/mem_1", params={"user_id": "usr_2"})
            assert missing_memory.status_code == 404

            contract_response = client.get(
                "/v1/users/usr_1/contract",
                params={"assistant_mode_id": "coding_debug"},
            )
            assert contract_response.status_code == 200
            assert "directness" in contract_response.json()

            state_response = client.get("/v1/users/usr_1/state")
            assert state_response.status_code == 200
            assert state_response.json() == {}

            scoped_state_response = client.get(
                "/v1/users/usr_1/state",
            params={"assistant_mode_id": "coding_debug", "conversation_id": "cnv_1"},
        )
        assert scoped_state_response.status_code == 200
        assert scoped_state_response.json() == {
            "focus_topic": "retries",
            "urgency": "high",
        }
