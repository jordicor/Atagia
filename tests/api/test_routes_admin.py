"""API tests for admin routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from fastapi.testclient import TestClient

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.consequence_repository import ConsequenceRepository
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.embeddings import EmbeddingIndex

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "admin-compaction-tests"

    def __init__(self, outputs: dict[str, list[str]]) -> None:
        self.outputs = {key: list(value) for key, value in outputs.items()}
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        queue = self.outputs.get(purpose, [])
        if not queue:
            if purpose == "summary_chunk_segmentation":
                prompt = request.messages[1].content
                message_sequences = [int(item) for item in re.findall(r'<message seq="(\d+)"', prompt)]
                if not message_sequences:
                    raise AssertionError("Expected message sequences in chunk segmentation prompt")
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text=json.dumps(
                        {
                            "episodes": [
                                {
                                    "start_seq": min(message_sequences),
                                    "end_seq": max(message_sequences),
                                    "summary_text": "Admin rebuild chunk summary.",
                                }
                            ]
                        }
                    ),
                )
            if purpose == "episode_synthesis":
                prompt = request.messages[1].content
                chunk_ids = re.findall(r'<conversation_chunk id="([^"]+)"', prompt)
                if not chunk_ids:
                    raise AssertionError("Expected conversation chunk IDs in episode synthesis prompt")
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text=json.dumps(
                        {
                            "episodes": [
                                {
                                    "episode_key": "admin_rebuild",
                                    "summary_text": "Admin rebuild episode summary.",
                                }
                            ],
                            "chunk_episode_keys": ["admin_rebuild"] * len(chunk_ids),
                        }
                    ),
                )
            if purpose == "thematic_profile_synthesis":
                prompt = request.messages[1].content
                episode_ids = re.findall(r'<episode id="([^"]+)"', prompt)
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text=json.dumps(
                        {
                            "profiles": (
                                [
                                    {
                                        "source_memory_ids": [episode_ids[0]],
                                        "summary_text": "Admin rebuild thematic profile.",
                                    }
                                ]
                                if episode_ids
                                else []
                            )
                        }
                    ),
                )
            raise AssertionError(f"No queued output left for purpose {purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=queue.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in admin route tests")


class RecordingEmbeddingIndex(EmbeddingIndex):
    def __init__(self) -> None:
        self.upsert_calls: list[tuple[str, str, dict[str, object]]] = []

    @property
    def vector_limit(self) -> int:
        return 1

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        self.upsert_calls.append((memory_id, text, metadata))

    async def search(self, query: str, user_id: str, top_k: int):
        return []

    async def delete(self, memory_id: str) -> None:
        return None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-admin.db"),
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


def test_admin_routes_require_admin_key_and_can_inspect_retrieval_event(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 2, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(
            provider_name=QueueProvider.name,
            providers=[
                QueueProvider(
                    {
                        "memory_extraction": [
                            json.dumps(
                                {
                                    "evidences": [],
                                    "beliefs": [],
                                    "contract_signals": [],
                                    "state_updates": [],
                                    "mode_guess": None,
                                    "nothing_durable": True,
                                }
                            ),
                            json.dumps(
                                {
                                    "evidences": [],
                                    "beliefs": [],
                                    "contract_signals": [],
                                    "state_updates": [],
                                    "mode_guess": None,
                                    "nothing_durable": True,
                                }
                            )
                        ],
                        "consequence_detection": [
                            json.dumps(
                                {
                                    "is_consequence": False,
                                    "action_description": "",
                                    "outcome_description": "",
                                    "outcome_sentiment": "neutral",
                                    "confidence": 0.0,
                                    "likely_action_message_id": None,
                                }
                            )
                        ],
                        "contract_projection": [
                            json.dumps({"signals": [], "nothing_durable": True})
                        ],
                    }
                )
            ],
        )
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
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
            client.portal.call(messages.create_message, "msg_1", "cnv_1", "user", 1, "Need help", 2, {})
            client.portal.call(messages.create_message, "msg_2", "cnv_1", "assistant", 2, "Try this", 2, {})
            event = client.portal.call(
                events.create_event,
                {
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_1",
                    "response_message_id": "msg_2",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["retry"]},
                    "selected_memory_ids_json": [],
                    "context_view_json": {"selected_memory_ids": [], "items_included": 0, "items_dropped": 0},
                    "outcome_json": {},
                },
            )
            client.portal.call(
                runtime.storage_backend.set_context_view,
                "ctx:1",
                {"user_id": "usr_1", "conversation_id": "cnv_1"},
                60,
            )
            client.portal.call(
                runtime.storage_backend.set_context_view,
                "ctx:2",
                {"user_id": "usr_2", "conversation_id": "cnv_2"},
                60,
            )

            unauthorized = client.get(f"/v1/admin/retrieval-events/{event['id']}")
            assert unauthorized.status_code == 401

            wrong_key = client.get(
                f"/v1/admin/retrieval-events/{event['id']}",
                headers={"Authorization": "Bearer service-key"},
            )
            assert wrong_key.status_code == 401

            authorized = client.get(
                f"/v1/admin/retrieval-events/{event['id']}",
                headers={"Authorization": "Bearer admin-key"},
            )
            assert authorized.status_code == 200
            assert authorized.json()["id"] == event["id"]

            rebuild = client.post(
                "/v1/admin/rebuild/user/usr_1",
                headers={"Authorization": "Bearer admin-key"},
            )
            assert rebuild.status_code == 200
            assert rebuild.json()["status"] == "rebuilt"
            assert rebuild.json()["user_id"] == "usr_1"
            assert rebuild.json()["conversation_ids"] == ["cnv_1"]
            assert client.portal.call(runtime.storage_backend.get_context_view, "ctx:1") is None
            assert client.portal.call(runtime.storage_backend.get_context_view, "ctx:2") == {
                "user_id": "usr_2",
                "conversation_id": "cnv_2",
            }


def test_admin_lifecycle_route_supports_dry_run(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            old_clock = FrozenClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
            runtime.clock = old_clock
            old_memories = MemoryObjectRepository(connection, runtime.clock)
            client.portal.call(
                lambda: old_memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.INTERACTION_CONTRACT,
                    scope=MemoryScope.ASSISTANT_MODE,
                    canonical_text="Low-value memory",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.2,
                    vitality=0.04,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_lifecycle",
                )
            )
            runtime.clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))

            response = client.post(
                "/v1/admin/lifecycle/run",
                params={"dry_run": "true"},
                headers={"Authorization": "Bearer admin-key"},
            )

            assert response.status_code == 200
            assert response.json()["decayed_count"] == 1
            assert response.json()["archived_count"] == 1
            stored = client.portal.call(memories.get_memory_object, "mem_lifecycle", "usr_1")
            assert stored is not None
            assert stored["status"] == MemoryStatus.ACTIVE.value


def test_admin_lifecycle_route_invalidates_cache_for_affected_users(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            client.portal.call(users.create_user, "usr_2")
            client.portal.call(
                conversations.create_conversation,
                "cnv_2",
                "usr_2",
                None,
                "coding_debug",
                "Chat",
            )
            old_clock = FrozenClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
            runtime.clock = old_clock
            old_memories = MemoryObjectRepository(connection, runtime.clock)
            client.portal.call(
                lambda: old_memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.INTERACTION_CONTRACT,
                    scope=MemoryScope.ASSISTANT_MODE,
                    canonical_text="Low-value memory",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.2,
                    vitality=0.04,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_lifecycle",
                )
            )
            runtime.clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
            client.portal.call(
                runtime.storage_backend.set_context_view,
                "ctx:usr_1",
                {"user_id": "usr_1", "conversation_id": "cnv_1"},
                60,
            )
            client.portal.call(
                runtime.storage_backend.set_context_view,
                "ctx:usr_2",
                {"user_id": "usr_2", "conversation_id": "cnv_2"},
                60,
            )

            response = client.post(
                "/v1/admin/lifecycle/run",
                headers={"Authorization": "Bearer admin-key"},
            )

            assert response.status_code == 200
            assert response.json()["archived_count"] == 1
            stored = client.portal.call(memories.get_memory_object, "mem_lifecycle", "usr_1")
            assert stored is not None
            assert stored["status"] == MemoryStatus.ARCHIVED.value
            assert client.portal.call(runtime.storage_backend.get_context_view, "ctx:usr_1") is None
            assert client.portal.call(runtime.storage_backend.get_context_view, "ctx:usr_2") == {
                "user_id": "usr_2",
                "conversation_id": "cnv_2",
            }


def test_admin_can_list_consequence_chains(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 2, 17, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            chains = ConsequenceRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            action = client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Suggested a large refactor.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_action",
                )
            )
            outcome = client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Regressions followed.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_outcome",
                )
            )
            tendency = client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.CONSEQUENCE_CHAIN,
                    scope=MemoryScope.ASSISTANT_MODE,
                    canonical_text="Prefer incremental patches.",
                    source_kind=MemorySourceKind.INFERRED,
                    confidence=0.64,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_tendency",
                )
            )
            client.portal.call(
                chains.create_chain,
                {
                    "id": "chn_1",
                    "user_id": "usr_1",
                    "workspace_id": None,
                    "conversation_id": "cnv_1",
                    "assistant_mode_id": "coding_debug",
                    "action_memory_id": action["id"],
                    "outcome_memory_id": outcome["id"],
                    "tendency_belief_id": tendency["id"],
                    "confidence": 0.8,
                    "status": "active",
                    "created_at": "2026-04-02T17:00:00+00:00",
                    "updated_at": "2026-04-02T17:00:00+00:00",
                },
            )

            response = client.get(
                "/v1/admin/consequence-chains/usr_1",
                headers={"Authorization": "Bearer admin-key"},
            )

            assert response.status_code == 200
            payload = response.json()
            assert len(payload) == 1
            assert payload[0]["id"] == "chn_1"
            assert payload[0]["action_canonical_text"] == "Suggested a large refactor."
            assert payload[0]["outcome_canonical_text"] == "Regressions followed."
            assert payload[0]["tendency_canonical_text"] == "Prefer incremental patches."


def test_admin_can_compact_conversation_and_workspace(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = QueueProvider(
        {
            "summary_chunk_segmentation": [
                json.dumps(
                    {
                        "episodes": [
                            {"start_seq": 1, "end_seq": 2, "summary_text": "Debugging episode summary."}
                        ]
                    }
                )
            ],
            "workspace_rollup_synthesis": [
                json.dumps(
                    {
                        "summary_text": "Workspace prefers patch-first debugging.",
                        "cited_memory_ids": [],
                    }
                )
            ],
        }
    )
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)

            workspace = client.post(
                "/v1/workspaces",
                headers={"Authorization": "Bearer service-key", "X-Atagia-User-Id": "usr_1"},
                json={"user_id": "usr_1", "name": "Workspace", "metadata": {}},
            ).json()
            conversation = client.post(
                "/v1/conversations",
                headers={"Authorization": "Bearer service-key", "X-Atagia-User-Id": "usr_1"},
                json={
                    "user_id": "usr_1",
                    "assistant_mode_id": "coding_debug",
                    "workspace_id": workspace["id"],
                    "title": "Chat",
                    "metadata": {},
                },
            ).json()
            client.portal.call(users.get_user, "usr_1")
            client.portal.call(messages.create_message, "msg_1", conversation["id"], "user", 1, "Try a patch first.", 4, {})
            client.portal.call(messages.create_message, "msg_2", conversation["id"], "assistant", 2, "Patch the retry guard.", 5, {})

            compact_conversation = client.post(
                f"/v1/admin/compact/conversation/{conversation['id']}",
                headers={"Authorization": "Bearer admin-key"},
            )
            assert compact_conversation.status_code == 200
            summary_ids = compact_conversation.json()["summary_ids"]
            assert len(summary_ids) == 1

            compact_workspace = client.post(
                f"/v1/admin/compact/workspace/{workspace['id']}",
                headers={"Authorization": "Bearer admin-key"},
            )
            assert compact_workspace.status_code == 200
            workspace_summary_id = compact_workspace.json()["summary_id"]
            assert workspace_summary_id is not None

            summaries = SummaryRepository(connection, runtime.clock)
            workspace_summary = client.portal.call(summaries.get_summary, workspace_summary_id, "usr_1")
            assert workspace_summary is not None
            assert workspace_summary["summary_kind"] == "workspace_rollup"
            assert workspace_summary["summary_text"] == "Workspace prefers patch-first debugging."


def test_admin_embeddings_backfill_route_returns_counters_and_honors_user_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))
        runtime.embedding_index = RecordingEmbeddingIndex()
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)

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
            client.portal.call(
                conversations.create_conversation,
                "cnv_2",
                "usr_2",
                None,
                "coding_debug",
                "Chat",
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Backfill me",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_usr_1",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_2",
                    conversation_id="cnv_2",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Leave me alone",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    status=MemoryStatus.ACTIVE,
                    memory_id="mem_usr_2",
                )
            )

            response = client.post(
                "/v1/admin/embeddings/backfill",
                headers={"Authorization": "Bearer admin-key"},
                json={"batch_size": 10, "delay_ms": 0, "user_id": "usr_1"},
            )

            assert response.status_code == 200
            assert response.json() == {
                "examined": 1,
                "embedded": 1,
                "skipped": 0,
                "failed": 0,
                "batch_size": 10,
                "delay_ms": 0,
                "user_id": "usr_1",
            }
            assert runtime.embedding_index.upsert_calls[0][0] == "mem_usr_1"
