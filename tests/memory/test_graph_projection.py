"""Integration-style tests for SQLite graph projection."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.entity_graph_repository import EntityGraphRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.graph_projection import GraphProjectionSourceChunk, GraphProjector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMPolicyBlockedError,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class SequentialGraphProvider(LLMProvider):
    name = "graph-projection"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("No canned graph payload left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payloads.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in graph projection tests")


class SharedChunkGraphProvider(LLMProvider):
    name = "shared-chunk-graph-projection"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            payload = {
                "entities": [
                    {
                        "local_id": "maria",
                        "entity_type": "person",
                        "display_name": "Maria",
                        "resolution": "new",
                        "confidence": 0.95,
                        "evidence_quote": "Maria is my sister",
                    }
                ],
                "relationships": [],
                "nothing_durable": False,
            }
        else:
            known_entities = _known_entities_from_prompt(request.messages[1].content)
            maria_id = known_entities["entities"][0]["id"]
            payload = {
                "entities": [
                    {
                        "local_id": "maria",
                        "entity_type": "person",
                        "display_name": "Maria",
                        "resolution": "existing",
                        "existing_entity_id": maria_id,
                        "confidence": 0.95,
                        "evidence_quote": "Maria",
                    },
                    {
                        "local_id": "alba",
                        "entity_type": "person",
                        "display_name": "Alba",
                        "resolution": "new",
                        "confidence": 0.94,
                        "evidence_quote": "Alba is Maria's daughter",
                    },
                ],
                "relationships": [
                    {
                        "source_local_id": "maria",
                        "predicate": "person.parent_of",
                        "target_local_id": "alba",
                        "scope": "global_user",
                        "confidence": 0.91,
                        "status": "active",
                        "evidence_quote": "Alba is Maria's daughter",
                    }
                ],
                "nothing_durable": False,
            }
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in graph projection tests")


class PolicyBlockedOnceGraphProvider(LLMProvider):
    name = "policy-blocked-graph-projection"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise LLMPolicyBlockedError("provider blocked graph projection chunk")
        payload = {
            "entities": [
                {
                    "local_id": "maria",
                    "entity_type": "person",
                    "display_name": "Maria",
                    "resolution": "new",
                    "confidence": 0.95,
                    "evidence_quote": "Maria",
                }
            ],
            "relationships": [],
            "nothing_durable": False,
        }
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in graph projection tests")


def _known_entities_from_prompt(prompt: str) -> dict[str, object]:
    start = prompt.rindex("<known_entities>") + len("<known_entities>")
    end = prompt.rindex("</known_entities>")
    return json.loads(html.unescape(prompt[start:end].strip()))


def _settings(**overrides: object) -> Settings:
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        llm_forced_global_model="openai/reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        topic_working_set_enabled=False,
    )
    return Settings(**{**asdict(settings), **overrides})


async def _build_runtime(payloads: list[dict[str, object]]):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 13, 0, tzinfo=timezone.utc))
    loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, loader.load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock, settings=_settings())
    graph = EntityGraphRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    message = await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "Maria is my sister and Alba is her daughter.",
        12,
        {},
    )
    provider = SequentialGraphProvider(payloads)
    projector = GraphProjector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        graph_repository=graph,
        settings=_settings(),
    )
    return connection, loader, graph, projector, provider, message


async def _build_runtime_with_provider(provider: LLMProvider):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 13, 0, tzinfo=timezone.utc))
    loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, loader.load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock, settings=_settings())
    graph = EntityGraphRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    message = await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "Maria is my sister.\n\nAlba is Maria's daughter.",
        12,
        {},
    )
    projector = GraphProjector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        graph_repository=graph,
        settings=_settings(),
    )
    return connection, loader, graph, projector, message


def _resolved_policy(loader: ManifestLoader, mode_id: str = "coding_debug"):
    return PolicyResolver().resolve(loader.load_all()[mode_id], None, None)


def _context(message_id: str) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id=message_id,
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


async def _count(connection, table_name: str) -> int:
    cursor = await connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}")
    row = await cursor.fetchone()
    return int(row["count"])


def _source_chunks(
    text: str,
    *,
    source_memory_ids: list[str] | None = None,
) -> list[GraphProjectionSourceChunk]:
    return [
        GraphProjectionSourceChunk(
            text=text,
            chunk_index=1,
            chunk_count=1,
            source_memory_ids=source_memory_ids or [],
        )
    ]


@pytest.mark.asyncio
async def test_graph_projector_persists_personal_relationships_idempotently() -> None:
    payload = {
        "entities": [
            {
                "local_id": "e1",
                "entity_type": "person",
                "display_name": "Maria",
                "aliases": ["my sister"],
                "resolution": "new",
                "confidence": 0.94,
                "evidence_quote": "Maria is my sister",
            },
            {
                "local_id": "e2",
                "entity_type": "person",
                "display_name": "Alba",
                "aliases": [],
                "resolution": "new",
                "confidence": 0.93,
                "evidence_quote": "Alba is her daughter",
            },
        ],
        "relationships": [
            {
                "source_local_id": "e1",
                "predicate": "person.parent_of",
                "target_local_id": "e2",
                "scope": "conversation",
                "confidence": 0.91,
                "status": "active",
                "evidence_quote": "Alba is her daughter",
            }
        ],
        "nothing_durable": False,
    }
    connection, loader, graph, projector, provider, message = await _build_runtime([payload, payload])
    try:
        first = await projector.project(
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=_resolved_policy(loader),
            user_id="usr_1",
            source_chunks=_source_chunks(str(message["text"])),
        )
        second = await projector.project(
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=_resolved_policy(loader),
            user_id="usr_1",
            source_chunks=_source_chunks(str(message["text"])),
        )

        entities = await graph.list_entities(user_id="usr_1", entity_type="person")
        maria = next(entity for entity in entities if entity["display_name"] == "Maria")
        relationships = await graph.list_relationships_for_entity(
            user_id="usr_1",
            entity_id=str(maria["id"]),
        )
        sources = await graph.list_relationship_sources(
            user_id="usr_1",
            relationship_id=str(relationships[0]["id"]),
        )

        assert first.entity_count == 2
        assert first.relationship_count == 1
        assert second.relationship_count == 1
        assert await _count(connection, "graph_entities") == 2
        assert await _count(connection, "graph_entity_mentions") == 2
        assert await _count(connection, "graph_relationships") == 1
        assert await _count(connection, "graph_relationship_sources") == 1
        assert relationships[0]["predicate"] == "person.parent_of"
        assert sources[0]["evidence_quote"] == "Alba is her daughter"
        assert provider.requests[0].metadata["purpose"] == "graph_projection"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_graph_projector_keeps_ambiguous_person_reference_out_of_active_edges() -> None:
    payload = {
        "entities": [
            {
                "local_id": "e1",
                "entity_type": "person",
                "display_name": "David",
                "resolution": "ambiguous",
                "confidence": 0.42,
                "status": "review_required",
                "evidence_quote": "David",
            }
        ],
        "relationships": [
            {
                "source_local_id": "e1",
                "predicate": "person.friend_of",
                "target_value": {"description": "the user"},
                "scope": "conversation",
                "confidence": 0.42,
                "status": "review_required",
                "evidence_quote": "David",
            }
        ],
        "nothing_durable": False,
    }
    connection, loader, graph, projector, _provider, message = await _build_runtime([payload])
    try:
        outcome = await projector.project(
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=_resolved_policy(loader),
            user_id="usr_1",
            source_chunks=_source_chunks(str(message["text"])),
        )
        mentions = await graph.list_mentions_for_source(
            user_id="usr_1",
            source_kind="message",
            source_id=str(message["id"]),
        )

        assert outcome.entity_count == 0
        assert outcome.relationship_count == 0
        assert outcome.skipped_count == 2
        assert await _count(connection, "graph_entities") == 0
        assert await _count(connection, "graph_relationships") == 0
        assert len(mentions) == 1
        assert mentions[0]["status"] == "review_required"
        assert mentions[0]["entity_id"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_graph_projector_uses_shared_ingest_chunks_and_persists_chunk_metadata() -> None:
    provider = SharedChunkGraphProvider()
    connection, loader, graph, projector, message = await _build_runtime_with_provider(provider)
    try:
        outcome = await projector.project(
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=_resolved_policy(loader),
            user_id="usr_1",
            source_chunks=[
                GraphProjectionSourceChunk(
                    text="Maria is my sister.",
                    chunk_index=1,
                    chunk_count=2,
                    chunking_strategy="level0",
                    source_memory_ids=["mem_chunk_1"],
                ),
                GraphProjectionSourceChunk(
                    text="Alba is Maria's daughter.",
                    chunk_index=2,
                    chunk_count=2,
                    chunking_strategy="level0",
                    source_memory_ids=["mem_chunk_2"],
                ),
            ],
            source_memory_ids=["mem_chunk_1", "mem_chunk_2"],
        )

        entities = await graph.list_entities(user_id="usr_1", entity_type="person")
        maria = next(entity for entity in entities if entity["display_name"] == "Maria")
        relationships = await graph.list_relationships_for_entity(
            user_id="usr_1",
            entity_id=str(maria["id"]),
        )
        mentions = await graph.list_mentions_for_source(
            user_id="usr_1",
            source_kind="message",
            source_id=str(message["id"]),
        )
        sources = await graph.list_relationship_sources(
            user_id="usr_1",
            relationship_id=str(relationships[0]["id"]),
        )

        assert outcome.entity_count == 2
        assert len(provider.requests) == 2
        assert "Alba is Maria's daughter." not in provider.requests[0].messages[1].content
        assert "Maria is my sister." not in provider.requests[1].messages[1].content
        assert _known_entities_from_prompt(provider.requests[1].messages[1].content)["entities"][0]["display_name"] == "Maria"
        assert sorted(entity["display_name"] for entity in entities) == ["Alba", "Maria"]
        assert relationships[0]["scope"] == "chat"
        assert relationships[0]["metadata_json"]["llm_requested_scope"] == "global_user"
        assert relationships[0]["metadata_json"]["chunk_index"] == 2
        assert {mention["metadata_json"]["chunk_index"] for mention in mentions} == {1, 2}
        assert sources[0]["metadata_json"]["source_memory_ids"] == ["mem_chunk_2"]
        assert sources[0]["source_occurrence_key"] == "msg_1:chunk:2:2"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_graph_projector_skips_policy_blocked_chunks() -> None:
    provider = PolicyBlockedOnceGraphProvider()
    connection, loader, graph, projector, message = await _build_runtime_with_provider(provider)
    try:
        outcome = await projector.project(
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=_resolved_policy(loader),
            user_id="usr_1",
            source_chunks=[
                GraphProjectionSourceChunk(
                    text="Sensitive chunk that the provider blocks.",
                    chunk_index=1,
                    chunk_count=2,
                    source_memory_ids=["mem_chunk_1"],
                ),
                GraphProjectionSourceChunk(
                    text="Maria is mentioned safely.",
                    chunk_index=2,
                    chunk_count=2,
                    source_memory_ids=["mem_chunk_2"],
                ),
            ],
            source_memory_ids=["mem_chunk_1", "mem_chunk_2"],
        )
        cursor = await connection.execute(
            "SELECT status, error FROM graph_projection_runs ORDER BY created_at DESC LIMIT 1"
        )
        run = await cursor.fetchone()

        assert outcome.entity_count == 1
        assert outcome.skipped_count == 1
        assert await _count(connection, "graph_entities") == 1
        assert run["status"] == "completed"
        assert "provider blocked graph projection chunk" in run["error"]
    finally:
        await connection.close()
