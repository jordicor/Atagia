"""Tests for stream-backed SQLite graph projection workers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.entity_graph_repository import EntityGraphRepository
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import (
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    GraphProjectionChunkPayload,
    GraphProjectionJobPayload,
    JobEnvelope,
    JobType,
    MessageJobPayload,
    WORKER_GROUP_NAME,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.workers.graph_sync_worker import GraphSyncWorker
from atagia.workers.ingest_worker import IngestWorker
from tests.extraction_payload_support import (
    is_memory_extraction_card_purpose,
    memory_extraction_card_output_from_payload,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class PurposeProvider(LLMProvider):
    name = "graph-worker"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata.get("purpose")
        if is_memory_extraction_card_purpose(purpose):
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=memory_extraction_card_output_from_payload(
                    {"candidates": [], "nothing_durable": True},
                    purpose,
                ),
            )
        if purpose == "consequence_gate_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="no",
            )
        if purpose == "graph_projection":
            return self._response(
                request,
                {
                    "entities": [
                        {
                            "local_id": "e1",
                            "entity_type": "person",
                            "display_name": "Maria",
                            "resolution": "new",
                            "confidence": 0.9,
                            "evidence_quote": "Maria is my sister",
                        }
                    ],
                    "relationships": [],
                    "nothing_durable": False,
                },
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    def _response(self, request: LLMCompletionRequest, payload: dict[str, object]) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in graph worker tests")


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


async def _build_runtime(*, graph_projection_enabled: bool = False):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 14, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    message = await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "Maria is my sister.",
        6,
        {},
    )
    backend = InProcessBackend()
    provider = PurposeProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])
    settings = _settings(graph_projection_enabled=graph_projection_enabled)
    return connection, clock, manifest_loader, backend, provider, client, settings, message


def _message_payload(message_id: str) -> MessageJobPayload:
    return MessageJobPayload(
        message_id=message_id,
        message_text="Maria is my sister.",
        role="user",
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


def _job(payload: MessageJobPayload, job_type: JobType) -> JobEnvelope:
    return JobEnvelope(
        job_id=f"job_{job_type.value}",
        job_type=job_type,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[payload.message_id],
        payload=payload.model_dump(mode="json"),
        created_at=datetime(2026, 5, 2, 14, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_ingest_worker_enqueues_graph_projection_when_enabled() -> None:
    connection, clock, loader, backend, _provider, client, settings, message = await _build_runtime(
        graph_projection_enabled=True
    )
    try:
        worker = IngestWorker(
            storage_backend=backend,
            connection=connection,
            llm_client=client,
            clock=clock,
            manifest_loader=loader,
            settings=settings,
        )
        payload = _message_payload(str(message["id"]))
        await backend.stream_add(EXTRACT_STREAM_NAME, _job(payload, JobType.EXTRACT_MEMORY_CANDIDATES).model_dump(mode="json"))

        result = await worker.run_once()
        graph_job = await backend.dequeue_job(f"stream:{GRAPH_STREAM_NAME}", timeout_seconds=0)

        assert result.acked == 1
        assert graph_job is not None
        envelope = JobEnvelope.model_validate(graph_job["payload"])
        assert envelope.job_type is JobType.SYNC_GRAPH
        assert envelope.payload["message_id"] == "msg_1"
        assert envelope.payload["chunks"][0]["text"] == "Maria is my sister."
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_graph_worker_processes_stream_job_and_acks() -> None:
    connection, clock, loader, backend, provider, client, settings, message = await _build_runtime()
    try:
        worker = GraphSyncWorker(
            storage_backend=backend,
            connection=connection,
            llm_client=client,
            clock=clock,
            manifest_loader=loader,
            settings=settings,
        )
        graph_payload = GraphProjectionJobPayload(
            **_message_payload(str(message["id"])).model_dump(mode="json"),
            source_memory_ids=[],
            chunks=[
                GraphProjectionChunkPayload(
                    text="Maria is my sister.",
                    chunk_index=1,
                    chunk_count=1,
                    source_memory_ids=[],
                )
            ],
        )
        await backend.stream_add(GRAPH_STREAM_NAME, _job(graph_payload, JobType.SYNC_GRAPH).model_dump(mode="json"))

        result = await worker.run_once()
        pending = backend._stream_pending[(GRAPH_STREAM_NAME, WORKER_GROUP_NAME)]
        entities = await EntityGraphRepository(connection, clock).list_entities(user_id="usr_1")

        assert result.received == 1
        assert result.acked == 1
        assert result.failed == 0
        assert pending == {}
        assert [entity["display_name"] for entity in entities] == ["Maria"]
        assert provider.requests[-1].metadata["purpose"] == "graph_projection"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_graph_worker_retries_instead_of_acking_when_projection_lock_is_held() -> None:
    connection, clock, loader, backend, provider, client, settings, message = await _build_runtime()
    try:
        worker = GraphSyncWorker(
            storage_backend=backend,
            connection=connection,
            llm_client=client,
            clock=clock,
            manifest_loader=loader,
            settings=settings,
        )
        graph_payload = GraphProjectionJobPayload(
            **_message_payload(str(message["id"])).model_dump(mode="json"),
            source_memory_ids=[],
            chunks=[
                GraphProjectionChunkPayload(
                    text="Maria is my sister.",
                    chunk_index=1,
                    chunk_count=1,
                    source_memory_ids=[],
                )
            ],
        )
        dedupe_key = GraphSyncWorker._graph_dedupe_key(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_id=str(message["id"]),
        )
        assert await backend.acquire_lock(f"{dedupe_key}:lock", ttl_seconds=60) is not None
        await backend.stream_add(GRAPH_STREAM_NAME, _job(graph_payload, JobType.SYNC_GRAPH).model_dump(mode="json"))

        result = await worker.run_once()
        second = await worker.run_once()
        third = await worker.run_once()
        pending = backend._stream_pending[(GRAPH_STREAM_NAME, WORKER_GROUP_NAME)]
        entities = await EntityGraphRepository(connection, clock).list_entities(user_id="usr_1")
        dead_letter_queue = backend._queues.get(f"dead_letter:{GRAPH_STREAM_NAME}")

        assert result.received == 1
        assert result.acked == 0
        assert result.failed == 1
        assert second.acked == 0
        assert third.acked == 0
        assert len(pending) == 1
        assert next(iter(pending.values()))["payload"]["payload"]["message_id"] == "msg_1"
        assert dead_letter_queue is None or dead_letter_queue.qsize() == 0
        assert entities == []
        assert provider.requests == []
    finally:
        await connection.close()
