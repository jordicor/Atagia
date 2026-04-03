"""Tests for the compaction worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import COMPACT_STREAM_NAME, JobEnvelope, JobType
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.workers.compaction_worker import CompactionWorker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "compaction-worker-tests"

    def __init__(
        self,
        outputs: dict[str, list[str]],
        *,
        failing_purposes: set[str] | None = None,
    ) -> None:
        self.outputs = {key: list(value) for key, value in outputs.items()}
        self.failing_purposes = failing_purposes or set()
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose in self.failing_purposes:
            raise RuntimeError(f"synthetic failure for {purpose}")
        queue = self.outputs.get(purpose, [])
        if not queue:
            raise AssertionError(f"No queued output left for purpose {purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=queue.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in compaction worker tests")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
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
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime(
    outputs: dict[str, list[str]],
    *,
    failing_purposes: set[str] | None = None,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 3, 14, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    backend = InProcessBackend()
    provider = QueueProvider(outputs, failing_purposes=failing_purposes)
    llm_client = LLMClient(provider_name=provider.name, providers=[provider])
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    worker = CompactionWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=llm_client,
        clock=clock,
        settings=_settings(),
    )
    return connection, backend, messages, summaries, worker


async def _seed_messages(messages: MessageRepository) -> None:
    await messages.create_message("msg_1", "cnv_1", "user", 1, "We should try a patch.", 6, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Patch the retry guard first.", 6, {})


def _compaction_job(*, job_kind: str) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_compact_1",
        job_type=JobType.COMPACT_SUMMARIES,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=["msg_1"],
        payload={
            "user_id": "usr_1",
            "workspace_id": "wrk_1",
            "conversation_id": "cnv_1",
            "job_kind": job_kind,
        },
        created_at=datetime(2026, 4, 3, 14, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_compaction_worker_processes_conversation_chunk_job() -> None:
    connection, backend, messages, summaries, worker = await _build_runtime(
        {
            "summary_chunk_segmentation": [
                json.dumps(
                    {
                        "episodes": [
                            {"start_seq": 1, "end_seq": 2, "summary_text": "Patch-first debugging summary."}
                        ]
                    }
                )
            ]
        }
    )
    try:
        await _seed_messages(messages)
        await backend.stream_add(COMPACT_STREAM_NAME, _compaction_job(job_kind="conversation_chunk").model_dump(mode="json"))

        result = await worker.run_once()
        rows = await summaries.list_conversation_chunks("usr_1", "cnv_1", limit=10)

        assert result.acked == 1
        assert len(rows) == 1
        assert rows[0]["summary_text"] == "Patch-first debugging summary."
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compaction_worker_processes_workspace_rollup_job() -> None:
    connection, backend, _messages, summaries, worker = await _build_runtime(
        {
            "workspace_rollup_synthesis": [
                json.dumps(
                    {
                        "summary_text": "Workspace prefers narrow debugging patches.",
                        "cited_memory_ids": [],
                    }
                )
            ]
        }
    )
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_chunk_1",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 1,
                "source_message_end_seq": 2,
                "summary_kind": "conversation_chunk",
                "summary_text": "Chunk summary.",
                "source_object_ids_json": [],
                "maya_score": 1.5,
                "model": "classify-test-model",
                "created_at": "2026-04-03T14:00:00+00:00",
            }
        )
        await backend.stream_add(COMPACT_STREAM_NAME, _compaction_job(job_kind="workspace_rollup").model_dump(mode="json"))

        result = await worker.run_once()
        rows = await summaries.list_workspace_rollups("usr_1", "wrk_1", limit=10)

        assert result.acked == 1
        assert len(rows) == 1
        assert rows[0]["summary_text"] == "Workspace prefers narrow debugging patches."
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compaction_worker_dead_letters_after_max_failed_deliveries() -> None:
    connection, backend, messages, _summaries, worker = await _build_runtime(
        {"summary_chunk_segmentation": ["not-json", "not-json", "not-json"]}
    )
    try:
        await _seed_messages(messages)
        await backend.stream_add(COMPACT_STREAM_NAME, _compaction_job(job_kind="conversation_chunk").model_dump(mode="json"))

        first = await worker.run_once()
        second = await worker.run_once()
        third = await worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{COMPACT_STREAM_NAME}", timeout_seconds=0)

        assert first.failed == 1
        assert second.failed == 1
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert dead_letter is not None
        assert dead_letter["delivery_count"] == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compaction_worker_handles_llm_failure_gracefully() -> None:
    connection, backend, messages, summaries, worker = await _build_runtime(
        {},
        failing_purposes={"summary_chunk_segmentation"},
    )
    try:
        await _seed_messages(messages)
        await backend.stream_add(COMPACT_STREAM_NAME, _compaction_job(job_kind="conversation_chunk").model_dump(mode="json"))

        result = await worker.run_once()
        rows = await summaries.list_conversation_chunks("usr_1", "cnv_1", limit=10)

        assert result.acked == 0
        assert result.failed == 1
        assert rows == []
    finally:
        await connection.close()
