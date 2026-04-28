"""Tests for the evaluation worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.metrics_repository import MetricsRepository
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import MemoryFeedbackRepository, RetrievalEventRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import EVALUATION_STREAM_NAME, JobEnvelope, JobType
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    StructuredOutputError,
)
from atagia.workers.evaluation_worker import EvaluationWorker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "evaluation-worker-tests"

    def __init__(self, outputs: list[dict[str, object]] | None = None) -> None:
        self.outputs = list(outputs or [])

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if not self.outputs:
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"compliance_score": 0.8, "reasoning": "ok"}),
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.outputs.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in evaluation worker tests")


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


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 31, 9, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    backend = InProcessBackend()
    provider = QueueProvider()
    llm_client = LLMClient(provider_name=provider.name, providers=[provider])
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    feedback = MemoryFeedbackRepository(connection, clock)
    metrics = MetricsRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    await messages.create_message("msg_1", "cnv_1", "user", 1, "Need help", 2, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Try this", 2, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Retry memory",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id="mem_1",
    )
    await events.create_event(
        {
            "id": "ret_1",
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry"]},
            "selected_memory_ids_json": ["mem_1"],
            "context_view_json": {"selected_memory_ids": ["mem_1"], "items_included": 1, "items_dropped": 0},
            "outcome_json": {},
            "created_at": "2026-03-31T09:05:00+00:00",
        }
    )
    await feedback.create_feedback(
        retrieval_event_id="ret_1",
        memory_id="mem_1",
        user_id="usr_1",
        feedback_type="useful",
        score=1.0,
        metadata={},
    )
    worker = EvaluationWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=llm_client,
        clock=clock,
        settings=_settings(),
    )
    return connection, backend, metrics, worker


def _job(metrics: list[str], *, job_type: JobType = JobType.RUN_EVALUATION) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_eval_1",
        job_type=job_type,
        user_id="usr_1",
        payload={
            "time_bucket": "2026-03-31",
            "user_id": "usr_1",
            "assistant_mode_id": "coding_debug",
            "metrics": metrics,
        },
        created_at=datetime(2026, 3, 31, 9, 10, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_evaluation_worker_processes_job_and_stores_metrics() -> None:
    connection, backend, metrics, worker = await _build_runtime()
    try:
        await backend.stream_add(EVALUATION_STREAM_NAME, _job(["mur"]).model_dump(mode="json"))

        result = await worker.run_once()
        stored = await metrics.get_metric(
            metric_name="mur",
            time_bucket="2026-03-31",
            user_id="usr_1",
            assistant_mode_id="coding_debug",
        )

        assert result.acked == 1
        assert stored is not None
        assert stored["metric_value"] == pytest.approx(1.0)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_evaluation_worker_handles_unknown_metric_names_gracefully() -> None:
    connection, _backend, _metrics, worker = await _build_runtime()
    try:
        result = await worker.process_job(_job(["mystery"]).model_dump(mode="json"))

        assert result["computed"] == {}
        assert result["skipped_metrics"] == ["mystery"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_evaluation_worker_dead_letters_after_max_retries() -> None:
    connection, backend, _metrics, worker = await _build_runtime()
    try:
        await backend.stream_add(
            EVALUATION_STREAM_NAME,
            _job(["mur"], job_type=JobType.EXTRACT_MEMORY_CANDIDATES).model_dump(mode="json"),
        )

        first = await worker.run_once()
        second = await worker.run_once()
        third = await worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{EVALUATION_STREAM_NAME}", timeout_seconds=0)

        assert first.failed == 1
        assert second.failed == 1
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert dead_letter is not None
        assert dead_letter["delivery_count"] == 3
        assert dead_letter["error_details"] == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_evaluation_worker_logs_structured_job_failure_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail_metric(*args, **kwargs):
        del args, kwargs
        raise StructuredOutputError(
            "Provider returned invalid structured output",
            details=("$.compliance_score: Field required",),
        )

    connection, backend, _metrics, worker = await _build_runtime()
    try:
        worker._metrics_computer.compute_named_metric = fail_metric
        await backend.stream_add(EVALUATION_STREAM_NAME, _job(["ccr"]).model_dump(mode="json"))

        with caplog.at_level("WARNING", logger="atagia.workers.evaluation_worker"):
            result = await worker.run_once()

        records = [
            record
            for record in caplog.records
            if "due to structured output" in record.getMessage()
        ]
        assert result.failed == 1
        assert records
        assert records[0].exc_info is None
        assert "Traceback" not in caplog.text
    finally:
        await connection.close()
