"""Tests for stream-backed ingest workers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.locking import acquire_belief_lock
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
    REVISE_STREAM_NAME,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import MemoryObjectType
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    StructuredOutputError,
)
from atagia.workers.contract_worker import ContractWorker
from atagia.workers.ingest_worker import IngestWorker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "queue-worker"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.metadata.get("purpose") == "need_detection":
            if not self.outputs:
                raise AssertionError("No queued output left for this test")
            output_text = self.outputs.pop(0)
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
        if request.metadata.get("purpose") == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_explicit": True,
                        "reasoning": "Test classifier response.",
                    }
                ),
            )
        if request.metadata.get("purpose") == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"equivalent": True}),
            )
        if request.metadata.get("purpose") == "consequence_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_consequence": False,
                        "action_description": "",
                        "outcome_description": "",
                        "outcome_sentiment": "neutral",
                        "confidence": 0.0,
                        "likely_action_message_id": None,
                    }
                ),
            )
        if request.metadata.get("purpose") == "consequence_tendency_inference":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"tendency_text": ""}),
            )
        if not self.outputs:
            raise AssertionError("No queued output left for this test")
        output_text = self.outputs.pop(0)
        if request.metadata.get("purpose") == "memory_extraction":
            output_text = _with_default_language_codes_json(output_text)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in worker tests")


def _settings(**overrides: object) -> Settings:
    settings = Settings(
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
    return Settings(**{**asdict(settings), **overrides})


def _with_default_language_codes_json(output_text: str) -> str:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        return output_text
    if not isinstance(payload, dict):
        return output_text
    for field_name in ("evidences", "beliefs", "contract_signals", "state_updates"):
        items = payload.get(field_name)
        if not isinstance(items, list):
            continue
        payload[field_name] = [
            (
                {
                    **item,
                    "language_codes": ["en"],
                }
                if isinstance(item, dict) and "canonical_text" in item and "language_codes" not in item
                else item
            )
            for item in items
        ]
    return json.dumps(payload)


async def _build_runtime(
    outputs: list[str],
    *,
    workspace_id: str | None = None,
    extra_messages: int = 0,
    settings: Settings | None = None,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    backend = InProcessBackend()
    provider = QueueProvider(outputs)
    client = LLMClient(provider_name=provider.name, providers=[provider])
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    if workspace_id is not None:
        await workspaces.create_workspace(workspace_id, "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", workspace_id, "coding_debug", "Chat")
    message = await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "I prefer concise debugging answers for retry issues.",
        10,
        {},
    )
    for index in range(extra_messages):
        seq = index + 2
        role = "assistant" if index % 2 == 0 else "user"
        await messages.create_message(
            f"msg_extra_{seq}",
            "cnv_1",
            role,
            seq,
            f"Extra message {seq}",
            3,
            {},
        )
    resolved_settings = settings or _settings()
    ingest_worker = IngestWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=client,
        clock=clock,
        manifest_loader=manifest_loader,
        settings=resolved_settings,
    )
    contract_worker = ContractWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=client,
        clock=clock,
        manifest_loader=manifest_loader,
        settings=resolved_settings,
    )
    return (
        connection,
        clock,
        backend,
        provider,
        memories,
        ingest_worker,
        contract_worker,
        message,
    )


def _extract_job(message_id: str, *, workspace_id: str | None = None) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_extract_1",
        job_type="extract_memory_candidates",
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload={
            "message_id": message_id,
            "message_text": "I prefer concise debugging answers for retry issues.",
            "role": "user",
            "assistant_mode_id": "coding_debug",
            "workspace_id": workspace_id,
            "recent_messages": [],
        },
        created_at=datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc),
    )


def _contract_job(message_id: str) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_contract_1",
        job_type="project_contract",
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload={
            "message_id": message_id,
            "message_text": "I prefer concise debugging answers for retry issues.",
            "role": "user",
            "assistant_mode_id": "coding_debug",
            "workspace_id": None,
            "recent_messages": [],
        },
        created_at=datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_ingest_worker_processes_stream_job_and_acks() -> None:
    payload = json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "scope": "assistant_mode",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 1,
                    "payload": {"kind": "preference"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload]
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        result = await ingest_worker.run_once()
        pending = backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)]

        assert result.received == 1
        assert result.acked == 1
        assert result.failed == 0
        assert pending == {}
        stored = await memories.list_for_user("usr_1")
        assert len(stored) == 1
        assert stored[0]["object_type"] == MemoryObjectType.EVIDENCE.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_does_not_ack_failed_job() -> None:
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        ["not-json"]
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        result = await ingest_worker.run_once()
        pending = backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)]

        assert result.received == 1
        assert result.acked == 0
        assert result.failed == 1
        assert len(pending) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_succeeds_when_extraction_retry_recovers_invalid_output() -> None:
    payload = json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "scope": "assistant_mode",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 1,
                    "payload": {"kind": "preference"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        ["not-json", payload]
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        result = await ingest_worker.run_once()

        assert result.received == 1
        assert result.acked == 1
        assert result.failed == 0
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        stored = await memories.list_for_user("usr_1")
        assert len(stored) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_dead_letters_after_max_failed_deliveries() -> None:
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        ["not-json", "not-json", "not-json"]
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        first = await ingest_worker.run_once()
        second = await ingest_worker.run_once()
        third = await ingest_worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{EXTRACT_STREAM_NAME}", timeout_seconds=0)

        assert first.failed == 1
        assert second.failed == 1
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        assert dead_letter is not None
        assert dead_letter["delivery_count"] == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_dead_letter_includes_structured_error_details_after_retry_budget_exhaustion() -> None:
    invalid_payload = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [
                {
                    "canonical_text": "I am on vacation this week.",
                    "scope": "conversation",
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "payload": {"focus": "vacation"},
                    "temporal_type": "bounded",
                    "valid_from_iso": "2023-05-12T00:00:00+00:00",
                    "valid_to_iso": "2023-05-08T00:00:00+00:00",
                    "temporal_confidence": 0.82,
                }
            ],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [invalid_payload] * 9
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        first = await ingest_worker.run_once()
        second = await ingest_worker.run_once()
        third = await ingest_worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{EXTRACT_STREAM_NAME}", timeout_seconds=0)

        assert first.failed == 1
        assert second.failed == 1
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        assert dead_letter is not None
        assert dead_letter["error"] == "Provider returned invalid structured output"
        assert dead_letter["error_details"] == [
            "$.state_updates[0]: Value error, valid_from_iso must be <= valid_to_iso"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_enqueues_compaction_job_for_workspace_after_threshold() -> None:
    payload = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload],
        workspace_id="wrk_1",
        extra_messages=9,
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), workspace_id="wrk_1").model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        compact_job = await backend.dequeue_job(f"stream:{COMPACT_STREAM_NAME}", timeout_seconds=0)

        assert result.acked == 1
        assert compact_job is not None
        envelope = JobEnvelope.model_validate(compact_job["payload"])
        assert envelope.job_type is JobType.COMPACT_SUMMARIES
        assert envelope.payload["job_kind"] == "conversation_chunk"
        assert envelope.payload["workspace_id"] == "wrk_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_enqueues_compaction_job_without_workspace_after_threshold() -> None:
    payload = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload],
        workspace_id=None,
        extra_messages=9,
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), workspace_id=None).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        compact_job = await backend.dequeue_job(f"stream:{COMPACT_STREAM_NAME}", timeout_seconds=0)

        assert result.acked == 1
        assert compact_job is not None
        envelope = JobEnvelope.model_validate(compact_job["payload"])
        assert envelope.job_type is JobType.COMPACT_SUMMARIES
        assert envelope.payload["job_kind"] == "conversation_chunk"
        assert envelope.payload["workspace_id"] is None
        assert envelope.payload["conversation_id"] == "cnv_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_skips_revision_when_disabled() -> None:
    persisted = [
        {
            "id": "mem_belief_1",
            "object_type": MemoryObjectType.BELIEF.value,
            "scope": "assistant_mode",
            "payload_json": {
                "claim_key": "response_style.debugging",
                "claim_value": "concise_actionable",
            },
        }
    ]

    async def fake_extract(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(nothing_durable=False), persisted

    async def skip_side_effects(*args, **kwargs) -> None:
        del args, kwargs

    control_connection, _clock, control_backend, _provider, _memories, control_worker, _contract_worker, control_message = await _build_runtime(
        [],
    )
    try:
        control_worker._extractor.extract_with_persistence_details = fake_extract
        control_worker._process_consequence_detection = skip_side_effects
        control_worker._maybe_enqueue_conversation_compaction = skip_side_effects
        await control_backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(control_message["id"])).model_dump(mode="json"),
        )

        control_result = await control_worker.run_once()
        control_revision_job = await control_backend.dequeue_job(
            f"stream:{REVISE_STREAM_NAME}",
            timeout_seconds=0,
        )

        assert control_result.acked == 1
        assert control_revision_job is not None
    finally:
        await control_connection.close()

    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [],
        settings=_settings(skip_belief_revision=True),
    )
    try:
        ingest_worker._extractor.extract_with_persistence_details = fake_extract
        ingest_worker._process_consequence_detection = skip_side_effects
        ingest_worker._maybe_enqueue_conversation_compaction = skip_side_effects
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        revision_job = await backend.dequeue_job(f"stream:{REVISE_STREAM_NAME}", timeout_seconds=0)

        assert result.acked == 1
        assert revision_job is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_treats_non_json_after_schema_fallback_as_noop() -> None:
    async def fake_extract(*args, **kwargs):
        del args, kwargs
        raise StructuredOutputError("Provider returned non-JSON structured output after schema fallback")

    async def skip_side_effects(*args, **kwargs) -> None:
        del args, kwargs

    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [],
    )
    try:
        ingest_worker._extractor.extract_with_persistence_details = fake_extract
        ingest_worker._process_consequence_detection = skip_side_effects
        ingest_worker._maybe_enqueue_conversation_compaction = skip_side_effects
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        stored = await memories.list_for_user("usr_1")

        assert result.acked == 1
        assert result.failed == 0
        assert stored == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_skips_compaction_when_disabled() -> None:
    payload = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload],
        workspace_id="wrk_1",
        extra_messages=9,
        settings=_settings(skip_compaction=True),
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), workspace_id="wrk_1").model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        compact_job = await backend.dequeue_job(f"stream:{COMPACT_STREAM_NAME}", timeout_seconds=0)

        assert result.acked == 1
        assert compact_job is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_is_idempotent_for_duplicate_jobs() -> None:
    payload = json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "scope": "assistant_mode",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 1,
                    "payload": {"kind": "preference"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload, payload]
    )
    try:
        job = _extract_job(str(message["id"])).model_dump(mode="json")
        await backend.stream_add(EXTRACT_STREAM_NAME, job)
        await backend.stream_add(EXTRACT_STREAM_NAME, job)

        first = await ingest_worker.run_once()
        second = await ingest_worker.run_once()

        assert first.acked == 1
        assert second.acked == 1
        stored = await memories.list_for_user("usr_1")
        assert len(stored) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_handles_empty_stream_gracefully() -> None:
    connection, _clock, _backend, _provider, _memories, ingest_worker, _contract_worker, _message = await _build_runtime(
        []
    )
    try:
        result = await ingest_worker.run_once(block_ms=10)
        assert result.received == 0
        assert result.acked == 0
        assert result.failed == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_belief_lock_utility_handles_acquire_release_and_contention() -> None:
    backend = InProcessBackend()

    first_lock = await acquire_belief_lock(backend, "blf_1", attempts=1)
    assert first_lock is not None
    assert await acquire_belief_lock(backend, "blf_1", attempts=2, base_delay_seconds=0.001) is None
    await backend.release_lock("belief:blf_1", "wrong-token")
    assert await acquire_belief_lock(backend, "blf_1", attempts=1) is None
    await backend.release_lock("belief:blf_1", first_lock)
    assert await acquire_belief_lock(backend, "blf_1", attempts=1) is not None


@pytest.mark.asyncio
async def test_ingest_worker_run_recovers_from_unexpected_loop_errors(monkeypatch) -> None:
    connection, _clock, _backend, _provider, _memories, ingest_worker, _contract_worker, _message = await _build_runtime(
        []
    )
    calls = 0
    sleeps: list[float] = []

    async def fake_run_once(*, consumer_name: str = "ingest-1", block_ms: int | None = 0):
        del consumer_name, block_ms
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("stream read failed")
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(ingest_worker, "run_once", fake_run_once)
    monkeypatch.setattr("atagia.workers.ingest_worker.asyncio.sleep", fake_sleep)
    try:
        with pytest.raises(asyncio.CancelledError):
            await ingest_worker.run()
        assert calls == 2
        assert sleeps == [1.0]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_worker_run_recovers_from_unexpected_loop_errors(monkeypatch) -> None:
    connection, _clock, _backend, _provider, _memories, _ingest_worker, contract_worker, _message = await _build_runtime(
        []
    )
    calls = 0
    sleeps: list[float] = []

    async def fake_run_once(*, consumer_name: str = "contract-1", block_ms: int | None = 0):
        del consumer_name, block_ms
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("stream read failed")
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(contract_worker, "run_once", fake_run_once)
    monkeypatch.setattr("atagia.workers.contract_worker.asyncio.sleep", fake_sleep)
    try:
        with pytest.raises(asyncio.CancelledError):
            await contract_worker.run()
        assert calls == 2
        assert sleeps == [1.0]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_worker_is_idempotent_for_duplicate_jobs() -> None:
    payload = json.dumps(
        {
            "signals": [
                {
                    "canonical_text": "I prefer concise debugging answers",
                    "dimension_name": "directness",
                    "value_json": {"label": "concise", "score": 0.88},
                    "confidence": 0.92,
                    "scope": "assistant_mode",
                    "source_kind": "inferred",
                    "privacy_level": 1,
                }
            ],
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, provider, memories, _ingest_worker, contract_worker, message = await _build_runtime(
        [payload, payload]
    )
    try:
        job = _contract_job(str(message["id"])).model_dump(mode="json")
        await backend.stream_add(CONTRACT_STREAM_NAME, job)
        await backend.stream_add(CONTRACT_STREAM_NAME, job)

        first = await contract_worker.run_once()
        second = await contract_worker.run_once()

        contract_memories = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.INTERACTION_CONTRACT.value
        ]
        assert first.acked == 1
        assert second.acked == 1
        assert len(contract_memories) == 1
        assert len(provider.requests) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_worker_projects_even_when_ingest_already_persisted_contract_memory() -> None:
    extract_payload = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "confidence": 0.93,
                    "scope": "assistant_mode",
                    "source_kind": "inferred",
                    "privacy_level": 1,
                    "payload": {
                        "dimension_name": "directness",
                        "value_json": {"label": "concise", "score": 0.9},
                    },
                }
            ],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    project_payload = json.dumps(
        {
            "signals": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "dimension_name": "directness",
                    "value_json": {"label": "concise", "score": 0.9},
                    "confidence": 0.93,
                    "scope": "assistant_mode",
                    "source_kind": "inferred",
                    "privacy_level": 1,
                }
            ],
            "nothing_durable": False,
        }
    )
    connection, clock, _backend, _provider, memories, ingest_worker, contract_worker, message = await _build_runtime(
        [extract_payload, project_payload]
    )
    contracts = ContractDimensionRepository(connection, clock)
    try:
        await ingest_worker.process_job(_extract_job(str(message["id"])).model_dump(mode="json"))

        raw_contracts = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.INTERACTION_CONTRACT.value
        ]
        before_projection = await contracts.list_for_context("usr_1", "coding_debug", None, "cnv_1")

        await contract_worker.process_job(_contract_job(str(message["id"])).model_dump(mode="json"))

        after_projection = await contracts.list_for_context("usr_1", "coding_debug", None, "cnv_1")

        assert len(raw_contracts) == 1
        assert before_projection == []
        assert len(after_projection) == 1
        assert after_projection[0]["dimension_name"] == "directness"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_worker_reclaims_failed_pending_job_and_retries_successfully() -> None:
    payload = json.dumps(
        {
            "signals": [
                {
                    "canonical_text": "I prefer concise debugging answers",
                    "dimension_name": "directness",
                    "value_json": {"label": "concise", "score": 0.88},
                    "confidence": 0.92,
                    "scope": "assistant_mode",
                    "source_kind": "inferred",
                    "privacy_level": 1,
                }
            ],
            "nothing_durable": False,
        }
    )
    connection, _clock, backend, provider, memories, _ingest_worker, contract_worker, message = await _build_runtime(
        ["not-json", payload]
    )
    try:
        await backend.stream_add(CONTRACT_STREAM_NAME, _contract_job(str(message["id"])).model_dump(mode="json"))

        first = await contract_worker.run_once()
        second = await contract_worker.run_once()

        contract_memories = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.INTERACTION_CONTRACT.value
        ]
        assert first.failed == 1
        assert second.acked == 1
        assert len(contract_memories) == 1
        assert len(provider.requests) == 2
    finally:
        await connection.close()


def test_chat_reply_enqueues_stream_jobs_and_worker_processes_extraction(tmp_path: Path) -> None:
    settings = Settings(
        sqlite_path=str(tmp_path / "atagia-step12.db"),
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
        small_corpus_token_threshold_ratio=0.0,
    )
    provider = QueueProvider(
        [
            json.dumps([]),
            "Check the retry guard first.",
            json.dumps(
                {
                    "evidences": [
                        {
                            "canonical_text": "I prefer concise debugging answers for retry issues",
                            "scope": "assistant_mode",
                            "confidence": 0.92,
                            "source_kind": "extracted",
                            "privacy_level": 1,
                            "payload": {"kind": "preference"},
                        }
                    ],
                    "beliefs": [],
                    "contract_signals": [],
                    "state_updates": [],
                    "mode_guess": None,
                    "nothing_durable": False,
                }
            ),
            json.dumps({"signals": [], "nothing_durable": True}),
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
        ]
    )
    app = create_app(settings)
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        connection = client.portal.call(runtime.open_connection)
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 30, tzinfo=timezone.utc))
        runtime.llm_client = LLMClient(provider_name=provider.name, providers=[provider])

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
                "message_text": "I prefer concise debugging answers for retry issues.",
                "include_thinking": False,
                "metadata": {},
                "debug": True,
            },
        )

        assert response.status_code == 200
        assert len(response.json()["debug"]["enqueued_job_ids"]) == 3

        ingest_worker = IngestWorker(
            storage_backend=runtime.storage_backend,
            connection=connection,
            llm_client=runtime.llm_client,
            clock=runtime.clock,
            manifest_loader=runtime.manifest_loader,
            settings=runtime.settings,
        )
        contract_worker = ContractWorker(
            storage_backend=runtime.storage_backend,
            connection=connection,
            llm_client=runtime.llm_client,
            clock=runtime.clock,
            manifest_loader=runtime.manifest_loader,
            settings=runtime.settings,
        )
        memories = MemoryObjectRepository(connection, runtime.clock)
        assert client.portal.call(memories.list_for_user, "usr_1") == []

        ingest_result = client.portal.call(ingest_worker.run_once)
        contract_result = client.portal.call(contract_worker.run_once)
        second_ingest_result = client.portal.call(ingest_worker.run_once)
        stored = client.portal.call(memories.list_for_user, "usr_1")

        assert ingest_result.acked == 1
        assert second_ingest_result.acked == 1
        assert contract_result.acked == 1
        assert len(stored) == 1
