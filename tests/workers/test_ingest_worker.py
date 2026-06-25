"""Tests for stream-backed ingest workers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.locking import acquire_belief_lock
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.extractor import ExtractionPersistenceDetails
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.memory.text_chunker import ChunkingPlan, TextChunk
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
    MessageJobPayload,
    REVISE_STREAM_NAME,
    RevisionJobPayload,
    WorkerControlMode,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    ExtractionResult,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    ScoredCandidate,
)
from atagia.services.llm_client import (
    LLMError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    StructuredOutputError,
    TransientLLMError,
)
from atagia.services.worker_control_service import WorkerControlService
from atagia.workers.contract_worker import ContractWorker
from atagia.workers.ingest_worker import IngestWorker
from tests.extraction_payload_support import (
    is_memory_extraction_card_purpose,
    memory_extraction_card_output_from_payload,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _is_need_detection_card_purpose(purpose: object) -> bool:
    value = str(purpose)
    return value.startswith("need_detection_") and value.endswith("_card")


_LANGUAGE_PROFILE_CARD_PURPOSES = {
    "user_language_profile_observed_card",
    "user_language_profile_preference_card",
    "user_language_profile_ability_card",
    "user_language_profile_norm_card",
}

_EMPTY_LANGUAGE_PROFILE_CARD_OUTPUTS = {
    "user_language_profile_observed_card": "none",
    "user_language_profile_preference_card": "none",
    "user_language_profile_ability_card": "none",
    "user_language_profile_norm_card": "none",
}

_MEMORY_EXTRACTION_ENRICHMENT_CARD_PURPOSES = {
    "memory_extraction_kind_scope_card",
    "memory_extraction_evidence_card",
    "memory_extraction_index_card",
    "memory_extraction_temporal_card",
    "memory_extraction_belief_card",
    "memory_extraction_coverage_members_card",
}


def _is_language_profile_card_purpose(purpose: object) -> bool:
    return str(purpose) in _LANGUAGE_PROFILE_CARD_PURPOSES


class QueueProvider(LLMProvider):
    name = "queue-worker"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []
        self._active_language_profile_outputs: dict[str, str] | None = None
        self._active_language_profile_consumed: set[str] = set()
        self._active_extraction_payload: str | None = None
        self._active_extraction_consumed: set[str] = set()

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if _is_need_detection_card_purpose(request.metadata.get("purpose")):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "retry loop",
                "need_detection_search_words_other_language_card": "none",
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=outputs[str(request.metadata.get("purpose"))],
            )
        if request.metadata.get("purpose") == "retrieval_surface_generation_dry_run":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {"surfaces": self._retrieval_surface_payloads(request)}
                ),
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
        if request.metadata.get("purpose") == "consequence_gate_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="no",
            )
        if request.metadata.get("purpose") == "consequence_tendency_inference":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"tendency_text": ""}),
            )
        if request.metadata.get("purpose") == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        if _is_language_profile_card_purpose(request.metadata.get("purpose")):
            purpose = str(request.metadata.get("purpose"))
            if self._active_language_profile_outputs is None:
                output_bundle = _EMPTY_LANGUAGE_PROFILE_CARD_OUTPUTS
                if self.outputs and _is_user_language_profile_card_output_bundle(
                    self.outputs[0]
                ):
                    output_bundle = {
                        **_EMPTY_LANGUAGE_PROFILE_CARD_OUTPUTS,
                        **json.loads(self.outputs.pop(0)),
                    }
                self._active_language_profile_outputs = output_bundle
                self._active_language_profile_consumed = set()
            output_text = self._active_language_profile_outputs[purpose]
            self._active_language_profile_consumed.add(purpose)
            if (
                self._active_language_profile_consumed
                == _LANGUAGE_PROFILE_CARD_PURPOSES
            ):
                self._active_language_profile_outputs = None
                self._active_language_profile_consumed = set()
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output_text,
            )
        if is_memory_extraction_card_purpose(request.metadata.get("purpose")):
            purpose = str(request.metadata.get("purpose"))
            if purpose == "memory_extraction_candidate_card":
                if not self.outputs:
                    raise AssertionError("No queued output left for this test")
                if self.outputs[0] == "__llm_error__":
                    raise LLMError("provider failed")
                if self.outputs[0] == "__transient_provider_unavailable__":
                    raise TransientLLMError(
                        "provider unavailable",
                        retry_after_seconds=120.0,
                    )
                self._active_extraction_payload = self.outputs.pop(0)
                self._active_extraction_consumed = set()
                output_text = memory_extraction_card_output_from_payload(
                    self._active_extraction_payload,
                    purpose,
                )
                if output_text == "none" or "|" not in output_text:
                    self._active_extraction_payload = None
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text=output_text,
                )
            output_text = memory_extraction_card_output_from_payload(
                self._active_extraction_payload or {"candidates": []},
                purpose,
            )
            self._active_extraction_consumed.add(purpose)
            if (
                self._active_extraction_consumed
                == _MEMORY_EXTRACTION_ENRICHMENT_CARD_PURPOSES
            ):
                self._active_extraction_payload = None
                self._active_extraction_consumed = set()
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output_text,
            )
        if not self.outputs:
            raise AssertionError("No queued output left for this test")
        if self.outputs[0] == "__llm_error__":
            raise LLMError("provider failed")
        if self.outputs[0] == "__transient_provider_unavailable__":
            raise TransientLLMError(
                "provider unavailable",
                retry_after_seconds=120.0,
            )
        output_text = self.outputs.pop(0)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in worker tests")

    def _retrieval_surface_payloads(
        self,
        request: LLMCompletionRequest,
    ) -> list[dict[str, object]]:
        prompt = request.messages[-1].content
        assert isinstance(prompt, str)
        start = prompt.index("<source_memories>") + len("<source_memories>")
        end = prompt.index("</source_memories>")
        source_memories = json.loads(prompt[start:end].strip())
        assert isinstance(source_memories, list)
        return [
            {
                "memory_id": str(memory["id"]),
                "surface_type": "alias",
                "surface_text": "respuestas depuracion concisas",
                "alias_kind": "domain_synonym",
                "language_code": "es",
                "preserve_verbatim": False,
                "non_evidential": True,
                "confidence": 0.74,
                "visibility_policy": "base_memory_gated",
                "derivation": {"reason": "worker fixture dry-run"},
            }
            for memory in source_memories
            if isinstance(memory, dict)
        ]


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


def _resolved_policy():
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    return PolicyResolver().resolve(manifest, None, None)


def _persisted_surface_plan(
    fts_query: str,
    *,
    query_type: str = "slot_fill",
    exact_recall_mode: bool = True,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=fts_query,
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(
                text=fts_query,
                fts_queries=[fts_query],
                fts_query_kinds=["surface_probe"],
            )
        ],
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        query_type=query_type,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        retrieval_levels=[0],
        exact_recall_mode=exact_recall_mode,
    )


def _is_user_language_profile_card_output_bundle(output_text: str) -> bool:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return bool(_LANGUAGE_PROFILE_CARD_PURPOSES & set(payload))


def _public_extraction_payload() -> str:
    return json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": "I prefer concise debugging answers for retry issues",
                    "scope": "user",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "sensitivity": "public",
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


def _high_risk_extraction_payload() -> str:
    return json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": "The retry issue recovery PIN is stored elsewhere",
                    "scope": "user",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 3,
                    "sensitivity": "secret",
                    "memory_category": "pin_or_password",
                    "payload": {"kind": "credential"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )


async def _retrieval_surface_counts(connection) -> tuple[int, int]:
    surface_cursor = await connection.execute(
        "SELECT COUNT(*) AS count FROM memory_retrieval_surfaces"
    )
    surface_row = await surface_cursor.fetchone()
    fts_cursor = await connection.execute(
        "SELECT COUNT(*) AS count FROM memory_retrieval_surfaces_fts"
    )
    fts_row = await fts_cursor.fetchone()
    return int(surface_row["count"]), int(fts_row["count"])


def _retrieval_packet_request_count(provider: QueueProvider) -> int:
    return sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "retrieval_surface_generation_dry_run"
    )


async def _build_runtime(
    outputs: list[str],
    *,
    workspace_id: str | None = None,
    extra_messages: int = 0,
    settings: Settings | None = None,
    structured_output_retry_attempts: int = 1,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    backend = InProcessBackend()
    provider = QueueProvider(outputs)
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        structured_output_retry_attempts=structured_output_retry_attempts,
    )
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


def _extract_job(
    message_id: str,
    *,
    workspace_id: str | None = None,
    message_text: str = "I prefer concise debugging answers for retry issues.",
) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_extract_1",
        job_type="extract_memory_candidates",
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload={
            "message_id": message_id,
            "message_text": message_text,
            "role": "user",
            "assistant_mode_id": "coding_debug",
            "workspace_id": workspace_id,
            "recent_messages": [],
        },
        created_at=datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc),
    )


def _test_chunk_plan(text: str = "I prefer concise debugging answers for retry issues.") -> ChunkingPlan:
    return ChunkingPlan(
        chunks=[TextChunk(text=text)],
        chunked=False,
        fallback_count=0,
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


def test_retrieval_packet_settings_default_off_and_parse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED", raising=False)

    defaults = Settings.from_env()

    assert defaults.retrieval_packets_dry_run_enabled is False
    assert defaults.retrieval_packets_write_enabled is False

    monkeypatch.setenv("ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED", "true")
    writer_only = Settings.from_env()

    assert writer_only.retrieval_packets_dry_run_enabled is False
    assert writer_only.retrieval_packets_write_enabled is True

    monkeypatch.setenv("ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED", "true")
    both = Settings.from_env()

    assert both.retrieval_packets_dry_run_enabled is True
    assert both.retrieval_packets_write_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings_overrides", "expect_dry_run", "expect_writer"),
    [
        ({}, False, False),
        ({"retrieval_packets_dry_run_enabled": True}, True, False),
        ({"retrieval_packets_write_enabled": True}, False, False),
        (
            {
                "retrieval_packets_dry_run_enabled": True,
                "retrieval_packets_write_enabled": True,
            },
            True,
            True,
        ),
    ],
)
async def test_ingest_worker_retrieval_packet_constructor_wiring(
    settings_overrides: dict[str, object],
    expect_dry_run: bool,
    expect_writer: bool,
) -> None:
    (
        connection,
        _clock,
        _backend,
        _provider,
        memories,
        ingest_worker,
        _contract_worker,
        _message,
    ) = await _build_runtime([], settings=_settings(**settings_overrides))
    try:
        extractor = ingest_worker._extractor

        assert extractor._retrieval_packet_dry_run_enabled is expect_dry_run
        assert (extractor._retrieval_packet_dry_run_generator is not None) is expect_dry_run
        assert extractor._retrieval_packet_surface_write_enabled is expect_writer
        assert (extractor._retrieval_packet_surface_writer is not None) is expect_writer
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings_overrides", "expected_packet_requests", "expected_surface_rows"),
    [
        ({}, 0, 0),
        ({"retrieval_packets_write_enabled": True}, 0, 0),
        ({"retrieval_packets_dry_run_enabled": True}, 1, 0),
        (
            {
                "retrieval_packets_dry_run_enabled": True,
                "retrieval_packets_write_enabled": True,
            },
            1,
            1,
        ),
    ],
)
async def test_ingest_worker_retrieval_packet_flags_control_calls_and_writes(
    settings_overrides: dict[str, object],
    expected_packet_requests: int,
    expected_surface_rows: int,
) -> None:
    (
        connection,
        _clock,
        backend,
        provider,
        memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        [_public_extraction_payload()],
        settings=_settings(**settings_overrides),
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        surface_count, fts_count = await _retrieval_surface_counts(connection)
        stored = await memories.list_for_user("usr_1")

        assert result.acked == 1
        assert len(stored) == 1
        assert _retrieval_packet_request_count(provider) == expected_packet_requests
        assert surface_count == expected_surface_rows
        assert fts_count == expected_surface_rows
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_updates_user_language_profile_after_ingest() -> None:
    language_profile_update = json.dumps(
        {
            "user_language_profile_observed_card": "es",
            "user_language_profile_preference_card": (
                "default_answer_language es ordinary_chat"
            ),
            "user_language_profile_ability_card": "none",
            "user_language_profile_norm_card": "none",
        }
    )
    (
        connection,
        clock,
        _backend,
        provider,
        _memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime([_public_extraction_payload(), language_profile_update])
    try:
        await ingest_worker.process_job(_extract_job(str(message["id"])).model_dump(mode="json"))

        repository = CommunicationProfileRepository(connection, clock)
        profile = await repository.get_user_language_profile_for_context(
            ExtractionConversationContext(
                user_id="usr_1",
                conversation_id="cnv_1",
                source_message_id=str(message["id"]),
                workspace_id=None,
                assistant_mode_id="coding_debug",
                platform_id="default",
                character_id=None,
            )
        )
        assert profile is not None
        assert [row.language_code for row in profile.observed_user_languages] == ["es"]
        assert profile.explicit_language_preferences[0].preference_kind == "default_answer_language"
        assert any(
            request.metadata.get("purpose") in _LANGUAGE_PROFILE_CARD_PURPOSES
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_retrieval_packet_surface_recovers_and_composes_base_memory() -> None:
    (
        connection,
        clock,
        backend,
        provider,
        memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        [_public_extraction_payload()],
        settings=_settings(
            retrieval_packets_dry_run_enabled=True,
            retrieval_packets_write_enabled=True,
        ),
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        surface_count, fts_count = await _retrieval_surface_counts(connection)
        stored = await memories.list_for_user("usr_1")

        assert result.acked == 1
        assert len(stored) == 1
        assert _retrieval_packet_request_count(provider) == 1
        assert surface_count == 1
        assert fts_count == 1

        fts_query_audit: list[dict[str, object]] = []
        candidates = await CandidateSearch(connection, clock).search(
            _persisted_surface_plan("respuestas depuracion concisas"),
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        memory_id = str(stored[0]["id"])
        assert [str(candidate["id"]) for candidate in candidates] == [memory_id]
        assert candidates[0]["canonical_text"] == (
            "I prefer concise debugging answers for retry issues"
        )
        persisted_matches = [
            match
            for match in candidates[0].get("fts_query_matches", [])
            if match.get("source") == "persisted_surface"
        ]
        assert persisted_matches
        assert persisted_matches[0]["non_evidential"] is True
        persisted_audit = [
            entry
            for entry in fts_query_audit
            if entry.get("source") == "persisted_surface"
        ]
        assert persisted_audit[0]["raw_rows"] == 1

        composed = ContextComposer(clock).compose(
            [
                ScoredCandidate(
                    memory_id=memory_id,
                    memory_object=candidates[0],
                    llm_applicability=1.0,
                    retrieval_score=float(candidates[0].get("rrf_score", 0.0)),
                    vitality_boost=0.0,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=1.0,
                )
            ],
            current_contract={},
            user_state=None,
            resolved_policy=_resolved_policy(),
            conversation_messages=[],
            query_text="respuestas depuracion concisas",
            query_type="slot_fill",
            exact_recall_mode=True,
        )

        assert composed.selected_memory_ids == [memory_id]
        assert "I prefer concise debugging answers for retry issues" in composed.memory_block
        assert "respuestas depuracion concisas" not in composed.memory_block.lower()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_lean_extraction_persists_formerly_high_risk_as_active() -> None:
    # Privacy enrichment is deferred (F1.2): the lean contract carries no
    # privacy_level/sensitivity/memory_category, so content that previously
    # registered as high-risk now persists active and public. The retrieval
    # packet writer's high-risk exclusion is therefore not triggered at
    # extraction time; that exclusion is covered by the dedicated retrieval
    # surface policy tests.
    (
        connection,
        clock,
        backend,
        provider,
        memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        [_high_risk_extraction_payload()],
        settings=_settings(
            retrieval_packets_dry_run_enabled=True,
            retrieval_packets_write_enabled=True,
        ),
    )
    try:
        message_text = "The retry issue recovery PIN is stored elsewhere"
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), message_text=message_text).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        stored = await memories.list_for_user("usr_1")

        assert result.acked == 1
        assert len(stored) == 1
        assert stored[0]["status"] == MemoryStatus.ACTIVE.value
        assert stored[0]["privacy_level"] == 0
        assert stored[0]["sensitivity"] == MemorySensitivity.PUBLIC.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_hard_pause_does_not_claim_stream_job() -> None:
    (
        connection,
        clock,
        backend,
        _provider,
        _memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime([])
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )
        await WorkerControlService(connection, clock).set_mode(
            WorkerControlMode.HARD_PAUSE,
            reason="restore",
        )

        result = await ingest_worker.run_once(block_ms=0)

        assert result.received == 0
        queued = await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "probe",
            count=1,
            block_ms=0,
        )
        assert len(queued) == 1
    finally:
        await connection.close()


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
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

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
        ["__llm_error__"]
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(message["id"]),
                message_text="I am on vacation this week.",
            ).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        pending = backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)]

        assert result.received == 1
        assert result.acked == 0
        assert result.failed == 1
        assert len(pending) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_defers_transient_provider_failures_without_dead_lettering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("atagia.services.llm_client.asyncio.sleep", fake_sleep)
    (
        connection,
        clock,
        backend,
        _provider,
        memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        ["__transient_provider_unavailable__"],
        settings=_settings(
            worker_transient_defer_seconds=60.0,
            worker_transient_defer_max_seconds=90.0,
        ),
    )
    try:
        job = _extract_job(str(message["id"]))
        await JobRunRepository(connection, clock).create_queued_job(
            job_id=job.job_id,
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            source_message_ids=job.message_ids,
            source_token_estimate=12,
            size_bucket="small",
        )
        await backend.stream_add(EXTRACT_STREAM_NAME, job.model_dump(mode="json"))

        result = await ingest_worker.run_once()
        deferred_job = await JobRunRepository(connection, clock).get_job(job.job_id)
        immediate = await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "probe",
            count=1,
            block_ms=0,
        )

        assert result.deferred == 1
        assert result.failed == 0
        assert result.dead_lettered == 0
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        assert len(backend._stream_deferred[EXTRACT_STREAM_NAME]) == 1
        assert immediate == []
        assert await memories.list_for_user("usr_1") == []
        assert deferred_job is not None
        assert deferred_job["status"] == "retrying"
        assert deferred_job["attempt_count"] == 1
        assert deferred_job["deferred_until"] == "2026-03-31T04:01:30+00:00"
        assert deferred_job["transient_defer_count"] == 1
        assert deferred_job["first_deferred_at"] == "2026-03-31T04:00:00+00:00"
        assert deferred_job["last_deferred_at"] == "2026-03-31T04:00:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_dead_letters_after_transient_defer_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("atagia.services.llm_client.asyncio.sleep", fake_sleep)
    (
        connection,
        clock,
        backend,
        _provider,
        _memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        ["__transient_provider_unavailable__"],
        settings=_settings(
            worker_transient_defer_seconds=60.0,
            worker_transient_defer_max_seconds=90.0,
            worker_transient_defer_max_count=2,
            worker_transient_defer_max_age_seconds=3600.0,
        ),
    )
    try:
        job = _extract_job(str(message["id"]))
        repository = JobRunRepository(connection, clock)
        await repository.create_queued_job(
            job_id=job.job_id,
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            source_message_ids=job.message_ids,
            source_token_estimate=12,
            size_bucket="small",
        )
        await backend.stream_add(EXTRACT_STREAM_NAME, job.model_dump(mode="json"))

        first = await ingest_worker.run_once()
        backend._stream_deferred[EXTRACT_STREAM_NAME][0]["due_at"] = 0.0
        second = await ingest_worker.run_once()
        backend._stream_deferred[EXTRACT_STREAM_NAME][0]["due_at"] = 0.0
        third = await ingest_worker.run_once()
        dead_letter = await backend.dequeue_job(
            f"dead_letter:{EXTRACT_STREAM_NAME}",
            timeout_seconds=0,
        )
        stored_job = await repository.get_job(job.job_id)

        assert first.deferred == 1
        assert second.deferred == 1
        assert third.deferred == 0
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        assert backend._stream_deferred.get(EXTRACT_STREAM_NAME) is None
        assert dead_letter is not None
        assert dead_letter["delivery_count"] == 1
        assert "transient defer budget exhausted" in dead_letter["error"]
        assert stored_job is not None
        assert stored_job["status"] == "dead_lettered"
        assert stored_job["transient_defer_count"] == 3
        assert stored_job["deferred_until"] is None
        assert stored_job["error_class"] == "TransientDeferBudgetExceededError"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_dead_letters_after_transient_defer_age_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("atagia.services.llm_client.asyncio.sleep", fake_sleep)
    (
        connection,
        clock,
        backend,
        _provider,
        _memories,
        ingest_worker,
        _contract_worker,
        message,
    ) = await _build_runtime(
        ["__transient_provider_unavailable__"],
        settings=_settings(
            worker_transient_defer_seconds=60.0,
            worker_transient_defer_max_seconds=90.0,
            worker_transient_defer_max_count=99,
            worker_transient_defer_max_age_seconds=30.0,
        ),
    )
    try:
        job = _extract_job(str(message["id"]))
        repository = JobRunRepository(connection, clock)
        await repository.create_queued_job(
            job_id=job.job_id,
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            source_message_ids=job.message_ids,
            source_token_estimate=12,
            size_bucket="small",
        )
        await backend.stream_add(EXTRACT_STREAM_NAME, job.model_dump(mode="json"))

        first = await ingest_worker.run_once()
        clock.advance(seconds=31)
        backend._stream_deferred[EXTRACT_STREAM_NAME][0]["due_at"] = 0.0
        second = await ingest_worker.run_once()
        dead_letter = await backend.dequeue_job(
            f"dead_letter:{EXTRACT_STREAM_NAME}",
            timeout_seconds=0,
        )
        stored_job = await repository.get_job(job.job_id)

        assert first.deferred == 1
        assert second.failed == 1
        assert second.dead_lettered == 1
        assert dead_letter is not None
        assert "transient defer budget exhausted" in dead_letter["error"]
        assert stored_job is not None
        assert stored_job["transient_defer_count"] == 2
        assert stored_job["status"] == "dead_lettered"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_tolerates_malformed_card_output_as_no_durable_memory() -> None:
    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        ["not-json"]
    )
    try:
        await backend.stream_add(EXTRACT_STREAM_NAME, _extract_job(str(message["id"])).model_dump(mode="json"))

        result = await ingest_worker.run_once()

        assert result.received == 1
        assert result.acked == 1
        assert result.failed == 0
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        stored = await memories.list_for_user("usr_1")
        assert stored == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_dead_letters_after_max_failed_deliveries() -> None:
    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        ["__llm_error__"]
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
async def test_ingest_worker_card_assembly_drops_invalid_temporal_bounds() -> None:
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
    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [invalid_payload],
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(message["id"]),
                message_text="I am on vacation this week.",
            ).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{EXTRACT_STREAM_NAME}", timeout_seconds=0)
        stored = await memories.list_for_user("usr_1")

        assert result.acked == 1
        assert result.failed == 0
        assert result.dead_lettered == 0
        assert backend._stream_pending[(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)] == {}
        assert dead_letter is None
        assert len(stored) == 1
        assert stored[0]["temporal_type"] == "unknown"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_logs_structured_job_failure_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract(*args, **kwargs):
        del args, kwargs
        raise StructuredOutputError("Provider returned invalid structured output")

    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [],
    )
    try:
        ingest_worker._extractor.extract_with_persistence_and_chunk_plan = fake_extract
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        with caplog.at_level("WARNING", logger="atagia.workers.ingest_worker"):
            result = await ingest_worker.run_once()

        failure_records = [
            record
            for record in caplog.records
            if "Failed to process extraction job" in record.message
        ]
        assert result.failed == 1
        assert failure_records
        assert all(record.exc_info is None for record in failure_records)
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
async def test_ingest_worker_dedupes_pending_compaction_for_same_conversation_window() -> None:
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
        [payload, payload],
        workspace_id=None,
        extra_messages=9,
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), workspace_id=None).model_dump(mode="json"),
        )
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"]), workspace_id=None).model_dump(mode="json"),
        )

        first = await ingest_worker.run_once()
        second = await ingest_worker.run_once()
        first_compact_job = await backend.dequeue_job(f"stream:{COMPACT_STREAM_NAME}", timeout_seconds=0)
        second_compact_job = await backend.dequeue_job(f"stream:{COMPACT_STREAM_NAME}", timeout_seconds=0)

        assert first.acked == 1
        assert second.acked == 1
        assert first_compact_job is not None
        assert second_compact_job is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_rolls_back_failed_topic_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    connection, _clock, _backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [payload],
        settings=_settings(topic_working_set_enabled=True),
    )

    class FailingTopicRefreshService:
        def __init__(self, **kwargs) -> None:
            self.connection = kwargs["connection"]

        async def maybe_refresh_after_message(self, **_kwargs):
            await self.connection.execute("BEGIN")
            await self.connection.execute(
                """
                INSERT INTO conversation_topic_events(
                    id,
                    user_id,
                    conversation_id,
                    event_type,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "tpe_uncommitted",
                    "usr_1",
                    "cnv_1",
                    "updated",
                    "{}",
                    "2026-03-31T04:00:00+00:00",
                ),
            )
            raise LLMError("topic refresh failed")

    monkeypatch.setattr(
        "atagia.workers.ingest_worker.TopicWorkingSetRefreshService",
        FailingTopicRefreshService,
    )

    try:
        job = _extract_job(str(message["id"]))
        await ingest_worker._maybe_refresh_topic_working_set(
            envelope=job,
            job_payload=MessageJobPayload.model_validate(job.payload),
        )
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_topic_events WHERE id = ?",
            ("tpe_uncommitted",),
        )
        row = await cursor.fetchone()

        assert int(row["count"]) == 0
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
        return ExtractionPersistenceDetails(
            result=ExtractionResult(nothing_durable=False),
            persisted=persisted,
            chunk_plan=_test_chunk_plan(),
        )

    async def skip_side_effects(*args, **kwargs) -> None:
        del args, kwargs

    control_connection, _clock, control_backend, _provider, _memories, control_worker, _contract_worker, control_message = await _build_runtime(
        [],
    )
    try:
        control_worker._extractor.extract_with_persistence_and_chunk_plan = fake_extract
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
        ingest_worker._extractor.extract_with_persistence_and_chunk_plan = fake_extract
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
@pytest.mark.parametrize(
    ("existing_belief_id", "expected_validated", "expected_belief_id"),
    [
        ("mem_existing_match", True, "mem_existing_match"),
        (None, False, ""),
    ],
)
async def test_ingest_marks_belief_revision_claim_key_validated_by_match(
    existing_belief_id: str | None,
    expected_validated: bool,
    expected_belief_id: str,
) -> None:
    """Producer-side: the belief-branch revision job must carry
    ``claim_key_already_validated=bool(existing_belief_id)`` so the consumer
    can skip the re-validation it has already implicitly performed."""
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
        return ExtractionPersistenceDetails(
            result=ExtractionResult(nothing_durable=False),
            persisted=persisted,
            chunk_plan=_test_chunk_plan(),
        )

    async def skip_side_effects(*args, **kwargs) -> None:
        del args, kwargs

    async def fake_find_existing_belief_id(*args, **kwargs) -> str | None:
        del args, kwargs
        return existing_belief_id

    connection, _clock, backend, _provider, _memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [],
    )
    try:
        ingest_worker._extractor.extract_with_persistence_and_chunk_plan = fake_extract
        ingest_worker._process_consequence_detection = skip_side_effects
        ingest_worker._maybe_enqueue_conversation_compaction = skip_side_effects
        ingest_worker._find_existing_belief_id = fake_find_existing_belief_id
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        result = await ingest_worker.run_once()
        revision_job = await backend.dequeue_job(
            f"stream:{REVISE_STREAM_NAME}",
            timeout_seconds=0,
        )

        assert result.acked == 1
        assert revision_job is not None

        payload = RevisionJobPayload.model_validate(revision_job["payload"]["payload"])
        assert payload.claim_key == "response_style.debugging"
        assert payload.claim_key_already_validated is expected_validated
        assert payload.belief_id == expected_belief_id
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_treats_non_json_after_schema_fallback_as_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract(*args, **kwargs):
        del args, kwargs
        raise StructuredOutputError(
            "Provider returned non-JSON structured output after schema fallback",
            details=("$: Response was not valid JSON.",),
            reason="schema_fallback_non_json",
        )

    async def skip_side_effects(*args, **kwargs) -> None:
        del args, kwargs

    connection, _clock, backend, _provider, memories, ingest_worker, _contract_worker, message = await _build_runtime(
        [],
    )
    try:
        ingest_worker._extractor.extract_with_persistence_and_chunk_plan = fake_extract
        ingest_worker._process_consequence_detection = skip_side_effects
        ingest_worker._maybe_enqueue_conversation_compaction = skip_side_effects
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(str(message["id"])).model_dump(mode="json"),
        )

        with caplog.at_level("WARNING", logger="atagia.workers.ingest_worker"):
            result = await ingest_worker.run_once()
        stored = await memories.list_for_user("usr_1")
        records = [
            record
            for record in caplog.records
            if "after schema fallback returned non-JSON output" in record.getMessage()
        ]

        assert result.acked == 1
        assert result.failed == 0
        assert stored == []
        assert records
        assert records[0].exc_info is None
        assert "Traceback" not in caplog.text
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
async def test_contract_worker_reclaims_failed_pending_job_and_retries_successfully(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        ["not-json", "still-not-json", payload],
        structured_output_retry_attempts=0,
    )
    try:
        await backend.stream_add(CONTRACT_STREAM_NAME, _contract_job(str(message["id"])).model_dump(mode="json"))

        with caplog.at_level("WARNING", logger="atagia.workers.contract_worker"):
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
        assert len(provider.requests) == 3
        compact_records = [
            record
            for record in caplog.records
            if "due to structured output" in record.getMessage()
        ]
        assert compact_records
        assert compact_records[0].exc_info is None
    finally:
        await connection.close()


def test_chat_reply_enqueues_stream_jobs_and_worker_processes_extraction(tmp_path: Path) -> None:
    settings = Settings(
        sqlite_path=str(tmp_path / "atagia-step12.db"),
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
        small_corpus_token_threshold_ratio=0.0,
    )
    provider = QueueProvider(
        [
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
        try:
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
            assert response.json()["debug"]["selected_memory_count"] == 0
            assert "enqueued_job_ids" not in response.json()["debug"]

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
        finally:
            client.portal.call(connection.close)
