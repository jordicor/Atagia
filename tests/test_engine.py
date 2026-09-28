"""Tests for the library-mode Atagia engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from time import monotonic

import pytest

from atagia import Atagia
from atagia.core.clock import FrozenClock
from atagia.core.mind_repository import DEFAULT_MIND_ID
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)
from atagia.models.schemas_jobs import JobEnvelope, JobType, WORKER_GROUP_NAME
from atagia.models.schemas_replay import AblationConfig
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.chat_support import default_operational_profile_snapshot
from atagia.services.errors import (
    ConversationNotFoundError,
    MessageIdConflictError,
    SourceSequenceConflictError,
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.memory.token_document_frequency import TokenDocumentFrequencyCache

from tests.turn_telemetry_support import sample_turn_telemetry
from tests.extraction_payload_support import (
    is_memory_extraction_card_purpose,
    memory_extraction_card_output_from_payload,
)


async def _block_library_memory_access_for_selected_transcript(
    engine: Atagia,
    *,
    user_id: str,
    conversation_id: str,
    state: str,
) -> None:
    runtime = engine.runtime
    assert runtime is not None
    now = runtime.clock.now().isoformat()
    workflow_id = f"trw_library_{state}"
    connection = await runtime.open_connection()
    try:
        await connection.execute(
            """
            INSERT INTO transcript_rebuild_workflows(
                id,
                operation_id,
                user_id,
                conversation_id,
                selection_epoch,
                transcript_hash,
                mutation_kind,
                selected_message_ids_json,
                abandoned_message_ids_json,
                supporting_message_ids_json,
                affected_memory_ids_json,
                affected_summary_ids_json,
                orchestrator_job_id,
                stage,
                start_derivation_revision,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                f"op_library_{state}",
                user_id,
                conversation_id,
                1,
                f"hash_library_{state}",
                "replace",
                "[]",
                "[]",
                "[]",
                "[]",
                "[]",
                f"job_library_{state}",
                "remediation_required"
                if state == "remediation_required"
                else "aggregates",
                0,
                now,
                now,
            ),
        )
        await connection.execute(
            """
            INSERT INTO conversation_transcript_selections(
                user_id,
                conversation_id,
                selection_epoch,
                transcript_hash,
                current_workflow_id,
                state,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                conversation_id,
                1,
                f"hash_library_{state}",
                workflow_id,
                state,
                now,
            ),
        )
        await connection.commit()
    finally:
        await connection.close()


_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _is_need_detection_card_purpose(purpose: object) -> bool:
    value = str(purpose)
    return value.startswith("need_detection_") and value.endswith("_card")


def _persisted_surface_exact_plan(
    fts_query: str,
    *,
    conversation_id: str,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=fts_query,
        assistant_mode_id="coding_debug",
        conversation_id=conversation_id,
        sub_query_plans=[
            PlannedSubQuery(
                text=fts_query,
                fts_queries=[fts_query],
                fts_query_kinds=["surface_probe"],
            )
        ],
        scope_filter=[MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        query_type="default",
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        retrieval_levels=[0],
        exact_recall_mode=True,
    )


class EngineProvider(LLMProvider):
    name = "engine-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if _is_need_detection_card_purpose(purpose):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_query_language_card": "en",
                "need_detection_answer_language_card": "en",
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
                output_text=outputs[purpose],
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} useful" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": True,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "consent_confirmation_intent":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": "confirm"}),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        if is_memory_extraction_card_purpose(purpose):
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=memory_extraction_card_output_from_payload(
                    {
                        "candidates": [],
                        "nothing_durable": True,
                    },
                    purpose,
                    prompt="\n".join(message.content for message in request.messages),
                ),
            )
        if purpose == "contract_projection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "signals": [],
                        "nothing_durable": True,
                    }
                ),
            )
        if purpose == "consequence_gate_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="no",
            )
        if purpose.startswith("user_language_profile_") and purpose.endswith("_card"):
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="none",
            )
        if purpose == "topic_working_set_route_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="none",
            )
        if purpose == "initial_context_package_curation":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"items": [], "nothing_to_add": True}),
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(
            f"Embeddings are not used in engine tests: {request.model}"
        )


class FailingEngineProvider(EngineProvider):
    def __init__(self, fail_purpose: str) -> None:
        super().__init__()
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if purpose == self._fail_purpose or (
            self._fail_purpose == "applicability_scoring"
            and purpose
            in {
                "applicability_relevance_card",
                "applicability_date_card",
            }
        ):
            raise LLMError(f"Injected failure for {self._fail_purpose}")
        return await super().complete(request)


def test_engine_build_settings_preserves_llm_debug_io_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    debug_dir = tmp_path / "llm-debug"
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO", "true")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_DIR", str(debug_dir))
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_PURPOSES", "applicability_scoring")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_RAW", "true")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_MAX_CHARS", "12345")
    engine = Atagia(
        db_path=tmp_path / "debug-settings.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    settings = engine._build_settings()

    assert settings.llm_debug_io_enabled is True
    assert settings.llm_debug_io_dir == str(debug_dir)
    assert settings.llm_debug_io_purposes == ("applicability_scoring",)
    assert settings.llm_debug_io_raw is True
    assert settings.llm_debug_io_max_chars == 12345


def test_engine_build_settings_preserves_answer_postcondition_guard_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED", "true")
    engine = Atagia(db_path=tmp_path / "guard-settings.db")

    settings = engine._build_settings()

    assert settings.answer_postcondition_guard_enabled is True


def test_engine_build_settings_preserves_response_mode_and_adaptive_retrieval_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_RESPONSE_MODE", "smart_fast")
    monkeypatch.setenv("ATAGIA_ADAPTIVE_RETRIEVAL", "true")
    engine = Atagia(db_path=tmp_path / "mode-settings.db")

    settings = engine._build_settings()

    assert settings.response_mode == "smart_fast"
    assert settings.adaptive_retrieval is True


def test_engine_build_settings_allows_explicit_answer_postcondition_guard_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED", "false")
    engine = Atagia(
        db_path=tmp_path / "guard-settings.db",
        answer_postcondition_guard_enabled=True,
    )

    settings = engine._build_settings()

    assert settings.answer_postcondition_guard_enabled is True


def test_engine_build_settings_allows_explicit_answer_stance_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE", "reactive")
    engine = Atagia(
        db_path=tmp_path / "answer-stance-settings.db",
        answer_stance="proactive",
    )

    settings = engine._build_settings()

    assert settings.answer_stance == "proactive"


def test_engine_build_settings_allows_explicit_answer_stance_prompt_variant_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE_PROMPT_VARIANT", "template_v1")
    engine = Atagia(
        db_path=tmp_path / "answer-stance-variant-settings.db",
        answer_stance_prompt_variant="baseline",
    )

    settings = engine._build_settings()

    assert settings.answer_stance_prompt_variant == "baseline"


def test_engine_build_settings_preserves_memory_bm25_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_MEMORY_FTS_CANONICAL_BM25_WEIGHT", "9.0")
    monkeypatch.setenv("ATAGIA_MEMORY_FTS_INDEX_BM25_WEIGHT", "0.2")
    engine = Atagia(db_path=tmp_path / "bm25-settings.db")

    settings = engine._build_settings()

    assert settings.memory_fts_canonical_bm25_weight == 9.0
    assert settings.memory_fts_index_bm25_weight == 0.2


def test_engine_build_settings_preserves_embedding_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec")
    monkeypatch.setenv("ATAGIA_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("ATAGIA_EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "17")
    monkeypatch.setenv("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "3")
    engine = Atagia(db_path=tmp_path / "embedding-settings.db")

    settings = engine._build_settings()

    assert settings.embedding_backend == "sqlite_vec"
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    assert settings.embedding_vector_limit_cap == 17
    assert settings.embedding_search_overfetch_multiplier == 3


def test_engine_build_settings_preserves_topic_working_set_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_MESSAGE_LAG", "11")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_STALE_MESSAGE_LAG", "13")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_TOKEN_LAG", "1700")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_STALE_TOKEN_LAG", "2300")
    monkeypatch.setenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_BATCH_MESSAGES", "5")
    engine = Atagia(db_path=tmp_path / "topic-settings.db")

    settings = engine._build_settings()

    assert settings.topic_working_set_enabled is False
    assert settings.topic_working_set_refresh_message_lag == 11
    assert settings.topic_working_set_stale_message_lag == 13
    assert settings.topic_working_set_refresh_token_lag == 1700
    assert settings.topic_working_set_stale_token_lag == 2300
    assert settings.topic_working_set_refresh_batch_messages == 5


def test_engine_build_settings_preserves_evidence_route_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_SURFACES_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_STRUCTURED_ONLY", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_SPAN_COADMISSION_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_FACT_FACET_RETRIEVAL_LIMIT", "7")
    monkeypatch.setenv("ATAGIA_FACT_FACET_RETRIEVAL_RRF_WEIGHT", "1.4")
    monkeypatch.setenv("ATAGIA_APPLICABILITY_GATE_MODE", "shadow")
    monkeypatch.setenv("ATAGIA_ANSWER_POSTCONDITION_RETRY_MAX_OUTPUT_TOKENS", "12288")
    engine = Atagia(db_path=tmp_path / "evidence-route-settings.db")

    settings = engine._build_settings()

    assert settings.retrieval_packets_dry_run_enabled is True
    assert settings.retrieval_packets_write_enabled is True
    assert settings.fact_facet_surfaces_enabled is True
    assert settings.fact_facet_retrieval_enabled is True
    assert settings.fact_facet_structured_only is True
    assert settings.fact_facet_span_coadmission_enabled is True
    assert settings.fact_facet_retrieval_limit == 7
    assert settings.fact_facet_retrieval_rrf_weight == 1.4
    assert settings.applicability_gate_mode == "shadow"
    assert settings.answer_postcondition_retry_max_output_tokens == 12288


def test_engine_build_settings_allows_explicit_embedding_backend_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec")
    engine = Atagia(
        db_path=tmp_path / "embedding-settings.db",
        embedding_backend="none",
    )

    settings = engine._build_settings()

    assert settings.embedding_backend == "none"


def test_engine_build_settings_resolves_resource_env_from_external_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("migrations", "manifests", "operational_profiles"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATAGIA_MIGRATIONS_PATH", "./migrations")
    monkeypatch.setenv("ATAGIA_MANIFESTS_PATH", "./manifests")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILES_PATH", "./operational_profiles")

    engine = Atagia(db_path=tmp_path / "external-cwd.db")
    settings = engine._build_settings()

    assert settings.migrations_dir().resolve() == tmp_path / "migrations"
    assert settings.manifests_dir().resolve() == tmp_path / "manifests"
    assert (
        settings.operational_profiles_dir().resolve()
        == tmp_path / "operational_profiles"
    )


def _install_stub_client(
    monkeypatch: pytest.MonkeyPatch, provider: EngineProvider
) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    # The engine tests assert the full retrieval pipeline runs (including
    # need detection), so disable the small-corpus shortcut for the duration
    # of the test. Individual tests can still override via setenv.
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


def _normal_operational_profile_token(engine: Atagia) -> str:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    return default_operational_profile_snapshot(
        loader=engine.runtime.operational_profile_loader,
        settings=engine.runtime.settings,
    ).token


async def _active_cache_identity(
    engine: Atagia,
    user_id: str,
) -> tuple[str, int, int]:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    connection = await engine.runtime.open_connection()
    try:
        identity = await UserLifecycleRepository(
            connection,
            engine.runtime.clock,
        ).get_active_identity(user_id)
    finally:
        await connection.close()
    if identity is None:
        raise AssertionError("Active user lifecycle identity should exist")
    return (
        identity.lifecycle_epoch,
        identity.cache_revision,
        identity.derivation_revision,
    )


@pytest.mark.asyncio
async def test_engine_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    assert engine.runtime is not None

    await engine.close()
    assert engine.runtime is None


@pytest.mark.asyncio
async def test_engine_setup_respects_env_context_cache_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "false")
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.context_cache_enabled is False
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_setup_respects_env_graph_projection_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_GRAPH_PROJECTION_ENABLED", "true")
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.graph_projection_enabled is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_setup_respects_chunking_disable_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    monkeypatch.setenv("ATAGIA_DISABLE_CHUNKING_EXTRACTION", "false")
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        disable_chunking_extraction=True,
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.disable_chunking_extraction is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_create_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_user("usr_1")
        await engine.create_workspace("usr_1", "wrk_1", "Workspace")
        conversation_id = await engine.create_conversation(
            "usr_1",
            "cnv_1",
            workspace_id="wrk_1",
            assistant_mode_id="coding_debug",
        )

        connection = await engine.runtime.open_connection()
        try:
            users = UserRepository(connection, engine.runtime.clock)
            workspaces = WorkspaceRepository(connection, engine.runtime.clock)
            conversations = ConversationRepository(connection, engine.runtime.clock)
            assert await users.get_user("usr_1") is not None
            assert await workspaces.get_workspace("wrk_1", "usr_1") is not None
            assert (
                await conversations.get_conversation(conversation_id, "usr_1")
                is not None
            )
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_lifecycle_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_close",
            assistant_mode_id="coding_debug",
            temporary=True,
            temporary_ttl_seconds=3600,
            purge_on_close=False,
        )
        closed = await engine.close_conversation("usr_1", "cnv_close")
        assert closed["status"] == "closed"

        await engine.create_conversation(
            "usr_1", "cnv_memory", assistant_mode_id="coding_debug"
        )
        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            await memories.create_memory_object(
                user_id="usr_1",
                conversation_id="cnv_memory",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="Original lifecycle memory.",
                source_kind=MemorySourceKind.VERBATIM,
                confidence=0.9,
                privacy_level=0,
                memory_id="mem_lifecycle",
                payload={"writer_kind": "manual"},
            )
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            await surfaces.upsert_surface(
                user_id="usr_1",
                memory_id="mem_lifecycle",
                surface_type="alias",
                surface_text="lifecycle surface",
            )
        finally:
            await connection.close()

        edited = await engine.edit_memory(
            "usr_1", "mem_lifecycle", "Updated lifecycle memory."
        )
        assert edited["canonical_text"] == "Updated lifecycle memory."
        connection = await engine.runtime.open_connection()
        try:
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            surface_rows = await surfaces.list_surfaces_for_memory(
                user_id="usr_1",
                memory_id="mem_lifecycle",
            )
            assert [row["status"] for row in surface_rows] == ["stale"]
            assert (
                await surfaces.search_active_surfaces(
                    user_id="usr_1",
                    fts_query="lifecycle",
                )
                == []
            )
        finally:
            await connection.close()
        memory_report = await engine.delete_memory(
            "usr_1", "mem_lifecycle", hard=True, confirmation="HARD_DELETE_MEMORY"
        )
        assert memory_report.deleted_memories == 1
        connection = await engine.runtime.open_connection()
        try:
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            assert (
                await surfaces.list_surfaces_for_memory(
                    user_id="usr_1",
                    memory_id="mem_lifecycle",
                )
                == []
            )
            assert (
                await surfaces.search_active_surfaces(
                    user_id="usr_1",
                    fts_query="lifecycle",
                )
                == []
            )
        finally:
            await connection.close()

        conversation_report = await engine.delete_conversation(
            "usr_1",
            "cnv_memory",
            confirmation="DELETE_CONVERSATION",
        )
        assert conversation_report.conversation_id == "cnv_memory"

        await engine.create_conversation(
            "usr_1", "cnv_erase", assistant_mode_id="coding_debug"
        )
        erase_report = await engine.erase_user_data(
            "usr_1", confirmation="ERASE_ALL_DATA"
        )
        assert erase_report.deleted_conversations == 3
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_conversation_delete_tombstones_retrieval_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_surface_delete",
            assistant_mode_id="coding_debug",
        )
        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            await memories.create_memory_object(
                user_id="usr_1",
                conversation_id="cnv_surface_delete",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="Conversation memory whose derived surface should be tombstoned.",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                memory_id="mem_conversation_surface",
            )
            await surfaces.upsert_surface(
                user_id="usr_1",
                memory_id="mem_conversation_surface",
                surface_type="alias",
                surface_text="conversationtombsurface",
            )
        finally:
            await connection.close()

        report = await engine.delete_conversation(
            "usr_1",
            "cnv_surface_delete",
            confirmation="DELETE_CONVERSATION",
        )
        assert report.deleted_memories == 1

        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            memory = await memories.get_memory_object(
                "mem_conversation_surface", "usr_1"
            )
            assert memory is not None
            assert memory["status"] == MemoryStatus.DELETED.value
            assert memory["archived_by_conversation_id"] == "cnv_surface_delete"

            surface_rows = await surfaces.list_surfaces_for_memory(
                user_id="usr_1",
                memory_id="mem_conversation_surface",
            )
            assert [row["status"] for row in surface_rows] == ["deleted"]
            assert (
                await surfaces.search_active_surfaces(
                    user_id="usr_1",
                    fts_query="conversationtombsurface",
                )
                == []
            )

            search = CandidateSearch(
                connection,
                engine.runtime.clock,
                token_document_frequency_cache=TokenDocumentFrequencyCache(),
            )
            candidates = await search.search(
                _persisted_surface_exact_plan(
                    "conversationtombsurface",
                    conversation_id="cnv_surface_delete",
                ),
                user_id="usr_1",
                fts_query_audit=[],
            )
            assert candidates == []

            cursor = await connection.execute(
                """
                SELECT mrs.id
                FROM memory_retrieval_surfaces_fts
                JOIN memory_retrieval_surfaces AS mrs
                  ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
                JOIN memory_objects AS mo
                  ON mo.id = mrs.memory_id
                WHERE memory_retrieval_surfaces_fts MATCH ?
                  AND mrs.user_id = ?
                  AND mo.user_id = ?
                  AND mrs.status = 'active'
                  AND mo.status = ?
                """,
                (
                    "conversationtombsurface",
                    "usr_1",
                    "usr_1",
                    MemoryStatus.ACTIVE.value,
                ),
            )
            assert await cursor.fetchall() == []
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_user_erasure_deletes_retrieval_surfaces_and_fts_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_surface_erase")
        await engine.create_conversation(
            "usr_surface_erase",
            "cnv_surface_erase",
            assistant_mode_id="coding_debug",
        )
        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            await memories.create_memory_object(
                user_id="usr_surface_erase",
                conversation_id="cnv_surface_erase",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="User erasure should remove derived retrieval surfaces.",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                memory_id="mem_user_surface_erase",
            )
            await surfaces.upsert_surface(
                user_id="usr_surface_erase",
                memory_id="mem_user_surface_erase",
                surface_type="alias",
                surface_text="usererasuresurface",
            )

            before_candidates = await CandidateSearch(
                connection,
                engine.runtime.clock,
                token_document_frequency_cache=TokenDocumentFrequencyCache(),
            ).search(
                _persisted_surface_exact_plan(
                    "usererasuresurface",
                    conversation_id="cnv_surface_erase",
                ),
                user_id="usr_surface_erase",
                fts_query_audit=[],
            )
            assert [candidate["id"] for candidate in before_candidates] == [
                "mem_user_surface_erase"
            ]
        finally:
            await connection.close()

        erase_report = await engine.erase_user_data(
            "usr_surface_erase",
            confirmation="ERASE_ALL_DATA",
        )
        assert erase_report.deleted_memories == 1

        connection = await engine.runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            surfaces = MemoryRetrievalSurfaceRepository(
                connection, engine.runtime.clock
            )
            assert (
                await memories.get_memory_object(
                    "mem_user_surface_erase",
                    "usr_surface_erase",
                )
                is None
            )
            assert (
                await surfaces.list_surfaces_for_memory(
                    user_id="usr_surface_erase",
                    memory_id="mem_user_surface_erase",
                )
                == []
            )
            assert (
                await surfaces.search_active_surfaces(
                    user_id="usr_surface_erase",
                    fts_query="usererasuresurface",
                )
                == []
            )
            assert (
                await CandidateSearch(
                    connection,
                    engine.runtime.clock,
                    token_document_frequency_cache=TokenDocumentFrequencyCache(),
                ).search(
                    _persisted_surface_exact_plan(
                        "usererasuresurface",
                        conversation_id="cnv_surface_erase",
                    ),
                    user_id="usr_surface_erase",
                    fts_query_audit=[],
                )
                == []
            )

            cursor = await connection.execute(
                """
                SELECT mrs.id
                FROM memory_retrieval_surfaces_fts
                JOIN memory_retrieval_surfaces AS mrs
                  ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
                JOIN memory_objects AS mo
                  ON mo.id = mrs.memory_id
                WHERE memory_retrieval_surfaces_fts MATCH ?
                  AND mrs.user_id = ?
                  AND mo.user_id = ?
                  AND mrs.status = 'active'
                  AND mo.status = ?
                """,
                (
                    "usererasuresurface",
                    "usr_surface_erase",
                    "usr_surface_erase",
                    MemoryStatus.ACTIVE.value,
                ),
            )
            assert await cursor.fetchall() == []

            cursor = await connection.execute(
                """
                SELECT rowid
                FROM memory_retrieval_surfaces_fts
                WHERE memory_retrieval_surfaces_fts MATCH ?
                """,
                ("usererasuresurface",),
            )
            assert await cursor.fetchall() == []
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_lifecycle_deletes_only_targeted_retrieval_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        for conversation_id in ("cnv_mem_1", "cnv_mem_2", "cnv_del_1", "cnv_del_2"):
            await engine.create_conversation(
                "usr_1", conversation_id, assistant_mode_id="coding_debug"
            )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            memories = MemoryObjectRepository(connection, engine.runtime.clock)
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            for conversation_id, message_id in (
                ("cnv_mem_1", "msg_mem_1"),
                ("cnv_mem_2", "msg_mem_2"),
                ("cnv_del_1", "msg_del_1"),
                ("cnv_del_2", "msg_del_2"),
            ):
                await messages.create_message(
                    message_id, conversation_id, "user", 1, "hello", 1, {}
                )
            for conversation_id, memory_id, text in (
                ("cnv_mem_1", "mem_target", "target memory"),
                ("cnv_mem_2", "mem_other", "other memory"),
            ):
                await memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id=conversation_id,
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text=text,
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id=memory_id,
                    payload={
                        "source_message_ids": [
                            "msg_mem_1" if memory_id == "mem_target" else "msg_mem_2"
                        ]
                    },
                )
            await memories.create_memory_object(
                user_id="usr_1",
                conversation_id=None,
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.USER,
                canonical_text="broad memory sourced by a deleted chat",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                payload={"source_message_ids": ["msg_del_1"]},
                memory_id="mem_del_broad",
            )
            for event_id, conversation_id, message_id, selected_ids in (
                ("evt_mem_target", "cnv_mem_1", "msg_mem_1", ["mem_target"]),
                ("evt_mem_other", "cnv_mem_2", "msg_mem_2", ["mem_other"]),
                ("evt_del_target", "cnv_del_1", "msg_del_1", []),
                ("evt_cross_selected", "cnv_del_2", "msg_del_2", ["mem_del_broad"]),
                ("evt_del_other", "cnv_del_2", "msg_del_2", []),
            ):
                await events.create_event(
                    {
                        "id": event_id,
                        "user_id": "usr_1",
                        "conversation_id": conversation_id,
                        "request_message_id": message_id,
                        "assistant_mode_id": "coding_debug",
                        "retrieval_plan_json": {},
                        "selected_memory_ids_json": selected_ids,
                        "context_view_json": {"event_id": event_id},
                        "outcome_json": {},
                    },
                    telemetry=sample_turn_telemetry(),
                )
        finally:
            await connection.close()

        await engine.delete_memory(
            "usr_1",
            "mem_target",
            hard=True,
            confirmation="HARD_DELETE_MEMORY",
        )
        lifecycle_epoch, cache_revision, derivation_revision = (
            await _active_cache_identity(engine, "usr_1")
        )
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_del_2",
            workspace_id=None,
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )
        await engine.runtime.storage_backend.set_context_view(
            cache_key,
            {"user_id": "usr_1", "conversation_id": "cnv_del_2"},
            ttl_seconds=60,
        )
        assert (
            await engine.runtime.storage_backend.get_context_view(cache_key) is not None
        )
        await engine.delete_conversation(
            "usr_1",
            "cnv_del_1",
            confirmation="DELETE_CONVERSATION",
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None

        connection = await engine.runtime.open_connection()
        try:
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            remaining_ids = {
                row["id"] for row in await events.list_events("usr_1", None, limit=20)
            }
            assert remaining_ids == {"evt_mem_other", "evt_del_other"}
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        answer_stance="proactive",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
            message_id="aurvek:msg:ctx-1",
        )

        assert isinstance(context.system_prompt, str)
        assert context.system_prompt
        assert "Answer stance: proactive" in context.system_prompt
        assert "related, not the same fact" in context.system_prompt
        assert context.request_message_id == "aurvek:msg:ctx-1"
        assert context.recent_transcript == []
        assert context.recent_transcript_trace is not None
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert stored_messages[-1]["role"] == "user"
            assert stored_messages[-1]["id"] == "aurvek:msg:ctx-1"
            assert (
                stored_messages[-1]["text"] == "Please help me debug this retry loop."
            )
            assert stored_messages[-1]["occurred_at"] == "2023-05-08T13:56:00"
        finally:
            await connection.close()

        duplicate_context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
            message_id="aurvek:msg:ctx-1",
        )

        assert duplicate_context.request_message_id == "aurvek:msg:ctx-1"
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert [message["id"] for message in stored_messages] == [
                "aurvek:msg:ctx-1"
            ]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_includes_recent_transcript_without_fts_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.ingest_message(
            "usr_1",
            "cnv_1",
            "user",
            "Yesterday I went to the bank on Carrer Major.",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What time does that usually close?",
        )

        transcript_texts = [entry.text for entry in context.recent_transcript]
        assert transcript_texts == ["Yesterday I went to the bank on Carrer Major."]
        assert "What time does that usually close?" not in transcript_texts
        assert "<recent_transcript_json>" in context.system_prompt
        assert "Carrer Major" in context.system_prompt
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_keeps_recent_transcript_inside_context_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The coding_debug manifest asks for 8000 transcript tokens; the default 8192
    # envelope allocates 1638 to that section. The envelope is the hard ceiling,
    # so the manifest value can only lower it, never raise it.
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did we decide?",
        )

        assert context.recent_transcript_trace is not None
        assert context.recent_transcript_trace.budget_tokens == 1_638
        assert context.context_envelope_trace is not None
        assert context.context_envelope_trace["total_budget_tokens"] == 8_192
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_uses_constructor_context_envelope_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_envelope_budget_tokens=10_000,
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did we decide?",
        )

        assert context.recent_transcript_trace is not None
        assert context.recent_transcript_trace.budget_tokens == 2_000
        assert context.context_envelope_trace is not None
        assert context.context_envelope_trace["total_budget_tokens"] == 10_000
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_uses_context_envelope_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did we decide?",
            ablation=AblationConfig(context_envelope_budget_tokens=10_000),
        )

        assert context.recent_transcript_trace is not None
        assert context.recent_transcript_trace.budget_tokens == 2_000
        assert context.context_envelope_trace is not None
        assert context.context_envelope_trace["total_budget_tokens"] == 10_000
        assert context.context_envelope_trace["reserve_tokens"] == 0
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_can_disable_recent_transcript_for_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.ingest_message(
            "usr_1",
            "cnv_1",
            "user",
            "This prior sentence should not be injected as raw transcript.",
        )

        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did I just say?",
        )

        assert context.recent_transcript == []
        assert context.recent_transcript_omissions == []
        assert context.recent_transcript_trace is None
        assert context.assistant_guidance == []
        assert "<recent_transcript_json>" not in context.system_prompt
        assert "This prior sentence should not be injected" not in context.system_prompt
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_add_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
            occurred_at="2023-05-09T14:10:00",
            message_id="aurvek:msg:assistant-1",
        )
        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
            occurred_at="2023-05-09T14:10:00",
            message_id="aurvek:msg:assistant-1",
        )
        assert await engine.flush(timeout_seconds=5.0) is True

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert stored_messages[-1]["role"] == "assistant"
            assert stored_messages[-1]["id"] == "aurvek:msg:assistant-1"
            assert stored_messages[-1]["text"] == "Check the retry guard first."
            assert stored_messages[-1]["occurred_at"] == "2023-05-09T14:10:00"
            assert [message["role"] for message in stored_messages] == [
                "user",
                "assistant",
            ]
        finally:
            await connection.close()

        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction_candidate_card") == 2
        assert purposes.count("contract_projection") == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    async with engine:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        context = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        assert context.system_prompt

    assert engine._closed is True
    assert engine.runtime is None


@pytest.mark.asyncio
async def test_engine_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        engine.runtime.clock = FrozenClock(
            datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc)
        )
        result = await engine.chat(
            user_id="usr_1",
            conversation_id="cnv_1",
            mode="coding_debug",
            message="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
        )

        assert result.response_text == "Check the retry guard first."
        assert result.retrieval_event_id is not None
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert stored_messages[0]["occurred_at"] == "2023-05-08T13:56:00"
            assert stored_messages[1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
            assert stored_messages[1]["created_at"] == "2026-03-31T04:00:00+00:00"
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_chat_can_disable_raw_recent_transcript_for_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.ingest_message(
            "usr_1",
            "cnv_1",
            "user",
            "This prior chat message must not reach the chat model as transcript.",
        )

        await engine.chat(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Answer from retrieved memory only.",
        )

        chat_request = next(
            request
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        )
        chat_prompt = "\n".join(message.content for message in chat_request.messages)
        assert "This prior chat message must not reach" not in chat_prompt
        assert [message.role for message in chat_request.messages] == ["system", "user"]
        assert chat_request.messages[-1].content == "Answer from retrieved memory only."
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_cache_hit_exposes_observability_and_records_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        first = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        second = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="continue",
        )

        assert first.from_cache is False
        assert second.from_cache is True
        assert second.need_detection_skipped is True
        assert second.detected_needs == []
        assert [entry.text for entry in second.recent_transcript] == [
            "Please help me debug this retry loop."
        ]
        assert "continue" not in [entry.text for entry in second.recent_transcript]
        # A retrieve-only call is still a retrieval, cache hit included, so both
        # calls persist a queryable event under the retrieval-only surface.
        assert first.retrieval_event_id is not None
        assert second.retrieval_event_id is not None
        assert first.retrieval_duration_ms > 0.0
        connection = await engine.runtime.open_connection()
        try:
            events = RetrievalEventRepository(connection, engine.runtime.clock)
            listed = await events.list_events("usr_1", "cnv_1", limit=10)
            assert {event["id"] for event in listed} == {
                first.retrieval_event_id,
                second.retrieval_event_id,
            }
            assert {event["turn_surface"] for event in listed} == {"context"}
            assert all(event["response_message_id"] is None for event in listed)
            assert all(event["retrieval_duration_ms"] > 0.0 for event in listed)
            assert all(event["turn_to_event_write_wall_ms"] > 0.0 for event in listed)
            # The retrieve-only surface counts retrieval-stage calls only: the
            # host runs the reply itself, so no reply round-trip is recorded.
            fresh = next(
                event for event in listed if event["id"] == first.retrieval_event_id
            )
            cached = next(
                event for event in listed if event["id"] == second.retrieval_event_id
            )
            assert fresh["llm_failed_calls"] == 0
            assert fresh["stage_timings_ms_json"]
            # The cache hit still costs a staleness decision, and that decision
            # is a provider round-trip the meter has to see -- the whole point
            # of metering this surface separately.
            assert cached["llm_total_calls"] >= 1
            assert "context_cache_signal_detection" in (
                cached["llm_by_purpose_json"]
            )
            assert cached["llm_total_latency_ms"] > 0.0
            assert "chat_reply" not in cached["llm_by_purpose_json"]
            for event in listed:
                by_purpose = event["llm_by_purpose_json"]
                assert sum(usage["calls"] for usage in by_purpose.values()) == (
                    event["llm_total_calls"]
                )
                # CS-1.5: the breakdown accounts for the turn's wall time too.
                assert sum(
                    usage["latency_ms"] for usage in by_purpose.values()
                ) == pytest.approx(event["llm_total_latency_ms"])
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_add_response_invalidates_stable_context_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        lifecycle_epoch, cache_revision, derivation_revision = (
            await _active_cache_identity(engine, "usr_1")
        )
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )
        assert (
            await engine.runtime.storage_backend.get_context_view(cache_key) is not None
        )

        await engine.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Check the retry guard first.",
        )

        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_invalidates_stable_context_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        lifecycle_epoch, cache_revision, derivation_revision = (
            await _active_cache_identity(engine, "usr_1")
        )
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="default_assistant",
            active_mind_id=DEFAULT_MIND_ID,
            mind_topology="unimind",
            operational_profile_token=_normal_operational_profile_token(engine),
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )
        assert (
            await engine.runtime.storage_backend.get_context_view(cache_key) is not None
        )

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Check the retry guard first.",
        )

        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-flush.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
    finally:
        await engine.close()


async def _cancel_runtime_dispatcher(engine: Atagia) -> None:
    assert engine.runtime is not None
    dispatcher_tasks = [
        task
        for task in engine.runtime.worker_tasks
        if task.get_name() == "atagia-durable-job-dispatcher"
    ]
    assert len(dispatcher_tasks) == 1
    dispatcher_tasks[0].cancel()
    await asyncio.gather(*dispatcher_tasks, return_exceptions=True)


async def _finish_flush_test_job(engine: Atagia, stream_name: str) -> None:
    assert engine.runtime is not None
    connection = await engine.runtime.open_connection()
    try:
        tracking = JobTrackingService(
            connection,
            engine.runtime.clock,
            workers_enabled=True,
            settings=engine.runtime.settings,
        )
        await tracking.dispatch_pending_jobs(engine.runtime.storage_backend)
        messages = await engine.runtime.storage_backend.stream_read(
            stream_name,
            WORKER_GROUP_NAME,
            "flush-test-consumer",
            count=1,
            block_ms=0,
        )
        assert len(messages) == 1
        claim = await tracking.claim_notification(
            messages[0],
            owner_id="flush-test-owner",
        )
        assert claim is not None
        assert await tracking.finish_claim_succeeded(claim)
        await engine.runtime.storage_backend.stream_ack(
            stream_name,
            WORKER_GROUP_NAME,
            messages[0].message_id,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_engine_flush_waits_for_durable_job_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-delayed-dispatch-flush.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    stream_name = "atagia:test_flush_delayed_dispatch"

    await engine.setup()
    try:
        assert engine.runtime is not None
        await _cancel_runtime_dispatcher(engine)
        await engine.create_user("usr_flush")
        await engine.runtime.storage_backend.stream_ensure_group(
            stream_name,
            WORKER_GROUP_NAME,
        )
        connection = await engine.runtime.open_connection()
        try:
            await JobTrackingService(
                connection,
                engine.runtime.clock,
                workers_enabled=True,
                settings=engine.runtime.settings,
            ).enqueue_job(
                engine.runtime.storage_backend,
                stream_name,
                JobEnvelope(
                    job_id="job_flush_delayed_dispatch",
                    job_type=JobType.RUN_EVALUATION,
                    user_id="usr_flush",
                ),
                dispatch=False,
            )
        finally:
            await connection.close()

        flush_task = asyncio.create_task(engine.flush(timeout_seconds=2.0))
        await asyncio.sleep(0.1)
        assert not flush_task.done()

        await _finish_flush_test_job(engine, stream_name)
        assert await flush_task is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_flush_waits_for_durable_job_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    database_path = tmp_path / "atagia-engine-restart-flush.db"
    stream_name = "atagia:test_flush_restart"
    first_engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await first_engine.setup()
    try:
        assert first_engine.runtime is not None
        await _cancel_runtime_dispatcher(first_engine)
        await first_engine.create_user("usr_flush")
        connection = await first_engine.runtime.open_connection()
        try:
            await JobTrackingService(
                connection,
                first_engine.runtime.clock,
                workers_enabled=True,
                settings=first_engine.runtime.settings,
            ).enqueue_job(
                first_engine.runtime.storage_backend,
                stream_name,
                JobEnvelope(
                    job_id="job_flush_restart",
                    job_type=JobType.RUN_EVALUATION,
                    user_id="usr_flush",
                ),
                dispatch=False,
            )
        finally:
            await connection.close()
    finally:
        await first_engine.close()

    second_engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await second_engine.setup()
    try:
        assert second_engine.runtime is not None
        await _cancel_runtime_dispatcher(second_engine)
        await second_engine.runtime.storage_backend.stream_ensure_group(
            stream_name,
            WORKER_GROUP_NAME,
        )
        flush_task = asyncio.create_task(second_engine.flush(timeout_seconds=2.0))
        await asyncio.sleep(0.1)
        assert not flush_task.done()

        await _finish_flush_test_job(second_engine, stream_name)
        assert await flush_task is True
    finally:
        await second_engine.close()


@pytest.mark.asyncio
async def test_engine_flush_retries_locked_job_table_until_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    stream_name = "atagia:test_flush_locked"

    await engine.setup()
    blocker = None
    try:
        assert engine.runtime is not None
        await _cancel_runtime_dispatcher(engine)
        await engine.create_user("usr_flush")
        await engine.runtime.storage_backend.stream_ensure_group(
            stream_name,
            WORKER_GROUP_NAME,
        )
        connection = await engine.runtime.open_connection()
        try:
            await JobTrackingService(
                connection,
                engine.runtime.clock,
                workers_enabled=True,
                settings=engine.runtime.settings,
            ).enqueue_job(
                engine.runtime.storage_backend,
                stream_name,
                JobEnvelope(
                    job_id="job_flush_locked",
                    job_type=JobType.RUN_EVALUATION,
                    user_id="usr_flush",
                ),
                dispatch=False,
            )
        finally:
            await connection.close()

        blocker = await engine.runtime.open_connection()
        await blocker.execute("BEGIN IMMEDIATE")
        await blocker.execute(
            """
            UPDATE worker_job_runs
            SET status = status
            WHERE job_id = 'job_flush_locked'
            """
        )
        flush_task = asyncio.create_task(engine.flush(timeout_seconds=2.0))
        await asyncio.sleep(0.1)
        assert not flush_task.done()

        await blocker.rollback()
        await _finish_flush_test_job(engine, stream_name)
        assert await flush_task is True
    finally:
        if blocker is not None:
            if blocker.in_transaction:
                await blocker.rollback()
            await blocker.close()
        await engine.close()


@pytest.mark.asyncio
async def test_engine_flush_locked_job_table_honors_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    blocker = None
    try:
        assert engine.runtime is not None
        await _cancel_runtime_dispatcher(engine)
        await engine.create_user("usr_flush")
        connection = await engine.runtime.open_connection()
        try:
            await JobTrackingService(
                connection,
                engine.runtime.clock,
                workers_enabled=True,
                settings=engine.runtime.settings,
            ).enqueue_job(
                engine.runtime.storage_backend,
                "atagia:test_flush_locked_timeout",
                JobEnvelope(
                    job_id="job_flush_locked_timeout",
                    job_type=JobType.RUN_EVALUATION,
                    user_id="usr_flush",
                ),
                dispatch=False,
            )
        finally:
            await connection.close()

        blocker = await engine.runtime.open_connection()
        await blocker.execute("BEGIN IMMEDIATE")
        await blocker.execute(
            """
            UPDATE worker_job_runs
            SET status = status
            WHERE job_id = 'job_flush_locked_timeout'
            """
        )
        started_at = monotonic()
        assert await engine.flush(timeout_seconds=0.15) is False
        assert monotonic() - started_at < 0.5
    finally:
        if blocker is not None:
            if blocker.in_transaction:
                await blocker.rollback()
            await blocker.close()
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ablation_switches_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        skip_belief_revision=True,
        skip_compaction=True,
    )

    await engine.setup()
    try:
        assert engine.runtime is not None
        assert engine.runtime.settings.skip_belief_revision is True
        assert engine.runtime.settings.skip_compaction is True
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        engine.runtime.clock = FrozenClock(
            datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc)
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please help me debug this retry loop.",
            occurred_at="2023-05-08T13:56:00",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Check the retry guard first.",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
        assert not any(
            request.metadata.get("purpose")
            in {
                "need_detection",
                "applicability_scoring",
                "applicability_relevance_card",
                "applicability_date_card",
            }
            for request in provider.requests
        )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert [message["role"] for message in stored_messages[-2:]] == [
                "user",
                "assistant",
            ]
            assert (
                stored_messages[-2]["text"] == "Please help me debug this retry loop."
            )
            assert stored_messages[-2]["occurred_at"] == "2023-05-08T13:56:00"
            assert stored_messages[-1]["text"] == "Check the retry guard first."
            assert stored_messages[-1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
        finally:
            await connection.close()

        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction_candidate_card") == 2
        assert purposes.count("contract_projection") == 1
        assert any(
            request.metadata.get("purpose") == "memory_extraction_candidate_card"
            and "<message_timestamp>2023-05-08T13:56:00</message_timestamp>"
            in request.messages[1].content
            for request in provider.requests
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_is_idempotent_by_message_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest-idempotent.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )

        assert await engine.flush(timeout_seconds=5.0) is True
        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert [message["id"] for message in stored_messages] == ["aurvek:msg:1"]
        finally:
            await connection.close()
        purposes = [request.metadata.get("purpose") for request in provider.requests]
        assert purposes.count("memory_extraction_candidate_card") == 1
        assert purposes.count("contract_projection") == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_source_seq_preserves_backfill_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-ingest-source-seq.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )

        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="First message.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
            source_seq=1,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Third message arrived before retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:3",
            source_seq=3,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Second message retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:2",
            source_seq=2,
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="assistant",
            text="Second message retry.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:2",
            source_seq=2,
        )

        with pytest.raises(
            SourceSequenceConflictError, match="source_seq already exists"
        ):
            await engine.ingest_message(
                user_id="usr_1",
                conversation_id="cnv_1",
                role="user",
                text="Different message for occupied source seq.",
                mode="coding_debug",
                platform_id="aurvek",
                message_id="aurvek:msg:other",
                source_seq=2,
            )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert [(message["id"], message["seq"]) for message in stored_messages] == [
                ("aurvek:msg:1", 1),
                ("aurvek:msg:2", 2),
                ("aurvek:msg:3", 3),
            ]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_message_id_conflict_rejects_incompatible_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-message-id-conflict.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            mode="coding_debug",
            platform_id="aurvek",
        )
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text="Original content.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="aurvek:msg:1",
        )

        with pytest.raises(MessageIdConflictError, match="different role or text"):
            await engine.ingest_message(
                user_id="usr_1",
                conversation_id="cnv_1",
                role="user",
                text="Changed content.",
                mode="coding_debug",
                platform_id="aurvek",
                message_id="aurvek:msg:1",
            )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_ingest_message_marks_large_plain_text_skip_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-engine-heavy-message.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        skip_compaction=True,
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        huge_text = "large biography segment " * 800
        await engine.ingest_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            role="user",
            text=huge_text,
        )

        assert await engine.flush(timeout_seconds=5.0) is True

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            stored = stored_messages[-1]
            assert stored["text"] == huge_text
            assert stored["include_raw"] == 0
            assert stored["skip_by_default"] == 1
            assert stored["heavy_content"] == 1
            assert stored["requires_explicit_request"] == 1
            assert stored["policy_reason"] == "mechanical_size_threshold"
            assert "large biography segment" not in stored["context_placeholder"]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_rolls_back_user_message_when_scoring_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingEngineProvider("applicability_scoring")
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )
        # Seed a memory so the shortlist is non-empty and scoring is invoked.
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            await memories.create_memory_object(
                user_id="usr_1",
                workspace_id=None,
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="retry loop websocket backoff",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
            )
        finally:
            await connection.close()

        with pytest.raises(LLMError):
            await engine.get_context(
                user_id="usr_1",
                conversation_id="cnv_1",
                message="Please help me debug this retry loop.",
            )

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            assert (
                await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0) == []
            )
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_engine_get_context_degrades_when_need_detector_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingEngineProvider("need_detection")
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
        )

        # Need detector failure should no longer break retrieval. The pipeline
        # falls back to the base search and the user message is persisted.
        result = await engine.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please help me debug this retry loop.",
        )
        assert result is not None

        connection = await engine.runtime.open_connection()
        try:
            messages = MessageRepository(connection, engine.runtime.clock)
            stored = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [row["text"] for row in stored] == [
                "Please help me debug this retry loop.",
            ]
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_library_optional_identity_hint_matrix(tmp_path: Path) -> None:
    engine = Atagia(
        db_path=tmp_path / "library-optional-identity.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("user_identity")
        await engine.create_conversation(
            "user_identity",
            "conversation_identity",
            platform_id="platform-a",
            user_persona_id="persona-a",
            character_id="character-a",
            mode="coding_debug",
        )

        assert (
            await engine.create_conversation(
                "user_identity",
                "conversation_identity",
                mode="coding_debug",
            )
            == "conversation_identity"
        )

        for field_name, persisted_value in (
            ("user_persona_id", "persona-a"),
            ("platform_id", "platform-a"),
            ("character_id", "character-a"),
        ):
            assert (
                await engine.create_conversation(
                    "user_identity",
                    "conversation_identity",
                    mode="coding_debug",
                    **{field_name: persisted_value},
                )
                == "conversation_identity"
            )
            with pytest.raises(
                ConversationNotFoundError,
                match="Conversation not found for user",
            ):
                await engine.create_conversation(
                    "user_identity",
                    "conversation_identity",
                    mode="coding_debug",
                    **{field_name: "conflicting-value"},
                )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_readme_library_quickstart_sequence_runs_with_omitted_identity_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)

    async with Atagia(
        db_path=tmp_path / "memory.db",
        anthropic_api_key="test-anthropic-key",
        llm_forced_global_model="anthropic/claude-sonnet-4-6",
    ) as engine:
        await engine.create_user("user_1")
        await engine.create_conversation(
            "user_1",
            "conv_1",
            platform_id="web",
            character_id="project_backend",
            mode="coding_debug",
        )

        context = await engine.get_context(
            user_id="user_1",
            conversation_id="conv_1",
            message="What did we decide about the migration?",
            mode="coding_debug",
        )

        result = await engine.chat(
            user_id="user_1",
            conversation_id="conv_1",
            message="Why is the test failing?",
            mode="coding_debug",
        )

    assert context.system_prompt
    assert result.response_text == "Check the retry guard first."


@pytest.mark.parametrize(
    ("state", "error_type", "error_message"),
    [
        (
            "rebuilding",
            TranscriptRebuildInProgressError,
            "Selected transcript rebuild is still in progress",
        ),
        (
            "remediation_required",
            TranscriptRebuildRemediationRequiredError,
            "Selected transcript rebuild requires remediation before memory access",
        ),
    ],
)
@pytest.mark.asyncio
async def test_library_memory_surfaces_fail_closed_during_selected_transcript_rebuild(
    tmp_path: Path,
    state: str,
    error_type: type[Exception],
    error_message: str,
) -> None:
    async with Atagia(
        db_path=tmp_path / f"library-selected-{state}.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    ) as engine:
        await engine.create_user("usr_selected")
        await engine.create_conversation(
            "usr_selected",
            "cnv_selected",
            platform_id="library",
        )
        await _block_library_memory_access_for_selected_transcript(
            engine,
            user_id="usr_selected",
            conversation_id="cnv_selected",
            state=state,
        )

        for operation in (
            engine.get_memory_preferences("usr_selected"),
            engine.set_memory_preferences(
                "usr_selected",
                remember_across_chats=False,
            ),
            engine.ingest_message(
                "usr_selected",
                "cnv_selected",
                "user",
                "This write must not become visible.",
            ),
        ):
            with pytest.raises(
                error_type,
                match=f"^{re.escape(error_message)}$",
            ):
                await operation

        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            preferences = await UserRepository(
                connection,
                runtime.clock,
            ).get_memory_preferences("usr_selected")
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation("cnv_selected", "usr_selected")
        finally:
            await connection.close()

        assert preferences is not None
        assert bool(preferences["remember_across_chats"]) is True
        assert messages == []
