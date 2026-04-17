"""Tests for the reusable retrieval pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    RetrievalTrace,
    SummaryViewKind,
    VerbatimPinTargetKind,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.services.retrieval_pipeline import RetrievalPipeline

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class PipelineProvider(LLMProvider):
    name = "retrieval-pipeline-tests"

    def __init__(
        self,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self.need_response = need_response or {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["retry loop websocket backoff"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "retry loop websocket backoff",
                    "fts_phrase": "retry loop websocket backoff",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        }
        self.score_map = dict(score_map or {})
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(self.need_response),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            payload = [
                {"memory_id": memory_id, "llm_applicability": self.score_map.get(memory_id, 0.5)}
                for memory_id in memory_ids
            ]
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(payload),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in retrieval pipeline tests")


class FailingPipelineProvider(PipelineProvider):
    """Provider that injects an LLMError for a chosen purpose."""

    def __init__(
        self,
        fail_purpose: str,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(need_response=need_response, score_map=score_map)
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if str(request.metadata.get("purpose")) == self._fail_purpose:
            self.requests.append(request)
            raise LLMError(f"Injected {self._fail_purpose} failure")
        return await super().complete(request)


def _settings(*, small_corpus_token_threshold_ratio: float = 0.0) -> Settings:
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
        small_corpus_token_threshold_ratio=small_corpus_token_threshold_ratio,
    )


async def _build_runtime(
    *,
    mode_id: str = "coding_debug",
    provider: PipelineProvider | None = None,
    settings: Settings | None = None,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    contracts = ContractDimensionRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", mode_id, "Chat")
    llm_provider = provider or PipelineProvider()
    resolved_settings = settings or _settings()
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=llm_provider.name, providers=[llm_provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=resolved_settings,
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()[mode_id]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id="wrk_1",
        assistant_mode_id=mode_id,
        recent_messages=[],
    )
    return connection, memories, contracts, pipeline, llm_provider, resolved_policy, context


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    scope: MemoryScope,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    assistant_mode_id: str = "coding_debug",
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> dict[str, object]:
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1" if scope is MemoryScope.CONVERSATION else None,
        assistant_mode_id=assistant_mode_id,
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED if object_type is not MemoryObjectType.INTERACTION_CONTRACT else MemorySourceKind.INFERRED,
        confidence=0.8,
        privacy_level=0,
        status=status,
        memory_id=memory_id,
    )


def _score_request_memory_ids(provider: PipelineProvider) -> list[str]:
    for request in provider.requests:
        if str(request.metadata.get("purpose")) != "applicability_scoring":
            continue
        return _MEMORY_ID_PATTERN.findall(request.messages[1].content)
    return []


def _slot_fill_plan(
    *,
    query_text: str = "What was the root cause of the outage?",
    exact_recall_mode: bool = False,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=query_text,
        assistant_mode_id="general_qa",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["root cause outage"],
        sub_query_plans=[
            PlannedSubQuery(
                text=query_text,
                sparse_phrase="root cause outage",
                quoted_phrases=[],
                must_keep_terms=["root", "cause"],
                fts_queries=["root cause outage"],
            )
        ],
        query_type="slot_fill",
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.WORKSPACE, MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=30,
        max_context_items=8,
        privacy_ceiling=3,
        retrieval_levels=[0, 1],
        exact_recall_mode=exact_recall_mode,
    )


def _candidate_record(
    *,
    memory_id: str,
    canonical_text: str,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    rrf_score: float = 0.0,
    payload_json: dict[str, object] | None = None,
    updated_at: str = "2026-04-05T12:00:00+00:00",
    retrieval_sources: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": memory_id,
        "object_type": object_type.value,
        "status": status.value,
        "scope": scope.value,
        "privacy_level": privacy_level,
        "assistant_mode_id": "general_qa",
        "conversation_id": "cnv_1" if scope is MemoryScope.CONVERSATION else None,
        "workspace_id": "wrk_1" if scope in {MemoryScope.CONVERSATION, MemoryScope.WORKSPACE} else None,
        "canonical_text": canonical_text,
        "payload_json": payload_json or {},
        "source_kind": MemorySourceKind.EXTRACTED.value,
        "confidence": 0.8,
        "stability": 0.5,
        "vitality": 0.0,
        "maya_score": 0.0,
        "rrf_score": rrf_score,
        "updated_at": updated_at,
        "created_at": updated_at,
        "valid_from": None,
        "valid_to": None,
        "temporal_type": "unknown",
        "channel_ranks": {
            "fts": 1 if rrf_score > 0.0 else None,
            "embedding": None,
            "consequence": None,
            "raw_message": None,
        },
        "matched_sub_queries": ["What was the root cause of the outage?"],
        "retrieval_sources": retrieval_sources or ["fts"],
    }


@pytest.mark.asyncio
async def test_pipeline_executes_full_flow() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(
        need_response={
            "needs": [
                {
                    "need_type": "follow_up_failure",
                    "confidence": 0.82,
                    "reasoning": "The user is describing an unresolved retry problem.",
                }
            ],
            "temporal_range": None,
            "sub_queries": [
                "retry loop websocket backoff",
                "production failure outcome",
            ],
            "sparse_query_hints": [
                {
                    "sub_query_text": "retry loop websocket backoff",
                    "fts_phrase": "retry loop websocket backoff",
                },
                {
                    "sub_query_text": "production failure outcome",
                    "fts_phrase": "production failure outcome",
                },
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.91},
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert [need.need_type.value for need in result.detected_needs] == ["follow_up_failure"]
        assert result.retrieval_plan.query_type == "broad_list"
        assert [sub_query.text for sub_query in result.retrieval_plan.sub_query_plans] == [
            "retry loop websocket backoff",
            "production failure outcome",
        ]
        assert result.retrieval_plan.max_candidates >= resolved_policy.retrieval_params.fts_limit
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == ["mem_1"]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
        assert {
            "need_detection",
            "planning",
            "candidate_search",
            "applicability_scoring",
            "contract_lookup",
            "state_lookup",
            "workspace_rollup_lookup",
            "context_composition",
        }.issubset(result.stage_timings)
        assert all(value >= 0.0 for value in result.stage_timings.values())
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_excludes_pending_and_declined_candidates_before_scoring() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(score_map={"mem_active": 0.9, "mem_pending": 0.99, "mem_declined": 0.99})
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.ACTIVE,
        )
        await _seed_memory(
            memories,
            memory_id="mem_pending",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_declined",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.DECLINED,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_active"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == ["mem_active"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_need_detection_returns_empty_needs_without_llm_call() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(score_map={"mem_1": 0.88})
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(memories, memory_id="mem_1", canonical_text="retry loop websocket backoff", scope=MemoryScope.CONVERSATION)

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(skip_need_detection=True),
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.detected_needs == []
        assert result.retrieval_plan.query_type == "default"
        assert [sub_query.text for sub_query in result.retrieval_plan.sub_query_plans] == [message_text]
        assert not any(request.metadata.get("purpose") == "need_detection" for request in provider.requests)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_normalizes_callback_hint_anchor_before_planning() -> None:
    message_text = "What was that citrus marinade you suggested?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": [message_text],
            "callback_bias": True,
            "sparse_query_hints": [
                {
                    "sub_query_text": message_text,
                    "fts_phrase": "citrus marinade",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.87},
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="The assistant recommended using a citrus marinade for grilled fish.",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.retrieval_plan.callback_bias is True
        assert result.retrieval_plan.sub_query_plans[0].quoted_phrases == ["citrus marinade"]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_applicability_scoring_uses_raw_scores() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(memories, memory_id="mem_1", canonical_text="retry loop websocket backoff", scope=MemoryScope.CONVERSATION)

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(skip_applicability_scoring=True),
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.scored_candidates[0].final_score == pytest.approx(result.scored_candidates[0].retrieval_score)
        assert not any(request.metadata.get("purpose") == "applicability_scoring" for request in provider.requests)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_contract_memory_clears_contract_block() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(score_map={"mem_1": 0.9})
    connection, memories, contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        source_memory = await _seed_memory(
            memories,
            memory_id="mem_contract_source",
            canonical_text="Prefer concise debugging answers",
            scope=MemoryScope.ASSISTANT_MODE,
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            scope=MemoryScope.CONVERSATION,
            dimension_name="brevity",
            value_json={"label": "short", "confidence": 0.9},
            confidence=0.9,
            source_memory_id=str(source_memory["id"]),
        )
        await _seed_memory(memories, memory_id="mem_1", canonical_text="retry loop websocket backoff", scope=MemoryScope.CONVERSATION)

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(skip_contract_memory=True),
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.current_contract == {}
        assert result.composed_context.contract_block == ""
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_force_all_scopes_overrides_scope_filter() -> None:
    provider = PipelineProvider()
    connection, _memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        result = await pipeline.execute(
            message_text="What should I do next?",
            conversation_context=context.model_copy(update={"assistant_mode_id": "general_qa"}),
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(force_all_scopes=True),
            conversation_messages=[{"role": "user", "text": "What should I do next?"}],
        )

        assert result.retrieval_plan.scope_filter == list(MemoryScope)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_high_stakes_filters_derived_memory_and_workspace_rollup() -> None:
    message_text = "database migration rollback safety"
    provider = PipelineProvider(
        need_response={
            "needs": [
                {
                    "need_type": "high_stakes",
                    "confidence": 0.94,
                    "reasoning": "The user is asking for a risky operational action.",
                }
            ],
            "temporal_range": None,
            "sub_queries": ["database migration rollback safety", "rollback safety"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "database migration rollback safety",
                    "fts_phrase": "database migration rollback safety",
                },
                {
                    "sub_query_text": "rollback safety",
                    "fts_phrase": "rollback safety",
                },
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        score_map={"mem_belief": 0.99, "mem_evidence": 0.51},
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_belief",
            canonical_text="Database migration rollback safety",
            scope=MemoryScope.WORKSPACE,
            object_type=MemoryObjectType.BELIEF,
        )
        await _seed_memory(
            memories,
            memory_id="mem_evidence",
            canonical_text="Database migration rollback safety",
            scope=MemoryScope.CONVERSATION,
            object_type=MemoryObjectType.EVIDENCE,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            workspace_rollup={"summary_text": "Derived workspace rollup that should be suppressed."},
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.retrieval_plan.require_evidence_regrounding is True
        assert result.retrieval_plan.query_type == "broad_list"
        assert len(result.retrieval_plan.sub_query_plans) == 2
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_evidence"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == ["mem_evidence"]
        assert result.composed_context.selected_memory_ids == ["mem_evidence"]
        assert result.composed_context.workspace_block == ""
        assert "mem_belief" not in result.composed_context.memory_block
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Wave 1-A: Small-corpus shortcut
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_returns_all_eligible_memories() -> None:
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_conv",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_mode",
            canonical_text="User prefers concise debugging answers.",
            scope=MemoryScope.ASSISTANT_MODE,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is True
        assert result.degraded_mode is False
        assert result.detected_needs == []
        assert not any(
            request.metadata.get("purpose") == "need_detection" for request in provider.requests
        )
        assert not any(
            request.metadata.get("purpose") == "applicability_scoring"
            for request in provider.requests
        )
        returned_ids = {candidate["id"] for candidate in result.raw_candidates}
        assert returned_ids == {"mem_conv", "mem_mode"}
        assert set(result.composed_context.selected_memory_ids) == {"mem_conv", "mem_mode"}
        assert result.stage_timings["need_detection"] == 0.0
        assert result.stage_timings["candidate_search"] >= 0.0
        assert result.stage_timings["applicability_scoring"] == 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_sets_trace_flag() -> None:
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text="retry loop websocket backoff",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
            trace=trace,
        )

        assert result.small_corpus_mode is True
        assert trace.small_corpus_mode is True
        assert trace.degraded_mode is False
        assert trace.need_detection is not None
        assert trace.need_detection.degraded_mode is False
        assert trace.need_detection.duration_ms == 0.0
        assert trace.candidate_search is not None
        assert trace.candidate_search.total_after_fusion == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_enforces_privacy_ceiling() -> None:
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="public note",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_public",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="secret pin 1234",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=3,
            memory_id="mem_private",
        )
        # Override the resolved policy so the privacy ceiling drops below the
        # private memory. This simulates a public assistant mode that should
        # never surface level-3 material.
        limited_policy = resolved_policy.model_copy(update={"privacy_ceiling": 1})

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=limited_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is True
        returned_ids = {candidate["id"] for candidate in result.raw_candidates}
        assert "mem_public" in returned_ids
        assert "mem_private" not in returned_ids
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_excludes_pending_memories() -> None:
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_pending",
            canonical_text="another note",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_declined",
            canonical_text="rejected note",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.DECLINED,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is True
        returned_ids = {candidate["id"] for candidate in result.raw_candidates}
        assert returned_ids == {"mem_active"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_large_corpus_takes_normal_path() -> None:
    # Seed more than 0.7 * context_budget_tokens worth of content so the
    # shortcut declines to fire (coding_debug budget is 5300 tokens).
    long_text = "retry loop websocket backoff " * 2000
    provider = PipelineProvider(score_map={"mem_large": 0.9})
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_large",
            canonical_text=long_text,
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is False
        assert any(
            request.metadata.get("purpose") == "need_detection" for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_disabled_when_ratio_is_zero() -> None:
    provider = PipelineProvider()
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.0),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is False
        assert any(
            request.metadata.get("purpose") == "need_detection" for request in provider.requests
        )
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Wave 1-B: Need detector as counselor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_degrades_when_need_detector_fails() -> None:
    provider = FailingPipelineProvider("need_detection", score_map={"mem_1": 0.85})
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.degraded_mode is True
        assert result.detected_needs == []
        # Base search still produced candidates even though the enrichment lane
        # collapsed, and the applicability scorer still ran over them.
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
        assert [scored.memory_id for scored in result.scored_candidates] == ["mem_1"]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
        assert any(
            request.metadata.get("purpose") == "applicability_scoring"
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_degraded_mode_trace_is_recorded() -> None:
    provider = FailingPipelineProvider("need_detection", score_map={"mem_1": 0.85})
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text="retry loop websocket backoff",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
            trace=trace,
        )

        assert result.degraded_mode is True
        assert trace.degraded_mode is True
        assert trace.small_corpus_mode is False
        assert trace.need_detection is not None
        assert trace.need_detection.degraded_mode is True
        assert trace.candidate_search is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_base_search_runs_even_when_need_detector_fails() -> None:
    provider = FailingPipelineProvider("need_detection")
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_base",
            canonical_text="retry loop websocket backoff only base",
            scope=MemoryScope.CONVERSATION,
        )
        # No enriched plan will be built because the need detector fails, but
        # the base plan alone must still surface this memory.
        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.degraded_mode is True
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_base"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_merges_base_and_enriched_candidates() -> None:
    # Enriched sub-query targets the entity, base targets only the generic
    # retry-loop lexical tokens. The merge must surface both memories.
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["anniversary dinner restaurant"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "anniversary dinner restaurant",
                    "fts_phrase": "anniversary dinner restaurant",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_base": 0.8, "mem_enriched": 0.9},
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_base",
            canonical_text="retry loop websocket backoff during deploy",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_enriched",
            canonical_text="anniversary dinner restaurant reservation",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.degraded_mode is False
        returned_ids = {candidate["id"] for candidate in result.raw_candidates}
        assert {"mem_base", "mem_enriched"}.issubset(returned_ids)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_base_candidates_are_deduped_against_enriched() -> None:
    # The enriched sub-query reuses the base query text so the same memory is
    # present in both lanes. The merge must collapse it to a single entry.
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["retry loop websocket backoff"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "retry loop websocket backoff",
                    "fts_phrase": "retry loop websocket backoff",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.9},
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
    finally:
        await connection.close()


def test_merge_candidates_prefers_higher_rrf_score() -> None:
    base = [
        {"id": "mem_a", "rrf_score": 0.3, "canonical_text": "A"},
        {"id": "mem_b", "rrf_score": 0.5, "canonical_text": "B"},
    ]
    enriched = [
        {"id": "mem_a", "rrf_score": 0.8, "canonical_text": "A enriched"},
        {"id": "mem_c", "rrf_score": 0.4, "canonical_text": "C"},
    ]
    merged = RetrievalPipeline._merge_candidates(base, enriched)
    assert [candidate["id"] for candidate in merged] == ["mem_a", "mem_b", "mem_c"]
    mem_a = next(candidate for candidate in merged if candidate["id"] == "mem_a")
    assert mem_a["rrf_score"] == 0.8
    assert mem_a["canonical_text"] == "A enriched"


def test_merge_candidates_handles_empty_lists() -> None:
    base = [{"id": "mem_a", "rrf_score": 0.3}]
    assert RetrievalPipeline._merge_candidates(base, []) == base
    assert RetrievalPipeline._merge_candidates([], base) == base
    assert RetrievalPipeline._merge_candidates([], []) == []


@pytest.mark.asyncio
async def test_pipeline_raw_message_channel_contributes_to_exact_recall_trace() -> None:
    """End-to-end Wave 1 batch 2 (1-C + 1-D): raw evidence reaches the trace.

    The need detector surfaces exact recall, the planner propagates it,
    the candidate search attaches the raw_message channel, and the
    trace records how many candidates came from raw evidence.
    """
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["tengo hijos"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "tengo hijos",
                    "fts_phrase": "hijos",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["quantity"],
        },
        score_map={},
    )
    connection, _memories, _contracts, pipeline, _provider, resolved_policy, context = (
        await _build_runtime(mode_id="general_qa", provider=provider)
    )
    try:
        messages = MessageRepository(
            connection,
            FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)),
        )
        await messages.create_message(
            message_id=None,
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text="Tengo tres hijos, se llaman Ana, Luis y Sara.",
        )
        await messages.create_message(
            message_id=None,
            conversation_id="cnv_1",
            role="assistant",
            seq=None,
            text="Gracias por contármelo.",
        )

        trace = RetrievalTrace(
            query_text="¿Cuántos hijos tengo?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text="¿Cuántos hijos tengo?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "¿Cuántos hijos tengo?"},
            ],
            trace=trace,
        )

        assert result.retrieval_plan.exact_recall_mode is True
        assert trace.need_detection is not None
        assert trace.need_detection.exact_recall_needed is True
        assert "quantity" in trace.need_detection.exact_facets
        assert trace.candidate_search is not None
        assert trace.candidate_search.raw_message_candidates_count >= 1
        raw_windows = [
            candidate
            for candidate in result.raw_candidates
            if candidate.get("is_raw_message_window")
        ]
        assert raw_windows, "raw message window should reach the raw_candidates list"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_exact_recall_prefers_verbatim_pins_over_summaries() -> None:
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["bank card PIN"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "bank card PIN",
                    "fts_phrase": "bank card PIN",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["code"],
            "raw_context_access_mode": "verbatim",
        },
        score_map={
            "sum_1": 0.35,
            "vbp_1": 0.95,
        },
    )
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_1",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=0,
            summary_text="Bank card PIN: 4512",
            source_object_ids=[],
            created_at="2026-04-05T11:00:00+00:00",
            index_text="bank card PIN",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_level=0,
        )
        pins = VerbatimPinRepository(connection, FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.CONVERSATION,
            target_kind=VerbatimPinTargetKind.MESSAGE,
            target_id="msg_pin",
            pin_id="vbp_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            canonical_text="Bank card PIN: 4512",
            index_text="bank card PIN",
            privacy_level=0,
            created_by="usr_1",
        )

        trace = RetrievalTrace(
            query_text="What is the bank card PIN?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-05T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text="What is the bank card PIN?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "What is the bank card PIN?"},
            ],
            trace=trace,
        )

        assert result.retrieval_plan.exact_recall_mode is True
        assert trace.candidate_search is not None
        assert trace.candidate_search.verbatim_pin_candidates_count >= 1
        assert result.scored_candidates[0].memory_id == "vbp_1"
        assert result.composed_context.selected_memory_ids[0] == "vbp_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_promotes_existing_filtered_support() -> None:
    connection, _memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa"
    )
    try:
        summary = _candidate_record(
            memory_id="sum_episode",
            canonical_text="Abstract outage summary.",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            rrf_score=1.0,
            payload_json={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": ["mem_support"],
            },
        )
        support = _candidate_record(
            memory_id="mem_support",
            canonical_text="Concrete support memory.",
            rrf_score=0.01,
        )

        updated_shortlist = await pipeline._reground_summary_support_shortlist(
            shortlist=[summary],
            filtered_candidates=[summary, support],
            conversation_context=context,
            resolved_policy=resolved_policy,
            detected_needs=[],
            retrieval_plan=_slot_fill_plan(),
        )

        assert [candidate["id"] for candidate in updated_shortlist] == [
            "sum_episode",
            "mem_support",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_fetches_missing_support_by_id() -> None:
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa"
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_support_a",
            canonical_text="Payment reconciliation worker leaked connections on timeout exceptions.",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_support_b",
            canonical_text="Fix was wrapping the connection logic in try/finally blocks.",
            scope=MemoryScope.CONVERSATION,
        )
        summary = _candidate_record(
            memory_id="sum_episode",
            canonical_text="Abstract outage summary.",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            rrf_score=1.0,
            payload_json={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": ["mem_support_a", "mem_support_b"],
            },
        )

        updated_shortlist = await pipeline._reground_summary_support_shortlist(
            shortlist=[summary],
            filtered_candidates=[summary],
            conversation_context=context,
            resolved_policy=resolved_policy,
            detected_needs=[],
            retrieval_plan=_slot_fill_plan(),
        )

        assert [candidate["id"] for candidate in updated_shortlist] == [
            "sum_episode",
            "mem_support_a",
            "mem_support_b",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_does_not_promote_filtered_source() -> None:
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa"
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Private source memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=3,
            memory_id="mem_private",
        )
        summary = _candidate_record(
            memory_id="sum_episode",
            canonical_text="Abstract outage summary.",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            rrf_score=1.0,
            payload_json={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": ["mem_private"],
            },
        )

        restricted_policy = resolved_policy.model_copy(update={"privacy_ceiling": 2})
        updated_shortlist = await pipeline._reground_summary_support_shortlist(
            shortlist=[summary],
            filtered_candidates=[summary],
            conversation_context=context,
            resolved_policy=restricted_policy,
            detected_needs=[],
            retrieval_plan=_slot_fill_plan(),
        )

        assert [candidate["id"] for candidate in updated_shortlist] == ["sum_episode"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_caps_total_promotions() -> None:
    connection, memories, _contracts, pipeline, _provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa"
    )
    try:
        support_ids = [
            "mem_support_a1",
            "mem_support_a2",
            "mem_support_b1",
            "mem_support_b2",
            "mem_support_c1",
            "mem_support_c2",
        ]
        for support_id in support_ids:
            await _seed_memory(
                memories,
                memory_id=support_id,
                canonical_text=f"Support memory {support_id}",
                scope=MemoryScope.CONVERSATION,
            )

        shortlist = [
            _candidate_record(
                memory_id="sum_a",
                canonical_text="Summary A",
                object_type=MemoryObjectType.SUMMARY_VIEW,
                rrf_score=1.0,
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_support_a1", "mem_support_a2"],
                },
            ),
            _candidate_record(
                memory_id="sum_b",
                canonical_text="Summary B",
                object_type=MemoryObjectType.SUMMARY_VIEW,
                rrf_score=0.99,
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_support_b1", "mem_support_b2"],
                },
            ),
            _candidate_record(
                memory_id="sum_c",
                canonical_text="Summary C",
                object_type=MemoryObjectType.SUMMARY_VIEW,
                rrf_score=0.98,
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_support_c1", "mem_support_c2"],
                },
            ),
        ]

        updated_shortlist = await pipeline._reground_summary_support_shortlist(
            shortlist=shortlist,
            filtered_candidates=shortlist,
            conversation_context=context,
            resolved_policy=resolved_policy,
            detected_needs=[],
            retrieval_plan=_slot_fill_plan(),
        )

        promoted_ids = [
            candidate["id"]
            for candidate in updated_shortlist
            if candidate["id"] in support_ids
        ]
        assert promoted_ids == [
            "mem_support_a1",
            "mem_support_a2",
            "mem_support_b1",
            "mem_support_b2",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_summary_support_regrounding_reaches_composer(monkeypatch: pytest.MonkeyPatch) -> None:
    message_text = "What was the root cause of the connection pool exhaustion?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": [message_text],
            "sparse_query_hints": [
                {
                    "sub_query_text": message_text,
                    "fts_phrase": "connection pool exhaustion root cause",
                    "must_keep_terms": ["connection", "pool", "root", "cause"],
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0, 1],
        },
        score_map={
            "sum_episode": 0.94,
            "mem_support_a": 0.91,
            "mem_support_b": 0.90,
        },
    )
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_support_a",
            canonical_text=(
                "The payment reconciliation worker was leaking connections on timeout exceptions."
            ),
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_support_b",
            canonical_text="The fix was adding proper try/finally blocks around the connection handling.",
            scope=MemoryScope.CONVERSATION,
        )

        rerank_top_k = resolved_policy.retrieval_params.rerank_top_k
        candidates = [
            _candidate_record(
                memory_id="sum_episode",
                canonical_text="Abstract summary of the connection-pool incident.",
                object_type=MemoryObjectType.SUMMARY_VIEW,
                rrf_score=1.0,
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_support_a", "mem_support_b"],
                },
            )
        ]
        candidates.extend(
            _candidate_record(
                memory_id=f"mem_state_{index}",
                canonical_text=f"Unrelated state snapshot {index}",
                object_type=MemoryObjectType.STATE_SNAPSHOT,
                rrf_score=0.90 - (index * 0.01),
            )
            for index in range(rerank_top_k + 2)
        )

        call_count = 0

        async def fake_search(_plan: RetrievalPlan, _user_id: str) -> list[dict[str, object]]:
            nonlocal call_count
            call_count += 1
            return [] if call_count == 1 else candidates

        monkeypatch.setattr(pipeline._candidate_search, "search", fake_search)

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        scored_ids = [candidate.memory_id for candidate in result.scored_candidates]
        assert "mem_support_a" in scored_ids
        assert "mem_support_b" in scored_ids
        assert "sum_episode" in result.composed_context.selected_memory_ids
        assert "mem_support_a" in result.composed_context.selected_memory_ids
        assert "mem_support_b" in _score_request_memory_ids(provider)
    finally:
        await connection.close()
