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
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
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
        need_response: list[dict[str, object]] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self.need_response = list(need_response or [])
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


async def _build_runtime(*, mode_id: str = "coding_debug", provider: PipelineProvider | None = None):
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
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=llm_provider.name, providers=[llm_provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=_settings(),
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


@pytest.mark.asyncio
async def test_pipeline_executes_full_flow() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(
        need_response=[
            {
                "need_type": "follow_up_failure",
                "confidence": 0.82,
                "reasoning": "The user is describing an unresolved retry problem.",
            }
        ],
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
        assert not any(request.metadata.get("purpose") == "need_detection" for request in provider.requests)
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
        need_response=[
            {
                "need_type": "high_stakes",
                "confidence": 0.94,
                "reasoning": "The user is asking for a risky operational action.",
            }
        ],
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
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_evidence"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == ["mem_evidence"]
        assert result.composed_context.selected_memory_ids == ["mem_evidence"]
        assert result.composed_context.workspace_block == ""
        assert "mem_belief" not in result.composed_context.memory_block
    finally:
        await connection.close()
