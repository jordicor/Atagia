"""Metamorphic retrieval invariants for pipeline-level custody boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    ResolvedPolicy,
    sync_assistant_modes,
)
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)
from atagia.models.schemas_replay import AblationConfig, PipelineResult
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
_QUERY = "shared invariant token"


class InvariantProvider(LLMProvider):
    name = "retrieval-invariant-tests"

    def __init__(
        self,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self.need_response = need_response or {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [_QUERY],
            "sparse_query_hints": [
                {
                    "sub_query_text": _QUERY,
                    "fts_phrase": _QUERY,
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
            payload = {
                "scores": [
                    {
                        "memory_id": memory_id,
                        "llm_applicability": self.score_map.get(memory_id, 0.7),
                    }
                    for memory_id in memory_ids
                ],
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(payload),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in retrieval invariant tests")


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
        raw_message_channel=False,
    )


async def _build_runtime(
    *,
    mode_id: str = "coding_debug",
    provider: InvariantProvider | None = None,
    settings: Settings | None = None,
) -> tuple[
    Any,
    MemoryObjectRepository,
    RetrievalPipeline,
    InvariantProvider,
    ResolvedPolicy,
    ExtractionConversationContext,
]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    manifests = ManifestLoader(MANIFESTS_DIR).load_all()
    await sync_assistant_modes(connection, manifests, clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)

    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await workspaces.create_workspace("wrk_2", "usr_2", "Other workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", mode_id, "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", "wrk_2", mode_id, "Other chat")

    llm_provider = provider or InvariantProvider()
    resolved_settings = settings or _settings()
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=llm_provider.name, providers=[llm_provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=resolved_settings,
    )
    resolved_policy = PolicyResolver().resolve(manifests[mode_id], None, None)
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id="wrk_1",
        assistant_mode_id=mode_id,
        recent_messages=[],
    )
    return connection, memories, pipeline, llm_provider, resolved_policy, context


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    user_id: str = "usr_1",
    workspace_id: str | None = "wrk_1",
    conversation_id: str | None = "cnv_1",
    assistant_mode_id: str = "coding_debug",
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    source_kind: MemorySourceKind = MemorySourceKind.EXTRACTED,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoped_workspace_scopes = {
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
        MemoryScope.EPHEMERAL_SESSION,
    }
    scoped_conversation_scopes = {
        MemoryScope.CONVERSATION,
        MemoryScope.EPHEMERAL_SESSION,
    }
    resolved_workspace_id = workspace_id if scope in scoped_workspace_scopes else None
    resolved_conversation_id = conversation_id if scope in scoped_conversation_scopes else None
    return await memories.create_memory_object(
        user_id=user_id,
        workspace_id=resolved_workspace_id,
        conversation_id=resolved_conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        index_text=canonical_text,
        source_kind=source_kind,
        confidence=0.8,
        privacy_level=privacy_level,
        payload=payload,
        status=status,
        memory_id=memory_id,
    )


async def _run_pipeline(
    pipeline: RetrievalPipeline,
    *,
    context: ExtractionConversationContext,
    policy: ResolvedPolicy,
    ablation: AblationConfig | None = None,
) -> PipelineResult:
    return await pipeline.execute(
        message_text=_QUERY,
        conversation_context=context,
        resolved_policy=policy,
        cold_start=False,
        ablation=ablation,
    )


def _candidate_ids(candidates: list[dict[str, Any]]) -> set[str]:
    return {str(candidate["id"]) for candidate in candidates}


def _scored_ids(result: PipelineResult) -> set[str]:
    return {candidate.memory_id for candidate in result.scored_candidates}


def _custody_ids(result: PipelineResult) -> set[str]:
    return {str(record["candidate_id"]) for record in result.candidate_custody}


def _score_request_memory_ids(provider: InvariantProvider) -> set[str]:
    memory_ids: set[str] = set()
    for request in provider.requests:
        if str(request.metadata.get("purpose")) == "applicability_scoring":
            memory_ids.update(_MEMORY_ID_PATTERN.findall(request.messages[1].content))
    return memory_ids


def _slot_fill_plan() -> RetrievalPlan:
    return RetrievalPlan(
        original_query="Which source memory is the summary based on?",
        assistant_mode_id="general_qa",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=[_QUERY],
        sub_query_plans=[
            PlannedSubQuery(
                text=_QUERY,
                sparse_phrase=_QUERY,
                quoted_phrases=[],
                must_keep_terms=[],
                fts_queries=[_QUERY],
            )
        ],
        query_type="slot_fill",
        scope_filter=[MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=30,
        max_context_items=8,
        privacy_ceiling=2,
        retrieval_levels=[0, 1],
    )


@pytest.mark.asyncio
async def test_hostile_cross_user_distractors_never_enter_retrieval_outputs() -> None:
    connection, memories, pipeline, provider, policy, context = await _build_runtime(
        provider=InvariantProvider(score_map={"mem_allowed": 0.9, "mem_usr2": 1.0}),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_allowed",
            canonical_text=f"{_QUERY} belongs to the active user.",
        )
        await _seed_memory(
            memories,
            memory_id="mem_usr2",
            canonical_text=f"{_QUERY} belongs to another user and must never leak.",
            user_id="usr_2",
            workspace_id="wrk_2",
            conversation_id="cnv_2",
        )

        result = await _run_pipeline(pipeline, context=context, policy=policy)

        assert "mem_allowed" in _candidate_ids(result.raw_candidates)
        assert "mem_usr2" not in _candidate_ids(result.raw_candidates)
        assert "mem_usr2" not in _scored_ids(result)
        assert "mem_usr2" not in result.composed_context.selected_memory_ids
        assert "mem_usr2" not in _custody_ids(result)
        assert {candidate["user_id"] for candidate in result.raw_candidates} == {"usr_1"}
        assert {
            candidate.memory_object["user_id"]
            for candidate in result.scored_candidates
            if "user_id" in candidate.memory_object
        } <= {"usr_1"}
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [MemoryScope.WORKSPACE, MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION],
)
@pytest.mark.parametrize("small_corpus_ratio", [0.0, 0.99])
async def test_assistant_mode_boundary_matches_normal_and_small_corpus_paths(
    scope: MemoryScope,
    small_corpus_ratio: float,
) -> None:
    connection, memories, pipeline, provider, policy, context = await _build_runtime(
        provider=InvariantProvider(score_map={"mem_allowed": 0.9, "mem_other_mode": 1.0}),
        settings=_settings(small_corpus_token_threshold_ratio=small_corpus_ratio),
    )
    try:
        restricted_policy = policy.model_copy(update={"allowed_scopes": [scope]})
        await _seed_memory(
            memories,
            memory_id="mem_allowed",
            canonical_text=f"{_QUERY} visible within the current assistant mode.",
            scope=scope,
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_mode",
            canonical_text=f"{_QUERY} shares the same local scope ids but another mode.",
            scope=scope,
            assistant_mode_id="general_qa",
        )

        result = await _run_pipeline(pipeline, context=context, policy=restricted_policy)

        assert result.small_corpus_mode is (small_corpus_ratio > 0.0)
        assert "mem_allowed" in _candidate_ids(result.raw_candidates)
        assert "mem_other_mode" not in _candidate_ids(result.raw_candidates)
        assert "mem_other_mode" not in _scored_ids(result)
        assert "mem_other_mode" not in result.composed_context.selected_memory_ids
        assert "mem_other_mode" not in _custody_ids(result)
        assert "mem_other_mode" not in _score_request_memory_ids(provider)
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ablation",
    [
        None,
        AblationConfig(force_all_scopes=True),
        AblationConfig(skip_need_detection=True),
        AblationConfig(skip_applicability_scoring=True),
    ],
)
async def test_status_and_privacy_gates_survive_retrieval_ablations(
    ablation: AblationConfig | None,
) -> None:
    provider = InvariantProvider(
        score_map={
            "mem_allowed": 0.9,
            "mem_pending": 1.0,
            "mem_declined": 1.0,
            "mem_review": 1.0,
            "mem_private": 1.0,
        }
    )
    connection, memories, pipeline, provider, policy, context = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_allowed",
            canonical_text=f"{_QUERY} active public candidate.",
        )
        await _seed_memory(
            memories,
            memory_id="mem_pending",
            canonical_text=f"{_QUERY} pending candidate.",
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_declined",
            canonical_text=f"{_QUERY} declined candidate.",
            status=MemoryStatus.DECLINED,
        )
        await _seed_memory(
            memories,
            memory_id="mem_review",
            canonical_text=f"{_QUERY} review candidate.",
            status=MemoryStatus.REVIEW_REQUIRED,
        )
        await _seed_memory(
            memories,
            memory_id="mem_private",
            canonical_text=f"{_QUERY} private candidate.",
            privacy_level=3,
        )

        result = await _run_pipeline(pipeline, context=context, policy=policy, ablation=ablation)

        gated_ids = {"mem_pending", "mem_declined", "mem_review", "mem_private"}
        assert "mem_allowed" in _candidate_ids(result.raw_candidates)
        assert _candidate_ids(result.raw_candidates).isdisjoint(gated_ids)
        assert _scored_ids(result).isdisjoint(gated_ids)
        assert set(result.composed_context.selected_memory_ids).isdisjoint(gated_ids)
        assert _custody_ids(result).isdisjoint(gated_ids)
        assert _score_request_memory_ids(provider).isdisjoint(gated_ids)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_promotion_filters_ineligible_fetched_support_rows() -> None:
    connection, memories, pipeline, _provider, policy, context = await _build_runtime(mode_id="general_qa")
    try:
        await _seed_memory(
            memories,
            memory_id="mem_support_good",
            canonical_text="Eligible source evidence for the episode summary.",
            scope=MemoryScope.CONVERSATION,
            assistant_mode_id="general_qa",
        )
        await _seed_memory(
            memories,
            memory_id="mem_support_pending",
            canonical_text="Pending source evidence should not ground a summary.",
            scope=MemoryScope.CONVERSATION,
            assistant_mode_id="general_qa",
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_support_private",
            canonical_text="Private source evidence should not ground a summary.",
            scope=MemoryScope.CONVERSATION,
            assistant_mode_id="general_qa",
            privacy_level=3,
        )
        summary = await _seed_memory(
            memories,
            memory_id="sum_episode",
            canonical_text="Episode summary that cites mixed-quality source rows.",
            scope=MemoryScope.CONVERSATION,
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            source_kind=MemorySourceKind.SUMMARIZED,
            payload={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": [
                    "mem_support_good",
                    "mem_support_pending",
                    "mem_support_private",
                ],
            },
        )
        summary["rrf_score"] = 1.0
        summary["channel_ranks"] = {"fts": 1, "embedding": None, "consequence": None, "raw_message": None}
        summary["matched_sub_queries"] = [_QUERY]
        summary["retrieval_sources"] = ["fts"]

        promoted = await pipeline._reground_summary_support_shortlist(
            shortlist=[summary],
            filtered_candidates=[summary],
            conversation_context=context,
            resolved_policy=policy,
            detected_needs=[],
            retrieval_plan=_slot_fill_plan(),
        )

        assert [candidate["id"] for candidate in promoted] == ["sum_episode", "mem_support_good"]
    finally:
        await connection.close()
