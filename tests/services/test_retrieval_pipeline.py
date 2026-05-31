"""Tests for the reusable retrieval pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    sync_assistant_modes,
)
from atagia.memory.context_composer import ContextComposer
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExplicitLanguagePreference,
    ExtractionConversationContext,
    IntimacyBoundary,
    LanguageProfileSourceRef,
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    NeedTrigger,
    PlannedSubQuery,
    RetrievalPlan,
    RetrievalTrace,
    SpaceBoundaryMode,
    SummaryViewKind,
    UserCommunicationProfile,
    ObservedUserLanguage,
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
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _score_keys_by_memory_id(prompt: str) -> dict[str, str]:
    return {
        memory_id: score_key
        for memory_id, score_key in _CANDIDATE_SCORE_KEY_PATTERN.findall(prompt)
    }


def _scores_for_prompt(
    prompt: str,
    score_map: dict[str, float],
    *,
    omitted_memory_ids: set[str] | None = None,
) -> list[dict[str, float | str]]:
    omitted = omitted_memory_ids or set()
    return [
        {"score_key": score_key, "llm_applicability": score_map.get(memory_id, 0.5)}
        for memory_id, score_key in _score_keys_by_memory_id(prompt).items()
        if memory_id not in omitted
    ]


def test_policy_filter_audit_separates_actual_and_would_filter_reasons() -> None:
    regrounding_reasons = {"mem_1": "regrounding_filtered"}
    policy_reasons = {"mem_2": "policy_filtered_sensitivity"}

    assert RetrievalPipeline._actual_filter_reasons(
        regrounding_reasons,
        policy_reasons,
        privacy_enforcement="enforce",
    ) == {
        "mem_1": "regrounding_filtered",
        "mem_2": "policy_filtered_sensitivity",
    }
    assert RetrievalPipeline._actual_filter_reasons(
        regrounding_reasons,
        policy_reasons,
        privacy_enforcement="off",
    ) == {"mem_1": "regrounding_filtered"}

    audit = RetrievalPipeline._build_policy_filter_audit("off", policy_reasons)

    assert audit == {
        "privacy_enforcement": "off",
        "enforced": False,
        "high_risk_secret_literal_redaction_enforced": False,
        "high_risk_secret_literal_redaction_disabled": True,
        "would_filter_count": 1,
        "would_filter_reason_counts": {"policy_filtered_sensitivity": 1},
        "would_filter_by_candidate_id": {"mem_2": "policy_filtered_sensitivity"},
    }


def test_policy_filter_audit_renders_all_phase8_policy_modes() -> None:
    policy_reasons = {
        "mem_private": "policy_filtered_privacy",
        "mem_secret": "policy_filtered_sensitivity",
    }

    off = RetrievalPipeline._build_policy_filter_audit("off", policy_reasons)
    audit_only = RetrievalPipeline._build_policy_filter_audit(
        "audit_only",
        policy_reasons,
    )
    enforce = RetrievalPipeline._build_policy_filter_audit("enforce", policy_reasons)

    assert off["privacy_enforcement"] == "off"
    assert off["enforced"] is False
    assert off["high_risk_secret_literal_redaction_enforced"] is False
    assert off["high_risk_secret_literal_redaction_disabled"] is True
    assert off["would_filter_count"] == 2

    assert audit_only["privacy_enforcement"] == "audit_only"
    assert audit_only["enforced"] is False
    assert audit_only["high_risk_secret_literal_redaction_enforced"] is True
    assert audit_only["high_risk_secret_literal_redaction_disabled"] is False
    assert audit_only["would_filter_reason_counts"] == {
        "policy_filtered_privacy": 1,
        "policy_filtered_sensitivity": 1,
    }

    assert enforce["privacy_enforcement"] == "enforce"
    assert enforce["enforced"] is True
    assert enforce["high_risk_secret_literal_redaction_enforced"] is True
    assert enforce["high_risk_secret_literal_redaction_disabled"] is False
    assert enforce["would_filter_by_candidate_id"] == policy_reasons


class PipelineProvider(LLMProvider):
    name = "retrieval-pipeline-tests"

    def __init__(
        self,
        *,
        need_response: dict[str, object] | None = None,
        anchor_review_response: dict[str, object] | None = None,
        coverage_response: dict[str, object] | None = None,
        unknown_exact_review_response: dict[str, object] | None = None,
        multi_facet_exact_review_response: dict[str, object] | None = None,
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
        self.anchor_review_response = anchor_review_response or {"anchors": []}
        self.coverage_response = coverage_response or {
            "should_expand": False,
            "missing_facets": [],
            "sub_queries": [],
        }
        self.unknown_exact_review_response = unknown_exact_review_response or {
            "is_exact_value_lookup": False,
            "exact_facets": [],
            "must_keep_terms": [],
            "quoted_phrases": [],
        }
        self.multi_facet_exact_review_response = multi_facet_exact_review_response or {
            "has_multiple_obligations": False,
            "sub_queries": [],
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
        if purpose == "need_detection_anchor_review":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(self.anchor_review_response),
            )
        if purpose == "applicability_scoring":
            payload = {
                "scores": _scores_for_prompt(
                    request.messages[1].content, self.score_map
                ),
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(payload),
            )
        if purpose == "coverage_expansion":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(self.coverage_response),
            )
        if purpose == "need_detection_unknown_only_contract_review":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(self.unknown_exact_review_response),
            )
        if purpose == "need_detection_multi_facet_exact_review":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(self.multi_facet_exact_review_response),
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


class OmitFirstApplicabilityScoreProvider(PipelineProvider):
    """Provider that omits one scored memory on the first scoring call."""

    def __init__(
        self,
        omitted_memory_id: str,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(need_response=need_response, score_map=score_map)
        self._omitted_memory_id = omitted_memory_id
        self._scoring_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if str(request.metadata.get("purpose")) != "applicability_scoring":
            return await super().complete(request)

        self.requests.append(request)
        self._scoring_calls += 1
        payload = {
            "scores": _scores_for_prompt(
                request.messages[1].content,
                self.score_map,
                omitted_memory_ids={self._omitted_memory_id}
                if self._scoring_calls == 1
                else set(),
            ),
        }
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(payload),
        )


class MalformedFirstApplicabilityScoreProvider(PipelineProvider):
    """Provider that returns one score without memory_id on the first scoring call."""

    def __init__(
        self,
        malformed_memory_id: str,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(need_response=need_response, score_map=score_map)
        self._malformed_memory_id = malformed_memory_id
        self._scoring_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if str(request.metadata.get("purpose")) != "applicability_scoring":
            return await super().complete(request)

        self.requests.append(request)
        self._scoring_calls += 1
        scores: list[dict[str, float | str]] = []
        for memory_id, score_key in _score_keys_by_memory_id(
            request.messages[1].content
        ).items():
            score = self.score_map.get(memory_id, 0.5)
            if self._scoring_calls == 1 and memory_id == self._malformed_memory_id:
                scores.append(
                    {"score_key": "candidate_999", "llm_applicability": score}
                )
            else:
                scores.append({"score_key": score_key, "llm_applicability": score})
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps({"scores": scores}),
        )


class InvalidFirstApplicabilityScoreProvider(PipelineProvider):
    """Provider that returns schema-invalid scores on the first scoring call."""

    def __init__(
        self,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(need_response=need_response, score_map=score_map)
        self._scoring_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if str(request.metadata.get("purpose")) != "applicability_scoring":
            return await super().complete(request)

        self.requests.append(request)
        self._scoring_calls += 1
        if self._scoring_calls == 1:
            score_key = next(
                iter(_score_keys_by_memory_id(request.messages[1].content).values())
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {"scores": [{"score_key": score_key, "llm_applicability": 2.0}]}
                ),
            )
        payload = {
            "scores": _scores_for_prompt(request.messages[1].content, self.score_map),
        }
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(payload),
        )


def _settings(*, small_corpus_token_threshold_ratio: float = 0.0) -> Settings:
    return Settings(
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
    await sync_assistant_modes(
        connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
    )
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    contracts = ContractDimensionRepository(connection, clock)
    spaces = SpaceRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    for space_id, boundary_mode in (
        ("space_focus", SpaceBoundaryMode.FOCUS),
        ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
        ("space_severed", SpaceBoundaryMode.SEVERANCE),
        ("space_tagged", SpaceBoundaryMode.TAGGED),
    ):
        await spaces.resolve_space(
            owner_user_id="usr_1",
            space_id=space_id,
            boundary_mode=boundary_mode,
            display_name=space_id,
            source_kind="explicit",
            source_id=space_id,
        )
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
    return (
        connection,
        memories,
        contracts,
        pipeline,
        llm_provider,
        resolved_policy,
        context,
    )


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    scope: MemoryScope,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    assistant_mode_id: str = "coding_debug",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    space_id: str | None = None,
    space_boundary_mode: SpaceBoundaryMode | None = None,
    language_codes: list[str] | None = None,
) -> dict[str, object]:
    scope_canonical = {
        MemoryScope.CONVERSATION: MemoryScope.CHAT.value,
        MemoryScope.EPHEMERAL_SESSION: MemoryScope.CHAT.value,
        MemoryScope.WORKSPACE: MemoryScope.CHARACTER.value,
        MemoryScope.GLOBAL_USER: MemoryScope.USER.value,
    }.get(scope)
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1" if scope is MemoryScope.CONVERSATION else None,
        assistant_mode_id=assistant_mode_id,
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED
        if object_type is not MemoryObjectType.INTERACTION_CONTRACT
        else MemorySourceKind.INFERRED,
        confidence=0.8,
        privacy_level=0,
        status=status,
        language_codes=language_codes,
        memory_id=memory_id,
        platform_id="default",
        character_id="wrk_1" if scope is MemoryScope.WORKSPACE else None,
        scope_canonical=scope_canonical,
        space_id=space_id,
        space_boundary_mode=space_boundary_mode.value
        if space_boundary_mode is not None
        else None,
    )


@pytest.mark.asyncio
async def test_empty_clean_first_turn_skips_llm_retrieval_work() -> None:
    connection, _memories, _contracts, pipeline, provider, resolved_policy, context = (
        await _build_runtime()
    )
    try:
        trace = RetrievalTrace(
            query_text="Hello, can you help me plan my day?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-05T12:00:00+00:00",
        )

        result = await pipeline.execute(
            message_text="Hello, can you help me plan my day?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=True,
            conversation_messages=[
                {
                    "id": "pending:cnv_1",
                    "role": "user",
                    "seq": 1,
                    "text": "Hello, can you help me plan my day?",
                }
            ],
            trace=trace,
        )

        assert provider.requests == []
        assert result.detected_needs == []
        assert result.raw_candidates == []
        assert result.scored_candidates == []
        assert "empty_clean_fast_path" in result.stage_timings
        assert result.trace is not None
        assert result.trace.need_detection is not None
        assert result.trace.need_detection.duration_ms == 0.0
    finally:
        await connection.close()


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
        scope_filter=[
            MemoryScope.CONVERSATION,
            MemoryScope.WORKSPACE,
            MemoryScope.GLOBAL_USER,
        ],
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
    scope_canonical = {
        MemoryScope.CONVERSATION: MemoryScope.CHAT.value,
        MemoryScope.EPHEMERAL_SESSION: MemoryScope.CHAT.value,
        MemoryScope.WORKSPACE: MemoryScope.CHARACTER.value,
        MemoryScope.GLOBAL_USER: MemoryScope.USER.value,
    }.get(scope, scope.value)
    return {
        "id": memory_id,
        "object_type": object_type.value,
        "status": status.value,
        "scope": scope.value,
        "scope_canonical": scope_canonical,
        "privacy_level": privacy_level,
        "sensitivity": "public",
        "user_persona_id": None,
        "platform_id": "default",
        "character_id": "wrk_1" if scope is MemoryScope.WORKSPACE else None,
        "platform_locked": 0,
        "platform_id_lock": None,
        "assistant_mode_id": "general_qa",
        "conversation_id": "cnv_1" if scope is MemoryScope.CONVERSATION else None,
        "workspace_id": "wrk_1"
        if scope in {MemoryScope.CONVERSATION, MemoryScope.WORKSPACE}
        else None,
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
            "verbatim_evidence_search": None,
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        trace = RetrievalTrace(
            query_text=message_text,
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            trace=trace,
        )

        assert [need.need_type.value for need in result.detected_needs] == [
            "follow_up_failure"
        ]
        assert result.retrieval_plan.query_type == "broad_list"
        assert [
            sub_query.text for sub_query in result.retrieval_plan.sub_query_plans
        ] == [
            "retry loop websocket backoff",
            "production failure outcome",
        ]
        assert (
            result.retrieval_plan.max_candidates
            >= resolved_policy.retrieval_params.fts_limit
        )
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == [
            "mem_1"
        ]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
        assert result.retrieval_sufficiency is not None
        assert result.retrieval_sufficiency.state == "retrieval_sufficient"
        assert result.retrieval_sufficiency.scored_candidate_count == 1
        assert result.candidate_custody[0]["candidate_id"] == "mem_1"
        assert result.candidate_custody[0]["shortlist_status"] == "shortlisted"
        assert result.candidate_custody[0]["score_status"] == "scored"
        assert result.candidate_custody[0]["composer_decision"] == "selected"
        assert result.candidate_custody[0]["matched_subquery_indexes"] == [0]
        assert trace.facet_support is not None
        assert [
            obligation.status for obligation in trace.facet_support.obligations
        ] == [
            "covered",
            "missing",
        ]
        assert trace.direct_vs_indirect_provenance is not None
        assert trace.direct_vs_indirect_provenance.direct_recovery_count == 1
        assert trace.direct_vs_indirect_provenance.evidence[0].proof_source == (
            "base_canonical"
        )
        assert trace.token_budget is not None
        assert (
            trace.token_budget.context_tokens
            == result.composed_context.total_tokens_estimate
        )
        assert trace.cross_conversation_raw_policy is not None
        assert trace.cross_conversation_raw_policy.enabled is False
        assert trace.custody.raw_candidate_count == 1
        assert trace.custody.post_user_id_candidate_count == 1
        assert trace.custody.post_scope_coordinate_lifecycle_candidate_count == 1
        assert trace.custody.scored_candidate_count == 1
        assert trace.custody.selected_candidate_count == 1
        assert trace.custody.candidate_count_by_channel == {"fts": 1}
        assert trace.custody.source_backed_candidate_count == 1
        assert trace.custody.summary_only_candidate_count == 0
        assert trace.custody.selected_evidence_ids == ["mem_1"]
        assert trace.custody.selected_source_evidence_count == 1
        assert trace.custody.selected_summary_count == 0
        assert trace.custody.rendered_evidence_ids == ["mem_1"]
        assert trace.custody.candidate_found_but_not_selected == []
        assert trace.custody.funnel_coverage_state == "complete"
        assert trace.runtime_diagnostics.stage_timings_ms["candidate_search"] >= 0.0
        assert trace.runtime_diagnostics.hydration_timings_ms["contract_lookup"] >= 0.0
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
async def test_pipeline_promotes_summary_source_window_for_privacy_off_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atagia.core.repositories import MessageRepository

    message_text = "What does Jon plan to do at the grand opening?"
    source_window_id = "ssw_sum_grand_opening_277_283"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": [message_text],
            "sparse_query_hints": [
                {
                    "sub_query_text": message_text,
                    "fts_phrase": "grand opening plan",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0, 1],
        },
        score_map={
            "sum_grand_opening": 0.72,
            source_window_id: 0.96,
        },
    )
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
        messages = MessageRepository(connection, clock)
        for seq, role, text in [
            (275, "assistant", "Jon: I need to sort out permits first."),
            (276, "user", "Gina: The lease papers finally cleared."),
            (277, "assistant", "Jon: Still working on opening a dance studio."),
            (278, "user", "Gina: When are you opening the studio?"),
            (279, "assistant", "Jon: The official opening night is tomorrow."),
            (280, "user", "Gina: Congrats, Jon! The studio looks amazing."),
            (281, "assistant", "Jon: Thanks, Gina! I'm excited!"),
            (282, "user", "Gina: Take some time to savor it."),
            (283, "assistant", "Jon: I want to savor all the good vibes."),
        ]:
            await messages.create_message(
                message_id=f"msg_{seq}",
                conversation_id="cnv_1",
                role=role,
                seq=seq,
                text=text,
                occurred_at="2023-06-19T10:04:00+00:00",
            )

        candidates = [
            _candidate_record(
                memory_id="sum_grand_opening",
                canonical_text="Jon and Gina discussed the dance studio grand opening.",
                object_type=MemoryObjectType.SUMMARY_VIEW,
                rrf_score=1.0,
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": [
                        "msg_275",
                        "msg_276",
                        "msg_277",
                        "msg_278",
                        "msg_279",
                        "msg_280",
                        "msg_281",
                        "msg_282",
                        "msg_283",
                    ],
                },
            )
        ]

        async def fake_search(
            _plan: RetrievalPlan,
            _user_id: str,
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            return candidates

        monkeypatch.setattr(pipeline._candidate_search, "search", fake_search)

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            ablation=AblationConfig(privacy_enforcement="off"),
        )

        scored_ids = [candidate.memory_id for candidate in result.scored_candidates]
        assert source_window_id in scored_ids
        assert source_window_id in _score_request_memory_ids(provider)
        source_window = next(
            candidate
            for candidate in result.scored_candidates
            if candidate.memory_id == source_window_id
        )
        assert source_window.memory_object["source_kind"] == "verbatim"
        assert (
            source_window.memory_object["payload_json"]["source_kind_variant"]
            == "summary_source_window"
        )
        assert "savor all the good vibes" in source_window.memory_object["canonical_text"]
        assert source_window_id in result.composed_context.selected_memory_ids
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_expands_broad_list_candidates_from_retrieved_source_message() -> (
    None
):
    message_text = "Which details were part of the campaign?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["launch promotion"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "launch promotion",
                    "fts_phrase": "launch promotion",
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        score_map={
            "mem_matched": 0.72,
            "mem_source_sibling": 0.91,
        },
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        source_payload = {"source_message_ids": ["msg_campaign"]}
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The launch promotion included a store-opening announcement.",
            payload=source_payload,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_matched",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Red hoodie video frame.",
            payload=source_payload,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_source_sibling",
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        raw_ids = [candidate["id"] for candidate in result.raw_candidates]
        assert raw_ids == ["mem_matched", "mem_source_sibling"]
        sibling = next(
            candidate
            for candidate in result.raw_candidates
            if candidate["id"] == "mem_source_sibling"
        )
        assert sibling["retrieval_sources"] == ["source_neighbor"]
        assert sibling["coverage_source_message_id"] == "msg_campaign"
        assert "mem_source_sibling" in _score_request_memory_ids(provider)
        assert result.stage_timings["coverage_candidate_expansion"] >= 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_llm_coverage_expansion_is_default_off() -> None:
    message_text = "Which rollout details are in memory?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["rollout store opening"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "rollout store opening",
                    "fts_phrase": "rollout store opening",
                    "must_keep_terms": ["rollout", "store"],
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        coverage_response={
            "should_expand": True,
            "missing_facets": ["visual detail"],
            "sub_queries": [
                {
                    "sub_query_text": "red hoodie video frame",
                    "fts_phrase": "red hoodie video frame",
                    "must_keep_terms": ["red", "hoodie", "frame"],
                }
            ],
        },
        score_map={"mem_rollout": 0.82, "mem_visual": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_rollout",
            canonical_text="The rollout notes mention the store-opening announcement.",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_visual",
            canonical_text="The red hoodie video frame showed the demo lead.",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert [candidate["id"] for candidate in result.raw_candidates] == [
            "mem_rollout"
        ]
        assert "coverage_expansion" not in {
            str(request.metadata.get("purpose")) for request in provider.requests
        }
        assert result.stage_timings["llm_coverage_expansion"] >= 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_adds_llm_coverage_expansion_candidates_when_enabled() -> None:
    message_text = "Which rollout details are in memory?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["rollout store opening"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "rollout store opening",
                    "fts_phrase": "rollout store opening",
                    "must_keep_terms": ["rollout", "store"],
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        coverage_response={
            "should_expand": True,
            "missing_facets": ["visual detail"],
            "sub_queries": [
                {
                    "sub_query_text": "red hoodie video frame",
                    "fts_phrase": "red hoodie video frame",
                    "must_keep_terms": ["red", "hoodie", "frame"],
                }
            ],
        },
        score_map={"mem_rollout": 0.82, "mem_visual": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_rollout",
            canonical_text="The rollout notes mention the store-opening announcement.",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_visual",
            canonical_text="The red hoodie video frame showed the demo lead.",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            ablation=AblationConfig(enable_llm_coverage_expansion=True),
        )

        raw_ids = [candidate["id"] for candidate in result.raw_candidates]
        assert raw_ids == ["mem_rollout", "mem_visual"]
        visual_candidate = next(
            candidate
            for candidate in result.raw_candidates
            if candidate["id"] == "mem_visual"
        )
        assert "llm_coverage_expansion" in visual_candidate["retrieval_sources"]
        assert visual_candidate["coverage_expansion_sub_queries"] == [
            "red hoodie video frame"
        ]
        assert "mem_visual" in _score_request_memory_ids(provider)
        assert "coverage_expansion" in {
            str(request.metadata.get("purpose")) for request in provider.requests
        }
        assert result.stage_timings["llm_coverage_expansion"] >= 0.0
        assert (
            result.stage_timings["candidate_search"]
            >= result.stage_timings["llm_coverage_expansion"]
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_filters_intimacy_boundary_during_source_message_expansion() -> (
    None
):
    message_text = "Which details were part of the campaign?"
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["launch promotion"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "launch promotion",
                    "fts_phrase": "launch promotion",
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        score_map={"mem_matched": 0.72, "mem_private_sibling": 0.99},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        source_payload = {"source_message_ids": ["msg_campaign"]}
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The launch promotion included a store-opening announcement.",
            payload=source_payload,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_matched",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Private source-neighbor continuity.",
            payload=source_payload,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
            memory_id="mem_private_sibling",
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert [candidate["id"] for candidate in result.raw_candidates] == [
            "mem_matched"
        ]
        assert "mem_private_sibling" not in _score_request_memory_ids(provider)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_privacy_enforcement_off_reaches_composer_for_restricted_candidates() -> (
    None
):
    provider = PipelineProvider(score_map={"mem_restricted_budget": 0.99})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="retry loop websocket backoff apartment budget was $2,800.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=3,
            memory_id="mem_restricted_budget",
            sensitivity=MemorySensitivity.SECRET,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
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
            ablation=AblationConfig(privacy_enforcement="off"),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
            trace=trace,
        )

        assert "mem_restricted_budget" in {
            candidate["id"] for candidate in result.raw_candidates
        }
        assert "mem_restricted_budget" in {
            candidate.memory_id for candidate in result.scored_candidates
        }
        assert "mem_restricted_budget" in result.composed_context.selected_memory_ids
        assert "$2,800" in result.composed_context.memory_block
        assert (
            trace.policy_filter_audit["would_filter_by_candidate_id"][
                "mem_restricted_budget"
            ]
            == "policy_filtered_sensitivity"
        )
        assert (
            trace.policy_filter_audit["high_risk_secret_literal_redaction_enforced"]
            is False
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_privacy_enforcement_off_includes_review_and_pending_statuses() -> None:
    provider = PipelineProvider(
        score_map={
            "mem_review": 0.99,
            "mem_pending": 0.98,
            "mem_declined": 1.0,
        }
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_review",
            canonical_text="retry loop websocket backoff review-required diagnostic fact",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.REVIEW_REQUIRED,
        )
        await _seed_memory(
            memories,
            memory_id="mem_pending",
            canonical_text="retry loop websocket backoff pending-confirmation diagnostic fact",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_declined",
            canonical_text="retry loop websocket backoff declined fact",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.DECLINED,
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
            ablation=AblationConfig(privacy_enforcement="off"),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
            trace=trace,
        )

        assert result.retrieval_plan.status_filter == [
            MemoryStatus.ACTIVE,
            MemoryStatus.REVIEW_REQUIRED,
            MemoryStatus.PENDING_USER_CONFIRMATION,
        ]
        assert {candidate["id"] for candidate in result.raw_candidates} == {
            "mem_review",
            "mem_pending",
        }
        assert {candidate.memory_id for candidate in result.scored_candidates} == {
            "mem_review",
            "mem_pending",
        }
        assert set(result.composed_context.selected_memory_ids) == {
            "mem_review",
            "mem_pending",
        }
        assert trace.scoring is not None
        assert trace.scoring.rejection_reasons == {}
        assert trace.policy_filter_audit["would_filter_reason_counts"] == {
            "policy_filtered_status": 2,
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_privacy_enforcement_off_can_search_cross_conversation_chat_memory() -> (
    None
):
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["therapy sessions"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "therapy sessions",
                    "fts_phrase": "therapy sessions",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
        },
        score_map={"mem_cross_chat_secret": 0.99},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
    )
    conversations = ConversationRepository(
        connection,
        FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)),
    )
    try:
        await conversations.create_conversation(
            "cnv_2",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Earlier personal chat",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Therapy sessions with Dr. Reeves happen Mondays at 5 PM.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=3,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_cross_chat_secret",
            sensitivity=MemorySensitivity.SECRET,
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
        )

        enforced = await pipeline.execute(
            message_text="therapy sessions",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "therapy sessions"},
            ],
        )
        diagnostic = await pipeline.execute(
            message_text="therapy sessions",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(privacy_enforcement="off"),
            conversation_messages=[
                {"role": "user", "text": "therapy sessions"},
            ],
        )

        assert "mem_cross_chat_secret" not in {
            candidate["id"] for candidate in enforced.raw_candidates
        }
        assert "mem_cross_chat_secret" in {
            candidate["id"] for candidate in diagnostic.raw_candidates
        }
        assert (
            "mem_cross_chat_secret" in diagnostic.composed_context.selected_memory_ids
        )
        assert "Dr. Reeves" in diagnostic.composed_context.memory_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_privacy_enforcement_off_disables_high_risk_redaction_in_context() -> (
    None
):
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["deployment pin"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "deployment pin",
                    "fts_phrase": "deployment pin",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_deployment_pin": 0.99},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The deployment PIN is 1234.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=3,
            status=MemoryStatus.ACTIVE,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
            preserve_verbatim=True,
            memory_id="mem_deployment_pin",
            sensitivity=MemorySensitivity.SECRET,
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
        )

        result = await pipeline.execute(
            message_text="deployment pin",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(privacy_enforcement="off"),
            conversation_messages=[
                {"role": "user", "text": "deployment pin"},
            ],
        )

        assert "mem_deployment_pin" in result.composed_context.selected_memory_ids
        assert "The deployment PIN is 1234." in result.composed_context.memory_block
        assert "raw value withheld" not in result.composed_context.memory_block
        assert (
            "privacy_restrictions_inactive: high_risk_secret_literal_unredacted"
            in result.composed_context.memory_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_forwards_budgeted_composer_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_strategies: list[str | None] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_strategies.append(kwargs.get("composer_strategy"))
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(score_map={"mem_1": 0.91})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(composer_strategy="budgeted_marginal"),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert observed_strategies == ["budgeted_marginal"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_forwards_evidence_obligation_composer_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_flags: list[bool | None] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_flags.append(kwargs.get("enable_evidence_obligation_coverage"))
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(score_map={"mem_1": 0.91})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(enable_evidence_obligation_coverage=True),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert observed_flags == [True]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_forwards_answer_coverage_shape_to_composer_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_shape_kwargs: list[tuple[str | None, str | None, str | None]] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_shape_kwargs.append(
            (
                kwargs.get("answer_shape"),
                kwargs.get("coverage_mode"),
                kwargs.get("source_precision"),
            )
        )
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Which cities did Caroline mention?"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Which cities did Caroline mention?",
                    "fts_phrase": "cities Caroline mention",
                    "must_keep_terms": ["Caroline", "cities"],
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.91},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="Caroline mentioned the cities Paris and Rome.",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text="Which cities did Caroline mention?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-05T12:00:00+00:00",
        )

        result = await pipeline.execute(
            message_text="Which cities did Caroline mention?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(applicability_gate_mode="shadow"),
            conversation_messages=[
                {"role": "user", "text": "Which cities did Caroline mention?"}
            ],
            trace=trace,
        )

        assert observed_shape_kwargs == [
            ("list", "exhaustive_known_set", "required")
        ]
        assert result.retrieval_plan.answer_shape == "list"
        assert result.retrieval_plan.coverage_mode == "exhaustive_known_set"
        assert result.retrieval_plan.source_precision == "required"
        assert result.composed_context.answer_shape == "list"
        assert result.composed_context.coverage_mode == "exhaustive_known_set"
        assert result.composed_context.source_precision == "required"
        assert trace.need_detection is not None
        assert trace.need_detection.answer_shape == "list"
        assert trace.need_detection.coverage_mode == "exhaustive_known_set"
        assert trace.need_detection.source_precision == "required"
        assert trace.composition is not None
        assert trace.composition.answer_shape == "list"
        assert trace.composition.coverage_mode == "exhaustive_known_set"
        assert trace.composition.source_precision == "required"
        assert trace.scoring is not None
        assert trace.scoring.applicability_gate_mode == "shadow"
        assert trace.scoring.eligible_candidate_count == 0
        assert trace.scoring.ineligible_reason_counts == {
            "missing_direct_source_support": 1
        }
        assert trace.scoring.estimated_calls_saved == 0
        assert trace.scoring.adjacent_rrf_delta_distribution["count"] == 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_hydrates_evidence_packets_only_when_ablation_enabled() -> None:
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(score_map={"mem_gina": 0.95})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
        messages = MessageRepository(connection, clock)
        await messages.create_message(
            message_id="msg_source",
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text="Gina: Contemporary dance really speaks to me.",
            occurred_at="2023-01-20T16:04:00+00:00",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Gina's favorite dance style is contemporary.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            payload={"source_message_ids": ["msg_source"]},
            memory_id="mem_gina",
        )
        await MemoryEvidenceRepository(connection, clock).create_support_edge_with_spans(
            user_id="usr_1",
            memory_id="mem_gina",
            support_kind="contextual_direct",
            evidence_polarity="supports",
            speaker_relation_to_subject="self_report",
            confidence=0.91,
            spans=[
                {
                    "span_role": "source",
                    "message_id": "msg_source",
                    "quote_text": "Contemporary dance really speaks to me.",
                }
            ],
        )

        default_packets = await pipeline.execute(
            message_text="What is Gina's favorite style of dance?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "What is Gina's favorite style of dance?"}
            ],
        )
        without_packets = await pipeline.execute(
            message_text="What is Gina's favorite style of dance?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(enable_evidence_packets=False),
            conversation_messages=[
                {"role": "user", "text": "What is Gina's favorite style of dance?"}
            ],
        )

        assert "evidence_packet:" not in without_packets.composed_context.memory_block
        assert "evidence_packet: support: contextual_direct" in default_packets.composed_context.memory_block
        assert "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 1: Contemporary dance really speaks to me." in default_packets.composed_context.memory_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_applies_structural_context_envelope_budget_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[int] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_policy = kwargs["resolved_policy"]
        observed_budgets.append(observed_policy.context_budget_tokens)
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(score_map={"mem_1": 0.91})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert observed_budgets == [2_744]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_applies_context_envelope_budget_to_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[int] = []
    observed_final_items: list[int] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_policy = kwargs["resolved_policy"]
        observed_budgets.append(observed_policy.context_budget_tokens)
        observed_final_items.append(
            observed_policy.retrieval_params.final_context_items
        )
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(score_map={"mem_1": 0.91})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(
                context_envelope_budget_tokens=10_000,
                override_retrieval_params={"final_context_items": 3},
            ),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert observed_budgets == [6_700]
        assert observed_final_items == [3]
        expanded_policy = resolved_policy.model_copy(
            update={
                "retrieval_params": resolved_policy.retrieval_params.model_copy(
                    update={"final_context_items": 70}
                )
            }
        )
        capped_policy = RetrievalPipeline._cap_explicit_final_context_items(
            expanded_policy,
            AblationConfig(override_retrieval_params={"final_context_items": 20}),
        )
        assert capped_policy.retrieval_params.final_context_items == 20
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_forwards_budgeted_composer_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_strategies: list[str | None] = []
    original_compose = ContextComposer.compose

    def _capture_compose(self: ContextComposer, *args, **kwargs):
        observed_strategies.append(kwargs.get("composer_strategy"))
        return original_compose(self, *args, **kwargs)

    monkeypatch.setattr(ContextComposer, "compose", _capture_compose)
    provider = PipelineProvider(score_map={"mem_1": 0.91})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=1.0),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
        )

        await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(composer_strategy="budgeted_marginal"),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert observed_strategies == ["budgeted_marginal"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_retries_partial_applicability_scores_without_crashing() -> None:
    message_text = "retry loop websocket backoff"
    provider = OmitFirstApplicabilityScoreProvider(
        "mem_missing_first",
        score_map={"mem_present": 0.7, "mem_missing_first": 0.86},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_present",
            canonical_text="retry loop websocket backoff present memory",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_missing_first",
            canonical_text="retry loop websocket backoff omitted on first scoring response",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        scored_ids = {candidate.memory_id for candidate in result.scored_candidates}
        assert scored_ids == {"mem_present", "mem_missing_first"}
        scoring_requests = [
            request
            for request in provider.requests
            if str(request.metadata.get("purpose")) == "applicability_scoring"
        ]
        assert len(scoring_requests) == 2
        assert _MEMORY_ID_PATTERN.findall(scoring_requests[1].messages[1].content) == [
            "mem_missing_first"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_retries_malformed_applicability_scores_without_crashing() -> (
    None
):
    message_text = "retry loop websocket backoff"
    provider = MalformedFirstApplicabilityScoreProvider(
        "mem_malformed_first",
        score_map={"mem_present": 0.7, "mem_malformed_first": 0.86},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_present",
            canonical_text="retry loop websocket backoff present memory",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_malformed_first",
            canonical_text="retry loop websocket backoff malformed on first scoring response",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        scored_ids = {candidate.memory_id for candidate in result.scored_candidates}
        assert scored_ids == {"mem_present", "mem_malformed_first"}
        scoring_requests = [
            request
            for request in provider.requests
            if str(request.metadata.get("purpose")) == "applicability_scoring"
        ]
        assert len(scoring_requests) == 2
        assert _MEMORY_ID_PATTERN.findall(scoring_requests[1].messages[1].content) == [
            "mem_malformed_first"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_retries_invalid_applicability_scores_without_crashing() -> None:
    message_text = "retry loop websocket backoff"
    provider = InvalidFirstApplicabilityScoreProvider(
        score_map={"mem_present": 0.7, "mem_retry": 0.86},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_present",
            canonical_text="retry loop websocket backoff present memory",
            scope=MemoryScope.CONVERSATION,
        )
        await _seed_memory(
            memories,
            memory_id="mem_retry",
            canonical_text="retry loop websocket backoff retry memory",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        scored_ids = {candidate.memory_id for candidate in result.scored_candidates}
        assert scored_ids == {"mem_present", "mem_retry"}
        scoring_requests = [
            request
            for request in provider.requests
            if str(request.metadata.get("purpose")) == "applicability_scoring"
        ]
        assert len(scoring_requests) == 2
        assert set(
            _MEMORY_ID_PATTERN.findall(scoring_requests[1].messages[1].content)
        ) == {
            "mem_present",
            "mem_retry",
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_excludes_pending_and_declined_candidates_before_scoring() -> (
    None
):
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(
        score_map={"mem_active": 0.9, "mem_pending": 0.99, "mem_declined": 0.99}
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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

        assert [candidate["id"] for candidate in result.raw_candidates] == [
            "mem_active"
        ]
        assert [candidate.memory_id for candidate in result.scored_candidates] == [
            "mem_active"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_need_detection_returns_empty_needs_without_llm_call() -> (
    None
):
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(score_map={"mem_1": 0.88})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
            ablation=AblationConfig(skip_need_detection=True),
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.detected_needs == []
        assert result.retrieval_plan.query_type == "default"
        assert [
            sub_query.text for sub_query in result.retrieval_plan.sub_query_plans
        ] == [message_text]
        assert not any(
            request.metadata.get("purpose") == "need_detection"
            for request in provider.requests
        )
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
        assert result.retrieval_plan.sub_query_plans[0].quoted_phrases == [
            "citrus marinade"
        ]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_applicability_scoring_uses_raw_scores() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
            ablation=AblationConfig(skip_applicability_scoring=True),
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.scored_candidates[0].final_score == pytest.approx(
            result.scored_candidates[0].retrieval_score
        )
        assert not any(
            request.metadata.get("purpose") == "applicability_scoring"
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_skip_contract_memory_clears_contract_block() -> None:
    message_text = "retry loop websocket backoff"
    provider = PipelineProvider(score_map={"mem_1": 0.9})
    (
        connection,
        memories,
        contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        result = await pipeline.execute(
            message_text="What should I do next?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            ablation=AblationConfig(force_all_scopes=True),
            conversation_messages=[{"role": "user", "text": "What should I do next?"}],
        )

        assert result.retrieval_plan.scope_filter == list(MemoryScope)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_resolves_character_rollup_with_namespace_gates() -> None:
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        _resolved_policy,
        context,
    ) = await _build_runtime()
    summaries = SummaryRepository(
        connection, FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    )
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_character_rollup",
                "workspace_id": "wrk_1",
                "user_persona_id": "persona_writer",
                "platform_id": "web",
                "character_id": "char_debug",
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "character_rollup",
                "summary_text": "Character-specific debugging style.",
                "source_object_ids_json": [],
                "sensitivity": "public",
                "scope_canonical": "character",
                "maya_score": 1.5,
                "model": "model-a",
                "created_at": "2026-04-05T12:00:00+00:00",
            },
        )
        plan = _slot_fill_plan().model_copy(
            update={
                "user_persona_id": "persona_writer",
                "platform_id": "web",
                "character_id": "char_debug",
                "remember_across_devices": False,
            }
        )
        gated_context = context.model_copy(
            update={
                "user_persona_id": "persona_writer",
                "platform_id": "web",
                "character_id": "char_debug",
                "remember_across_devices": False,
            }
        )

        row = await pipeline._resolve_workspace_rollup(  # noqa: SLF001
            conversation_context=gated_context,
            retrieval_plan=plan,
            workspace_rollup=None,
            ablation=AblationConfig(),
        )
        other_platform_row = await pipeline._resolve_workspace_rollup(  # noqa: SLF001
            conversation_context=gated_context.model_copy(
                update={"platform_id": "mobile"}
            ),
            retrieval_plan=plan.model_copy(update={"platform_id": "mobile"}),
            workspace_rollup=None,
            ablation=AblationConfig(),
        )

        assert row is not None
        assert row["summary_text"] == "Character-specific debugging style."
        assert other_platform_row is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_high_stakes_filters_derived_memory_and_workspace_rollup() -> (
    None
):
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
            workspace_rollup={
                "summary_text": "Derived workspace rollup that should be suppressed."
            },
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        assert result.retrieval_plan.require_evidence_regrounding is True
        assert result.retrieval_plan.query_type == "broad_list"
        assert len(result.retrieval_plan.sub_query_plans) == 2
        assert [candidate["id"] for candidate in result.raw_candidates] == [
            "mem_evidence"
        ]
        assert [candidate.memory_id for candidate in result.scored_candidates] == [
            "mem_evidence"
        ]
        assert result.composed_context.selected_memory_ids == ["mem_evidence"]
        assert result.composed_context.workspace_block == ""
        assert "mem_belief" not in result.composed_context.memory_block
        custody_by_id = {
            record["candidate_id"]: record for record in result.candidate_custody
        }
        assert custody_by_id["mem_belief"]["filter_reason"] == "regrounding_filtered"
        assert custody_by_id["mem_belief"]["score_status"] == "filtered_before_scoring"
        assert custody_by_id["mem_evidence"]["composer_decision"] == "selected"
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Wave 1-A: Small-corpus shortcut
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_scores_eligible_memories() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
            memory_id="mem_user",
            canonical_text="User prefers concise debugging answers.",
            scope=MemoryScope.GLOBAL_USER,
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
        assert any(
            request.metadata.get("purpose") == "need_detection"
            for request in provider.requests
        )
        assert any(
            request.metadata.get("purpose") == "applicability_scoring"
            for request in provider.requests
        )
        returned_ids = {candidate["id"] for candidate in result.raw_candidates}
        assert returned_ids == {"mem_conv", "mem_user"}
        assert set(result.composed_context.selected_memory_ids) == {
            "mem_conv",
            "mem_user",
        }
        assert result.retrieval_sufficiency is not None
        assert result.retrieval_sufficiency.state == "retrieval_sufficient"
        assert result.retrieval_sufficiency.scored_candidate_count == 2
        custody_by_id = {
            record["candidate_id"]: record for record in result.candidate_custody
        }
        assert set(custody_by_id) == {"mem_conv", "mem_user"}
        assert all(
            record["score_status"] == "scored" for record in custody_by_id.values()
        )
        assert result.stage_timings["need_detection"] >= 0.0
        assert result.stage_timings["candidate_search"] >= 0.0
        assert result.stage_timings["applicability_scoring"] >= 0.0
    finally:
        await connection.close()


@pytest.mark.parametrize(
    ("active_space_id", "boundary_mode", "expected_ids"),
    [
        (None, None, {"mem_unscoped", "mem_focus", "mem_tagged"}),
        (
            "space_focus",
            SpaceBoundaryMode.FOCUS,
            {"mem_unscoped", "mem_focus", "mem_tagged"},
        ),
        (
            "space_vault",
            SpaceBoundaryMode.PRIVACY_VAULT,
            {"mem_unscoped", "mem_vault", "mem_tagged"},
        ),
        ("space_severed", SpaceBoundaryMode.SEVERANCE, {"mem_severed"}),
        (
            "space_tagged",
            SpaceBoundaryMode.TAGGED,
            {"mem_unscoped", "mem_focus", "mem_tagged"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_enforces_space_boundaries(
    active_space_id: str | None,
    boundary_mode: SpaceBoundaryMode | None,
    expected_ids: set[str],
) -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_unscoped",
            canonical_text="unscoped broader preference",
            scope=MemoryScope.GLOBAL_USER,
        )
        await _seed_memory(
            memories,
            memory_id="mem_focus",
            canonical_text="focus space broader preference",
            scope=MemoryScope.GLOBAL_USER,
            space_id="space_focus",
            space_boundary_mode=SpaceBoundaryMode.FOCUS,
        )
        await _seed_memory(
            memories,
            memory_id="mem_vault",
            canonical_text="vault space broader preference",
            scope=MemoryScope.GLOBAL_USER,
            space_id="space_vault",
            space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
        )
        await _seed_memory(
            memories,
            memory_id="mem_severed",
            canonical_text="severed space broader preference",
            scope=MemoryScope.GLOBAL_USER,
            space_id="space_severed",
            space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
        )
        await _seed_memory(
            memories,
            memory_id="mem_tagged",
            canonical_text="tagged space broader preference",
            scope=MemoryScope.GLOBAL_USER,
            space_id="space_tagged",
            space_boundary_mode=SpaceBoundaryMode.TAGGED,
        )

        active_context = context.model_copy(
            update={
                "active_space_id": active_space_id,
                "active_space_boundary_mode": boundary_mode or SpaceBoundaryMode.FOCUS,
            }
        )
        result = await pipeline.execute(
            message_text="Which broader preferences apply?",
            conversation_context=active_context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "Which broader preferences apply?"},
            ],
        )

        assert result.small_corpus_mode is True
        returned_ids = {str(candidate["id"]) for candidate in result.raw_candidates}
        assert returned_ids == expected_ids
        assert set(result.composed_context.selected_memory_ids) == expected_ids
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_composes_all_eligible_memories() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        memory_count = resolved_policy.retrieval_params.final_context_items + 2
        for index in range(memory_count):
            await _seed_memory(
                memories,
                memory_id=f"mem_small_{index}",
                canonical_text=f"small corpus exact fact {index}",
                scope=MemoryScope.CONVERSATION,
            )

        result = await pipeline.execute(
            message_text="small corpus exact fact",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "small corpus exact fact"},
            ],
        )

        assert result.small_corpus_mode is True
        assert len(_score_request_memory_ids(provider)) == memory_count
        assert len(result.composed_context.selected_memory_ids) == memory_count
        assert result.composed_context.items_dropped == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_sets_trace_flag() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
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
        assert [
            row.model_dump(mode="json")
            for row in trace.need_detection.content_language_profile
        ] == [
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
        assert trace.need_detection.duration_ms >= 0.0
        assert trace.candidate_search is not None
        assert trace.candidate_search.total_after_fusion == 1
        assert trace.scoring is not None
        assert trace.scoring.candidates_scored == 1
        assert trace.retrieval_sufficiency is not None
        assert trace.retrieval_sufficiency.state == "retrieval_sufficient"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_degraded_trace_preserves_language_profile() -> (
    None
):
    provider = FailingPipelineProvider("need_detection")
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
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
        assert result.degraded_mode is True
        assert trace.need_detection is not None
        assert trace.need_detection.degraded_mode is True
        assert [
            row.model_dump(mode="json")
            for row in trace.need_detection.content_language_profile
        ] == [
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_passes_user_communication_profile_to_need_detection_and_trace() -> (
    None
):
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
            "query_language": "ca",
            "answer_language": "es",
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.9},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.0),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="retry loop websocket backoff",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
        )
        source_ref = LanguageProfileSourceRef(
            source_kind="source_message",
            conversation_id="cnv_1",
            source_message_id="msg_lang",
        )
        await CommunicationProfileRepository(
            connection,
            FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)),
        ).upsert_user_language_profile(
            context,
            UserCommunicationProfile(
                observed_user_languages=[
                    ObservedUserLanguage(
                        language_code="ca",
                        message_count=3,
                        source_refs=[source_ref],
                        confidence=0.85,
                    )
                ],
                explicit_language_preferences=[
                    ExplicitLanguagePreference(
                        language_code="es",
                        preference_kind="default_answer_language",
                        context_label="ordinary_chat",
                        source_refs=[source_ref],
                        confidence=0.92,
                    )
                ],
            ),
            scope=MemoryScope.CHARACTER,
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

        need_prompt = next(
            request.messages[1].content
            for request in provider.requests
            if request.metadata.get("purpose") == "need_detection"
        )
        assert "<user_communication_profile>" in need_prompt
        assert (
            "observed_user_languages: ca: 3 observed user-authored messages"
            in need_prompt
        )
        assert (
            "explicit_language_preferences: es/default_answer_language/ordinary_chat"
            in need_prompt
        )
        assert result.retrieval_plan.query_language == "ca"
        assert result.retrieval_plan.answer_language == "es"
        assert trace.need_detection is not None
        assert trace.need_detection.user_communication_profile is not None
        assert (
            trace.need_detection.user_communication_profile.observed_language_codes
            == ["ca"]
        )
        assert (
            trace.need_detection.user_communication_profile.preference_language_codes
            == ["es"]
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_exact_recall_includes_verbatim_evidence_search_windows() -> (
    None
):
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["panique prochaine étape"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "panique prochaine étape",
                    "fts_phrase": "panique prochaine étape",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["other_verbatim"],
            "raw_context_access_mode": "verbatim",
        },
    )
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.7),
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
            text="Quand je panique, aide-moi avec une seule prochaine étape.",
        )

        trace = RetrievalTrace(
            query_text="Comment m'aider quand je panique ?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )
        result = await pipeline.execute(
            message_text="Comment m'aider quand je panique ?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "Comment m'aider quand je panique ?"},
            ],
            trace=trace,
        )

        assert result.small_corpus_mode is True
        assert trace.candidate_search is not None
        assert trace.candidate_search.verbatim_evidence_search_candidates_count >= 1
        assert any(
            candidate.get("is_verbatim_evidence_window")
            for candidate in result.raw_candidates
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_hydrates_default_source_quote_for_selected_memory_source_message() -> (
    None
):
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Nia cedar notebook"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Nia cedar notebook",
                    "fts_phrase": "Nia cedar notebook",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_notebook": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        messages = MessageRepository(
            connection,
            FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)),
        )
        await messages.create_message(
            message_id="msg_notebook_source",
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text="Nia keeps the cedar notebook in her carry-on because it has the gate checklist.",
            occurred_at="2026-02-02T15:45:00+00:00",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Nia keeps a cedar notebook with a gate checklist.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            payload={"source_message_ids": ["msg_notebook_source"]},
            memory_id="mem_notebook",
        )

        result = await pipeline.execute(
            message_text="Why does Nia keep that notebook nearby?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {
                    "id": "msg_current",
                    "conversation_id": "cnv_1",
                    "role": "user",
                    "text": "Why does Nia keep that notebook nearby?",
                },
            ],
        )

        assert "mem_notebook" in result.composed_context.selected_memory_ids
        assert (
            "source_quote: user @ 2026-02-02T15:45:00+00:00 seq 1: "
            "Nia keeps the cedar notebook in her carry-on because it has the gate checklist."
        ) in result.composed_context.memory_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_backfills_source_quote_for_selected_memory_source_message() -> (
    None
):
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Jon banker job"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Jon banker job",
                    "fts_phrase": "Jon banker job",
                }
            ],
            "query_type": "temporal",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["date"],
        },
        score_map={"mem_job_loss": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        messages = MessageRepository(
            connection,
            FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)),
        )
        await messages.create_message(
            message_id="msg_source",
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text="Jon: Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
            occurred_at="2023-01-20T16:04:00+00:00",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Jon is no longer in a secure banker job and is starting a business.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            payload={
                "source_message_ids": ["msg_source"],
                "source_message_window_start_occurred_at": "2023-01-20T16:04:00+00:00",
                "source_message_window_end_occurred_at": "2023-01-20T16:04:00+00:00",
            },
            memory_id="mem_job_loss",
        )

        result = await pipeline.execute(
            message_text="When did Jon lose his job as a banker?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "When did Jon lose his job as a banker?"},
            ],
        )

        assert "mem_job_loss" in result.composed_context.selected_memory_ids
        assert (
            "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 1: Jon: Lost my job as a banker yesterday"
            in result.composed_context.memory_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_does_not_backfill_source_quote_from_other_conversation() -> (
    None
):
    from atagia.core.repositories import MessageRepository

    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Jon banker job"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Jon banker job",
                    "fts_phrase": "Jon banker job",
                }
            ],
            "query_type": "temporal",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["date"],
        },
        score_map={"mem_cross_chat_source": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        mode_id="general_qa",
        provider=provider,
    )
    try:
        clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        await conversations.create_conversation(
            "cnv_other", "usr_1", "wrk_1", "general_qa", "Other Chat"
        )
        await messages.create_message(
            message_id="msg_other_source",
            conversation_id="cnv_other",
            role="user",
            seq=None,
            text="Jon: Lost my job as a banker yesterday.",
            occurred_at="2023-01-20T16:04:00+00:00",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Jon lost his banker job before starting a business.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            payload={
                "source_message_ids": ["msg_other_source"],
                "source_message_window_start_occurred_at": "2023-01-20T16:04:00+00:00",
                "source_message_window_end_occurred_at": "2023-01-20T16:04:00+00:00",
            },
            memory_id="mem_cross_chat_source",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
        )

        result = await pipeline.execute(
            message_text="When did Jon lose his job as a banker?",
            conversation_context=context.model_copy(
                update={"assistant_mode_id": "general_qa"}
            ),
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {
                    "id": "msg_current",
                    "conversation_id": "cnv_1",
                    "role": "user",
                    "text": "When did Jon lose his job as a banker?",
                },
            ],
        )

        assert "mem_cross_chat_source" in result.composed_context.selected_memory_ids
        assert "source_quote:" not in result.composed_context.memory_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_enforces_privacy_ceiling() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
async def test_trusted_private_retrieval_bypasses_public_only_cold_start_and_small_corpus() -> (
    None
):
    provider = PipelineProvider(score_map={"mem_private": 0.95})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=1.0),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="retry loop websocket backoff private note",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=2,
            sensitivity=MemorySensitivity.PRIVATE,
            memory_id="mem_private",
        )

        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=True,
            ablation=AblationConfig(
                override_retrieval_params={
                    "privacy_ceiling": 3,
                    "allow_private_sensitivity": True,
                }
            ),
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"},
            ],
        )

        assert result.small_corpus_mode is False
        assert "mem_private" in {candidate["id"] for candidate in result.raw_candidates}
        assert "mem_private" in result.composed_context.selected_memory_ids
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_shortcut_excludes_pending_memories() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
async def test_pipeline_exact_recall_composes_scored_candidates_up_to_rerank_budget() -> (
    None
):
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["needle exact detail"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "needle exact detail",
                    "fts_phrase": "needle exact detail",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["other_verbatim"],
        },
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.0),
    )
    try:
        memory_count = resolved_policy.retrieval_params.rerank_top_k + 2
        assert memory_count <= resolved_policy.retrieval_params.fts_limit
        for index in range(memory_count):
            await _seed_memory(
                memories,
                memory_id=f"mem_exact_{index}",
                canonical_text=f"needle exact detail {index}",
                scope=MemoryScope.CONVERSATION,
            )

        result = await pipeline.execute(
            message_text="Which needle exact details are available?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "Which needle exact details are available?"},
            ],
        )

        assert result.small_corpus_mode is False
        assert result.retrieval_plan.exact_recall_mode is True
        assert len(_score_request_memory_ids(provider)) == memory_count
        assert len(result.composed_context.selected_memory_ids) == memory_count
        assert result.composed_context.items_dropped == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_exact_recall_plan_exposes_anchor_only_fts_materialization() -> (
    None
):
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["service account for Falcon"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "service account for Falcon",
                    "fts_phrase": "service account Falcon rotation owner",
                    "must_keep_terms": ["Falcon", "SA42"],
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["code"],
        },
        score_map={"mem_falcon_service_account": 0.95},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.0),
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_falcon_service_account",
            canonical_text="Falcon uses SA42 for deployment approvals.",
            scope=MemoryScope.CONVERSATION,
        )

        result = await pipeline.execute(
            message_text="Which service account did Falcon use?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "Which service account did Falcon use?"},
            ],
        )

        sub_query = result.retrieval_plan.sub_query_plans[0]
        assert result.retrieval_plan.exact_recall_mode is True
        assert sub_query.fts_queries == [
            "falcon sa42 service account rotation owner",
            "service account falcon rotation owner sa42",
            "falcon sa42",
            "service OR account OR falcon OR rotation OR owner OR sa42",
        ]
        assert sub_query.fts_query_kinds == [
            "anchor_first_and",
            "sparse_and",
            "anchor_only_and",
            "broad_or",
        ]
        assert {candidate["id"] for candidate in result.raw_candidates} == {
            "mem_falcon_service_account",
        }
        fts_matches = result.raw_candidates[0]["fts_query_matches"]
        assert {
            ("falcon sa42", "anchor_only_and", "implicit_and"),
            (
                "service OR account OR falcon OR rotation OR owner OR sa42",
                "broad_or",
                "explicit_or",
            ),
        }.issubset(
            {
                (match["query"], match["kind"], match["match_mode"])
                for match in fts_matches
            }
        )
        assert result.composed_context.selected_memory_ids == [
            "mem_falcon_service_account",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_degraded_need_detection_composes_scored_candidates_for_recovery() -> (
    None
):
    provider = FailingPipelineProvider("need_detection")
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
        provider=provider,
        settings=_settings(small_corpus_token_threshold_ratio=0.0),
    )
    try:
        memory_count = resolved_policy.retrieval_params.rerank_top_k + 2
        assert memory_count <= resolved_policy.retrieval_params.fts_limit
        for index in range(memory_count):
            await _seed_memory(
                memories,
                memory_id=f"mem_degraded_{index}",
                canonical_text=f"needle degraded recovery detail {index}",
                scope=MemoryScope.CONVERSATION,
            )

        result = await pipeline.execute(
            message_text="needle degraded recovery detail",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "needle degraded recovery detail"},
            ],
        )

        assert result.degraded_mode is True
        assert len(_score_request_memory_ids(provider)) == memory_count
        assert len(result.composed_context.selected_memory_ids) == memory_count
        assert result.composed_context.items_dropped == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_large_corpus_takes_normal_path() -> None:
    # Seed more than 0.7 * context_budget_tokens worth of content so the
    # shortcut declines to fire (coding_debug budget is 5300 tokens).
    long_text = "retry loop websocket backoff " * 2000
    provider = PipelineProvider(score_map={"mem_large": 0.9})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
            request.metadata.get("purpose") == "need_detection"
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_small_corpus_disabled_when_ratio_is_zero() -> None:
    provider = PipelineProvider()
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
            request.metadata.get("purpose") == "need_detection"
            for request in provider.requests
        )
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Wave 1-B: Need detector as counselor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_degrades_when_need_detector_fails() -> None:
    provider = FailingPipelineProvider("need_detection", score_map={"mem_1": 0.85})
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
        assert trace.retrieval_sufficiency is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_base_search_runs_even_when_need_detector_fails() -> None:
    provider = FailingPipelineProvider("need_detection")
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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


def test_regrounding_requirements_keep_grounded_l0_summary_chunks() -> None:
    plan = _slot_fill_plan()
    plan.require_evidence_regrounding = True
    evidence = _candidate_record(
        memory_id="mem_evidence",
        canonical_text="Concrete evidence.",
    )
    interaction_contract = _candidate_record(
        memory_id="mem_contract",
        canonical_text="Only disclose this value in the matching active context.",
        object_type=MemoryObjectType.INTERACTION_CONTRACT,
    )
    grounded_chunk = _candidate_record(
        memory_id="sum_chunk",
        canonical_text="Grounded chunk summary.",
        object_type=MemoryObjectType.SUMMARY_VIEW,
        payload_json={
            "summary_kind": SummaryViewKind.CONVERSATION_CHUNK.value,
            "hierarchy_level": 0,
            "source_excerpt_messages": [
                {"role": "user", "text": "Concrete transcript."}
            ],
        },
    )
    episode = _candidate_record(
        memory_id="sum_episode",
        canonical_text="Abstract episode summary.",
        object_type=MemoryObjectType.SUMMARY_VIEW,
        payload_json={
            "summary_kind": SummaryViewKind.EPISODE.value,
            "hierarchy_level": 1,
            "source_excerpt_messages": [
                {"role": "user", "text": "Concrete transcript."}
            ],
        },
    )

    filtered = RetrievalPipeline._apply_regrounding_requirements(
        [evidence, interaction_contract, grounded_chunk, episode],
        plan,
    )

    assert [candidate["id"] for candidate in filtered] == [
        "mem_evidence",
        "mem_contract",
        "sum_chunk",
    ]


def test_ambiguity_recovery_expands_scoring_budget() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["general_qa"]
    policy = PolicyResolver().resolve(manifest, None, None)
    plan = _slot_fill_plan(exact_recall_mode=False)

    expanded = RetrievalPipeline._expand_recall_or_recovery_scoring_budget(
        policy,
        plan,
        degraded_mode=False,
        detected_needs=[
            DetectedNeed(
                need_type=NeedTrigger.AMBIGUITY,
                confidence=0.8,
                reasoning="Several memories may match this underspecified request.",
            )
        ],
        item_count=32,
    )

    assert policy.retrieval_params.rerank_top_k < 32
    assert expanded.retrieval_params.rerank_top_k == 32


def test_under_specified_recovery_expands_context_items_with_low_cap() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["general_qa"]
    policy = PolicyResolver().resolve(manifest, None, None)
    scoring_policy = policy.model_copy(
        update={
            "retrieval_params": policy.retrieval_params.model_copy(
                update={"rerank_top_k": 32}
            )
        }
    )
    plan = _slot_fill_plan(exact_recall_mode=False)

    expanded = RetrievalPipeline._expand_recall_or_recovery_context_items(
        scoring_policy,
        plan,
        degraded_mode=False,
        detected_needs=[
            DetectedNeed(
                need_type=NeedTrigger.UNDER_SPECIFIED_REQUEST,
                confidence=0.8,
                reasoning="The query needs a few complementary memories.",
            )
        ],
        item_count=32,
    )

    assert policy.retrieval_params.final_context_items == 8
    assert expanded.retrieval_params.final_context_items == 12


def test_default_queries_do_not_expand_context_items_without_recovery_need() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["general_qa"]
    policy = PolicyResolver().resolve(manifest, None, None)
    scoring_policy = policy.model_copy(
        update={
            "retrieval_params": policy.retrieval_params.model_copy(
                update={"rerank_top_k": 32}
            )
        }
    )
    plan = _slot_fill_plan(exact_recall_mode=False)

    expanded = RetrievalPipeline._expand_recall_or_recovery_context_items(
        scoring_policy,
        plan,
        degraded_mode=False,
        detected_needs=[],
        item_count=32,
    )

    assert expanded.retrieval_params.final_context_items == 8


@pytest.mark.asyncio
async def test_pipeline_verbatim_evidence_search_enabled_contributes_to_exact_recall_trace() -> (
    None
):
    """End-to-end Wave 1 batch 2 (1-C + 1-D): raw evidence reaches the trace.

    The need detector surfaces exact recall, the planner propagates it,
    the candidate search attaches the verbatim_evidence_search channel, and the
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
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa", provider=provider)
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
        assert trace.candidate_search.verbatim_evidence_search_candidates_count >= 1
        evidence_windows = [
            candidate
            for candidate in result.raw_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, (
            "verbatim evidence search window should reach the raw_candidates list"
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_traces_temporary_exact_recall_scaffolding() -> None:
    provider = PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["What country is Caroline's grandma from?"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "What country is Caroline's grandma from?",
                    "fts_phrase": "What country is Caroline's grandma from?",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
            "exact_recall_needed": False,
            "exact_facets": [],
        },
        unknown_exact_review_response={
            "is_exact_value_lookup": True,
            "sub_query_text": "Caroline grandma country",
            "fts_phrase": "Caroline grandma country",
            "exact_facets": ["location", "person_name"],
            "must_keep_terms": ["Caroline", "grandma", "country"],
            "quoted_phrases": [],
        },
        score_map={"mem_country": 0.9},
    )
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa", provider=provider)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_country",
            canonical_text="Caroline's grandma is from Norway.",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text="What country is Caroline's grandma from?",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text="What country is Caroline's grandma from?",
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "What country is Caroline's grandma from?"},
            ],
            trace=trace,
        )

        assert result.retrieval_plan.exact_recall_mode is True
        assert trace.need_detection is not None
        need_mechanisms = {
            event.mechanism for event in trace.need_detection.temporary_scaffolding
        }
        root_mechanisms = {event.mechanism for event in trace.temporary_scaffolding}
        assert "unknown_only_exact_value_contract_review" in need_mechanisms
        assert "unknown_only_exact_value_contract_review" in root_mechanisms
        assert "must_keep_tail_exact_recall_backoff" in root_mechanisms
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(provider=provider)
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
        pins = VerbatimPinRepository(
            connection, FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
        )
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
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
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
async def test_summary_support_regrounding_runs_for_under_specified_requests() -> None:
    (
        connection,
        _memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
    try:
        summary = _candidate_record(
            memory_id="sum_episode",
            canonical_text="Abstract shared-history summary.",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            rrf_score=1.0,
            payload_json={
                "summary_kind": SummaryViewKind.CONVERSATION_CHUNK.value,
                "hierarchy_level": 0,
                "source_object_ids": ["mem_support"],
            },
        )
        support = _candidate_record(
            memory_id="mem_support",
            canonical_text="Concrete source memory for the shared history.",
            rrf_score=0.01,
        )
        default_plan = _slot_fill_plan().model_copy(
            update={
                "query_type": "default",
                "exact_recall_mode": False,
            }
        )

        updated_shortlist = await pipeline._reground_summary_support_shortlist(
            shortlist=[summary],
            filtered_candidates=[summary, support],
            conversation_context=context,
            resolved_policy=resolved_policy,
            detected_needs=[
                DetectedNeed(
                    need_type=NeedTrigger.UNDER_SPECIFIED_REQUEST,
                    confidence=0.8,
                    reasoning="The query needs broad shared context.",
                )
            ],
            retrieval_plan=default_plan,
        )

        assert [candidate["id"] for candidate in updated_shortlist] == [
            "sum_episode",
            "mem_support",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_fetches_missing_support_by_id() -> None:
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
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
async def test_summary_support_regrounding_rejects_cross_persona_fetched_support() -> (
    None
):
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Persona-specific source memory from another namespace.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_other_persona",
            user_persona_id="persona_b",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
        )
        summary = _candidate_record(
            memory_id="sum_episode",
            canonical_text="Abstract outage summary.",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            rrf_score=1.0,
            payload_json={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": ["mem_other_persona"],
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

        assert [candidate["id"] for candidate in updated_shortlist] == ["sum_episode"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_support_regrounding_does_not_promote_filtered_source() -> None:
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        _provider,
        resolved_policy,
        context,
    ) = await _build_runtime(mode_id="general_qa")
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
async def test_pipeline_summary_support_regrounding_reaches_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    (
        connection,
        memories,
        _contracts,
        pipeline,
        provider,
        resolved_policy,
        context,
    ) = await _build_runtime(
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

        async def fake_search(
            _plan: RetrievalPlan,
            _user_id: str,
            **_kwargs: object,
        ) -> list[dict[str, object]]:
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
