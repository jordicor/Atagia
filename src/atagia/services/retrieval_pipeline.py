"""Reusable retrieval pipeline for chat and replay flows."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.memory.retrieval_planner import RetrievalPlanner
from atagia.models.schemas_memory import (
    ComposedContext,
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    RetrievalParams,
    RetrievalPlan,
    ScoredCandidate,
)
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.llm_client import LLMClient


class RetrievalPipeline:
    """Execute the retrieval stages used by chat and replay flows."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        embedding_index: EmbeddingIndex,
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._settings = settings or Settings.from_env()
        self._llm_client = llm_client
        self._embedding_index = embedding_index
        self._message_repository = MessageRepository(connection, clock)
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._contract_repository = ContractDimensionRepository(connection, clock)
        self._summary_repository = SummaryRepository(connection, clock)
        self._need_detector = NeedDetector(llm_client=llm_client, settings=self._settings)
        self._planner = RetrievalPlanner(clock)
        self._candidate_search = CandidateSearch(
            connection,
            clock,
            embedding_index=embedding_index,
            settings=self._settings,
        )
        self._scorer = ApplicabilityScorer(llm_client=llm_client, clock=clock, settings=self._settings)
        self._context_composer = ContextComposer(clock)
        self._contract_projector = ContractProjector(
            llm_client=llm_client,
            clock=clock,
            message_repository=self._message_repository,
            memory_repository=self._memory_repository,
            contract_repository=self._contract_repository,
            settings=self._settings,
        )

    async def execute(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        cold_start: bool,
        ablation: AblationConfig | None = None,
        workspace_rollup: dict[str, Any] | None = None,
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        effective_ablation = ablation or AblationConfig()
        effective_policy = self._override_policy(resolved_policy, effective_ablation)
        transcript = conversation_messages or []
        stage_timings: dict[str, float] = {}

        if effective_ablation.skip_need_detection:
            detected_needs = []
            stage_timings["need_detection"] = 0.0
        else:
            detected_needs = await self._measure_stage(
                stage_timings,
                "need_detection",
                self._need_detector.detect(
                    message_text=message_text,
                    role="user",
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                ),
            )

        retrieval_plan = await self._measure_stage(
            stage_timings,
            "planning",
            self._build_plan(
                message_text=message_text,
                conversation_context=conversation_context,
                resolved_policy=effective_policy,
                detected_needs=detected_needs,
                cold_start=cold_start,
                ablation=effective_ablation,
            ),
        )
        raw_candidates = await self._measure_stage(
            stage_timings,
            "candidate_search",
            self._candidate_search.search(
                retrieval_plan,
                conversation_context.user_id,
                query_text=message_text,
            ),
        )
        raw_candidates = self._apply_regrounding_requirements(raw_candidates, retrieval_plan)

        if effective_ablation.skip_applicability_scoring:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._score_without_llm(
                    raw_candidates,
                    effective_policy,
                    detected_needs,
                    retrieval_plan,
                ),
            )
        else:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._scorer.score(
                    raw_candidates,
                    message_text=message_text,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    detected_needs=detected_needs,
                    retrieval_plan=retrieval_plan,
                ),
            )

        if effective_ablation.skip_contract_memory:
            current_contract = {}
            stage_timings["contract_lookup"] = 0.0
        else:
            current_contract = await self._measure_stage(
                stage_timings,
                "contract_lookup",
                self._contract_projector.get_current_contract(
                    conversation_context.user_id,
                    conversation_context.assistant_mode_id,
                    conversation_context.workspace_id,
                    conversation_context.conversation_id,
                ),
            )

        user_state = await self._measure_stage(
            stage_timings,
            "state_lookup",
            self._memory_repository.get_state_snapshot(
                conversation_context.user_id,
                assistant_mode_id=conversation_context.assistant_mode_id,
                workspace_id=conversation_context.workspace_id,
                conversation_id=conversation_context.conversation_id,
            ),
        )

        effective_workspace_rollup = await self._measure_stage(
            stage_timings,
                "workspace_rollup_lookup",
                self._resolve_workspace_rollup(
                    conversation_context=conversation_context,
                    retrieval_plan=retrieval_plan,
                    workspace_rollup=workspace_rollup,
                    ablation=effective_ablation,
                ),
        )

        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                message_text=message_text,
                scored_candidates=scored_candidates,
                current_contract=current_contract,
                workspace_rollup=effective_workspace_rollup,
                user_state=user_state,
                resolved_policy=effective_policy,
                conversation_messages=transcript,
            ),
        )
        if effective_ablation.skip_contract_memory:
            composed_context = self._without_contract_block(composed_context)

        return PipelineResult(
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
        )

    async def _build_plan(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        detected_needs: list[Any],
        cold_start: bool,
        ablation: AblationConfig,
    ):
        plan = self._planner.build_plan(
            message_text=message_text,
            conversation_context=conversation_context,
            resolved_policy=resolved_policy,
            detected_needs=detected_needs,
            cold_start=cold_start,
        )
        if ablation.force_all_scopes:
            plan.scope_filter = list(MemoryScope)
        override_params = ablation.override_retrieval_params or {}
        if "max_candidates" in override_params:
            plan.max_candidates = max(0, int(override_params["max_candidates"]))
        if "max_context_items" in override_params:
            plan.max_context_items = max(1, int(override_params["max_context_items"]))
        if "vector_limit" in override_params:
            plan.vector_limit = max(0, int(override_params["vector_limit"]))
        if "privacy_ceiling" in override_params:
            plan.privacy_ceiling = max(0, min(3, int(override_params["privacy_ceiling"])))
        return plan

    async def _score_without_llm(
        self,
        candidates: list[dict[str, Any]],
        resolved_policy: ResolvedPolicy,
        detected_needs: list[Any],
        retrieval_plan: RetrievalPlan,
    ) -> list[ScoredCandidate]:
        filtered = self._scorer.filter_candidates(
            candidates,
            resolved_policy,
            detected_needs,
            retrieval_plan=retrieval_plan,
        )
        shortlist = filtered[: resolved_policy.retrieval_params.rerank_top_k]
        scored: list[ScoredCandidate] = []
        for candidate in shortlist:
            retrieval_score = self._normalized_retrieval_score(candidate.get("rrf_score"))
            scored.append(
                ScoredCandidate(
                    memory_id=str(candidate["id"]),
                    memory_object=dict(candidate),
                    llm_applicability=retrieval_score,
                    retrieval_score=retrieval_score,
                    vitality_boost=0.0,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=retrieval_score,
                )
            )
        return sorted(scored, key=lambda item: (-item.final_score, item.memory_id))

    async def _resolve_workspace_rollup(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        retrieval_plan: RetrievalPlan,
        workspace_rollup: dict[str, Any] | None,
        ablation: AblationConfig,
    ) -> dict[str, Any] | None:
        # Enhancement over the original routes_chat.py inline flow: the pipeline
        # actively resolves workspace rollups. The original flow passed None.
        # This was added during the P2.6 extraction as the correct intended behavior.
        if ablation.skip_workspace_rollup:
            return None
        if retrieval_plan.require_evidence_regrounding:
            return None
        if workspace_rollup is not None:
            return workspace_rollup
        if conversation_context.workspace_id is None:
            return None
        return await self._summary_repository.get_latest_workspace_rollup(
            conversation_context.user_id,
            conversation_context.workspace_id,
        )

    @staticmethod
    def _apply_regrounding_requirements(
        candidates: list[dict[str, Any]],
        retrieval_plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        if not retrieval_plan.require_evidence_regrounding:
            return candidates
        allowed_types = {
            MemoryObjectType.EVIDENCE.value,
            MemoryObjectType.STATE_SNAPSHOT.value,
        }
        return [
            candidate
            for candidate in candidates
            if str(candidate.get("object_type")) in allowed_types
        ]

    async def _compose_context(
        self,
        *,
        message_text: str,
        scored_candidates: list[ScoredCandidate],
        current_contract: dict[str, dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        user_state: dict[str, Any],
        resolved_policy: ResolvedPolicy,
        conversation_messages: list[dict[str, Any]],
    ):
        return self._context_composer.compose(
            scored_candidates=scored_candidates,
            current_contract=current_contract,
            workspace_rollup=workspace_rollup,
            user_state=user_state,
            resolved_policy=resolved_policy,
            conversation_messages=conversation_messages,
            query_text=message_text,
        )

    def _override_policy(
        self,
        resolved_policy: ResolvedPolicy,
        ablation: AblationConfig,
    ) -> ResolvedPolicy:
        override_params = ablation.override_retrieval_params or {}
        if not override_params:
            return resolved_policy

        retrieval_updates: dict[str, Any] = {}
        for field_name in RetrievalParams.model_fields:
            if field_name in override_params:
                retrieval_updates[field_name] = override_params[field_name]
        retrieval_params = (
            resolved_policy.retrieval_params.model_copy(update=retrieval_updates)
            if retrieval_updates
            else resolved_policy.retrieval_params
        )
        updates: dict[str, Any] = {"retrieval_params": retrieval_params}
        if "privacy_ceiling" in override_params:
            updates["privacy_ceiling"] = max(0, min(3, int(override_params["privacy_ceiling"])))
        return resolved_policy.model_copy(update=updates)

    async def _measure_stage(self, stage_timings: dict[str, float], name: str, awaitable: Any) -> Any:
        started_at = perf_counter()
        result = await awaitable
        stage_timings[name] = (perf_counter() - started_at) * 1000.0
        return result

    @staticmethod
    def _normalized_retrieval_score(rrf_score: Any) -> float:
        if rrf_score is None:
            return 0.0
        return max(0.0, min(1.0, float(rrf_score)))

    def _without_contract_block(self, composed_context: ComposedContext) -> ComposedContext:
        contract_tokens = self._context_composer.estimate_tokens(composed_context.contract_block)
        return composed_context.model_copy(
            update={
                "contract_block": "",
                "total_tokens_estimate": max(0, composed_context.total_tokens_estimate - contract_tokens),
            }
        )
