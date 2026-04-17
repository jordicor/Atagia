"""Reusable retrieval pipeline for chat and replay flows."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from time import perf_counter
from typing import Any, Final

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.candidate_diversity import early_diversity_select
from atagia.memory.context_composer import ContextComposer
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.memory.retrieval_planner import RetrievalPlanner
from atagia.models.schemas_memory import (
    CandidateSearchTrace,
    ComposedContext,
    CompositionTrace,
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    NeedDetectionTrace,
    QueryIntelligenceResult,
    RetrievalParams,
    RetrievalPlan,
    RetrievalTrace,
    ScoredCandidate,
    ScoringTrace,
    SubQuerySearchCount,
)
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.llm_client import LLMClient


logger = logging.getLogger(__name__)
PROFILE_TOP_N: Final[int] = 5


def _default_query_intelligence(message_text: str) -> QueryIntelligenceResult:
    """Minimal query intelligence for the base search lane."""
    return QueryIntelligenceResult(
        needs=[],
        temporal_range=None,
        sub_queries=[message_text],
        query_type="default",
        raw_context_access_mode="normal",
        retrieval_levels=[0],
    )


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
        self._memory_repository = MemoryObjectRepository(connection, clock, settings=self._settings)
        self._contract_repository = ContractDimensionRepository(connection, clock)
        self._summary_repository = SummaryRepository(connection, clock)
        self._need_detector = NeedDetector(llm_client=llm_client, clock=clock, settings=self._settings)
        self._planner = RetrievalPlanner()
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
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        effective_ablation = ablation or AblationConfig()
        effective_policy = self._override_policy(resolved_policy, effective_ablation)
        transcript = conversation_messages or []
        stage_timings: dict[str, float] = {}
        pipeline_start = perf_counter() if trace is not None else 0.0

        # Wave 1-A: Small-corpus shortcut. When the full eligible corpus fits
        # inside the context budget there is nothing to rank, so we build the
        # context directly and skip need detection, candidate search, and
        # scoring entirely.
        if await self._is_small_corpus(
            conversation_context=conversation_context,
            resolved_policy=effective_policy,
        ):
            return await self._execute_small_corpus(
                message_text=message_text,
                conversation_context=conversation_context,
                resolved_policy=effective_policy,
                effective_ablation=effective_ablation,
                transcript=transcript,
                workspace_rollup=workspace_rollup,
                stage_timings=stage_timings,
                trace=trace,
                pipeline_start=pipeline_start,
            )

        # Wave 1-B: base search always runs. The raw user query is tokenised
        # into a single-sub-query plan with no need-driven expansions. This
        # guarantees at least one retrieval lane even if need detection fails.
        base_plan = await self._measure_stage(
            stage_timings,
            "base_planning",
            self._build_plan(
                message_text=message_text,
                query_intelligence=_default_query_intelligence(message_text),
                conversation_context=conversation_context,
                resolved_policy=effective_policy,
                cold_start=cold_start,
                ablation=effective_ablation,
            ),
        )
        base_candidates = await self._measure_stage(
            stage_timings,
            "base_candidate_search",
            self._candidate_search.search(
                base_plan,
                conversation_context.user_id,
            ),
        )

        # Wave 1-B: enrichment lane — try need detection and an enriched
        # search. Failures log, mark the result as degraded, and fall back
        # to the base candidates. Ablation skip is a deliberate bypass, not a
        # degradation.
        degraded_mode = False
        detected_needs: list[Any] = []
        enriched_candidates: list[dict[str, Any]] = []
        enriched_plan: RetrievalPlan | None = None
        query_intelligence: QueryIntelligenceResult = _default_query_intelligence(message_text)

        if effective_ablation.skip_need_detection:
            stage_timings["need_detection"] = 0.0
            stage_timings["enriched_planning"] = 0.0
            stage_timings["enriched_candidate_search"] = 0.0
            if trace is not None:
                trace.need_detection = NeedDetectionTrace(
                    detected_needs=[],
                    sub_queries=[message_text],
                    sparse_hints=[],
                    query_type="default",
                    raw_context_access_mode="normal",
                    temporal_range=None,
                    retrieval_levels=[0],
                    degraded_mode=False,
                    duration_ms=0.0,
                )
        else:
            need_start = perf_counter() if trace is not None else 0.0
            try:
                user_language_profile = await self._candidate_search.aggregate_retrievable_language_mix(
                    user_id=conversation_context.user_id,
                    scope_filter=base_plan.scope_filter,
                    assistant_mode_id=base_plan.assistant_mode_id,
                    workspace_id=base_plan.workspace_id,
                    conversation_id=base_plan.conversation_id,
                    privacy_ceiling=base_plan.privacy_ceiling,
                    limit=PROFILE_TOP_N,
                )
                query_intelligence = await self._measure_stage(
                    stage_timings,
                    "need_detection",
                    self._need_detector.detect(
                        message_text=message_text,
                        role="user",
                        conversation_context=conversation_context,
                        resolved_policy=effective_policy,
                        user_language_profile=user_language_profile,
                    ),
                )
            except Exception as exc:
                degraded_mode = True
                stage_timings.setdefault("need_detection", (perf_counter() - need_start) * 1000.0)
                stage_timings["enriched_planning"] = 0.0
                stage_timings["enriched_candidate_search"] = 0.0
                logger.warning(
                    "need_detector_failed_using_base_search_only",
                    extra={
                        "user_id": conversation_context.user_id,
                        "conversation_id": conversation_context.conversation_id,
                        "error": str(exc),
                    },
                )
                if trace is not None:
                    need_elapsed = (perf_counter() - need_start) * 1000.0
                    trace.need_detection = NeedDetectionTrace(
                        degraded_mode=True,
                        duration_ms=need_elapsed,
                        raw_context_access_mode="normal",
                    )
            else:
                detected_needs = list(query_intelligence.needs)
                if trace is not None:
                    need_elapsed = (perf_counter() - need_start) * 1000.0
                    trace.need_detection = NeedDetectionTrace(
                        detected_needs=[
                            need.need_type.value for need in query_intelligence.needs
                        ],
                        sub_queries=list(query_intelligence.sub_queries),
                        sparse_hints=[
                            hint.fts_phrase or hint.sub_query_text
                            for hint in query_intelligence.sparse_query_hints
                        ],
                        query_type=query_intelligence.query_type,
                        raw_context_access_mode=query_intelligence.raw_context_access_mode,
                        temporal_range=(
                            f"{query_intelligence.temporal_range.start.isoformat()}/{query_intelligence.temporal_range.end.isoformat()}"
                            if query_intelligence.temporal_range is not None
                            else None
                        ),
                        retrieval_levels=list(query_intelligence.retrieval_levels),
                        degraded_mode=False,
                        duration_ms=need_elapsed,
                        exact_recall_needed=bool(query_intelligence.exact_recall_needed),
                        exact_facets=[
                            facet.value for facet in query_intelligence.exact_facets
                        ],
                    )
            if trace is not None:
                trace.raw_context_access_mode = query_intelligence.raw_context_access_mode
            enriched_plan = await self._measure_stage(
                stage_timings,
                "enriched_planning",
                self._build_plan(
                    message_text=message_text,
                    query_intelligence=query_intelligence,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    cold_start=cold_start,
                    ablation=effective_ablation,
                ),
            )
            enriched_candidates = await self._measure_stage(
                stage_timings,
                "enriched_candidate_search",
                self._candidate_search.search(
                    enriched_plan,
                    conversation_context.user_id,
                ),
            )

        retrieval_plan = enriched_plan or base_plan
        raw_candidates = self._merge_candidates(base_candidates, enriched_candidates)
        # Regrounding is decided by the winning plan (enriched when available)
        # so the base search does not inject derived memories when high-stakes
        # needs require direct evidence.
        raw_candidates = self._apply_regrounding_requirements(raw_candidates, retrieval_plan)
        # Aggregate the two lanes into single "planning" and "candidate_search"
        # keys so downstream telemetry stays consistent across pipeline variants.
        stage_timings["planning"] = stage_timings.get("base_planning", 0.0) + stage_timings.get(
            "enriched_planning", 0.0
        )
        candidate_total_ms = stage_timings.get("base_candidate_search", 0.0) + stage_timings.get(
            "enriched_candidate_search", 0.0
        )
        stage_timings["candidate_search"] = candidate_total_ms
        if trace is not None:
            trace.candidate_search = self._build_candidate_search_trace(
                raw_candidates,
                retrieval_plan,
                candidate_total_ms,
            )

        filtered_candidates = self._scorer.filter_candidates(
            raw_candidates,
            effective_policy,
            detected_needs,
            retrieval_plan=retrieval_plan,
        )
        shortlist = early_diversity_select(
            filtered_candidates,
            query_type=retrieval_plan.query_type,
            shortlist_k=effective_policy.retrieval_params.rerank_top_k,
        )
        shortlist = await self._reground_summary_support_shortlist(
            shortlist=shortlist,
            filtered_candidates=filtered_candidates,
            conversation_context=conversation_context,
            resolved_policy=effective_policy,
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
        )

        scoring_start = perf_counter() if trace is not None else 0.0
        if effective_ablation.skip_applicability_scoring:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._score_without_llm(
                    shortlist,
                ),
            )
        else:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._scorer.score_shortlist(
                    shortlist,
                    message_text=message_text,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    detected_needs=detected_needs,
                    retrieval_plan=retrieval_plan,
                ),
            )
        if trace is not None:
            scoring_elapsed = (perf_counter() - scoring_start) * 1000.0
            trace.scoring = self._build_scoring_trace(
                raw_candidates,
                filtered_candidates,
                scored_candidates,
                scoring_elapsed,
            )

        if effective_ablation.skip_contract_memory:
            current_contract: dict[str, dict[str, Any]] = {}
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

        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                message_text=message_text,
                retrieval_plan=retrieval_plan,
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

        if trace is not None:
            composition_elapsed = (perf_counter() - composition_start) * 1000.0
            trace.composition = self._build_composition_trace(
                composed_context,
                resolved_policy,
                composition_elapsed,
            )
            trace.degraded_mode = degraded_mode
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0

        return PipelineResult(
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=False,
            degraded_mode=degraded_mode,
        )

    # ------------------------------------------------------------------
    # Wave 1-A: small-corpus shortcut
    # ------------------------------------------------------------------

    async def _is_small_corpus(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
    ) -> bool:
        """Return True if the full eligible corpus fits the small-corpus budget."""
        threshold_ratio = self._settings.small_corpus_token_threshold_ratio
        if threshold_ratio <= 0.0:
            return False
        threshold_tokens = int(resolved_policy.context_budget_tokens * threshold_ratio)
        if threshold_tokens <= 0:
            return False
        memory_chars = await self._memory_repository.sum_canonical_text_length_for_context(
            conversation_context.user_id,
            resolved_policy.allowed_scopes,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            assistant_mode_id=conversation_context.assistant_mode_id,
            privacy_ceiling=resolved_policy.privacy_ceiling,
        )
        message_chars = await self._message_repository.sum_text_length_for_context(
            conversation_context.user_id,
            resolved_policy.allowed_scopes,
            conversation_id=conversation_context.conversation_id,
            workspace_id=conversation_context.workspace_id,
            assistant_mode_id=conversation_context.assistant_mode_id,
        )
        estimated_tokens = self._estimate_tokens_from_chars(memory_chars + message_chars)
        return estimated_tokens < threshold_tokens

    async def _execute_small_corpus(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        effective_ablation: AblationConfig,
        transcript: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        stage_timings: dict[str, float],
        trace: RetrievalTrace | None,
        pipeline_start: float,
    ) -> PipelineResult:
        """Small-corpus shortcut: pass all eligible memories to the composer."""
        if trace is not None:
            trace.small_corpus_mode = True
            trace.degraded_mode = False
            trace.raw_context_access_mode = "normal"
            trace.need_detection = NeedDetectionTrace(
                detected_needs=[],
                sub_queries=[message_text],
                sparse_hints=[],
                query_type="default",
                raw_context_access_mode="normal",
                temporal_range=None,
                retrieval_levels=[0],
                degraded_mode=False,
                duration_ms=0.0,
            )
        stage_timings["need_detection"] = 0.0
        stage_timings["base_planning"] = 0.0
        stage_timings["base_candidate_search"] = 0.0
        stage_timings["enriched_planning"] = 0.0
        stage_timings["enriched_candidate_search"] = 0.0
        stage_timings["applicability_scoring"] = 0.0

        retrieval_plan = await self._measure_stage(
            stage_timings,
            "planning",
            self._build_plan(
                message_text=message_text,
                query_intelligence=_default_query_intelligence(message_text),
                conversation_context=conversation_context,
                resolved_policy=resolved_policy,
                cold_start=False,
                ablation=effective_ablation,
            ),
        )

        candidate_start = perf_counter() if trace is not None else 0.0
        raw_candidates = await self._measure_stage(
            stage_timings,
            "candidate_search",
            self._memory_repository.list_eligible_for_context(
                conversation_context.user_id,
                resolved_policy.allowed_scopes,
                workspace_id=conversation_context.workspace_id,
                conversation_id=conversation_context.conversation_id,
                assistant_mode_id=conversation_context.assistant_mode_id,
                privacy_ceiling=resolved_policy.privacy_ceiling,
            ),
        )
        raw_candidates = self._apply_regrounding_requirements(raw_candidates, retrieval_plan)
        if trace is not None:
            candidate_elapsed = (perf_counter() - candidate_start) * 1000.0
            trace.candidate_search = CandidateSearchTrace(
                fts_candidates_count=0,
                embedding_candidates_count=0,
                consequence_candidates_count=0,
                raw_message_candidates_count=0,
                entity_candidates_count=0,
                total_before_fusion=len(raw_candidates),
                total_after_fusion=len(raw_candidates),
                per_subquery_counts=[],
                duration_ms=candidate_elapsed,
            )

        scored_candidates = self._score_small_corpus_candidates(raw_candidates)
        if trace is not None:
            trace.scoring = ScoringTrace(
                candidates_received=len(raw_candidates),
                candidates_scored=len(scored_candidates),
                candidates_rejected=0,
                rejection_reasons={},
                top_score=1.0 if scored_candidates else 0.0,
                median_score=1.0 if scored_candidates else 0.0,
                min_score=1.0 if scored_candidates else 0.0,
                duration_ms=0.0,
            )

        if effective_ablation.skip_contract_memory:
            current_contract: dict[str, dict[str, Any]] = {}
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

        composition_policy = self._expand_final_context_items(
            resolved_policy,
            item_count=len(scored_candidates),
        )
        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                message_text=message_text,
                retrieval_plan=retrieval_plan,
                scored_candidates=scored_candidates,
                current_contract=current_contract,
                workspace_rollup=effective_workspace_rollup,
                user_state=user_state,
                resolved_policy=composition_policy,
                conversation_messages=transcript,
            ),
        )
        if effective_ablation.skip_contract_memory:
            composed_context = self._without_contract_block(composed_context)

        if trace is not None:
            composition_elapsed = (perf_counter() - composition_start) * 1000.0
            trace.composition = self._build_composition_trace(
                composed_context,
                resolved_policy,
                composition_elapsed,
            )
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0

        return PipelineResult(
            detected_needs=[],
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=True,
            degraded_mode=False,
        )

    @staticmethod
    def _expand_final_context_items(
        resolved_policy: ResolvedPolicy,
        *,
        item_count: int,
    ) -> ResolvedPolicy:
        """Allow the composer to emit every eligible small-corpus memory."""
        if item_count <= resolved_policy.retrieval_params.final_context_items:
            return resolved_policy
        expanded_retrieval = resolved_policy.retrieval_params.model_copy(
            update={"final_context_items": item_count}
        )
        return resolved_policy.model_copy(update={"retrieval_params": expanded_retrieval})

    @staticmethod
    def _estimate_tokens_from_chars(char_count: int) -> int:
        if char_count <= 0:
            return 0
        return max(1, (char_count + 3) // 4)

    @staticmethod
    def _score_small_corpus_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        """Produce neutral scored candidates for the small-corpus shortcut."""
        scored: list[ScoredCandidate] = []
        for candidate in candidates:
            scored.append(
                ScoredCandidate(
                    memory_id=str(candidate["id"]),
                    memory_object=dict(candidate),
                    llm_applicability=1.0,
                    retrieval_score=1.0,
                    vitality_boost=0.0,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=1.0,
                )
            )
        return scored

    @staticmethod
    def _merge_candidates(
        base: list[dict[str, Any]],
        enriched: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge base and enriched candidate lists, deduping by memory_id.

        Preserves the entry with the higher ``rrf_score`` when a memory
        appears in both lists. Base comes first so its ordering survives
        ties, which keeps degraded-mode outputs deterministic.
        """
        if not enriched:
            return list(base)
        if not base:
            return list(enriched)
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for candidate in base:
            memory_id = str(candidate["id"])
            merged[memory_id] = candidate
            order.append(memory_id)
        for candidate in enriched:
            memory_id = str(candidate["id"])
            existing = merged.get(memory_id)
            if existing is None:
                merged[memory_id] = candidate
                order.append(memory_id)
                continue
            existing_score = float(existing.get("rrf_score") or 0.0)
            candidate_score = float(candidate.get("rrf_score") or 0.0)
            if candidate_score > existing_score:
                merged[memory_id] = candidate
        return [merged[memory_id] for memory_id in order]

    async def _build_plan(
        self,
        *,
        message_text: str,
        query_intelligence: QueryIntelligenceResult,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        cold_start: bool,
        ablation: AblationConfig,
    ) -> RetrievalPlan:
        plan = self._planner.build_plan(
            original_query=message_text,
            query_intelligence=query_intelligence,
            conversation_context=conversation_context,
            resolved_policy=resolved_policy,
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
    ) -> list[ScoredCandidate]:
        """Fallback scoring path used when applicability scoring is ablated.

        Used only when ``AblationConfig.skip_applicability_scoring`` is
        set. This intentionally bypasses ``ApplicabilityScorer``, which
        means benchmark runs with this ablation also lose the exact
        recall boost, vitality boost, confirmation boost, and need boost.
        That is by design: the ablation opts out of scoring as a whole
        to measure the impact of the scorer, and adding individual
        boosts back inline would muddy the signal. Exact-recall
        benchmarks that care about routing should run without this
        ablation.
        """
        scored: list[ScoredCandidate] = []
        for candidate in candidates:
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

    async def _reground_summary_support_shortlist(
        self,
        *,
        shortlist: list[dict[str, Any]],
        filtered_candidates: list[dict[str, Any]],
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        detected_needs: list[Any],
        retrieval_plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        """Add bounded support candidates for shortlist summaries when needed.

        This is an additive safeguard for specific-answer queries. When a
        hierarchical summary survives into the shortlist but its supporting
        source memories did not, we carry forward a small number of those
        source memories so the scorer/composer can ground the summary.
        """
        if not shortlist:
            return shortlist
        if retrieval_plan.query_type != "slot_fill" and not retrieval_plan.exact_recall_mode:
            return shortlist

        per_summary_cap = 2
        total_cap = 4
        shortlisted_ids = {str(candidate["id"]) for candidate in shortlist}
        promoted_ids: set[str] = set()
        promoted: list[dict[str, Any]] = []
        filtered_by_id = {
            str(candidate["id"]): candidate
            for candidate in filtered_candidates
        }

        for candidate in shortlist:
            if len(promoted) >= total_cap:
                break
            if not self._is_hierarchical_summary_candidate(candidate):
                continue
            source_ids = self._candidate_source_ids(candidate)
            if not source_ids:
                continue

            source_positions = {
                source_id: index
                for index, source_id in enumerate(source_ids)
            }
            support_candidates: list[dict[str, Any]] = []
            support_ids: set[str] = set()

            for source_id in source_ids:
                existing = filtered_by_id.get(source_id)
                if existing is None:
                    continue
                if source_id in shortlisted_ids or source_id in promoted_ids:
                    continue
                if not self._is_summary_support_candidate(existing):
                    continue
                support_candidates.append(existing)
                support_ids.add(source_id)

            remaining_ids = [
                source_id
                for source_id in source_ids
                if source_id not in filtered_by_id
                and source_id not in shortlisted_ids
                and source_id not in promoted_ids
                and source_id not in support_ids
            ]
            if remaining_ids:
                fetched_rows = await self._memory_repository.list_memory_objects_by_ids(
                    conversation_context.user_id,
                    remaining_ids,
                )
                annotated_rows = [
                    self._annotate_summary_support_candidate(row)
                    for row in fetched_rows
                    if self._is_summary_support_candidate(row)
                ]
                eligible_rows = self._scorer.filter_candidates(
                    annotated_rows,
                    resolved_policy,
                    detected_needs,
                    retrieval_plan=retrieval_plan,
                )
                for row in eligible_rows:
                    source_id = str(row["id"])
                    if source_id in shortlisted_ids or source_id in promoted_ids or source_id in support_ids:
                        continue
                    support_candidates.append(row)
                    support_ids.add(source_id)
                    filtered_by_id[source_id] = row

            if not support_candidates:
                continue

            allowed_for_summary = min(per_summary_cap, total_cap - len(promoted))
            ordered_support = self._order_summary_support_candidates(
                support_candidates,
                source_positions=source_positions,
            )
            summary_promoted = 0
            for support_candidate in ordered_support:
                support_id = str(support_candidate["id"])
                if support_id in shortlisted_ids or support_id in promoted_ids:
                    continue
                promoted.append(support_candidate)
                promoted_ids.add(support_id)
                summary_promoted += 1
                if summary_promoted >= allowed_for_summary or len(promoted) >= total_cap:
                    break

        if not promoted:
            return shortlist
        return [*shortlist, *promoted]

    @classmethod
    def _is_hierarchical_summary_candidate(cls, candidate: dict[str, Any]) -> bool:
        if str(candidate.get("object_type")) != MemoryObjectType.SUMMARY_VIEW.value:
            return False
        payload_json = candidate.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        try:
            hierarchy_level = int(payload_json.get("hierarchy_level", -1))
        except (TypeError, ValueError):
            return False
        return hierarchy_level in {1, 2}

    @classmethod
    def _is_summary_support_candidate(cls, candidate: dict[str, Any]) -> bool:
        if str(candidate.get("object_type")) != MemoryObjectType.SUMMARY_VIEW.value:
            return True
        return bool(cls._candidate_source_ids(candidate))

    @staticmethod
    def _candidate_source_ids(candidate: dict[str, Any]) -> list[str]:
        payload_json = candidate.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return []
        return [
            str(item).strip()
            for item in payload_json.get("source_object_ids", [])
            if str(item).strip()
        ]

    @staticmethod
    def _annotate_summary_support_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        annotated = dict(candidate)
        annotated.setdefault("rrf_score", 0.0)
        annotated.setdefault(
            "channel_ranks",
            {
                "fts": None,
                "embedding": None,
                "consequence": None,
                "raw_message": None,
            },
        )
        annotated.setdefault("matched_sub_queries", [])
        retrieval_sources = list(annotated.get("retrieval_sources") or [])
        if "summary_support" not in retrieval_sources:
            retrieval_sources.append("summary_support")
        annotated["retrieval_sources"] = retrieval_sources
        return annotated

    @classmethod
    def _order_summary_support_candidates(
        cls,
        candidates: list[dict[str, Any]],
        *,
        source_positions: dict[str, int],
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            deduped[str(candidate["id"])] = candidate
        return sorted(
            deduped.values(),
            key=lambda candidate: (
                1 if str(candidate.get("object_type")) == MemoryObjectType.SUMMARY_VIEW.value else 0,
                -cls._candidate_timestamp(candidate.get("updated_at")),
                -cls._normalized_retrieval_score(candidate.get("rrf_score")),
                source_positions.get(str(candidate["id"]), len(source_positions)),
                str(candidate["id"]),
            ),
        )

    @staticmethod
    def _candidate_timestamp(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    async def _compose_context(
        self,
        *,
        message_text: str,
        retrieval_plan: RetrievalPlan,
        scored_candidates: list[ScoredCandidate],
        current_contract: dict[str, dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        user_state: dict[str, Any],
        resolved_policy: ResolvedPolicy,
        conversation_messages: list[dict[str, Any]],
    ) -> ComposedContext:
        return self._context_composer.compose(
            scored_candidates=scored_candidates,
            current_contract=current_contract,
            workspace_rollup=workspace_rollup,
            user_state=user_state,
            resolved_policy=resolved_policy,
            conversation_messages=conversation_messages,
            query_text=message_text,
            query_type=retrieval_plan.query_type,
            exact_recall_mode=retrieval_plan.exact_recall_mode,
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

    # ------------------------------------------------------------------
    # Trace builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_search_trace(
        raw_candidates: list[dict[str, Any]],
        retrieval_plan: RetrievalPlan,
        duration_ms: float,
    ) -> CandidateSearchTrace:
        fts_count = 0
        verbatim_pin_count = 0
        artifact_chunk_count = 0
        embedding_count = 0
        consequence_count = 0
        raw_message_count = 0
        for candidate in raw_candidates:
            channel_ranks = candidate.get("channel_ranks") or {}
            if channel_ranks.get("verbatim_pin") is not None or candidate.get("is_verbatim_pin"):
                verbatim_pin_count += 1
            if channel_ranks.get("artifact_chunk") is not None or candidate.get("is_artifact_chunk"):
                artifact_chunk_count += 1
            if channel_ranks.get("fts") is not None:
                fts_count += 1
            if channel_ranks.get("embedding") is not None:
                embedding_count += 1
            if channel_ranks.get("consequence") is not None:
                consequence_count += 1
            if (
                channel_ranks.get("raw_message") is not None
                or candidate.get("is_raw_message_window")
            ):
                raw_message_count += 1
        total_before_fusion = (
            fts_count + artifact_chunk_count + embedding_count + consequence_count + raw_message_count
        )
        per_subquery_counts: list[SubQuerySearchCount] = []
        for sub_query in retrieval_plan.sub_query_plans:
            sub_verbatim = 0
            sub_artifact = 0
            sub_fts = 0
            sub_emb = 0
            sub_raw = 0
            for candidate in raw_candidates:
                matched = candidate.get("matched_sub_queries") or []
                if sub_query.text in matched:
                    channel_ranks = candidate.get("channel_ranks") or {}
                    if channel_ranks.get("verbatim_pin") is not None or candidate.get("is_verbatim_pin"):
                        sub_verbatim += 1
                    if channel_ranks.get("artifact_chunk") is not None or candidate.get("is_artifact_chunk"):
                        sub_artifact += 1
                    if channel_ranks.get("fts") is not None:
                        sub_fts += 1
                    if channel_ranks.get("embedding") is not None:
                        sub_emb += 1
                    if (
                        channel_ranks.get("raw_message") is not None
                        or candidate.get("is_raw_message_window")
                    ):
                        sub_raw += 1
            per_subquery_counts.append(SubQuerySearchCount(
                subquery=sub_query.text,
                verbatim_pin=sub_verbatim,
                artifact_chunk=sub_artifact,
                fts=sub_fts,
                embedding=sub_emb,
                raw_message=sub_raw,
            ))
        return CandidateSearchTrace(
            fts_candidates_count=fts_count,
            verbatim_pin_candidates_count=verbatim_pin_count,
            artifact_chunk_candidates_count=artifact_chunk_count,
            embedding_candidates_count=embedding_count,
            consequence_candidates_count=consequence_count,
            raw_message_candidates_count=raw_message_count,
            entity_candidates_count=0,
            total_before_fusion=total_before_fusion,
            total_after_fusion=len(raw_candidates),
            per_subquery_counts=per_subquery_counts,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_scoring_trace(
        raw_candidates: list[dict[str, Any]],
        filtered_candidates: list[dict[str, Any]],
        scored_candidates: list[ScoredCandidate],
        duration_ms: float,
    ) -> ScoringTrace:
        candidates_received = len(raw_candidates)
        candidates_rejected = candidates_received - len(filtered_candidates)
        scores = [candidate.final_score for candidate in scored_candidates]
        top_score = max(scores) if scores else 0.0
        min_passing = min(scores) if scores else 0.0
        if scores:
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            median_score = (
                sorted_scores[mid]
                if len(sorted_scores) % 2 == 1
                else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
            )
        else:
            median_score = 0.0
        return ScoringTrace(
            candidates_received=candidates_received,
            candidates_scored=len(scored_candidates),
            candidates_rejected=candidates_rejected,
            rejection_reasons={},
            top_score=top_score,
            median_score=median_score,
            min_score=min_passing,
            duration_ms=duration_ms,
        )

    def _build_composition_trace(
        self,
        composed_context: ComposedContext,
        resolved_policy: ResolvedPolicy,
        duration_ms: float,
    ) -> CompositionTrace:
        contract_tokens = self._context_composer.estimate_tokens(composed_context.contract_block)
        workspace_tokens = self._context_composer.estimate_tokens(composed_context.workspace_block)
        memory_tokens = self._context_composer.estimate_tokens(composed_context.memory_block)
        state_tokens = self._context_composer.estimate_tokens(composed_context.state_block)
        return CompositionTrace(
            candidates_selected=composed_context.items_included,
            token_budget_total=resolved_policy.context_budget_tokens,
            token_budget_used=composed_context.total_tokens_estimate,
            contract_tokens=contract_tokens,
            workspace_tokens=workspace_tokens,
            memory_tokens=memory_tokens,
            state_tokens=state_tokens,
            diversity_penalties_applied=0,
            support_level="UNKNOWN",
            duration_ms=duration_ms,
        )

    def _without_contract_block(self, composed_context: ComposedContext) -> ComposedContext:
        contract_tokens = self._context_composer.estimate_tokens(composed_context.contract_block)
        return composed_context.model_copy(
            update={
                "contract_block": "",
                "total_tokens_estimate": max(0, composed_context.total_tokens_estimate - contract_tokens),
            }
        )
