"""Reusable retrieval pipeline for chat and replay flows."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import logging
from time import perf_counter
from typing import Any, Final

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.candidate_diversity import early_diversity_select
from atagia.memory.context_composer import ContextComposer
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.context_envelope import (
    ContextEnvelopeBudget,
    allocate_context_envelope_budget,
)
from atagia.memory.coverage_expander import CoverageExpansionPlan, CoverageExpander
from atagia.memory.need_detector import NeedCardCall, NeedDetector
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.memory.retrieval_diagnostics import build_retrieval_sufficiency_diagnostic
from atagia.memory.retrieval_custody import build_candidate_custody
from atagia.memory.retrieval_planner import (
    RetrievalPlanner,
    build_retrieval_fts_queries,
)
from atagia.models.schemas_memory import (
    AdaptiveGateStatus,
    CandidateSearchTrace,
    ComposedContext,
    CompositionTrace,
    CrossConversationRawPolicyTrace,
    DirectVsIndirectProvenanceTrace,
    ExtractionConversationContext,
    FacetSupportObligationTrace,
    FacetSupportTrace,
    FtsQueryExecutionCount,
    MemoryDependence,
    MemoryObjectType,
    MemoryScope,
    MemoryStatus,
    NeedTrigger,
    NeedCardCallTrace,
    NeedDetectionTrace,
    PlannedSubQuery,
    QueryIntelligenceResult,
    RuntimeAliasGroupTrace,
    RuntimeAliasSurfaceTrace,
    RetrievalParams,
    RetrievalPlan,
    RetrievalCustodyTrace,
    RetrievalSufficiencyDiagnostic,
    RetrievalSufficiencyState,
    RetrievalTrace,
    RequestRuntimeDiagnosticsTrace,
    RuntimeAnchor,
    ScoredCandidate,
    ScoringTrace,
    SubQuerySearchCount,
    SummaryViewKind,
    TemporaryScaffoldingTrace,
    TokenBudgetTrace,
    ProvenanceEvidenceTrace,
    ContentLanguageProfileTraceRow,
    UserCommunicationProfile,
    UserCommunicationProfileTrace,
)
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.llm_client import LLMClient
from atagia.services.prompt_authority import (
    effective_allow_private_for_sql_repository,
    privacy_sql_filters_disabled,
    process_authority_context,
)


logger = logging.getLogger(__name__)
PROFILE_TOP_N: Final[int] = 5
SOURCE_MESSAGE_FETCH_LIMIT: Final[int] = 80
COVERAGE_SOURCE_MESSAGE_LIMIT: Final[int] = 8
COVERAGE_CANDIDATES_PER_SOURCE_MESSAGE: Final[int] = 3
_INSUFFICIENT_RETRIEVAL_SUFFICIENCY_STATES: Final[
    frozenset[RetrievalSufficiencyState]
] = frozenset(
    {
        "retrieval_insufficient",
        "insufficient_no_candidates",
        "insufficient_no_scored_candidates",
        "insufficient_need_more_raw_evidence",
        "insufficient_need_artifact",
        "insufficient_summary_support",
    }
)
COVERAGE_CANDIDATE_LIMIT: Final[int] = 12
COVERAGE_INHERITED_SCORE_CAP: Final[float] = 0.70
COVERAGE_INHERITED_SCORE_FLOOR: Final[float] = 0.05
SUMMARY_SOURCE_WINDOW_PER_RUN_LIMIT: Final[int] = 4
SUMMARY_SOURCE_WINDOW_SIZE: Final[int] = 7
LLM_COVERAGE_CANDIDATE_LIMIT: Final[int] = 12
LLM_COVERAGE_MAX_SUBQUERIES: Final[int] = 3
PRIVACY_OFF_RETRIEVAL_STATUSES: Final[tuple[MemoryStatus, ...]] = (
    MemoryStatus.ACTIVE,
    MemoryStatus.REVIEW_REQUIRED,
    MemoryStatus.PENDING_USER_CONFIRMATION,
)


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


def _build_need_card_call_traces(
    calls: list[NeedCardCall],
) -> list[NeedCardCallTrace]:
    return [
        NeedCardCallTrace(
            card_name=call.card_name,
            model=call.model,
            prompt=call.prompt,
            raw_output=call.raw_output,
            parsed=call.parsed,
            parse_valid=call.parse_valid,
            error=call.error,
        )
        for call in calls
    ]


def _build_runtime_alias_groups(
    anchors: list[RuntimeAnchor],
) -> list[RuntimeAliasGroupTrace]:
    groups: list[RuntimeAliasGroupTrace] = []
    for anchor in anchors:
        if not anchor.aliases:
            continue
        groups.append(
            RuntimeAliasGroupTrace(
                sub_query_text=anchor.sub_query_text,
                anchor_type=anchor.anchor_type,
                original_surface=anchor.original_surface,
                normalized_surface=anchor.normalized_surface,
                preserve_verbatim=anchor.preserve_verbatim,
                anchor_confidence=anchor.confidence,
                anchor_non_evidential=anchor.non_evidential,
                aliases=[
                    RuntimeAliasSurfaceTrace(
                        surface=alias.surface,
                        alias_kind=alias.alias_kind,
                        alias_language=alias.alias_language,
                        confidence=alias.confidence,
                        non_evidential=alias.non_evidential,
                    )
                    for alias in anchor.aliases
                ],
            )
        )
    return groups


def _build_content_language_profile_trace(
    rows: list[dict[str, Any]],
) -> list[ContentLanguageProfileTraceRow]:
    trace_rows: list[ContentLanguageProfileTraceRow] = []
    for row in rows:
        language_code = str(row.get("language_code") or "").strip()
        if not language_code:
            continue
        trace_rows.append(
            ContentLanguageProfileTraceRow(
                language_code=language_code,
                memory_count=int(row.get("memory_count") or 0),
                last_seen_at=(
                    str(row.get("last_seen_at")).strip()
                    if row.get("last_seen_at") is not None
                    else None
                ),
            )
        )
    return trace_rows


def _build_user_communication_profile_trace(
    profile: UserCommunicationProfile | None,
) -> UserCommunicationProfileTrace | None:
    if profile is None:
        return None
    return UserCommunicationProfileTrace(
        profile_kind=profile.profile_kind,
        profile_version=profile.profile_version,
        stale=profile.stale,
        observed_language_codes=[
            row.language_code for row in profile.observed_user_languages
        ],
        preference_language_codes=[
            row.language_code for row in profile.explicit_language_preferences
        ],
        ability_language_codes=[
            row.language_code for row in profile.explicit_language_abilities
        ],
        contextual_norm_language_codes=[
            row.language_code for row in profile.contextual_norms
        ],
    )


def _effective_character_id(
    conversation_context: ExtractionConversationContext,
) -> str | None:
    return (
        conversation_context.character_id
        if conversation_context.character_id is not None
        else conversation_context.workspace_id
    )


def _safe_trace_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


_PLANNER_SCAFFOLDING_BY_FTS_KIND: Final[dict[str, TemporaryScaffoldingTrace]] = {
    "must_keep_tail_and": TemporaryScaffoldingTrace(
        component="retrieval_planner",
        mechanism="must_keep_tail_exact_recall_backoff",
        trace_flag="fts_query_kind:must_keep_tail_and",
        intended_metric="critical_evidence_raw_candidate_recall_for_exact_values",
        replacement_architecture=(
            "materialized retrieval surfaces and calibrated exact-recall route planner"
        ),
        retirement_condition=(
            "retire when retained replay preserves exact-value raw recall without "
            "this FTS backoff"
        ),
    ),
    "non_evidential_person_anchor_backoff_prefix": TemporaryScaffoldingTrace(
        component="retrieval_planner",
        mechanism="person_anchor_prefix_backoff",
        trace_flag="fts_query_kind:non_evidential_person_anchor_backoff_prefix",
        intended_metric="critical_evidence_raw_candidate_recall_for_person_anchored_exact_recall",
        replacement_architecture=(
            "source-backed person/entity retrieval surfaces and calibrated recall lanes"
        ),
        retirement_condition=(
            "retire when source-backed surfaces or calibrated lanes recover the same "
            "person-anchored evidence without this prefix backoff"
        ),
    ),
}


def _dedupe_scaffolding_events(
    events: list[TemporaryScaffoldingTrace],
) -> list[TemporaryScaffoldingTrace]:
    deduped: list[TemporaryScaffoldingTrace] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        key = (event.component, event.mechanism, event.trace_flag)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


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
        self._memory_repository = MemoryObjectRepository(
            connection, clock, settings=self._settings
        )
        self._communication_profile_repository = CommunicationProfileRepository(
            connection, clock
        )
        self._contract_repository = ContractDimensionRepository(connection, clock)
        self._summary_repository = SummaryRepository(connection, clock)
        self._memory_evidence_repository = MemoryEvidenceRepository(connection, clock)
        self._need_detector = NeedDetector(
            llm_client=llm_client, clock=clock, settings=self._settings
        )
        self._coverage_expander = CoverageExpander(
            llm_client=llm_client, settings=self._settings
        )
        self._planner = RetrievalPlanner()
        self._candidate_search = CandidateSearch(
            connection,
            clock,
            embedding_index=embedding_index,
            settings=self._settings,
        )
        self._scorer = ApplicabilityScorer(
            llm_client=llm_client, clock=clock, settings=self._settings
        )
        self._context_composer = ContextComposer(clock)
        self._contract_projector = ContractProjector(
            llm_client=llm_client,
            clock=clock,
            message_repository=self._message_repository,
            memory_repository=self._memory_repository,
            contract_repository=self._contract_repository,
            settings=self._settings,
        )

    @staticmethod
    def _append_temporary_scaffolding(
        trace: RetrievalTrace,
        events: list[TemporaryScaffoldingTrace],
    ) -> None:
        trace.temporary_scaffolding = _dedupe_scaffolding_events(
            [*trace.temporary_scaffolding, *events]
        )

    @staticmethod
    def _planner_temporary_scaffolding(
        retrieval_plan: RetrievalPlan,
    ) -> list[TemporaryScaffoldingTrace]:
        events: list[TemporaryScaffoldingTrace] = []
        for sub_query_plan in retrieval_plan.sub_query_plans:
            for kind in sub_query_plan.fts_query_kinds:
                event = _PLANNER_SCAFFOLDING_BY_FTS_KIND.get(kind)
                if event is not None:
                    events.append(event)
        return _dedupe_scaffolding_events(events)

    async def execute(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        cold_start: bool,
        ablation: AblationConfig | None = None,
        workspace_rollup: dict[str, Any] | None = None,
        conversation_messages: list[dict[str, Any]] | None = None,
        trace: RetrievalTrace | None = None,
        adaptive_retrieval: bool = False,
    ) -> PipelineResult:
        effective_ablation = ablation or AblationConfig()
        configured_policy = self._override_policy(resolved_policy, effective_ablation)
        effective_policy = self._policy_for_sql_privacy_mode(
            configured_policy,
            effective_ablation,
        )
        effective_cold_start = (
            False if effective_policy.allow_private_sensitivity else cold_start
        )
        transcript = conversation_messages or []
        stage_timings: dict[str, float] = {}
        pipeline_start = perf_counter() if trace is not None else 0.0
        character_id = _effective_character_id(conversation_context)
        if trace is not None:
            trace.privacy_enforcement = effective_ablation.privacy_enforcement
        if effective_ablation.privacy_enforcement != "enforce":
            logger.warning(
                "retrieval_privacy_enforcement_not_enforced_for_request",
                extra={
                    "user_id": conversation_context.user_id,
                    "conversation_id": conversation_context.conversation_id,
                    "privacy_enforcement": effective_ablation.privacy_enforcement,
                },
            )

        # The empty-clean shortcut must still surface a seeded character/
        # workspace rollup on a cold first turn, so resolve the effective
        # rollup once up front and feed it into both the classification and
        # the fast-path composition. On warm turns the empty-clean guard short
        # circuits before this lookup runs.
        empty_clean_candidate = (
            effective_cold_start
            and workspace_rollup is None
            and not conversation_context.recent_messages
            and self._transcript_is_first_user_turn(transcript, message_text)
        )
        empty_clean_rollup: dict[str, Any] | None = None
        if empty_clean_candidate:
            empty_clean_rollup = await self._measure_stage(
                stage_timings,
                "workspace_rollup_lookup",
                self._resolve_workspace_rollup(
                    conversation_context=conversation_context,
                    ablation=effective_ablation,
                ),
            )
            if await self._is_empty_clean_turn(
                conversation_context=conversation_context,
                resolved_policy=effective_policy,
                ablation=effective_ablation,
            ):
                return await self._execute_empty_clean_turn(
                    message_text=message_text,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    effective_ablation=effective_ablation,
                    transcript=transcript,
                    workspace_rollup=empty_clean_rollup,
                    stage_timings=stage_timings,
                    trace=trace,
                    pipeline_start=pipeline_start,
                )

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
                audit_policy=configured_policy,
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
                cold_start=effective_cold_start,
                ablation=effective_ablation,
            ),
        )
        base_fts_query_audit: list[dict[str, Any]] = []

        # Wave 1-B: enrichment lane — try need detection and an enriched
        # search. Failures log, mark the result as degraded, and fall back
        # to the base candidates. Ablation skip is a deliberate bypass, not a
        # degradation.
        degraded_mode = False
        detected_needs: list[Any] = []
        enriched_candidates: list[dict[str, Any]] = []
        enriched_plan: RetrievalPlan | None = None
        enriched_fts_query_audit: list[dict[str, Any]] = []
        query_intelligence: QueryIntelligenceResult = _default_query_intelligence(
            message_text
        )
        runtime_alias_groups: list[RuntimeAliasGroupTrace] = []
        content_language_profile: list[dict[str, Any]] = []
        user_communication_profile: UserCommunicationProfile | None = None

        if effective_ablation.skip_need_detection:
            # No need-detection LLM call to overlap with, so base search runs
            # sequentially as the sole DB user.
            base_candidates = await self._measure_stage(
                stage_timings,
                "base_candidate_search",
                self._candidate_search.search(
                    base_plan,
                    conversation_context.user_id,
                    fts_query_audit=base_fts_query_audit,
                ),
            )
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
                    user_communication_profile=None,
                    degraded_mode=False,
                    duration_ms=0.0,
                )
        else:
            # Overlap A: the content-language and user-communication profiles
            # feed the need-detection prompt, so they must finish first; they
            # run sequentially as the sole DB user. Once they are done, base
            # candidate search is launched as a task (the only DB toucher in
            # this window) so its SQL runs concurrently with the pure-LLM
            # need-detection round-trip.
            content_language_profile = await self._safe_content_language_profile(
                conversation_context=conversation_context,
                resolved_policy=configured_policy,
            )
            user_communication_profile = (
                await self._safe_user_communication_profile(
                    conversation_context=conversation_context,
                )
            )
            base_search_started_at = perf_counter()
            base_search_task = asyncio.create_task(
                self._candidate_search.search(
                    base_plan,
                    conversation_context.user_id,
                    fts_query_audit=base_fts_query_audit,
                )
            )
            need_card_calls: list[NeedCardCall] = []
            need_start = perf_counter() if trace is not None else 0.0
            try:
                try:
                    query_intelligence = await self._measure_stage(
                        stage_timings,
                        "need_detection",
                        self._need_detector.detect(
                            message_text=message_text,
                            role="user",
                            conversation_context=conversation_context,
                            resolved_policy=effective_policy,
                            content_language_profile=content_language_profile,
                            user_communication_profile=user_communication_profile,
                            prompt_authority_context=process_authority_context(
                                privacy_enforcement=effective_ablation.privacy_enforcement,
                                user_id=conversation_context.user_id,
                                privilege_level=(
                                    "atagia_master"
                                    if effective_ablation.privacy_enforcement == "off"
                                    else None
                                ),
                                is_atagia_master=(
                                    effective_ablation.privacy_enforcement == "off"
                                ),
                                purpose="need_detection",
                            ),
                            card_call_trace_sink=need_card_calls,
                        ),
                    )
                except Exception as exc:
                    # Need detection failed: keep the existing degraded
                    # fallback. The base-search task is still in flight; it is
                    # joined below so the base candidates feed the fallback.
                    degraded_mode = True
                    stage_timings.setdefault(
                        "need_detection", (perf_counter() - need_start) * 1000.0
                    )
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
                            content_language_profile=_build_content_language_profile_trace(
                                content_language_profile,
                            ),
                            user_communication_profile=_build_user_communication_profile_trace(
                                user_communication_profile,
                            ),
                            card_calls=_build_need_card_call_traces(need_card_calls),
                        )
                else:
                    detected_needs = list(query_intelligence.needs)
                    runtime_alias_groups = _build_runtime_alias_groups(
                        list(query_intelligence.anchors)
                    )
                    if trace is not None:
                        need_elapsed = (perf_counter() - need_start) * 1000.0
                        trace.need_detection = NeedDetectionTrace(
                            detected_needs=[
                                need.need_type.value
                                for need in query_intelligence.needs
                            ],
                            sub_queries=list(query_intelligence.sub_queries),
                            sparse_hints=[
                                hint.fts_phrase or hint.sub_query_text
                                for hint in query_intelligence.sparse_query_hints
                            ],
                            query_language=query_intelligence.query_language,
                            answer_language=query_intelligence.answer_language,
                            memory_dependence=query_intelligence.memory_dependence,
                            content_language_profile=_build_content_language_profile_trace(
                                content_language_profile,
                            ),
                            user_communication_profile=_build_user_communication_profile_trace(
                                user_communication_profile,
                            ),
                            anchors=list(query_intelligence.anchors),
                            alias_groups=runtime_alias_groups,
                            query_type=query_intelligence.query_type,
                            raw_context_access_mode=query_intelligence.raw_context_access_mode,
                            answer_shape=query_intelligence.answer_shape,
                            coverage_mode=query_intelligence.coverage_mode,
                            source_precision=query_intelligence.source_precision,
                            temporal_range=(
                                f"{query_intelligence.temporal_range.start.isoformat()}/{query_intelligence.temporal_range.end.isoformat()}"
                                if query_intelligence.temporal_range is not None
                                else None
                            ),
                            retrieval_levels=list(query_intelligence.retrieval_levels),
                            degraded_mode=False,
                            duration_ms=need_elapsed,
                            exact_recall_needed=bool(
                                query_intelligence.exact_recall_needed
                            ),
                            exact_facets=[
                                facet.value
                                for facet in query_intelligence.exact_facets
                            ],
                            temporary_scaffolding=query_intelligence.temporary_scaffolding,
                            card_calls=_build_need_card_call_traces(need_card_calls),
                        )
                # Need detection (LLM) has finished. Join the base-search task
                # NOW, before any further DB work (enriched planning/search)
                # begins, so only one coroutine ever touches the DB at a time.
                base_candidates = await self._await_measured_task(
                    stage_timings,
                    "base_candidate_search",
                    base_search_task,
                    base_search_started_at,
                )
                self._log_stage_overlaps(
                    conversation_context,
                    {"base_candidate_search": "need_detection"},
                )
            finally:
                # Safety net: if need detection raised something that escaped
                # the handled fallback (e.g. cancellation) or the join was
                # interrupted, never leave the base-search task orphaned. This
                # is a no-op once the task has already been awaited above.
                await self._drain_overlap_task(base_search_task)
            # Adaptive retrieval gate (decision point). Need detection succeeded
            # and the base-search task is joined; the expensive enriched
            # planning/search/scoring/composition stages have NOT started. When
            # the flag is on, the turn is non-degraded, and the classification
            # is world/conversation (no dependence on stored memory), short-
            # circuit them entirely. Exact recall is a stronger signal than a
            # coarse world/conversation classification: if another card says the
            # answer needs a specific remembered detail, retrieve. Degraded mode
            # never reaches here with gate authority because the gate requires
            # ``not degraded_mode``, and the MIXED/PERSONAL classes fall through
            # to full retrieval (uncertainty -> retrieve).
            gate_skip = (
                adaptive_retrieval
                and not degraded_mode
                and not query_intelligence.exact_recall_needed
                and query_intelligence.memory_dependence
                in (MemoryDependence.WORLD, MemoryDependence.CONVERSATION)
            )
            if gate_skip:
                logger.info(
                    "adaptive_gate_skip user_id=%s conversation_id=%s "
                    "classification=%s discarded_base_candidates=%d",
                    conversation_context.user_id,
                    conversation_context.conversation_id,
                    query_intelligence.memory_dependence.value,
                    len(base_candidates),
                )
                return await self._execute_adaptive_gate_skip(
                    message_text=message_text,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    effective_ablation=effective_ablation,
                    transcript=transcript,
                    query_intelligence=query_intelligence,
                    detected_needs=detected_needs,
                    stage_timings=stage_timings,
                    trace=trace,
                    pipeline_start=pipeline_start,
                )
            if trace is not None:
                trace.raw_context_access_mode = (
                    query_intelligence.raw_context_access_mode
                )
            enriched_plan = await self._measure_stage(
                stage_timings,
                "enriched_planning",
                self._build_plan(
                    message_text=message_text,
                    query_intelligence=query_intelligence,
                    conversation_context=conversation_context,
                    resolved_policy=effective_policy,
                    cold_start=effective_cold_start,
                    ablation=effective_ablation,
                ),
            )
            enriched_candidates = await self._measure_stage(
                stage_timings,
                "enriched_candidate_search",
                self._candidate_search.search(
                    enriched_plan,
                    conversation_context.user_id,
                    fts_query_audit=enriched_fts_query_audit,
                    runtime_alias_groups=runtime_alias_groups,
                ),
            )

        retrieval_plan = enriched_plan or base_plan
        if trace is not None:
            self._append_temporary_scaffolding(
                trace,
                [
                    *query_intelligence.temporary_scaffolding,
                    *self._planner_temporary_scaffolding(retrieval_plan),
                ],
            )
        fts_query_audit = (
            enriched_fts_query_audit
            if enriched_plan is not None
            else base_fts_query_audit
        )
        raw_candidates = self._merge_candidates(base_candidates, enriched_candidates)
        coverage_candidates = await self._measure_stage(
            stage_timings,
            "coverage_candidate_expansion",
            self._source_message_coverage_candidates(
                raw_candidates=raw_candidates,
                conversation_context=conversation_context,
                retrieval_plan=retrieval_plan,
            ),
        )
        raw_candidates = self._merge_candidates(raw_candidates, coverage_candidates)
        llm_coverage_candidates = await self._measure_stage(
            stage_timings,
            "llm_coverage_expansion",
            self._llm_coverage_expansion_candidates(
                message_text=message_text,
                raw_candidates=raw_candidates,
                conversation_context=conversation_context,
                retrieval_plan=retrieval_plan,
                ablation=effective_ablation,
            ),
        )
        raw_candidates = self._merge_candidates(raw_candidates, llm_coverage_candidates)
        # Regrounding is decided by the winning plan (enriched when available)
        # so the base search does not inject derived memories when high-stakes
        # needs require direct evidence.
        pre_regrounding_candidates = list(raw_candidates)
        raw_candidates = self._apply_regrounding_requirements(
            raw_candidates, retrieval_plan
        )
        regrounding_filter_reasons = self._regrounding_filter_reasons(
            pre_regrounding_candidates,
            raw_candidates,
        )
        audit_plan = self._plan_for_policy_audit(retrieval_plan, configured_policy)
        candidate_policy_filter_reasons = self._candidate_filter_reasons(
            raw_candidates,
            configured_policy,
            detected_needs,
            audit_plan,
        )
        filter_reasons_by_id = self._actual_filter_reasons(
            regrounding_filter_reasons,
            candidate_policy_filter_reasons,
            privacy_enforcement=effective_ablation.privacy_enforcement,
        )
        # Aggregate the two lanes into single "planning" and "candidate_search"
        # keys so downstream telemetry stays consistent across pipeline variants.
        stage_timings["planning"] = stage_timings.get(
            "base_planning", 0.0
        ) + stage_timings.get("enriched_planning", 0.0)
        candidate_total_ms = (
            stage_timings.get("base_candidate_search", 0.0)
            + stage_timings.get("enriched_candidate_search", 0.0)
            + stage_timings.get("coverage_candidate_expansion", 0.0)
            + stage_timings.get("llm_coverage_expansion", 0.0)
        )
        stage_timings["candidate_search"] = candidate_total_ms
        if trace is not None:
            trace.candidate_search = self._build_candidate_search_trace(
                raw_candidates,
                retrieval_plan,
                candidate_total_ms,
                fts_query_audit,
            )

        if trace is not None:
            trace.policy_filter_audit = self._build_policy_filter_audit(
                effective_ablation.privacy_enforcement,
                candidate_policy_filter_reasons,
            )

        filtered_candidates = self._filter_candidates_for_policy_mode(
            raw_candidates,
            effective_policy,
            detected_needs,
            retrieval_plan=retrieval_plan,
            privacy_enforcement=effective_ablation.privacy_enforcement,
        )
        scoring_policy = self._expand_recall_or_recovery_scoring_budget(
            effective_policy,
            retrieval_plan,
            degraded_mode=degraded_mode,
            detected_needs=detected_needs,
            item_count=len(filtered_candidates),
        )
        scoring_policy = self._policy_for_late_privacy_mode(
            scoring_policy,
            effective_ablation,
        )
        shortlist = early_diversity_select(
            filtered_candidates,
            query_type=retrieval_plan.query_type,
            shortlist_k=scoring_policy.retrieval_params.rerank_top_k,
        )
        shortlist = await self._reground_summary_support_shortlist(
            shortlist=shortlist,
            filtered_candidates=filtered_candidates,
            conversation_context=conversation_context,
            resolved_policy=scoring_policy,
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
        )
        shortlist = await self._reground_summary_source_window_shortlist(
            shortlist=shortlist,
            conversation_context=conversation_context,
            retrieval_plan=retrieval_plan,
            query_text=message_text,
        )

        # Overlap B: hoist the contract/state/workspace-rollup lookups so their
        # SQL runs concurrently with the applicability-scoring LLM call. The
        # lookups are bundled into ONE coroutine that touches the DB
        # sequentially, making it the sole DB owner for this window; the scoring
        # call is pure LLM (no DB access). None of the three lookups read
        # ``scored_candidates``, so they are safe to start before scoring.
        lookups_task = asyncio.create_task(
            self._post_scoring_lookups(
                conversation_context=conversation_context,
                effective_policy=effective_policy,
                effective_ablation=effective_ablation,
                character_id=character_id,
                stage_timings=stage_timings,
            )
        )

        scoring_start = perf_counter() if trace is not None else 0.0
        try:
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
                        resolved_policy=scoring_policy,
                        detected_needs=detected_needs,
                        retrieval_plan=retrieval_plan,
                        trace=trace,
                        applicability_gate_mode=(
                            effective_ablation.applicability_gate_mode
                        ),
                    ),
                )
        except BaseException:
            # Scoring failed: drain the overlap task so it is never orphaned,
            # then re-raise the scoring error (fail fast).
            await self._drain_overlap_task(lookups_task)
            raise
        if trace is not None:
            scoring_elapsed = (perf_counter() - scoring_start) * 1000.0
            trace.scoring = self._build_scoring_trace(
                pre_regrounding_candidates,
                filtered_candidates,
                scored_candidates,
                scoring_elapsed,
                rejection_reasons=self._rejection_reason_counts(filter_reasons_by_id),
                policy_audit_reason_counts=self._rejection_reason_counts(
                    candidate_policy_filter_reasons
                ),
                scoring_candidates=shortlist,
            )

        retrieval_sufficiency = build_retrieval_sufficiency_diagnostic(
            raw_candidates=pre_regrounding_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=shortlist,
            scored_candidates=scored_candidates,
            retrieval_plan=retrieval_plan,
            contradiction_tension_threshold=self._settings.belief_tension_threshold,
        )
        if trace is not None:
            trace.retrieval_sufficiency = retrieval_sufficiency

        # Join the Overlap B lookups. By this point the only awaits since the
        # task was created were the (DB-free) scoring call and pure CPU trace
        # work, so the lookups task was the sole DB owner throughout.
        (
            current_contract,
            user_state,
            effective_workspace_rollup,
        ) = await lookups_task
        self._log_stage_overlaps(
            conversation_context,
            {
                "contract_lookup": "applicability_scoring",
                "state_lookup": "applicability_scoring",
                "workspace_rollup_lookup": "applicability_scoring",
            },
        )

        composition_policy = self._expand_recall_or_recovery_context_items(
            scoring_policy,
            retrieval_plan,
            degraded_mode=degraded_mode,
            detected_needs=detected_needs,
            item_count=len(scored_candidates),
        )
        composition_policy = self._expand_exhaustive_coverage_budget(
            composition_policy,
            retrieval_plan,
            scored_candidates,
        )
        composition_policy = self._cap_explicit_final_context_items(
            composition_policy,
            effective_ablation,
        )
        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                user_id=conversation_context.user_id,
                message_text=message_text,
                retrieval_plan=retrieval_plan,
                scored_candidates=scored_candidates,
                current_contract=current_contract,
                workspace_rollup=effective_workspace_rollup,
                user_state=user_state,
                resolved_policy=self._policy_for_late_privacy_mode(
                    composition_policy,
                    effective_ablation,
                ),
                conversation_messages=transcript,
                composer_strategy=effective_ablation.composer_strategy,
                enable_evidence_obligation_coverage=(
                    effective_ablation.enable_evidence_obligation_coverage
                ),
                enable_evidence_packets=effective_ablation.enable_evidence_packets,
                enable_final_answer_evidence_pack=(
                    effective_ablation.enable_final_answer_evidence_pack
                ),
            ),
        )
        if effective_ablation.skip_contract_memory:
            composed_context = self._without_contract_block(composed_context)

        candidate_custody = build_candidate_custody(
            raw_candidates=pre_regrounding_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=shortlist,
            scored_candidates=scored_candidates,
            selected_memory_ids=list(composed_context.selected_memory_ids),
            retrieval_plan=retrieval_plan,
            filter_reasons_by_id=filter_reasons_by_id,
        )

        if trace is not None:
            composition_elapsed = (perf_counter() - composition_start) * 1000.0
            trace.composition = self._build_composition_trace(
                composed_context,
                composition_policy,
                composition_elapsed,
            )
            self._attach_stability_diagnostics(
                trace=trace,
                retrieval_plan=retrieval_plan,
                candidate_custody=candidate_custody,
                candidate_rows=[
                    *pre_regrounding_candidates,
                    *filtered_candidates,
                    *shortlist,
                    *(candidate.memory_object for candidate in scored_candidates),
                ],
                composed_context=composed_context,
            )
            trace.custody = self._build_custody_trace(
                raw_candidates=pre_regrounding_candidates,
                filtered_candidates=filtered_candidates,
                scored_candidates=scored_candidates,
                candidate_custody=candidate_custody,
                selected_memory_ids=list(composed_context.selected_memory_ids),
                retrieval_sufficiency=retrieval_sufficiency,
            )
            trace.runtime_diagnostics = RequestRuntimeDiagnosticsTrace(
                stage_timings_ms=self._nonnegative_float_map(stage_timings),
                hydration_timings_ms=self._hydration_timings(stage_timings),
            )
            trace.degraded_mode = degraded_mode
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0

        # Full retrieval ran. When the flag is on the gate had authority but
        # kept retrieval (PERSONAL/MIXED, or WORLD/CONVERSATION under degraded
        # mode where the gate has no authority); when the flag is off the
        # classification is recorded but no action was taken (shadow mode).
        return PipelineResult(
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            candidate_custody=candidate_custody,
            retrieval_sufficiency=retrieval_sufficiency,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=False,
            degraded_mode=degraded_mode,
            adaptive_gate_status=(
                AdaptiveGateStatus.RETRIEVED
                if adaptive_retrieval
                else AdaptiveGateStatus.OFF_SHADOW
            ),
            adaptive_gate_classification=query_intelligence.memory_dependence,
        )

    async def _post_scoring_lookups(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        effective_policy: ResolvedRetrievalPolicy,
        effective_ablation: AblationConfig,
        character_id: str | None,
        stage_timings: dict[str, float],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, Any],
        dict[str, Any] | None,
    ]:
        """Run the contract, state, and workspace-rollup lookups sequentially.

        This coroutine is the *sole* DB toucher during Overlap B: it runs
        concurrently with the applicability-scoring LLM call. The three lookups
        share one aiosqlite connection, so they must run one after another
        (never as concurrent SQL). None of them consume ``scored_candidates`` —
        they only need ``conversation_context`` — so they are safe to start
        before scoring completes.
        """
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
                    user_persona_id=conversation_context.user_persona_id,
                    platform_id=conversation_context.platform_id,
                    character_id=character_id,
                    incognito=conversation_context.incognito,
                    remember_across_chats=conversation_context.remember_across_chats,
                    remember_across_devices=conversation_context.remember_across_devices,
                    sensitivity_gates_enabled=privacy_sql_filters_disabled(
                        effective_ablation,
                    ),
                    allow_private_sensitivity=effective_allow_private_for_sql_repository(
                        effective_policy,
                        effective_ablation,
                    ),
                    active_space_id=conversation_context.active_space_id,
                    active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                    active_mind_id=conversation_context.active_mind_id,
                    mind_topology=conversation_context.mind_topology,
                    active_embodiment_id=conversation_context.active_embodiment_id,
                    active_realm_id=conversation_context.active_realm_id,
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
                allow_intimacy_context=effective_policy.allow_intimacy_context,
                user_persona_id=conversation_context.user_persona_id,
                platform_id=conversation_context.platform_id,
                character_id=character_id,
                incognito=conversation_context.incognito,
                remember_across_chats=conversation_context.remember_across_chats,
                remember_across_devices=conversation_context.remember_across_devices,
                sensitivity_gates_enabled=privacy_sql_filters_disabled(
                    effective_ablation,
                ),
                active_space_id=conversation_context.active_space_id,
                active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                active_mind_id=conversation_context.active_mind_id,
                mind_topology=conversation_context.mind_topology,
                active_embodiment_id=conversation_context.active_embodiment_id,
                active_realm_id=conversation_context.active_realm_id,
            ),
        )

        effective_workspace_rollup = await self._measure_stage(
            stage_timings,
            "workspace_rollup_lookup",
            self._resolve_workspace_rollup(
                conversation_context=conversation_context,
                ablation=effective_ablation,
            ),
        )
        return current_contract, user_state, effective_workspace_rollup

    async def _is_empty_clean_turn(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> bool:
        """Return True when the fast path can compose context without retrieval.

        The cold-start / first-user-turn / no-recent-messages gate is evaluated
        by the caller; a seeded character/workspace rollup does not disqualify
        the fast path because it is rendered directly without any retrieval LLM
        calls. Only a visible interaction contract forces the full pipeline.
        """
        return not await self._has_visible_contract_context(
            conversation_context=conversation_context,
            resolved_policy=resolved_policy,
            ablation=ablation,
        )

    @staticmethod
    def _transcript_is_first_user_turn(
        transcript: list[dict[str, Any]],
        message_text: str,
    ) -> bool:
        if len(transcript) != 1:
            return False
        message = transcript[0]
        return (
            str(message.get("role")) == "user"
            and str(message.get("text") or "") == message_text
        )

    async def _has_visible_contract_context(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> bool:
        character_id = _effective_character_id(conversation_context)
        count = await self._contract_repository.count_for_context(
            conversation_context.user_id,
            conversation_context.assistant_mode_id,
            conversation_context.workspace_id,
            conversation_context.conversation_id,
            user_persona_id=conversation_context.user_persona_id,
            platform_id=conversation_context.platform_id,
            character_id=character_id,
            incognito=conversation_context.incognito,
            remember_across_chats=conversation_context.remember_across_chats,
            remember_across_devices=conversation_context.remember_across_devices,
            sensitivity_gates_enabled=privacy_sql_filters_disabled(ablation),
            allow_private_sensitivity=effective_allow_private_for_sql_repository(
                resolved_policy,
                ablation,
            ),
            active_space_id=conversation_context.active_space_id,
            active_space_boundary_mode=conversation_context.active_space_boundary_mode,
            active_mind_id=conversation_context.active_mind_id,
            mind_topology=conversation_context.mind_topology,
            active_embodiment_id=conversation_context.active_embodiment_id,
            active_realm_id=conversation_context.active_realm_id,
        )
        return count > 0

    async def _execute_empty_clean_turn(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        effective_ablation: AblationConfig,
        transcript: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        stage_timings: dict[str, float],
        trace: RetrievalTrace | None,
        pipeline_start: float,
    ) -> PipelineResult:
        """Compose context without any LLM-backed retrieval stages.

        A pre-resolved character/workspace rollup is still rendered here so a
        seeded greeting turn surfaces its ``[Workspace Context]`` block; only
        the retrieval LLM stages are skipped.
        """
        query_intelligence = _default_query_intelligence(message_text)
        retrieval_plan = await self._measure_stage(
            stage_timings,
            "planning",
            self._build_plan(
                message_text=message_text,
                query_intelligence=query_intelligence,
                conversation_context=conversation_context,
                resolved_policy=resolved_policy,
                cold_start=True,
                ablation=effective_ablation,
            ),
        )
        stage_timings["base_planning"] = stage_timings["planning"]
        stage_timings["base_candidate_search"] = 0.0
        stage_timings["need_detection"] = 0.0
        stage_timings["enriched_planning"] = 0.0
        stage_timings["enriched_candidate_search"] = 0.0
        stage_timings["coverage_candidate_expansion"] = 0.0
        stage_timings["llm_coverage_expansion"] = 0.0
        stage_timings["candidate_search"] = 0.0
        stage_timings["applicability_scoring"] = 0.0
        stage_timings["contract_lookup"] = 0.0
        stage_timings["state_lookup"] = 0.0
        stage_timings.setdefault("workspace_rollup_lookup", 0.0)

        current_contract: dict[str, dict[str, Any]] = {}
        user_state: dict[str, Any] = {}
        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                user_id=conversation_context.user_id,
                message_text=message_text,
                retrieval_plan=retrieval_plan,
                scored_candidates=[],
                current_contract=current_contract,
                workspace_rollup=workspace_rollup,
                user_state=user_state,
                resolved_policy=resolved_policy,
                conversation_messages=transcript,
                composer_strategy=effective_ablation.composer_strategy,
                enable_evidence_obligation_coverage=(
                    effective_ablation.enable_evidence_obligation_coverage
                ),
                enable_evidence_packets=False,
                enable_final_answer_evidence_pack=(
                    effective_ablation.enable_final_answer_evidence_pack
                ),
            ),
        )
        stage_timings["empty_clean_fast_path"] = (
            (perf_counter() - pipeline_start) * 1000.0
            if trace is not None
            else 0.0
        )

        raw_candidates: list[dict[str, Any]] = []
        filtered_candidates: list[dict[str, Any]] = []
        scored_candidates: list[ScoredCandidate] = []
        candidate_custody = build_candidate_custody(
            raw_candidates=raw_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=[],
            scored_candidates=scored_candidates,
            selected_memory_ids=list(composed_context.selected_memory_ids),
            retrieval_plan=retrieval_plan,
            filter_reasons_by_id={},
        )
        retrieval_sufficiency = build_retrieval_sufficiency_diagnostic(
            raw_candidates=raw_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=[],
            scored_candidates=scored_candidates,
            retrieval_plan=retrieval_plan,
            contradiction_tension_threshold=self._settings.belief_tension_threshold,
        )

        if trace is not None:
            trace.raw_context_access_mode = query_intelligence.raw_context_access_mode
            trace.need_detection = NeedDetectionTrace(
                detected_needs=[],
                sub_queries=[message_text],
                sparse_hints=[],
                query_type="default",
                raw_context_access_mode="normal",
                temporal_range=None,
                retrieval_levels=[0],
                user_communication_profile=None,
                degraded_mode=False,
                duration_ms=0.0,
            )
            trace.candidate_search = self._build_candidate_search_trace(
                raw_candidates,
                retrieval_plan,
                0.0,
                [],
            )
            trace.policy_filter_audit = self._build_policy_filter_audit(
                effective_ablation.privacy_enforcement,
                {},
            )
            trace.scoring = self._build_scoring_trace(
                raw_candidates,
                filtered_candidates,
                scored_candidates,
                0.0,
            )
            trace.retrieval_sufficiency = retrieval_sufficiency
            trace.composition = self._build_composition_trace(
                composed_context,
                resolved_policy,
                (perf_counter() - composition_start) * 1000.0,
            )
            trace.custody = self._build_custody_trace(
                raw_candidates=raw_candidates,
                filtered_candidates=filtered_candidates,
                scored_candidates=scored_candidates,
                candidate_custody=candidate_custody,
                selected_memory_ids=list(composed_context.selected_memory_ids),
                retrieval_sufficiency=retrieval_sufficiency,
            )
            trace.runtime_diagnostics = RequestRuntimeDiagnosticsTrace(
                stage_timings_ms=self._nonnegative_float_map(stage_timings),
                hydration_timings_ms=self._hydration_timings(stage_timings),
            )
            trace.degraded_mode = False
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0

        return PipelineResult(
            detected_needs=[],
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            candidate_custody=candidate_custody,
            retrieval_sufficiency=retrieval_sufficiency,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=False,
            degraded_mode=False,
            # This shortcut runs before need detection, so the gate never had a
            # classification to act on.
            adaptive_gate_status=AdaptiveGateStatus.NOT_APPLICABLE,
            adaptive_gate_classification=None,
        )

    async def _execute_adaptive_gate_skip(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        effective_ablation: AblationConfig,
        transcript: list[dict[str, Any]],
        query_intelligence: QueryIntelligenceResult,
        detected_needs: list[Any],
        stage_timings: dict[str, float],
        trace: RetrievalTrace | None,
        pipeline_start: float,
    ) -> PipelineResult:
        """Compose context for a gate-skipped turn (D4).

        Need detection already ran and classified the turn as not depending on
        stored memory (``world``/``conversation``). The base candidates are
        DISCARDED (injecting them is the distractor failure mode the gate exists
        to avoid). The cheap interaction-contract SQL read still runs so the
        contract block survives; state and workspace-rollup lookups are skipped
        (matching the fast path: contract only). Composition uses the REAL
        ``query_intelligence`` so ``query_language``/``answer_language`` flow
        into ``source_retrieval_plan`` and the answer-language guidance keeps
        working truthfully.
        """
        character_id = _effective_character_id(conversation_context)
        # Build the plan from the real query intelligence so the response
        # language guidance and source plan reflect the actual turn.
        retrieval_plan = await self._measure_stage(
            stage_timings,
            "enriched_planning",
            self._build_plan(
                message_text=message_text,
                query_intelligence=query_intelligence,
                conversation_context=conversation_context,
                resolved_policy=resolved_policy,
                cold_start=False,
                ablation=effective_ablation,
            ),
        )
        # Zero the stages the gate short-circuited so downstream telemetry stays
        # consistent with the full and fast variants. Base planning genuinely ran
        # for this turn, so the aggregate "planning" key sums both lanes exactly
        # like the full path (see the aggregate block in ``execute``).
        stage_timings["planning"] = stage_timings.get(
            "base_planning", 0.0
        ) + stage_timings.get("enriched_planning", 0.0)
        stage_timings["enriched_candidate_search"] = 0.0
        stage_timings["coverage_candidate_expansion"] = 0.0
        stage_timings["llm_coverage_expansion"] = 0.0
        # Base candidates were searched (timing already recorded) but are
        # discarded; the effective candidate_search cost is the base lane only.
        stage_timings["candidate_search"] = stage_timings.get(
            "base_candidate_search", 0.0
        )
        stage_timings["applicability_scoring"] = 0.0
        stage_timings["state_lookup"] = 0.0
        stage_timings["workspace_rollup_lookup"] = 0.0

        # Contract-only lookup: same stage and predicates the full path uses, so
        # the contract block admits exactly what normal retrieval would.
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
                    user_persona_id=conversation_context.user_persona_id,
                    platform_id=conversation_context.platform_id,
                    character_id=character_id,
                    incognito=conversation_context.incognito,
                    remember_across_chats=conversation_context.remember_across_chats,
                    remember_across_devices=conversation_context.remember_across_devices,
                    sensitivity_gates_enabled=privacy_sql_filters_disabled(
                        effective_ablation,
                    ),
                    allow_private_sensitivity=effective_allow_private_for_sql_repository(
                        resolved_policy,
                        effective_ablation,
                    ),
                    active_space_id=conversation_context.active_space_id,
                    active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                    active_mind_id=conversation_context.active_mind_id,
                    mind_topology=conversation_context.mind_topology,
                    active_embodiment_id=conversation_context.active_embodiment_id,
                    active_realm_id=conversation_context.active_realm_id,
                ),
            )

        user_state: dict[str, Any] = {}
        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                user_id=conversation_context.user_id,
                message_text=message_text,
                retrieval_plan=retrieval_plan,
                scored_candidates=[],
                current_contract=current_contract,
                workspace_rollup=None,
                user_state=user_state,
                resolved_policy=resolved_policy,
                conversation_messages=transcript,
                composer_strategy=effective_ablation.composer_strategy,
                enable_evidence_obligation_coverage=(
                    effective_ablation.enable_evidence_obligation_coverage
                ),
                enable_evidence_packets=False,
                enable_final_answer_evidence_pack=(
                    effective_ablation.enable_final_answer_evidence_pack
                ),
            ),
        )
        if effective_ablation.skip_contract_memory:
            composed_context = self._without_contract_block(composed_context)

        raw_candidates: list[dict[str, Any]] = []
        filtered_candidates: list[dict[str, Any]] = []
        scored_candidates: list[ScoredCandidate] = []
        candidate_custody = build_candidate_custody(
            raw_candidates=raw_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=[],
            scored_candidates=scored_candidates,
            selected_memory_ids=list(composed_context.selected_memory_ids),
            retrieval_plan=retrieval_plan,
            filter_reasons_by_id={},
        )
        retrieval_sufficiency = build_retrieval_sufficiency_diagnostic(
            raw_candidates=raw_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=[],
            scored_candidates=scored_candidates,
            retrieval_plan=retrieval_plan,
            contradiction_tension_threshold=self._settings.belief_tension_threshold,
        )

        if trace is not None:
            trace.raw_context_access_mode = query_intelligence.raw_context_access_mode
            # The real need-detection trace was already recorded during the
            # overlap window; only fill in the remaining sections.
            trace.candidate_search = self._build_candidate_search_trace(
                raw_candidates,
                retrieval_plan,
                0.0,
                [],
            )
            trace.policy_filter_audit = self._build_policy_filter_audit(
                effective_ablation.privacy_enforcement,
                {},
            )
            trace.scoring = self._build_scoring_trace(
                raw_candidates,
                filtered_candidates,
                scored_candidates,
                0.0,
            )
            trace.retrieval_sufficiency = retrieval_sufficiency
            trace.composition = self._build_composition_trace(
                composed_context,
                resolved_policy,
                (perf_counter() - composition_start) * 1000.0,
            )
            trace.custody = self._build_custody_trace(
                raw_candidates=raw_candidates,
                filtered_candidates=filtered_candidates,
                scored_candidates=scored_candidates,
                candidate_custody=candidate_custody,
                selected_memory_ids=list(composed_context.selected_memory_ids),
                retrieval_sufficiency=retrieval_sufficiency,
            )
            trace.runtime_diagnostics = RequestRuntimeDiagnosticsTrace(
                stage_timings_ms=self._nonnegative_float_map(stage_timings),
                hydration_timings_ms=self._hydration_timings(stage_timings),
            )
            trace.degraded_mode = False
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0

        return PipelineResult(
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            candidate_custody=candidate_custody,
            retrieval_sufficiency=retrieval_sufficiency,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=False,
            degraded_mode=False,
            adaptive_gate_status=AdaptiveGateStatus.SKIPPED,
            adaptive_gate_classification=query_intelligence.memory_dependence,
        )

    # ------------------------------------------------------------------
    # Wave 1-A: small-corpus shortcut
    # ------------------------------------------------------------------

    async def _is_small_corpus(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
    ) -> bool:
        """Return True if the full eligible corpus fits the small-corpus budget."""
        if resolved_policy.allow_private_sensitivity:
            return False
        threshold_ratio = self._settings.small_corpus_token_threshold_ratio
        if threshold_ratio <= 0.0:
            return False
        threshold_tokens = int(resolved_policy.context_budget_tokens * threshold_ratio)
        if threshold_tokens <= 0:
            return False
        character_id = _effective_character_id(conversation_context)
        memory_chars = await self._memory_repository.sum_canonical_text_length_for_context(
            conversation_context.user_id,
            resolved_policy.allowed_scopes,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            assistant_mode_id=conversation_context.assistant_mode_id,
            privacy_ceiling=resolved_policy.privacy_ceiling,
            allow_intimacy_context=resolved_policy.allow_intimacy_context,
            user_persona_id=conversation_context.user_persona_id,
            platform_id=conversation_context.platform_id,
            character_id=character_id,
            incognito=conversation_context.incognito,
            remember_across_chats=conversation_context.remember_across_chats,
            remember_across_devices=conversation_context.remember_across_devices,
            sensitivity_gates_enabled=resolved_policy.allow_private_sensitivity,
            active_space_id=conversation_context.active_space_id,
            active_space_boundary_mode=conversation_context.active_space_boundary_mode,
            active_mind_id=conversation_context.active_mind_id,
            mind_topology=conversation_context.mind_topology,
            active_embodiment_id=conversation_context.active_embodiment_id,
            active_realm_id=conversation_context.active_realm_id,
        )
        message_chars = await self._message_repository.sum_text_length_for_context(
            conversation_context.user_id,
            resolved_policy.allowed_scopes,
            conversation_id=conversation_context.conversation_id,
            workspace_id=conversation_context.workspace_id,
            assistant_mode_id=conversation_context.assistant_mode_id,
        )
        estimated_tokens = self._estimate_tokens_from_chars(
            memory_chars + message_chars
        )
        return estimated_tokens < threshold_tokens

    async def _execute_small_corpus(
        self,
        *,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        audit_policy: ResolvedRetrievalPolicy,
        effective_ablation: AblationConfig,
        transcript: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        stage_timings: dict[str, float],
        trace: RetrievalTrace | None,
        pipeline_start: float,
    ) -> PipelineResult:
        """Small-corpus shortcut: skip search fan-out, but keep applicability gates."""
        if trace is not None:
            trace.small_corpus_mode = True
        stage_timings["base_planning"] = 0.0
        stage_timings["base_candidate_search"] = 0.0
        stage_timings["enriched_planning"] = 0.0
        stage_timings["enriched_candidate_search"] = 0.0
        character_id = _effective_character_id(conversation_context)

        base_plan = await self._measure_stage(
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
        retrieval_plan = base_plan
        detected_needs: list[Any] = []
        query_intelligence: QueryIntelligenceResult = _default_query_intelligence(
            message_text
        )
        runtime_alias_groups: list[RuntimeAliasGroupTrace] = []
        content_language_profile: list[dict[str, Any]] = []
        user_communication_profile: UserCommunicationProfile | None = None
        degraded_mode = False

        if effective_ablation.skip_need_detection:
            stage_timings["need_detection"] = 0.0
            if trace is not None:
                trace.need_detection = NeedDetectionTrace(
                    detected_needs=[],
                    sub_queries=[message_text],
                    sparse_hints=[],
                    query_type="default",
                    raw_context_access_mode="normal",
                    temporal_range=None,
                    retrieval_levels=[0],
                    user_communication_profile=None,
                    degraded_mode=False,
                    duration_ms=0.0,
                )
        else:
            need_start = perf_counter() if trace is not None else 0.0
            try:
                content_language_profile = await self._safe_content_language_profile(
                    conversation_context=conversation_context,
                    resolved_policy=audit_policy,
                )
                user_communication_profile = (
                    await self._safe_user_communication_profile(
                        conversation_context=conversation_context,
                    )
                )
                query_intelligence = await self._measure_stage(
                    stage_timings,
                    "need_detection",
                    self._need_detector.detect(
                        message_text=message_text,
                        role="user",
                        conversation_context=conversation_context,
                        resolved_policy=resolved_policy,
                        content_language_profile=content_language_profile,
                        user_communication_profile=user_communication_profile,
                        prompt_authority_context=process_authority_context(
                            privacy_enforcement=effective_ablation.privacy_enforcement,
                            user_id=conversation_context.user_id,
                            privilege_level=(
                                "atagia_master"
                                if effective_ablation.privacy_enforcement == "off"
                                else None
                            ),
                            is_atagia_master=(
                                effective_ablation.privacy_enforcement == "off"
                            ),
                            purpose="need_detection",
                        ),
                    ),
                )
            except Exception as exc:
                degraded_mode = True
                stage_timings.setdefault(
                    "need_detection", (perf_counter() - need_start) * 1000.0
                )
                logger.warning(
                    "small_corpus_need_detector_failed_using_default_plan",
                    extra={
                        "user_id": conversation_context.user_id,
                        "conversation_id": conversation_context.conversation_id,
                        "error": str(exc),
                    },
                )
                if trace is not None:
                    trace.need_detection = NeedDetectionTrace(
                        degraded_mode=True,
                        duration_ms=(perf_counter() - need_start) * 1000.0,
                        raw_context_access_mode="normal",
                        content_language_profile=_build_content_language_profile_trace(
                            content_language_profile,
                        ),
                        user_communication_profile=_build_user_communication_profile_trace(
                            user_communication_profile,
                        ),
                    )
            else:
                detected_needs = list(query_intelligence.needs)
                runtime_alias_groups = _build_runtime_alias_groups(
                    list(query_intelligence.anchors)
                )
                retrieval_plan = await self._measure_stage(
                    stage_timings,
                    "planning",
                    self._build_plan(
                        message_text=message_text,
                        query_intelligence=query_intelligence,
                        conversation_context=conversation_context,
                        resolved_policy=resolved_policy,
                        cold_start=False,
                        ablation=effective_ablation,
                    ),
                )
                if trace is not None:
                    trace.need_detection = NeedDetectionTrace(
                        detected_needs=[
                            need.need_type.value for need in query_intelligence.needs
                        ],
                        sub_queries=list(query_intelligence.sub_queries),
                        sparse_hints=[
                            hint.fts_phrase or hint.sub_query_text
                            for hint in query_intelligence.sparse_query_hints
                        ],
                        query_language=query_intelligence.query_language,
                        answer_language=query_intelligence.answer_language,
                        content_language_profile=_build_content_language_profile_trace(
                            content_language_profile,
                        ),
                        user_communication_profile=_build_user_communication_profile_trace(
                            user_communication_profile,
                        ),
                        anchors=list(query_intelligence.anchors),
                        alias_groups=runtime_alias_groups,
                        query_type=query_intelligence.query_type,
                        raw_context_access_mode=query_intelligence.raw_context_access_mode,
                        answer_shape=query_intelligence.answer_shape,
                        coverage_mode=query_intelligence.coverage_mode,
                        source_precision=query_intelligence.source_precision,
                        temporal_range=(
                            f"{query_intelligence.temporal_range.start.isoformat()}/{query_intelligence.temporal_range.end.isoformat()}"
                            if query_intelligence.temporal_range is not None
                            else None
                        ),
                        retrieval_levels=list(query_intelligence.retrieval_levels),
                        degraded_mode=False,
                        duration_ms=(perf_counter() - need_start) * 1000.0,
                        exact_recall_needed=bool(
                            query_intelligence.exact_recall_needed
                        ),
                        exact_facets=[
                            facet.value for facet in query_intelligence.exact_facets
                        ],
                        temporary_scaffolding=query_intelligence.temporary_scaffolding,
                    )

        if trace is not None:
            trace.raw_context_access_mode = query_intelligence.raw_context_access_mode
            self._append_temporary_scaffolding(
                trace,
                [
                    *query_intelligence.temporary_scaffolding,
                    *self._planner_temporary_scaffolding(retrieval_plan),
                ],
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
                privacy_ceiling=self._effective_privacy_ceiling(
                    resolved_policy,
                    effective_ablation,
                ),
                allow_intimacy_context=self._effective_allow_intimacy_context(
                    resolved_policy,
                    effective_ablation,
                ),
                user_persona_id=conversation_context.user_persona_id,
                platform_id=conversation_context.platform_id,
                character_id=character_id,
                incognito=conversation_context.incognito,
                remember_across_chats=conversation_context.remember_across_chats,
                remember_across_devices=conversation_context.remember_across_devices,
                sensitivity_gates_enabled=privacy_sql_filters_disabled(
                    effective_ablation,
                ),
                active_space_id=conversation_context.active_space_id,
                active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                active_mind_id=conversation_context.active_mind_id,
                mind_topology=conversation_context.mind_topology,
                active_embodiment_id=conversation_context.active_embodiment_id,
                active_realm_id=conversation_context.active_realm_id,
            ),
        )
        fts_query_audit: list[dict[str, Any]] = []
        if (
            retrieval_plan.exact_recall_mode
            or retrieval_plan.raw_context_access_mode in {"verbatim", "artifact"}
        ):
            searched_candidates = await self._measure_stage(
                stage_timings,
                "candidate_search_exact_recall",
                self._candidate_search.search(
                    retrieval_plan,
                    conversation_context.user_id,
                    fts_query_audit=fts_query_audit,
                    runtime_alias_groups=runtime_alias_groups,
                ),
            )
            raw_candidates = self._merge_candidates(raw_candidates, searched_candidates)
        pre_regrounding_candidates = list(raw_candidates)
        raw_candidates = self._apply_regrounding_requirements(
            raw_candidates, retrieval_plan
        )
        regrounding_filter_reasons = self._regrounding_filter_reasons(
            pre_regrounding_candidates,
            raw_candidates,
        )
        audit_plan = self._plan_for_policy_audit(retrieval_plan, audit_policy)
        candidate_policy_filter_reasons = self._candidate_filter_reasons(
            raw_candidates,
            audit_policy,
            detected_needs,
            audit_plan,
        )
        filter_reasons_by_id = self._actual_filter_reasons(
            regrounding_filter_reasons,
            candidate_policy_filter_reasons,
            privacy_enforcement=effective_ablation.privacy_enforcement,
        )
        if trace is not None:
            candidate_elapsed = (perf_counter() - candidate_start) * 1000.0
            trace.candidate_search = self._build_candidate_search_trace(
                raw_candidates,
                retrieval_plan,
                candidate_elapsed,
                fts_query_audit,
            )
            trace.policy_filter_audit = self._build_policy_filter_audit(
                effective_ablation.privacy_enforcement,
                candidate_policy_filter_reasons,
            )

        filtered_candidates = self._filter_candidates_for_policy_mode(
            raw_candidates,
            resolved_policy,
            detected_needs,
            retrieval_plan=retrieval_plan,
            privacy_enforcement=effective_ablation.privacy_enforcement,
        )
        scoring_policy = self._expand_candidate_budget(
            resolved_policy,
            item_count=len(filtered_candidates),
        )
        scoring_policy = self._policy_for_late_privacy_mode(
            scoring_policy,
            effective_ablation,
        )
        shortlist = early_diversity_select(
            filtered_candidates,
            query_type=retrieval_plan.query_type,
            shortlist_k=scoring_policy.retrieval_params.rerank_top_k,
        )
        shortlist = await self._reground_summary_support_shortlist(
            shortlist=shortlist,
            filtered_candidates=filtered_candidates,
            conversation_context=conversation_context,
            resolved_policy=scoring_policy,
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
        )
        shortlist = await self._reground_summary_source_window_shortlist(
            shortlist=shortlist,
            conversation_context=conversation_context,
            retrieval_plan=retrieval_plan,
            query_text=message_text,
        )

        scoring_start = perf_counter() if trace is not None else 0.0
        if effective_ablation.skip_applicability_scoring:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._score_without_llm(shortlist),
            )
        else:
            scored_candidates = await self._measure_stage(
                stage_timings,
                "applicability_scoring",
                self._scorer.score_shortlist(
                    shortlist,
                    message_text=message_text,
                    conversation_context=conversation_context,
                    resolved_policy=scoring_policy,
                    detected_needs=detected_needs,
                    retrieval_plan=retrieval_plan,
                    trace=trace,
                    applicability_gate_mode=(
                        effective_ablation.applicability_gate_mode
                    ),
                ),
            )
        if trace is not None:
            trace.scoring = self._build_scoring_trace(
                pre_regrounding_candidates,
                filtered_candidates,
                scored_candidates,
                (perf_counter() - scoring_start) * 1000.0,
                rejection_reasons=self._rejection_reason_counts(filter_reasons_by_id),
                policy_audit_reason_counts=self._rejection_reason_counts(
                    candidate_policy_filter_reasons
                ),
                scoring_candidates=shortlist,
            )

        retrieval_sufficiency = build_retrieval_sufficiency_diagnostic(
            raw_candidates=pre_regrounding_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=shortlist,
            scored_candidates=scored_candidates,
            retrieval_plan=retrieval_plan,
            contradiction_tension_threshold=self._settings.belief_tension_threshold,
        )
        if trace is not None:
            trace.retrieval_sufficiency = retrieval_sufficiency

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
                    user_persona_id=conversation_context.user_persona_id,
                    platform_id=conversation_context.platform_id,
                    character_id=character_id,
                    incognito=conversation_context.incognito,
                    remember_across_chats=conversation_context.remember_across_chats,
                    remember_across_devices=conversation_context.remember_across_devices,
                    sensitivity_gates_enabled=privacy_sql_filters_disabled(
                        effective_ablation,
                    ),
                    allow_private_sensitivity=effective_allow_private_for_sql_repository(
                        resolved_policy,
                        effective_ablation,
                    ),
                    active_space_id=conversation_context.active_space_id,
                    active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                    active_mind_id=conversation_context.active_mind_id,
                    mind_topology=conversation_context.mind_topology,
                    active_embodiment_id=conversation_context.active_embodiment_id,
                    active_realm_id=conversation_context.active_realm_id,
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
                allow_intimacy_context=resolved_policy.allow_intimacy_context,
                user_persona_id=conversation_context.user_persona_id,
                platform_id=conversation_context.platform_id,
                character_id=character_id,
                incognito=conversation_context.incognito,
                remember_across_chats=conversation_context.remember_across_chats,
                remember_across_devices=conversation_context.remember_across_devices,
                sensitivity_gates_enabled=privacy_sql_filters_disabled(
                    effective_ablation,
                ),
                active_space_id=conversation_context.active_space_id,
                active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                active_mind_id=conversation_context.active_mind_id,
                mind_topology=conversation_context.mind_topology,
                active_embodiment_id=conversation_context.active_embodiment_id,
                active_realm_id=conversation_context.active_realm_id,
            ),
        )

        effective_workspace_rollup = await self._measure_stage(
            stage_timings,
            "workspace_rollup_lookup",
            self._resolve_workspace_rollup(
                conversation_context=conversation_context,
                ablation=effective_ablation,
            ),
        )

        composition_policy = self._cap_explicit_final_context_items(
            scoring_policy,
            effective_ablation,
        )
        composition_start = perf_counter() if trace is not None else 0.0
        composed_context = await self._measure_stage(
            stage_timings,
            "context_composition",
            self._compose_context(
                user_id=conversation_context.user_id,
                message_text=message_text,
                retrieval_plan=retrieval_plan,
                scored_candidates=scored_candidates,
                current_contract=current_contract,
                workspace_rollup=effective_workspace_rollup,
                user_state=user_state,
                resolved_policy=self._policy_for_late_privacy_mode(
                    composition_policy,
                    effective_ablation,
                ),
                conversation_messages=transcript,
                composer_strategy=effective_ablation.composer_strategy,
                enable_evidence_obligation_coverage=(
                    effective_ablation.enable_evidence_obligation_coverage
                ),
                enable_evidence_packets=effective_ablation.enable_evidence_packets,
                enable_final_answer_evidence_pack=(
                    effective_ablation.enable_final_answer_evidence_pack
                ),
            ),
        )
        if effective_ablation.skip_contract_memory:
            composed_context = self._without_contract_block(composed_context)

        candidate_custody = build_candidate_custody(
            raw_candidates=pre_regrounding_candidates,
            filtered_candidates=filtered_candidates,
            shortlist=shortlist,
            scored_candidates=scored_candidates,
            selected_memory_ids=list(composed_context.selected_memory_ids),
            retrieval_plan=retrieval_plan,
            filter_reasons_by_id=filter_reasons_by_id,
        )

        if trace is not None:
            composition_elapsed = (perf_counter() - composition_start) * 1000.0
            trace.composition = self._build_composition_trace(
                composed_context,
                scoring_policy,
                composition_elapsed,
            )
            self._attach_stability_diagnostics(
                trace=trace,
                retrieval_plan=retrieval_plan,
                candidate_custody=candidate_custody,
                candidate_rows=[
                    *pre_regrounding_candidates,
                    *filtered_candidates,
                    *shortlist,
                    *(candidate.memory_object for candidate in scored_candidates),
                ],
                composed_context=composed_context,
            )
            trace.custody = self._build_custody_trace(
                raw_candidates=pre_regrounding_candidates,
                filtered_candidates=filtered_candidates,
                scored_candidates=scored_candidates,
                candidate_custody=candidate_custody,
                selected_memory_ids=list(composed_context.selected_memory_ids),
                retrieval_sufficiency=retrieval_sufficiency,
            )
            trace.total_duration_ms = (perf_counter() - pipeline_start) * 1000.0
            trace.degraded_mode = degraded_mode

        return PipelineResult(
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            scored_candidates=scored_candidates,
            candidate_custody=candidate_custody,
            retrieval_sufficiency=retrieval_sufficiency,
            composed_context=composed_context,
            current_contract=current_contract,
            user_state=user_state,
            stage_timings=stage_timings,
            trace=trace,
            small_corpus_mode=True,
            degraded_mode=degraded_mode,
            # The small-corpus shortcut runs before need detection, so the gate
            # never had a classification to act on.
            adaptive_gate_status=AdaptiveGateStatus.NOT_APPLICABLE,
            adaptive_gate_classification=None,
        )

    @staticmethod
    def _expand_candidate_budget(
        resolved_policy: ResolvedRetrievalPolicy,
        *,
        item_count: int,
    ) -> ResolvedRetrievalPolicy:
        """Allow small-corpus scoring/composition to consider every eligible memory."""
        retrieval_params = resolved_policy.retrieval_params
        if (
            item_count <= retrieval_params.final_context_items
            and item_count <= retrieval_params.rerank_top_k
        ):
            return resolved_policy
        expanded_retrieval = retrieval_params.model_copy(
            update={
                "final_context_items": max(
                    retrieval_params.final_context_items, item_count
                ),
                "rerank_top_k": max(retrieval_params.rerank_top_k, item_count),
            }
        )
        return resolved_policy.model_copy(
            update={"retrieval_params": expanded_retrieval}
        )

    @staticmethod
    def _expand_recall_or_recovery_scoring_budget(
        resolved_policy: ResolvedRetrievalPolicy,
        retrieval_plan: RetrievalPlan,
        *,
        degraded_mode: bool,
        detected_needs: list[Any],
        item_count: int,
    ) -> ResolvedRetrievalPolicy:
        """Score all recall/recovery candidates while the candidate set is bounded."""
        recovery_needs = {
            NeedTrigger.AMBIGUITY,
            NeedTrigger.FOLLOW_UP_FAILURE,
            NeedTrigger.UNDER_SPECIFIED_REQUEST,
        }
        need_expansion = any(
            getattr(need, "need_type", None) in recovery_needs
            for need in detected_needs
        )
        # Exhaustive known-set lists must score every candidate so members ranked
        # beyond the default rerank_top_k survive into scoring (rerank_top_k feeds
        # early_diversity_select, which caps the shortlist before scoring).
        exhaustive_coverage = (
            retrieval_plan.coverage_mode == "exhaustive_known_set"
        )
        if (
            not retrieval_plan.exact_recall_mode
            and not degraded_mode
            and not need_expansion
            and not exhaustive_coverage
        ):
            return resolved_policy
        if item_count <= resolved_policy.retrieval_params.rerank_top_k:
            return resolved_policy
        expanded_retrieval = resolved_policy.retrieval_params.model_copy(
            update={"rerank_top_k": item_count}
        )
        return resolved_policy.model_copy(
            update={"retrieval_params": expanded_retrieval}
        )

    @staticmethod
    def _expand_recall_or_recovery_context_items(
        resolved_policy: ResolvedRetrievalPolicy,
        retrieval_plan: RetrievalPlan,
        *,
        degraded_mode: bool,
        detected_needs: list[Any],
        item_count: int,
    ) -> ResolvedRetrievalPolicy:
        """Use already-scored recall/recovery candidates when token budget permits."""
        recovery_needs = {
            NeedTrigger.AMBIGUITY,
            NeedTrigger.FOLLOW_UP_FAILURE,
            NeedTrigger.UNDER_SPECIFIED_REQUEST,
        }
        need_expansion = any(
            getattr(need, "need_type", None) in recovery_needs
            for need in detected_needs
        )
        if (
            not retrieval_plan.exact_recall_mode
            and not degraded_mode
            and not need_expansion
        ):
            return resolved_policy
        if retrieval_plan.exact_recall_mode or degraded_mode:
            target_ceiling = resolved_policy.retrieval_params.rerank_top_k
        else:
            target_ceiling = min(
                resolved_policy.retrieval_params.rerank_top_k,
                resolved_policy.retrieval_params.final_context_items + 4,
            )
        target_items = min(
            item_count,
            max(
                resolved_policy.retrieval_params.final_context_items,
                target_ceiling,
            ),
        )
        if target_items <= resolved_policy.retrieval_params.final_context_items:
            return resolved_policy
        expanded_retrieval = resolved_policy.retrieval_params.model_copy(
            update={"final_context_items": target_items}
        )
        return resolved_policy.model_copy(
            update={"retrieval_params": expanded_retrieval}
        )

    @staticmethod
    def _expand_exhaustive_coverage_budget(
        resolved_policy: ResolvedRetrievalPolicy,
        retrieval_plan: RetrievalPlan,
        scored_candidates: list[ScoredCandidate],
    ) -> ResolvedRetrievalPolicy:
        """Raise final_context_items to fit exhaustive reserved coverage slots.

        For ``exhaustive_known_set`` mode, mirror the composer's reservation over
        the same admissible candidate set and raise ``final_context_items`` so
        near-tie verbatim evidence windows do not consume slots needed by member
        carriers. The physical token budget still governs admission via the
        selector, so members that do not fit become missing_slots, never a crash.
        This runs before the explicit-cap helper so ablation overrides still win.
        """
        if retrieval_plan.coverage_mode != "exhaustive_known_set":
            return resolved_policy
        coverage_floor = ContextComposer.exhaustive_coverage_floor(
            scored_candidates,
            active_presence_id=retrieval_plan.active_presence_id,
            allow_intimacy_context=resolved_policy.allow_intimacy_context,
        )
        if coverage_floor <= 0:
            return resolved_policy
        target_items = max(
            resolved_policy.retrieval_params.final_context_items,
            coverage_floor,
        )
        if target_items <= resolved_policy.retrieval_params.final_context_items:
            return resolved_policy
        expanded_retrieval = resolved_policy.retrieval_params.model_copy(
            update={"final_context_items": target_items}
        )
        return resolved_policy.model_copy(
            update={"retrieval_params": expanded_retrieval}
        )

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
                replacement = dict(candidate)
                CandidateSearch._merge_fts_query_matches(replacement, existing)
                merged[memory_id] = replacement
            else:
                CandidateSearch._merge_fts_query_matches(existing, candidate)
        return [merged[memory_id] for memory_id in order]

    async def _source_message_coverage_candidates(
        self,
        *,
        raw_candidates: list[dict[str, Any]],
        conversation_context: ExtractionConversationContext,
        retrieval_plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        """Pull bounded sibling memories from source messages already retrieved."""
        if not self._should_expand_source_message_coverage(
            retrieval_plan, raw_candidates
        ):
            return []

        existing_ids = {str(candidate.get("id") or "") for candidate in raw_candidates}
        source_messages = self._coverage_source_messages(raw_candidates)
        if not source_messages:
            return []

        expanded: list[dict[str, Any]] = []
        expanded_ids: set[str] = set()
        for source_message_id, source_metadata in source_messages[
            :COVERAGE_SOURCE_MESSAGE_LIMIT
        ]:
            rows = await self._memory_repository.list_for_source_message(
                user_id=conversation_context.user_id,
                source_message_id=source_message_id,
                statuses=tuple(retrieval_plan.status_filter),
            )
            per_message_count = 0
            for row in rows:
                memory_id = str(row.get("id") or "")
                if memory_id in existing_ids or memory_id in expanded_ids:
                    continue
                if not self._candidate_matches_retrieval_plan(row, retrieval_plan):
                    continue
                expanded.append(
                    self._annotate_source_message_coverage_candidate(
                        row,
                        source_message_id=source_message_id,
                        source_metadata=source_metadata,
                    )
                )
                expanded_ids.add(memory_id)
                per_message_count += 1
                if (
                    per_message_count >= COVERAGE_CANDIDATES_PER_SOURCE_MESSAGE
                    or len(expanded) >= COVERAGE_CANDIDATE_LIMIT
                ):
                    break
            if len(expanded) >= COVERAGE_CANDIDATE_LIMIT:
                break
        return expanded

    async def _llm_coverage_expansion_candidates(
        self,
        *,
        message_text: str,
        raw_candidates: list[dict[str, Any]],
        conversation_context: ExtractionConversationContext,
        retrieval_plan: RetrievalPlan,
        ablation: AblationConfig,
    ) -> list[dict[str, Any]]:
        """Run the default-off LLM coverage experiment and search its subqueries."""
        if not self._should_attempt_llm_coverage_expansion(
            retrieval_plan=retrieval_plan,
            raw_candidates=raw_candidates,
            ablation=ablation,
        ):
            return []
        try:
            expansion_plan = await self._coverage_expander.plan(
                message_text=message_text,
                conversation_context=conversation_context,
                retrieval_plan=retrieval_plan,
                raw_candidates=raw_candidates,
            )
        except Exception as exc:
            logger.warning(
                "llm_coverage_expansion_failed",
                extra={
                    "user_id": conversation_context.user_id,
                    "conversation_id": conversation_context.conversation_id,
                    "error": str(exc),
                },
            )
            return []
        if not expansion_plan.should_expand:
            return []

        coverage_plan = self._build_llm_coverage_retrieval_plan(
            retrieval_plan=retrieval_plan,
            expansion_plan=expansion_plan,
            ablation=ablation,
        )
        if coverage_plan is None:
            return []

        candidates = await self._candidate_search.search(
            coverage_plan,
            conversation_context.user_id,
        )
        existing_ids = {str(candidate.get("id") or "") for candidate in raw_candidates}
        expanded: list[dict[str, Any]] = []
        for candidate in candidates:
            memory_id = str(candidate.get("id") or "")
            if not memory_id or memory_id in existing_ids:
                continue
            expanded.append(
                self._annotate_llm_coverage_candidate(
                    candidate,
                    expansion_plan=expansion_plan,
                )
            )
            existing_ids.add(memory_id)
            if len(expanded) >= coverage_plan.max_candidates:
                break
        return expanded

    @staticmethod
    def _should_attempt_llm_coverage_expansion(
        *,
        retrieval_plan: RetrievalPlan,
        raw_candidates: list[dict[str, Any]],
        ablation: AblationConfig,
    ) -> bool:
        if not ablation.enable_llm_coverage_expansion:
            return False
        if retrieval_plan.skip_retrieval or retrieval_plan.max_candidates <= 0:
            return False
        if not retrieval_plan.sub_query_plans:
            return False
        if not raw_candidates:
            return True
        if retrieval_plan.query_type == "broad_list":
            return True
        if retrieval_plan.exact_recall_mode:
            return True
        if retrieval_plan.raw_context_access_mode in {
            "artifact",
            "verbatim",
            "skipped_raw",
        }:
            return True
        matched_subqueries = {
            str(sub_query)
            for candidate in raw_candidates
            for sub_query in candidate.get("matched_sub_queries", [])
            if str(sub_query).strip()
        }
        return len(matched_subqueries) < len(retrieval_plan.sub_query_plans)

    @classmethod
    def _build_llm_coverage_retrieval_plan(
        cls,
        *,
        retrieval_plan: RetrievalPlan,
        expansion_plan: CoverageExpansionPlan,
        ablation: AblationConfig,
    ) -> RetrievalPlan | None:
        sub_query_plans: list[PlannedSubQuery] = []
        for sub_query in expansion_plan.sub_queries[
            : cls._llm_coverage_max_subqueries(ablation)
        ]:
            sparse_phrase = sub_query.fts_phrase or sub_query.sub_query_text
            fts_queries = build_retrieval_fts_queries(
                sparse_phrase,
                quoted_phrases=sub_query.quoted_phrases,
                must_keep_terms=sub_query.must_keep_terms,
            )
            if not fts_queries:
                continue
            sub_query_plans.append(
                PlannedSubQuery(
                    text=sub_query.sub_query_text,
                    sparse_phrase=sparse_phrase,
                    quoted_phrases=list(sub_query.quoted_phrases),
                    must_keep_terms=list(sub_query.must_keep_terms),
                    fts_queries=fts_queries,
                )
            )
        if not sub_query_plans:
            return None
        candidate_limit = min(
            retrieval_plan.max_candidates,
            cls._llm_coverage_candidate_limit(ablation),
        )
        if candidate_limit <= 0:
            return None
        fts_queries: list[str] = []
        seen_queries: set[str] = set()
        for sub_query in sub_query_plans:
            for fts_query in sub_query.fts_queries:
                if fts_query in seen_queries:
                    continue
                seen_queries.add(fts_query)
                fts_queries.append(fts_query)
        vector_limit = min(retrieval_plan.vector_limit, candidate_limit)
        return retrieval_plan.model_copy(
            update={
                "fts_queries": fts_queries,
                "sub_query_plans": sub_query_plans,
                "max_candidates": candidate_limit,
                "vector_limit": vector_limit,
                "skip_retrieval": False,
            }
        )

    @staticmethod
    def _annotate_llm_coverage_candidate(
        candidate: dict[str, Any],
        *,
        expansion_plan: CoverageExpansionPlan,
    ) -> dict[str, Any]:
        annotated = dict(candidate)
        retrieval_sources = list(annotated.get("retrieval_sources") or [])
        if "llm_coverage_expansion" not in retrieval_sources:
            retrieval_sources.append("llm_coverage_expansion")
        annotated["retrieval_sources"] = retrieval_sources
        annotated["retrieval_source"] = "+".join(retrieval_sources)
        annotated["coverage_expansion_sub_queries"] = [
            sub_query.sub_query_text for sub_query in expansion_plan.sub_queries
        ]
        annotated["coverage_expansion_missing_facets"] = list(
            expansion_plan.missing_facets
        )
        return annotated

    @staticmethod
    def _llm_coverage_candidate_limit(ablation: AblationConfig) -> int:
        override_params = ablation.override_retrieval_params or {}
        if "llm_coverage_candidate_limit" not in override_params:
            return LLM_COVERAGE_CANDIDATE_LIMIT
        return max(1, int(override_params["llm_coverage_candidate_limit"]))

    @staticmethod
    def _llm_coverage_max_subqueries(ablation: AblationConfig) -> int:
        override_params = ablation.override_retrieval_params or {}
        if "llm_coverage_max_subqueries" not in override_params:
            return LLM_COVERAGE_MAX_SUBQUERIES
        return max(1, min(3, int(override_params["llm_coverage_max_subqueries"])))

    @staticmethod
    def _should_expand_source_message_coverage(
        retrieval_plan: RetrievalPlan,
        raw_candidates: list[dict[str, Any]],
    ) -> bool:
        if not raw_candidates:
            return False
        return (
            retrieval_plan.query_type == "broad_list"
            or retrieval_plan.exact_recall_mode
            or retrieval_plan.raw_context_access_mode == "artifact"
        )

    @classmethod
    def _coverage_source_messages(
        cls,
        raw_candidates: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any]]]:
        by_message_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for candidate in raw_candidates:
            message_ids = cls._raw_candidate_source_message_ids(candidate)
            if not message_ids:
                continue
            candidate_score = cls._normalized_retrieval_score(
                candidate.get("rrf_score")
            )
            matched_sub_queries = [
                str(item)
                for item in candidate.get("matched_sub_queries", [])
                if str(item).strip()
            ]
            for message_id in message_ids:
                metadata = by_message_id.get(message_id)
                if metadata is None:
                    metadata = {
                        "rrf_score": candidate_score,
                        "matched_sub_queries": list(matched_sub_queries),
                    }
                    by_message_id[message_id] = metadata
                    order.append(message_id)
                    continue
                if candidate_score > float(metadata.get("rrf_score") or 0.0):
                    metadata["rrf_score"] = candidate_score
                existing_queries = set(metadata.get("matched_sub_queries") or [])
                for sub_query in matched_sub_queries:
                    if sub_query in existing_queries:
                        continue
                    metadata["matched_sub_queries"].append(sub_query)
                    existing_queries.add(sub_query)
        return [(message_id, by_message_id[message_id]) for message_id in order]

    @staticmethod
    def _raw_candidate_source_message_ids(candidate: dict[str, Any]) -> list[str]:
        raw_ids = candidate.get("verbatim_evidence_window_message_ids")
        if not raw_ids:
            payload = candidate.get("payload_json") or {}
            if isinstance(payload, dict):
                raw_ids = payload.get("source_message_ids")
        if not raw_ids and candidate.get("message_id"):
            raw_ids = [candidate.get("message_id")]
        if not isinstance(raw_ids, list):
            return []
        return [
            str(message_id).strip() for message_id in raw_ids if str(message_id).strip()
        ]

    @classmethod
    def _annotate_source_message_coverage_candidate(
        cls,
        candidate: dict[str, Any],
        *,
        source_message_id: str,
        source_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        annotated = dict(candidate)
        source_score = cls._normalized_retrieval_score(source_metadata.get("rrf_score"))
        inherited_score = min(
            COVERAGE_INHERITED_SCORE_CAP,
            max(COVERAGE_INHERITED_SCORE_FLOOR, source_score * 0.8),
        )
        annotated["rrf_score"] = max(
            cls._normalized_retrieval_score(annotated.get("rrf_score")),
            inherited_score,
        )
        annotated.setdefault("channel_ranks", {})
        annotated["coverage_source_message_id"] = source_message_id
        annotated["matched_sub_queries"] = list(
            source_metadata.get("matched_sub_queries") or []
        )
        retrieval_sources = list(annotated.get("retrieval_sources") or [])
        if "source_neighbor" not in retrieval_sources:
            retrieval_sources.append("source_neighbor")
        annotated["retrieval_sources"] = retrieval_sources
        annotated["retrieval_source"] = "+".join(retrieval_sources)
        return annotated

    @staticmethod
    def _candidate_matches_retrieval_plan(
        candidate: dict[str, Any], plan: RetrievalPlan
    ) -> bool:
        return CandidateSearch._matches_plan_filters(candidate, plan)

    @staticmethod
    def _candidate_matches_retrieval_levels(
        candidate: dict[str, Any], plan: RetrievalPlan
    ) -> bool:
        if candidate.get("object_type") != MemoryObjectType.SUMMARY_VIEW.value:
            return 0 in plan.retrieval_levels
        payload_json = candidate.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        hierarchy_level = int(payload_json.get("hierarchy_level", -1))
        summary_kind = str(payload_json.get("summary_kind", "")).strip()
        if hierarchy_level == 0:
            return (
                0 in plan.retrieval_levels
                and summary_kind == SummaryViewKind.CONVERSATION_CHUNK.value
            )
        if hierarchy_level == 1:
            return (
                1 in plan.retrieval_levels
                and summary_kind == SummaryViewKind.EPISODE.value
            )
        if hierarchy_level == 2:
            return (
                2 in plan.retrieval_levels
                and summary_kind == SummaryViewKind.THEMATIC_PROFILE.value
            )
        return False

    @staticmethod
    def _candidate_matches_scope(
        candidate: dict[str, Any], plan: RetrievalPlan
    ) -> bool:
        scope = str(candidate.get("scope", ""))
        if scope == MemoryScope.GLOBAL_USER.value:
            return MemoryScope.GLOBAL_USER in plan.scope_filter
        if scope == MemoryScope.WORKSPACE.value:
            return (
                MemoryScope.WORKSPACE in plan.scope_filter
                and candidate.get("workspace_id") == plan.workspace_id
            )
        if scope == MemoryScope.CONVERSATION.value:
            return (
                MemoryScope.CONVERSATION in plan.scope_filter
                and candidate.get("conversation_id") == plan.conversation_id
            )
        if scope == MemoryScope.EPHEMERAL_SESSION.value:
            return (
                MemoryScope.EPHEMERAL_SESSION in plan.scope_filter
                and candidate.get("conversation_id") == plan.conversation_id
            )
        return False

    async def _build_plan(
        self,
        *,
        message_text: str,
        query_intelligence: QueryIntelligenceResult,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
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
        plan.privacy_enforcement = ablation.privacy_enforcement
        if privacy_sql_filters_disabled(ablation):
            plan.status_filter = self._privacy_off_status_filter(plan.status_filter)
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
            plan.privacy_ceiling = max(
                0, min(3, int(override_params["privacy_ceiling"]))
            )
        if "allow_private_sensitivity" in override_params:
            plan.allow_private_sensitivity = bool(
                override_params["allow_private_sensitivity"]
            )
        return plan

    async def _safe_content_language_profile(
        self,
        *,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
    ) -> list[dict[str, Any]]:
        """Return policy-filtered language counts for query intelligence."""
        try:
            return await self._candidate_search.aggregate_retrievable_content_language_mix(
                user_id=conversation_context.user_id,
                scope_filter=resolved_policy.allowed_scopes,
                assistant_mode_id=conversation_context.assistant_mode_id,
                workspace_id=conversation_context.workspace_id,
                conversation_id=conversation_context.conversation_id,
                privacy_ceiling=resolved_policy.privacy_ceiling,
                allow_intimacy_context=resolved_policy.allow_intimacy_context,
                user_persona_id=conversation_context.user_persona_id,
                platform_id=conversation_context.platform_id,
                character_id=_effective_character_id(conversation_context),
                active_space_id=conversation_context.active_space_id,
                active_space_boundary_mode=conversation_context.active_space_boundary_mode,
                active_mind_id=conversation_context.active_mind_id,
                mind_topology=conversation_context.mind_topology,
                active_embodiment_id=conversation_context.active_embodiment_id,
                active_realm_id=conversation_context.active_realm_id,
                incognito=conversation_context.incognito,
                remember_across_chats=conversation_context.remember_across_chats,
                remember_across_devices=conversation_context.remember_across_devices,
                limit=PROFILE_TOP_N,
            )
        except Exception as exc:
            logger.warning(
                "language_profile_aggregation_failed_using_empty_profile",
                extra={
                    "user_id": conversation_context.user_id,
                    "conversation_id": conversation_context.conversation_id,
                    "error": str(exc),
                },
            )
            return []

    async def _safe_user_communication_profile(
        self,
        *,
        conversation_context: ExtractionConversationContext,
    ) -> UserCommunicationProfile | None:
        """Return the control-plane language profile visible to this context."""
        try:
            return await self._communication_profile_repository.get_user_language_profile_for_context(
                conversation_context
            )
        except Exception as exc:
            logger.warning(
                "user_communication_profile_lookup_failed_using_empty_profile",
                extra={
                    "user_id": conversation_context.user_id,
                    "conversation_id": conversation_context.conversation_id,
                    "error": str(exc),
                },
            )
            return None

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
            retrieval_score = self._normalized_retrieval_score(
                candidate.get("rrf_score")
            )
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
        ablation: AblationConfig,
    ) -> dict[str, Any] | None:
        if ablation.skip_workspace_rollup:
            return None
        if (
            conversation_context.incognito
            or not conversation_context.remember_across_chats
        ):
            return None
        character_id = _effective_character_id(conversation_context)
        if character_id is None:
            return None
        row = await self._summary_repository.get_latest_character_rollup_for_persona(
            conversation_context.user_id,
            character_id,
            conversation_context.user_persona_id,
        )
        if row is None:
            return None
        if str(row.get("scope_canonical") or "") != MemoryScope.CHARACTER.value:
            return None
        if str(
            row.get("sensitivity") or "unknown"
        ) != "public" and not privacy_sql_filters_disabled(ablation):
            return None
        if row.get("user_persona_id") != conversation_context.user_persona_id:
            return None
        platform_id = conversation_context.platform_id or "default"
        if conversation_context.remember_across_devices:
            if (
                bool(row.get("platform_locked"))
                and row.get("platform_id_lock") != platform_id
            ):
                return None
        elif row.get("platform_id_lock") == platform_id:
            return row
        elif bool(row.get("platform_locked")) or row.get("platform_id") != platform_id:
            return None
        return row

    @staticmethod
    def _apply_regrounding_requirements(
        candidates: list[dict[str, Any]],
        retrieval_plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        if not retrieval_plan.require_evidence_regrounding:
            return candidates
        allowed_types = {
            MemoryObjectType.EVIDENCE.value,
            MemoryObjectType.INTERACTION_CONTRACT.value,
            MemoryObjectType.STATE_SNAPSHOT.value,
        }
        return [
            candidate
            for candidate in candidates
            if str(candidate.get("object_type")) in allowed_types
            or RetrievalPipeline._is_grounded_conversation_chunk(candidate)
        ]

    @staticmethod
    def _regrounding_filter_reasons(
        before_candidates: list[dict[str, Any]],
        after_candidates: list[dict[str, Any]],
    ) -> dict[str, str]:
        after_ids = {str(candidate.get("id") or "") for candidate in after_candidates}
        return {
            str(candidate.get("id") or ""): "regrounding_filtered"
            for candidate in before_candidates
            if str(candidate.get("id") or "") not in after_ids
        }

    def _candidate_filter_reasons(
        self,
        candidates: list[dict[str, Any]],
        resolved_policy: ResolvedRetrievalPolicy,
        detected_needs: list[Any],
        retrieval_plan: RetrievalPlan,
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            reason = self._scorer.candidate_filter_reason(
                candidate,
                resolved_policy,
                detected_needs,
                retrieval_plan=retrieval_plan,
            )
            if reason is not None:
                reasons[candidate_id] = reason
        return reasons

    @staticmethod
    def _policy_filters_enforced(privacy_enforcement: str) -> bool:
        return privacy_enforcement == "enforce"

    @staticmethod
    def _privacy_off_status_filter(
        statuses: list[MemoryStatus],
    ) -> list[MemoryStatus]:
        return list(dict.fromkeys([*statuses, *PRIVACY_OFF_RETRIEVAL_STATUSES]))

    @staticmethod
    def _privacy_relaxed_policy(
        policy: ResolvedRetrievalPolicy,
    ) -> ResolvedRetrievalPolicy:
        return policy.model_copy(
            update={
                "privacy_ceiling": 3,
                "allow_intimacy_context": True,
                "allow_private_sensitivity": True,
            }
        )

    @staticmethod
    def _policy_for_sql_privacy_mode(
        policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> ResolvedRetrievalPolicy:
        if privacy_sql_filters_disabled(ablation):
            return RetrievalPipeline._privacy_relaxed_policy(policy)
        return policy

    @staticmethod
    def _policy_for_late_privacy_mode(
        policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> ResolvedRetrievalPolicy:
        if not RetrievalPipeline._policy_filters_enforced(ablation.privacy_enforcement):
            return RetrievalPipeline._privacy_relaxed_policy(policy)
        return policy

    @staticmethod
    def _plan_for_policy_audit(
        plan: RetrievalPlan,
        policy: ResolvedRetrievalPolicy,
    ) -> RetrievalPlan:
        return plan.model_copy(
            update={
                "privacy_ceiling": policy.privacy_ceiling,
                "allow_intimacy_context": policy.allow_intimacy_context,
                "allow_private_sensitivity": policy.allow_private_sensitivity,
                "privacy_enforcement": "enforce",
            }
        )

    @staticmethod
    def _actual_filter_reasons(
        regrounding_filter_reasons: dict[str, str],
        candidate_policy_filter_reasons: dict[str, str],
        *,
        privacy_enforcement: str,
    ) -> dict[str, str]:
        if RetrievalPipeline._policy_filters_enforced(privacy_enforcement):
            return {
                **regrounding_filter_reasons,
                **candidate_policy_filter_reasons,
            }
        return dict(regrounding_filter_reasons)

    @staticmethod
    def _build_policy_filter_audit(
        privacy_enforcement: str,
        candidate_policy_filter_reasons: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "privacy_enforcement": privacy_enforcement,
            "enforced": RetrievalPipeline._policy_filters_enforced(privacy_enforcement),
            "high_risk_secret_literal_redaction_enforced": (
                privacy_enforcement != "off"
            ),
            "high_risk_secret_literal_redaction_disabled": (
                privacy_enforcement == "off"
            ),
            "would_filter_count": len(candidate_policy_filter_reasons),
            "would_filter_reason_counts": RetrievalPipeline._rejection_reason_counts(
                candidate_policy_filter_reasons
            ),
            "would_filter_by_candidate_id": dict(
                sorted(candidate_policy_filter_reasons.items())
            ),
        }

    def _filter_candidates_for_policy_mode(
        self,
        candidates: list[dict[str, Any]],
        resolved_policy: ResolvedRetrievalPolicy,
        detected_needs: list[Any],
        *,
        retrieval_plan: RetrievalPlan,
        privacy_enforcement: str,
    ) -> list[dict[str, Any]]:
        if not self._policy_filters_enforced(privacy_enforcement):
            return list(candidates)
        return self._scorer.filter_candidates(
            candidates,
            resolved_policy,
            detected_needs,
            retrieval_plan=retrieval_plan,
        )

    @staticmethod
    def _effective_privacy_ceiling(
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> int:
        if privacy_sql_filters_disabled(ablation):
            return 3
        return resolved_policy.privacy_ceiling

    @staticmethod
    def _effective_allow_intimacy_context(
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> bool:
        if privacy_sql_filters_disabled(ablation):
            return True
        return resolved_policy.allow_intimacy_context

    @staticmethod
    def _rejection_reason_counts(
        filter_reasons_by_id: dict[str, str],
    ) -> dict[str, int]:
        return dict(sorted(Counter(filter_reasons_by_id.values()).items()))

    @staticmethod
    def _is_grounded_conversation_chunk(candidate: dict[str, Any]) -> bool:
        if str(candidate.get("object_type")) != MemoryObjectType.SUMMARY_VIEW.value:
            return False
        payload_json = candidate.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        try:
            hierarchy_level = int(payload_json.get("hierarchy_level", -1))
        except (TypeError, ValueError):
            return False
        return (
            hierarchy_level == 0
            and payload_json.get("summary_kind")
            == SummaryViewKind.CONVERSATION_CHUNK.value
            and bool(payload_json.get("source_excerpt_messages"))
        )

    async def _reground_summary_source_window_shortlist(
        self,
        *,
        shortlist: list[dict[str, Any]],
        conversation_context: ExtractionConversationContext,
        retrieval_plan: RetrievalPlan,
        query_text: str,
    ) -> list[dict[str, Any]]:
        """Add bounded raw source windows for source-backed summaries.

        This is intentionally narrow: it only runs with benchmark/default-readiness
        privacy enforcement disabled, and it only materializes windows from source
        messages already referenced by summaries that reached the shortlist.
        """
        if (
            not shortlist
            or retrieval_plan.privacy_enforcement != "off"
            or retrieval_plan.query_type
            not in {"temporal", "slot_fill", "broad_list"}
        ):
            return shortlist

        existing_ids = {str(candidate.get("id") or "") for candidate in shortlist}
        promoted: list[dict[str, Any]] = []
        promoted_ids: set[str] = set()
        for candidate in shortlist:
            if len(promoted) >= SUMMARY_SOURCE_WINDOW_PER_RUN_LIMIT:
                break
            if str(candidate.get("object_type") or "") != MemoryObjectType.SUMMARY_VIEW.value:
                continue
            payload_json = candidate.get("payload_json") or {}
            if not isinstance(payload_json, dict):
                continue
            source_message_ids = [
                str(item).strip()
                for item in payload_json.get("source_message_ids") or []
                if str(item).strip()
            ]
            if len(source_message_ids) <= 1:
                continue
            anchor = await self._summary_source_window_anchor(
                source_message_ids=source_message_ids,
                user_id=conversation_context.user_id,
                query_text=query_text,
                active_conversation_id=retrieval_plan.conversation_id,
            )
            if anchor is None:
                continue
            window_candidate = await self._build_summary_source_window_candidate(
                summary_candidate=candidate,
                anchor_message=anchor,
                user_id=conversation_context.user_id,
                retrieval_plan=retrieval_plan,
            )
            if window_candidate is None:
                continue
            window_id = str(window_candidate.get("id") or "")
            if window_id in existing_ids or window_id in promoted_ids:
                continue
            promoted.append(window_candidate)
            promoted_ids.add(window_id)

        if not promoted:
            return shortlist
        return [*shortlist, *promoted]

    async def _summary_source_window_anchor(
        self,
        *,
        source_message_ids: list[str],
        user_id: str,
        query_text: str,
        active_conversation_id: str,
    ) -> dict[str, Any] | None:
        for message_id in source_message_ids[:SOURCE_MESSAGE_FETCH_LIMIT]:
            message = await self._message_repository.get_message(message_id, user_id)
            if message is None:
                continue
            if str(message.get("conversation_id") or "") != active_conversation_id:
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            if ContextComposer._quote_query_relevance(text, query_text) > 0.0:
                return message
        return None

    async def _build_summary_source_window_candidate(
        self,
        *,
        summary_candidate: dict[str, Any],
        anchor_message: dict[str, Any],
        user_id: str,
        retrieval_plan: RetrievalPlan,
    ) -> dict[str, Any] | None:
        conversation_id = str(anchor_message.get("conversation_id") or "")
        if not conversation_id:
            return None
        try:
            center_seq = int(anchor_message.get("seq"))
        except (TypeError, ValueError):
            return None
        window_messages = await self._message_repository.fetch_message_window(
            conversation_id=conversation_id,
            user_id=user_id,
            center_seq=center_seq + (SUMMARY_SOURCE_WINDOW_SIZE // 2),
            window_size=SUMMARY_SOURCE_WINDOW_SIZE,
        )
        if not window_messages:
            return None
        source_message_ids = [
            str(message.get("id") or "").strip()
            for message in window_messages
            if str(message.get("id") or "").strip()
        ]
        if not source_message_ids:
            return None
        start_seq = int(window_messages[0]["seq"])
        end_seq = int(window_messages[-1]["seq"])
        source_summary_id = str(summary_candidate.get("id") or "")
        candidate_id = f"ssw_{source_summary_id}_{start_seq}_{end_seq}"
        canonical_text = self._format_summary_source_window_text(window_messages)
        if not canonical_text:
            return None
        source_score = self._normalized_retrieval_score(summary_candidate.get("rrf_score"))
        retrieval_sources = list(summary_candidate.get("retrieval_sources") or [])
        if "summary_source_window" not in retrieval_sources:
            retrieval_sources.append("summary_source_window")
        occurred_at = str(
            window_messages[-1].get("occurred_at")
            or window_messages[-1].get("created_at")
            or ""
        )
        start_occurred_at = str(
            window_messages[0].get("occurred_at")
            or window_messages[0].get("created_at")
            or ""
        )
        payload_json = {
            "source_kind_variant": "summary_source_window",
            "source_summary_id": source_summary_id,
            "source_message_ids": source_message_ids,
            "window_start_seq": start_seq,
            "window_end_seq": end_seq,
            "source_message_window_start_occurred_at": start_occurred_at,
            "source_message_window_end_occurred_at": occurred_at,
        }
        return {
            "_rowid": None,
            "id": candidate_id,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "status": MemoryStatus.ACTIVE.value,
            "scope": summary_candidate.get("scope"),
            "scope_canonical": summary_candidate.get("scope_canonical"),
            "privacy_level": summary_candidate.get("privacy_level", 0),
            "sensitivity": summary_candidate.get("sensitivity", "public"),
            "user_persona_id": summary_candidate.get("user_persona_id"),
            "platform_id": summary_candidate.get("platform_id"),
            "character_id": summary_candidate.get("character_id"),
            "platform_locked": summary_candidate.get("platform_locked", 0),
            "platform_id_lock": summary_candidate.get("platform_id_lock"),
            "assistant_mode_id": summary_candidate.get("assistant_mode_id"),
            "conversation_id": conversation_id,
            "workspace_id": summary_candidate.get("workspace_id"),
            "canonical_text": canonical_text,
            "payload_json": payload_json,
            "source_kind": "verbatim",
            "confidence": 1.0,
            "stability": 0.5,
            "vitality": 0.0,
            "maya_score": 0.0,
            "rrf_score": max(
                self._normalized_retrieval_score(summary_candidate.get("rrf_score")),
                min(COVERAGE_INHERITED_SCORE_CAP, max(0.1, source_score * 0.9)),
            ),
            "updated_at": occurred_at or summary_candidate.get("updated_at"),
            "created_at": start_occurred_at or summary_candidate.get("created_at"),
            "valid_from": None,
            "valid_to": None,
            "temporal_type": "unknown",
            "channel_ranks": {
                "fts": None,
                "embedding": None,
                "consequence": None,
                "verbatim_evidence_search": None,
            },
            "matched_sub_queries": list(summary_candidate.get("matched_sub_queries") or []),
            "retrieval_sources": retrieval_sources,
        }

    @staticmethod
    def _format_summary_source_window_text(
        messages: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for message in messages:
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            prefix_parts = [str(message.get("role") or "unknown").strip() or "unknown"]
            occurred_at = str(message.get("occurred_at") or "").strip()
            if occurred_at:
                prefix_parts.append(f"@ {occurred_at}")
            seq = message.get("seq")
            if seq is not None:
                prefix_parts.append(f"seq {seq}")
            lines.append(
                f"{' '.join(prefix_parts)}: "
                + ContextComposer._truncate_inline(text, 260)
            )
        return "\n".join(lines)

    async def _reground_summary_support_shortlist(
        self,
        *,
        shortlist: list[dict[str, Any]],
        filtered_candidates: list[dict[str, Any]],
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
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
        recovery_needs = {
            NeedTrigger.AMBIGUITY,
            NeedTrigger.UNDER_SPECIFIED_REQUEST,
        }
        need_expansion = any(
            getattr(need, "need_type", None) in recovery_needs
            for need in detected_needs
        )
        if (
            retrieval_plan.query_type != "slot_fill"
            and not retrieval_plan.exact_recall_mode
            and not need_expansion
        ):
            return shortlist

        per_summary_cap = 4 if need_expansion else 2
        total_cap = 6 if need_expansion else 4
        shortlisted_ids = {str(candidate["id"]) for candidate in shortlist}
        promoted_ids: set[str] = set()
        promoted: list[dict[str, Any]] = []
        filtered_by_id = {
            str(candidate["id"]): candidate for candidate in filtered_candidates
        }

        for candidate in shortlist:
            if len(promoted) >= total_cap:
                break
            if not self._is_hierarchical_summary_candidate(candidate) and not (
                need_expansion
                and str(candidate.get("object_type"))
                == MemoryObjectType.SUMMARY_VIEW.value
            ):
                continue
            source_ids = self._candidate_source_ids(candidate)
            if not source_ids:
                continue

            source_positions = {
                source_id: index for index, source_id in enumerate(source_ids)
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
                    and self._candidate_matches_retrieval_plan(row, retrieval_plan)
                ]
                eligible_rows = self._scorer.filter_candidates(
                    annotated_rows,
                    resolved_policy,
                    detected_needs,
                    retrieval_plan=retrieval_plan,
                )
                for row in eligible_rows:
                    source_id = str(row["id"])
                    if (
                        source_id in shortlisted_ids
                        or source_id in promoted_ids
                        or source_id in support_ids
                    ):
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
                if (
                    summary_promoted >= allowed_for_summary
                    or len(promoted) >= total_cap
                ):
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
    def _annotate_summary_support_candidate(
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        annotated = dict(candidate)
        annotated.setdefault("rrf_score", 0.0)
        annotated.setdefault(
            "channel_ranks",
            {
                "fts": None,
                "embedding": None,
                "consequence": None,
                "verbatim_evidence_search": None,
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
                1
                if str(candidate.get("object_type"))
                == MemoryObjectType.SUMMARY_VIEW.value
                else 0,
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
        user_id: str,
        message_text: str,
        retrieval_plan: RetrievalPlan,
        scored_candidates: list[ScoredCandidate],
        current_contract: dict[str, dict[str, Any]],
        workspace_rollup: dict[str, Any] | None,
        user_state: dict[str, Any],
        resolved_policy: ResolvedRetrievalPolicy,
        conversation_messages: list[dict[str, Any]],
        composer_strategy: str | None = None,
        enable_evidence_obligation_coverage: bool = True,
        enable_evidence_packets: bool = True,
        enable_final_answer_evidence_pack: bool = False,
    ) -> ComposedContext:
        if enable_evidence_packets:
            scored_candidates = await self._with_evidence_packets(
                user_id=user_id,
                scored_candidates=scored_candidates,
            )
        source_messages = conversation_messages
        if retrieval_plan.exact_recall_mode or retrieval_plan.query_type in {
            "temporal",
            "slot_fill",
            "broad_list",
            "default",
        }:
            source_messages = await self._source_messages_for_candidates(
                user_id=user_id,
                scored_candidates=scored_candidates,
                existing_messages=conversation_messages,
                active_conversation_id=retrieval_plan.conversation_id,
            )
        context_policy = (
            self._privacy_relaxed_policy(resolved_policy)
            if not self._policy_filters_enforced(retrieval_plan.privacy_enforcement)
            else resolved_policy
        )
        return self._context_composer.compose(
            scored_candidates=scored_candidates,
            current_contract=current_contract,
            workspace_rollup=workspace_rollup,
            user_state=user_state,
            resolved_policy=context_policy,
            conversation_messages=source_messages,
            query_text=message_text,
            query_type=retrieval_plan.query_type,
            answer_shape=retrieval_plan.answer_shape,
            coverage_mode=retrieval_plan.coverage_mode,
            source_precision=retrieval_plan.source_precision,
            exact_recall_mode=retrieval_plan.exact_recall_mode,
            composer_strategy=composer_strategy,
            enable_evidence_obligation_coverage=enable_evidence_obligation_coverage,
            enable_final_answer_evidence_pack=enable_final_answer_evidence_pack,
            fact_facet_span_coadmission_enabled=(
                self._settings.fact_facet_span_coadmission_enabled
            ),
            active_presence_id=retrieval_plan.active_presence_id,
            active_realm_id=retrieval_plan.active_realm_id,
            redact_high_risk_secret_literals=(
                retrieval_plan.privacy_enforcement != "off"
            ),
        )

    async def _with_evidence_packets(
        self,
        *,
        user_id: str,
        scored_candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        memory_ids = [candidate.memory_id for candidate in scored_candidates]
        packets_by_memory = await self._memory_evidence_repository.list_packets_for_memory_ids(
            user_id=user_id,
            memory_ids=memory_ids,
            limit_per_memory=2,
        )
        if not packets_by_memory:
            return scored_candidates
        hydrated: list[ScoredCandidate] = []
        for candidate in scored_candidates:
            packets = packets_by_memory.get(candidate.memory_id)
            if not packets:
                hydrated.append(candidate)
                continue
            memory_object = dict(candidate.memory_object)
            memory_object["evidence_packets"] = packets
            hydrated.append(candidate.model_copy(update={"memory_object": memory_object}))
        return hydrated

    async def _source_messages_for_candidates(
        self,
        *,
        user_id: str,
        scored_candidates: list[ScoredCandidate],
        existing_messages: list[dict[str, Any]],
        active_conversation_id: str,
    ) -> list[dict[str, Any]]:
        messages_by_id = {
            str(message.get("id")): message
            for message in existing_messages
            if str(message.get("id") or "").strip()
        }
        missing_ids: list[str] = []
        seen = set(messages_by_id)
        for candidate in scored_candidates:
            for message_id in self._candidate_source_message_ids(candidate):
                if message_id in seen:
                    continue
                seen.add(message_id)
                missing_ids.append(message_id)
                if len(missing_ids) >= SOURCE_MESSAGE_FETCH_LIMIT:
                    break
            if len(missing_ids) >= SOURCE_MESSAGE_FETCH_LIMIT:
                break

        fetched_messages: list[dict[str, Any]] = []
        for message_id in missing_ids:
            message = await self._message_repository.get_message(message_id, user_id)
            if (
                message is not None
                and str(message.get("conversation_id") or "") == active_conversation_id
            ):
                fetched_messages.append(message)
        return [*existing_messages, *fetched_messages]

    @staticmethod
    def _candidate_source_message_ids(candidate: ScoredCandidate) -> list[str]:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return []
        raw_ids = payload_json.get("source_message_ids") or []
        if not isinstance(raw_ids, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_id in raw_ids:
            message_id = str(raw_id).strip()
            if not message_id or message_id in seen:
                continue
            seen.add(message_id)
            normalized.append(message_id)
        return normalized

    def _override_policy(
        self,
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> ResolvedRetrievalPolicy:
        override_params = ablation.override_retrieval_params or {}

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
        envelope_budget = self._effective_context_envelope_budget(ablation)
        updates["context_budget_tokens"] = (
            envelope_budget.retrieved_context_budget_tokens
        )
        if "context_budget_tokens" in override_params:
            updates["context_budget_tokens"] = max(
                1,
                int(override_params["context_budget_tokens"]),
            )
        if "transcript_budget_tokens" in override_params:
            updates["transcript_budget_tokens"] = max(
                1,
                int(override_params["transcript_budget_tokens"]),
            )
        if "privacy_ceiling" in override_params:
            updates["privacy_ceiling"] = max(
                0, min(3, int(override_params["privacy_ceiling"]))
            )
        if "allow_private_sensitivity" in override_params:
            updates["allow_private_sensitivity"] = bool(
                override_params["allow_private_sensitivity"]
            )
        return resolved_policy.model_copy(update=updates)

    def _effective_context_envelope_budget(
        self,
        ablation: AblationConfig,
    ) -> ContextEnvelopeBudget:
        return allocate_context_envelope_budget(
            (
                ablation.context_envelope_budget_tokens
                if ablation.context_envelope_budget_tokens is not None
                else self._settings.context_envelope_budget_tokens
            ),
            (
                ablation.context_envelope_ratios
                if ablation.context_envelope_ratios is not None
                else self._settings.context_envelope_ratios
            ),
        )

    @staticmethod
    def _cap_explicit_final_context_items(
        resolved_policy: ResolvedRetrievalPolicy,
        ablation: AblationConfig,
    ) -> ResolvedRetrievalPolicy:
        override_params = ablation.override_retrieval_params or {}
        if "final_context_items" not in override_params:
            return resolved_policy
        cap = max(1, int(override_params["final_context_items"]))
        if resolved_policy.retrieval_params.final_context_items <= cap:
            return resolved_policy
        retrieval_params = resolved_policy.retrieval_params.model_copy(
            update={"final_context_items": cap}
        )
        return resolved_policy.model_copy(update={"retrieval_params": retrieval_params})

    @staticmethod
    def _log_stage_overlaps(
        conversation_context: ExtractionConversationContext,
        overlaps: dict[str, str],
    ) -> None:
        """Record which stages ran concurrently for trace visibility.

        ``stage_timings`` is a strict ``dict[str, float]`` consumed widely
        downstream, so the overlap relationship (a stage name -> the partner
        stage it overlapped with) is surfaced through the structured log rather
        than mixed into the float map. Each concurrent stage still records its
        own duration in ``stage_timings`` via ``_measure_stage`` /
        ``_await_measured_task``.
        """
        logger.info(
            "retrieval_stage_overlap",
            extra={
                "user_id": conversation_context.user_id,
                "conversation_id": conversation_context.conversation_id,
                "stage_overlaps": dict(overlaps),
            },
        )

    async def _measure_stage(
        self, stage_timings: dict[str, float], name: str, awaitable: Any
    ) -> Any:
        started_at = perf_counter()
        result = await awaitable
        stage_timings[name] = (perf_counter() - started_at) * 1000.0
        return result

    @staticmethod
    async def _await_measured_task(
        stage_timings: dict[str, float],
        name: str,
        task: asyncio.Task[Any],
        started_at: float,
    ) -> Any:
        """Await a stage already running as a task and record its own duration.

        Used by the concurrency overlaps so a stage that ran while an LLM
        round-trip was in flight still reports its individual wall-clock
        duration (measured from task creation, not from the await point).
        """
        result = await task
        stage_timings[name] = (perf_counter() - started_at) * 1000.0
        return result

    @staticmethod
    async def _drain_overlap_task(task: asyncio.Task[Any]) -> None:
        """Cancel and settle an overlap task on an error exit path.

        Guarantees the task is never orphaned (no "Task was destroyed but it
        is pending" warnings) when the partnered LLM stage raises before the
        DB-owning task is awaited. The DB task's own exception is intentionally
        suppressed here because the LLM-side error is the one that surfaces.

        ``asyncio.wait`` is used instead of ``await task`` so the drained task's
        exception is never re-raised into this frame and a cancellation of the
        *current* task is not masked by the drained task's own cancellation.
        """
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            # Retrieve any stored exception so it is marked as consumed and
            # does not trigger a "Future exception was never retrieved" warning.
            drained_exception = task.exception()
            if drained_exception is not None:
                logger.debug(
                    "Drained overlap task exception (suppressed in favor of the "
                    "partnered stage's error): %r",
                    drained_exception,
                )

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
        fts_query_audit: list[dict[str, Any]] | None = None,
    ) -> CandidateSearchTrace:
        fts_count = 0
        verbatim_pin_count = 0
        artifact_chunk_count = 0
        embedding_count = 0
        fact_facet_count = 0
        consequence_count = 0
        verbatim_evidence_search_count = 0
        for candidate in raw_candidates:
            channel_ranks = candidate.get("channel_ranks") or {}
            if channel_ranks.get("verbatim_pin") is not None or candidate.get(
                "is_verbatim_pin"
            ):
                verbatim_pin_count += 1
            if channel_ranks.get("artifact_chunk") is not None or candidate.get(
                "is_artifact_chunk"
            ):
                artifact_chunk_count += 1
            if channel_ranks.get("fts") is not None:
                fts_count += 1
            if channel_ranks.get("fact_facet") is not None or candidate.get(
                "is_fact_facet_candidate"
            ):
                fact_facet_count += 1
            if channel_ranks.get("embedding") is not None:
                embedding_count += 1
            if channel_ranks.get("consequence") is not None:
                consequence_count += 1
            if channel_ranks.get(
                "verbatim_evidence_search"
            ) is not None or candidate.get("is_verbatim_evidence_window"):
                verbatim_evidence_search_count += 1
        total_before_fusion = (
            fts_count
            + artifact_chunk_count
            + fact_facet_count
            + embedding_count
            + consequence_count
            + verbatim_evidence_search_count
        )
        per_subquery_counts: list[SubQuerySearchCount] = []
        for sub_query in retrieval_plan.sub_query_plans:
            sub_verbatim = 0
            sub_artifact = 0
            sub_fts = 0
            sub_fact_facet = 0
            sub_emb = 0
            sub_raw = 0
            for candidate in raw_candidates:
                matched = candidate.get("matched_sub_queries") or []
                if sub_query.text in matched:
                    channel_ranks = candidate.get("channel_ranks") or {}
                    if channel_ranks.get("verbatim_pin") is not None or candidate.get(
                        "is_verbatim_pin"
                    ):
                        sub_verbatim += 1
                    if channel_ranks.get("artifact_chunk") is not None or candidate.get(
                        "is_artifact_chunk"
                    ):
                        sub_artifact += 1
                    if channel_ranks.get("fts") is not None:
                        sub_fts += 1
                    if channel_ranks.get("fact_facet") is not None or candidate.get(
                        "is_fact_facet_candidate"
                    ):
                        sub_fact_facet += 1
                    if channel_ranks.get("embedding") is not None:
                        sub_emb += 1
                    if channel_ranks.get(
                        "verbatim_evidence_search"
                    ) is not None or candidate.get("is_verbatim_evidence_window"):
                        sub_raw += 1
            per_subquery_counts.append(
                SubQuerySearchCount(
                    subquery=sub_query.text,
                    verbatim_pin=sub_verbatim,
                    artifact_chunk=sub_artifact,
                    fts=sub_fts,
                    fact_facet=sub_fact_facet,
                    embedding=sub_emb,
                    verbatim_evidence_search=sub_raw,
                    fts_queries=list(sub_query.fts_queries),
                    fts_query_kinds=list(sub_query.fts_query_kinds),
                    fts_query_executions=RetrievalPipeline._build_fts_query_execution_counts(
                        raw_candidates,
                        sub_query,
                        fts_query_audit,
                    ),
                )
            )
        return CandidateSearchTrace(
            fts_candidates_count=fts_count,
            verbatim_pin_candidates_count=verbatim_pin_count,
            artifact_chunk_candidates_count=artifact_chunk_count,
            fact_facet_candidates_count=fact_facet_count,
            embedding_candidates_count=embedding_count,
            consequence_candidates_count=consequence_count,
            verbatim_evidence_search_candidates_count=verbatim_evidence_search_count,
            entity_candidates_count=0,
            total_before_fusion=total_before_fusion,
            total_after_fusion=len(raw_candidates),
            per_subquery_counts=per_subquery_counts,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_fts_query_execution_counts(
        raw_candidates: list[dict[str, Any]],
        sub_query: PlannedSubQuery,
        fts_query_audit: list[dict[str, Any]] | None = None,
    ) -> list[FtsQueryExecutionCount]:
        query_kinds = list(sub_query.fts_query_kinds)
        raw_rows_by_query = RetrievalPipeline._raw_fts_query_rows_by_signature(
            fts_query_audit,
        )
        executions: list[FtsQueryExecutionCount] = []
        seen_signatures: set[tuple[str, str, str]] = set()
        for index, fts_query in enumerate(sub_query.fts_queries):
            kind = str(query_kinds[index]) if index < len(query_kinds) else "unknown"
            signature = (sub_query.text, fts_query, kind)
            seen_signatures.add(signature)
            executions.append(
                FtsQueryExecutionCount(
                    query=fts_query,
                    kind=kind,
                    match_mode=CandidateSearch._fts_query_match_mode(fts_query),
                    source="planned",
                    non_evidential=True,
                    raw_rows=raw_rows_by_query.get(
                        (
                            sub_query.text,
                            fts_query,
                            kind,
                        ),
                        0,
                    ),
                    candidates=RetrievalPipeline._count_fts_query_matched_candidates(
                        raw_candidates,
                        subquery=sub_query.text,
                        query=fts_query,
                        kind=kind,
                    ),
                )
            )
        for entry in fts_query_audit or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("subquery") or "") != sub_query.text:
                continue
            query = str(entry.get("query") or "")
            kind = str(entry.get("kind") or "unknown")
            signature = (sub_query.text, query, kind)
            if not query or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            executions.append(
                FtsQueryExecutionCount(
                    query=query,
                    kind=kind,
                    match_mode=str(
                        entry.get("match_mode")
                        or CandidateSearch._fts_query_match_mode(query)
                    ),
                    source=str(entry.get("source") or "dynamic"),
                    non_evidential=bool(entry.get("non_evidential", True)),
                    raw_rows=raw_rows_by_query.get(signature, 0),
                    candidates=RetrievalPipeline._count_fts_query_matched_candidates(
                        raw_candidates,
                        subquery=sub_query.text,
                        query=query,
                        kind=kind,
                    ),
                )
            )
        return executions

    @staticmethod
    def _count_fts_query_matched_candidates(
        raw_candidates: list[dict[str, Any]],
        *,
        subquery: str,
        query: str,
        kind: str,
    ) -> int:
        matched_candidate_ids: set[str] = set()
        for candidate in raw_candidates:
            for match in candidate.get("fts_query_matches") or []:
                if not isinstance(match, dict):
                    continue
                if str(match.get("subquery") or "") != subquery:
                    continue
                if str(match.get("query") or "") != query:
                    continue
                if str(match.get("kind") or "unknown") != kind:
                    continue
                matched_candidate_ids.add(str(candidate.get("id") or ""))
        return len(matched_candidate_ids)

    @staticmethod
    def _raw_fts_query_rows_by_signature(
        fts_query_audit: list[dict[str, Any]] | None,
    ) -> Counter[tuple[str, str, str]]:
        raw_rows_by_query: Counter[tuple[str, str, str]] = Counter()
        for entry in fts_query_audit or []:
            if not isinstance(entry, dict):
                continue
            signature = (
                str(entry.get("subquery") or ""),
                str(entry.get("query") or ""),
                str(entry.get("kind") or "unknown"),
            )
            raw_rows_by_query[signature] += max(0, int(entry.get("raw_rows") or 0))
        return raw_rows_by_query

    @staticmethod
    def _build_scoring_trace(
        raw_candidates: list[dict[str, Any]],
        filtered_candidates: list[dict[str, Any]],
        scored_candidates: list[ScoredCandidate],
        duration_ms: float,
        rejection_reasons: dict[str, int] | None = None,
        policy_audit_reason_counts: dict[str, int] | None = None,
        scoring_candidates: list[dict[str, Any]] | None = None,
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
        gate_trace = RetrievalPipeline._build_applicability_gate_trace(
            scoring_candidates or []
        )
        return ScoringTrace(
            candidates_received=candidates_received,
            candidates_scored=len(scored_candidates),
            candidates_rejected=candidates_rejected,
            rejection_reasons=rejection_reasons or {},
            policy_audit_reason_counts=policy_audit_reason_counts or {},
            top_score=top_score,
            median_score=median_score,
            min_score=min_passing,
            **gate_trace,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_applicability_gate_trace(
        scoring_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata_rows = [
            metadata
            for candidate in scoring_candidates
            if isinstance(metadata := candidate.get("_applicability_gate"), dict)
        ]
        if not metadata_rows:
            return {
                "applicability_gate_mode": "off",
                "eligible_candidate_count": 0,
                "ineligible_reason_counts": {},
                "llm_applicability_skipped_count": 0,
                "shadow_disagreement_count": 0,
                "shadow_harmful_disagreement_count": 0,
                "estimated_calls_saved": 0,
                "adjacent_rrf_delta_distribution": {},
                "gate_reason": "mode_off",
            }
        modes = {str(row.get("mode") or "off") for row in metadata_rows}
        if "enforced" in modes:
            gate_mode = "enforced"
        elif "shadow" in modes:
            gate_mode = "shadow"
        else:
            gate_mode = "off"
        ineligible_reason_counts: dict[str, int] = {}
        eligible_candidate_count = 0
        llm_skipped_count = 0
        shadow_disagreement_count = 0
        shadow_harmful_disagreement_count = 0
        estimated_calls_saved = 0
        adjacent_rrf_delta_distribution: dict[str, float] = {}
        for row in metadata_rows:
            if bool(row.get("eligible")):
                eligible_candidate_count += 1
            else:
                reason = str(row.get("ineligible_reason") or "unknown")
                ineligible_reason_counts[reason] = (
                    ineligible_reason_counts.get(reason, 0) + 1
                )
            if bool(row.get("llm_skipped")):
                llm_skipped_count += 1
            if bool(row.get("shadow_disagreement")):
                shadow_disagreement_count += 1
            if bool(row.get("shadow_harmful_disagreement")):
                shadow_harmful_disagreement_count += 1
            estimated_calls_saved += int(row.get("estimated_calls_saved") or 0)
            if (
                not adjacent_rrf_delta_distribution
                and isinstance(row.get("adjacent_rrf_delta_distribution"), dict)
            ):
                adjacent_rrf_delta_distribution = {
                    str(key): float(value)
                    for key, value in row["adjacent_rrf_delta_distribution"].items()
                    if isinstance(value, (float, int))
                }
        gate_reason = "mode_off"
        if gate_mode == "enforced" and llm_skipped_count:
            gate_reason = "enforced_skipped"
        elif gate_mode == "enforced":
            gate_reason = "enforced_no_eligible_candidates"
        elif gate_mode == "shadow" and shadow_harmful_disagreement_count:
            gate_reason = "shadow_harmful_disagreement"
        elif gate_mode == "shadow" and shadow_disagreement_count:
            gate_reason = "shadow_disagreement"
        elif gate_mode == "shadow" and eligible_candidate_count:
            gate_reason = "shadow_eligible"
        elif gate_mode == "shadow":
            gate_reason = "shadow_no_eligible_candidates"
        return {
            "applicability_gate_mode": gate_mode,
            "eligible_candidate_count": eligible_candidate_count,
            "ineligible_reason_counts": ineligible_reason_counts,
            "llm_applicability_skipped_count": llm_skipped_count,
            "shadow_disagreement_count": shadow_disagreement_count,
            "shadow_harmful_disagreement_count": shadow_harmful_disagreement_count,
            "estimated_calls_saved": estimated_calls_saved,
            "adjacent_rrf_delta_distribution": adjacent_rrf_delta_distribution,
            "gate_reason": gate_reason,
        }

    def _build_composition_trace(
        self,
        composed_context: ComposedContext,
        resolved_policy: ResolvedRetrievalPolicy,
        duration_ms: float,
    ) -> CompositionTrace:
        contract_tokens = self._context_composer.estimate_tokens(
            composed_context.contract_block
        )
        workspace_tokens = self._context_composer.estimate_tokens(
            composed_context.workspace_block
        )
        memory_tokens = self._context_composer.estimate_tokens(
            composed_context.memory_block
        )
        state_tokens = self._context_composer.estimate_tokens(
            composed_context.state_block
        )
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
            answer_shape=composed_context.answer_shape,
            coverage_mode=composed_context.coverage_mode,
            source_precision=composed_context.source_precision,
            coverage_state=composed_context.coverage_state,
            duration_ms=duration_ms,
        )

    def _attach_stability_diagnostics(
        self,
        *,
        trace: RetrievalTrace,
        retrieval_plan: RetrievalPlan,
        candidate_custody: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        composed_context: ComposedContext,
    ) -> None:
        candidate_by_id = self._candidate_rows_by_id(candidate_rows)
        trace.facet_support = self._build_facet_support_trace(
            retrieval_plan=retrieval_plan,
            candidate_custody=candidate_custody,
            candidate_by_id=candidate_by_id,
        )
        trace.direct_vs_indirect_provenance = self._build_provenance_trace(
            retrieval_plan=retrieval_plan,
            candidate_custody=candidate_custody,
            candidate_by_id=candidate_by_id,
        )
        trace.token_budget = self._build_token_budget_trace(
            trace.composition,
            composed_context=composed_context,
        )
        trace.cross_conversation_raw_policy = self._build_cross_raw_policy_trace(
            retrieval_plan=retrieval_plan,
            candidate_custody=candidate_custody,
        )

    @staticmethod
    def _candidate_rows_by_id(
        candidate_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for row in candidate_rows:
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("id") or "")
            if candidate_id and candidate_id not in by_id:
                by_id[candidate_id] = row
        return by_id

    @classmethod
    def _build_facet_support_trace(
        cls,
        *,
        retrieval_plan: RetrievalPlan,
        candidate_custody: list[dict[str, Any]],
        candidate_by_id: dict[str, dict[str, Any]],
    ) -> FacetSupportTrace:
        obligations: list[FacetSupportObligationTrace] = []
        subqueries = list(retrieval_plan.sub_query_plans)
        for index, subquery in enumerate(subqueries):
            selected_records = [
                record
                for record in candidate_custody
                if record.get("selected") is True
                and cls._record_matches_obligation(record, index, len(subqueries))
            ]
            composed_ids = [
                str(record.get("candidate_id"))
                for record in selected_records
                if str(record.get("candidate_id") or "")
            ]
            supported_ids = [
                candidate_id
                for candidate_id in composed_ids
                if cls._proof_joined_to_base(
                    candidate_by_id.get(candidate_id, {}),
                    cls._proof_source_for_record(
                        next(
                            record
                            for record in selected_records
                            if str(record.get("candidate_id")) == candidate_id
                        ),
                        candidate_by_id.get(candidate_id, {}),
                    ),
                )
            ]
            status = (
                "covered"
                if supported_ids
                else ("partial" if composed_ids else "missing")
            )
            obligations.append(
                FacetSupportObligationTrace(
                    id=f"f{index + 1}",
                    description=subquery.text,
                    status=status,
                    selected_memory_ids=composed_ids,
                    composed_memory_ids=composed_ids,
                    support_verdict="supported" if supported_ids else "unsupported",
                )
            )
        return FacetSupportTrace(obligations=obligations)

    @classmethod
    def _build_provenance_trace(
        cls,
        *,
        retrieval_plan: RetrievalPlan,
        candidate_custody: list[dict[str, Any]],
        candidate_by_id: dict[str, dict[str, Any]],
    ) -> DirectVsIndirectProvenanceTrace:
        evidence: list[ProvenanceEvidenceTrace] = []
        direct_count = 0
        indirect_count = 0
        summary_only_count = 0
        raw_cross_count = 0
        for record in candidate_custody:
            if record.get("selected") is not True:
                continue
            candidate_id = str(record.get("candidate_id") or "")
            if not candidate_id:
                continue
            row = candidate_by_id.get(candidate_id, {})
            recovery_channels = cls._provenance_recovery_channels(
                record,
                retrieval_plan=retrieval_plan,
            )
            proof_source = cls._proof_source_for_record(record, row)
            joined_to_base = cls._proof_joined_to_base(row, proof_source)
            if cls._is_direct_recovery(recovery_channels):
                direct_count += 1
            else:
                indirect_count += 1
            if proof_source == "summary_only":
                summary_only_count += 1
            if "raw_cross_conversation" in recovery_channels:
                raw_cross_count += 1
            evidence.append(
                ProvenanceEvidenceTrace(
                    memory_id=candidate_id,
                    recovery_channels=recovery_channels,
                    proof_source=proof_source,
                    joined_to_base=joined_to_base,
                    selected=True,
                    matched_subquery_indexes=[
                        int(index)
                        for index in record.get("matched_subquery_indexes") or []
                    ],
                    scope=_safe_trace_str(record.get("scope")),
                    scope_canonical=_safe_trace_str(record.get("scope_canonical")),
                    conversation_id=_safe_trace_str(record.get("conversation_id")),
                )
            )
        return DirectVsIndirectProvenanceTrace(
            evidence=evidence,
            direct_recovery_count=direct_count,
            indirect_recovery_count=indirect_count,
            summary_only_count=summary_only_count,
            raw_cross_conversation_count=raw_cross_count,
        )

    @staticmethod
    def _build_token_budget_trace(
        composition: CompositionTrace | None,
        *,
        composed_context: ComposedContext,
    ) -> TokenBudgetTrace:
        if composition is None:
            return TokenBudgetTrace(
                context_tokens=composed_context.total_tokens_estimate
            )
        return TokenBudgetTrace(
            context_tokens=composed_context.total_tokens_estimate,
            token_budget_total=composition.token_budget_total,
            token_budget_used=composition.token_budget_used,
            contract_tokens=composition.contract_tokens,
            workspace_tokens=composition.workspace_tokens,
            memory_tokens=composition.memory_tokens,
            state_tokens=composition.state_tokens,
        )

    def _build_cross_raw_policy_trace(
        self,
        *,
        retrieval_plan: RetrievalPlan,
        candidate_custody: list[dict[str, Any]],
    ) -> CrossConversationRawPolicyTrace:
        enabled = CandidateSearch._allow_cross_conversation_verbatim_evidence(
            retrieval_plan
        )
        cross_raw_records = [
            record
            for record in candidate_custody
            if record.get("candidate_kind") == "verbatim_evidence_search_window"
            and _safe_trace_str(record.get("conversation_id"))
            not in {None, retrieval_plan.conversation_id}
        ]
        violations = [
            record
            for record in cross_raw_records
            if not self._cross_raw_record_policy_ok(record, retrieval_plan)
        ]
        activation_reason = None
        if enabled:
            if (
                retrieval_plan.query_type == "broad_list"
                or len(retrieval_plan.sub_query_plans) > 1
            ):
                activation_reason = "exact_recall_multi_obligation"
            elif len(retrieval_plan.exact_facets) > 1:
                activation_reason = "exact_recall_multi_exact_facets"
            else:
                activation_reason = "exact_recall_cross_conversation_allowed"
        return CrossConversationRawPolicyTrace(
            enabled=enabled,
            activation_reason=activation_reason,
            candidate_count=len(cross_raw_records),
            selected_count=sum(
                1 for record in cross_raw_records if record.get("selected") is True
            ),
            violation_count=len(violations),
            max_windows_per_subquery=int(self._settings.verbatim_evidence_search_limit),
            policy_filters=[
                "user_id",
                "visibility",
                "scope",
                "status",
                "privacy_ceiling",
                "sensitivity",
                "pending_consent",
                "incognito",
                "remember_across_chats",
                "platform_lock",
                "lifecycle_status",
            ],
        )

    @staticmethod
    def _record_matches_obligation(
        record: dict[str, Any],
        index: int,
        subquery_count: int,
    ) -> bool:
        indexes = record.get("matched_subquery_indexes") or []
        if index in indexes:
            return True
        return subquery_count == 1 and not indexes

    @staticmethod
    def _provenance_recovery_channels(
        record: dict[str, Any],
        *,
        retrieval_plan: RetrievalPlan,
    ) -> list[str]:
        channels = {
            str(channel)
            for channel in [
                *(record.get("channels") or []),
                *(record.get("retrieval_sources") or []),
            ]
            if str(channel)
        }
        if str(record.get("candidate_kind") or "") == "summary_view":
            channels.add("summary_fusion")
        if str(
            record.get("candidate_kind") or ""
        ) == "verbatim_evidence_search_window" and _safe_trace_str(
            record.get("conversation_id")
        ) not in {None, retrieval_plan.conversation_id}:
            channels.add("raw_cross_conversation")
        return sorted(channels)

    @staticmethod
    def _proof_source_for_record(
        record: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        candidate_kind = str(record.get("candidate_kind") or "")
        payload = candidate.get("payload_json") if isinstance(candidate, dict) else {}
        payload_json = payload if isinstance(payload, dict) else {}
        if candidate_kind == "verbatim_evidence_search_window":
            return (
                "raw_source_span"
                if payload_json.get("source_message_ids")
                else "raw_only"
            )
        if candidate_kind == "summary_view":
            return (
                "summary_joined"
                if payload_json.get("source_object_ids")
                else "summary_only"
            )
        if candidate_kind in {
            "evidence",
            "belief",
            "interaction_contract",
            "state_snapshot",
            "consequence_chain",
            "verbatim_pin",
            "artifact_chunk",
        }:
            return "base_canonical"
        return "derived_only"

    @staticmethod
    def _proof_joined_to_base(candidate: dict[str, Any], proof_source: str) -> bool:
        if proof_source == "base_canonical":
            return True
        payload = candidate.get("payload_json") if isinstance(candidate, dict) else {}
        payload_json = payload if isinstance(payload, dict) else {}
        if proof_source == "summary_joined":
            return bool(payload_json.get("source_object_ids"))
        if proof_source == "raw_source_span":
            return bool(payload_json.get("source_message_ids"))
        return False

    @staticmethod
    def _is_direct_recovery(recovery_channels: list[str]) -> bool:
        direct_channels = {
            "fts",
            "verbatim_evidence_search",
            "verbatim_pin",
            "artifact_chunk",
            "raw_cross_conversation",
        }
        return bool(direct_channels.intersection(recovery_channels))

    @staticmethod
    def _cross_raw_record_policy_ok(
        record: dict[str, Any],
        retrieval_plan: RetrievalPlan,
    ) -> bool:
        return (
            CandidateSearch._allow_cross_conversation_verbatim_evidence(retrieval_plan)
            and _safe_trace_str(record.get("scope_canonical")) == MemoryScope.USER.value
            and _safe_trace_str(record.get("status")) == MemoryStatus.ACTIVE.value
            and (
                record.get("privacy_level") is None
                or int(record.get("privacy_level") or 0)
                <= retrieval_plan.privacy_ceiling
            )
            and record.get("filter_reason") is None
        )

    @classmethod
    def _build_custody_trace(
        cls,
        *,
        raw_candidates: list[dict[str, Any]],
        filtered_candidates: list[dict[str, Any]],
        scored_candidates: list[ScoredCandidate],
        candidate_custody: list[dict[str, Any]],
        selected_memory_ids: list[str],
        retrieval_sufficiency: RetrievalSufficiencyDiagnostic | None = None,
    ) -> RetrievalCustodyTrace:
        candidate_count_by_channel = cls._candidate_count_by_channel(raw_candidates)
        selected_evidence_ids = cls._selected_evidence_ids_from_custody(
            candidate_custody
        )
        rendered_evidence_ids = cls._rendered_evidence_ids_from_custody(
            candidate_custody
        )
        high_value_rejected_reasons = cls._high_value_rejected_reasons(
            candidate_custody
        )
        selected_source_evidence_count = cls._selected_source_evidence_count(
            candidate_custody
        )
        high_value_rejected_count = sum(high_value_rejected_reasons.values())
        selected_candidate_count = len(selected_memory_ids)
        return RetrievalCustodyTrace(
            raw_candidate_count=len(raw_candidates),
            candidate_count_by_channel=candidate_count_by_channel,
            source_backed_candidate_count=cls._custody_bool_count(
                candidate_custody,
                field_name="source_backed",
            ),
            summary_only_candidate_count=cls._custody_bool_count(
                candidate_custody,
                field_name="summary_only",
            ),
            post_user_id_candidate_count=len(raw_candidates),
            post_scope_coordinate_lifecycle_candidate_count=len(filtered_candidates),
            scored_candidate_count=len(scored_candidates),
            selected_candidate_count=selected_candidate_count,
            selected_evidence_ids=selected_evidence_ids,
            selected_source_evidence_count=selected_source_evidence_count,
            selected_summary_count=cls._selected_summary_count(candidate_custody),
            high_value_rejected_candidate_count=high_value_rejected_count,
            high_value_rejected_reasons=high_value_rejected_reasons,
            candidate_found_but_not_selected=cls._candidate_found_but_not_selected(
                candidate_custody
            ),
            rendered_evidence_ids=rendered_evidence_ids,
            funnel_coverage_state=cls._funnel_coverage_state(
                retrieval_sufficiency=retrieval_sufficiency,
                selected_source_evidence_count=selected_source_evidence_count,
                selected_candidate_count=selected_candidate_count,
                high_value_rejected_count=high_value_rejected_count,
            ),
            source_window_ids=cls._source_window_ids_from_custody(
                candidate_custody,
                selected_only=False,
            ),
            selected_source_window_ids=cls._source_window_ids_from_custody(
                candidate_custody,
                selected_only=True,
            ),
            drop_counts_by_stage=cls._custody_drop_counts(
                candidate_custody,
                field_name="drop_stage",
            ),
            drop_counts_by_reason=cls._custody_drop_counts(
                candidate_custody,
                field_name="drop_reason",
            ),
        )

    @staticmethod
    def _candidate_count_by_channel(
        raw_candidates: list[dict[str, Any]],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for candidate in raw_candidates:
            candidate_channels: set[str] = set()
            channels = candidate.get("retrieval_sources")
            if isinstance(channels, list):
                candidate_channels.update(
                    str(channel) for channel in channels if str(channel)
                )
            channel_ranks = candidate.get("channel_ranks")
            if isinstance(channel_ranks, dict):
                candidate_channels.update(
                    str(channel)
                    for channel, rank in channel_ranks.items()
                    if str(channel) and rank is not None
                )
            if candidate.get("is_verbatim_pin"):
                candidate_channels.add("verbatim_pin")
            if candidate.get("is_artifact_chunk"):
                candidate_channels.add("artifact_chunk")
            if candidate.get("is_verbatim_evidence_window"):
                candidate_channels.add("verbatim_evidence_search")
            for channel in candidate_channels:
                counts[channel] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _custody_bool_count(
        candidate_custody: list[dict[str, Any]],
        *,
        field_name: str,
    ) -> int:
        return sum(1 for record in candidate_custody if record.get(field_name) is True)

    @staticmethod
    def _selected_evidence_ids_from_custody(
        candidate_custody: list[dict[str, Any]],
    ) -> list[str]:
        evidence_kinds = {
            "evidence",
            "verbatim_evidence_search_window",
            "raw_source_span",
        }
        ids: list[str] = []
        seen: set[str] = set()
        for record in candidate_custody:
            if record.get("selected") is not True:
                continue
            if str(record.get("candidate_kind") or "") not in evidence_kinds:
                continue
            candidate_id = str(record.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in seen:
                ids.append(candidate_id)
                seen.add(candidate_id)
        return ids

    @staticmethod
    def _rendered_evidence_ids_from_custody(
        candidate_custody: list[dict[str, Any]],
    ) -> list[str]:
        evidence_kinds = {
            "evidence",
            "verbatim_evidence_search_window",
            "raw_source_span",
            "artifact_chunk",
            "verbatim_pin",
        }
        ids: list[str] = []
        seen: set[str] = set()
        for record in candidate_custody:
            if record.get("rendered") is not True:
                continue
            if str(record.get("candidate_kind") or "") not in evidence_kinds:
                continue
            candidate_id = str(record.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in seen:
                ids.append(candidate_id)
                seen.add(candidate_id)
        return ids

    @staticmethod
    def _selected_source_evidence_count(
        candidate_custody: list[dict[str, Any]],
    ) -> int:
        return sum(
            1
            for record in candidate_custody
            if record.get("selected") is True and record.get("source_backed") is True
        )

    @staticmethod
    def _selected_summary_count(candidate_custody: list[dict[str, Any]]) -> int:
        return sum(
            1
            for record in candidate_custody
            if record.get("selected") is True and record.get("summary_only") is True
        )

    @staticmethod
    def _candidate_found_but_not_selected(
        candidate_custody: list[dict[str, Any]],
    ) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for record in candidate_custody:
            if record.get("selected") is True:
                continue
            if record.get("source_backed") is not True:
                continue
            candidate_id = str(record.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in seen:
                ids.append(candidate_id)
                seen.add(candidate_id)
        return ids

    @staticmethod
    def _high_value_rejected_reasons(
        candidate_custody: list[dict[str, Any]],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in candidate_custody:
            if record.get("high_value_rejected") is not True:
                continue
            reason = str(record.get("eviction_reason") or "unknown").strip()
            counts[reason or "unknown"] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _funnel_coverage_state(
        *,
        retrieval_sufficiency: RetrievalSufficiencyDiagnostic | None,
        selected_source_evidence_count: int,
        selected_candidate_count: int,
        high_value_rejected_count: int,
    ) -> str:
        if retrieval_sufficiency is not None:
            if retrieval_sufficiency.state == "contradictory_candidates":
                return "conflicting"
            if (
                retrieval_sufficiency.state
                in _INSUFFICIENT_RETRIEVAL_SUFFICIENCY_STATES
            ):
                return "insufficient"
        if high_value_rejected_count > 0:
            return "partial"
        if selected_source_evidence_count > 0:
            return "complete"
        if selected_candidate_count > 0:
            return "unknown"
        return "insufficient"

    @staticmethod
    def _source_window_ids_from_custody(
        candidate_custody: list[dict[str, Any]],
        *,
        selected_only: bool,
    ) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for record in candidate_custody:
            if selected_only and record.get("selected") is not True:
                continue
            source_window_id = str(record.get("source_window_id") or "").strip()
            if source_window_id and source_window_id not in seen:
                ids.append(source_window_id)
                seen.add(source_window_id)
        return ids

    @staticmethod
    def _custody_drop_counts(
        candidate_custody: list[dict[str, Any]],
        *,
        field_name: str,
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for record in candidate_custody:
            value = str(record.get(field_name) or "").strip()
            if value:
                counts[value] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _nonnegative_float_map(values: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in values.items():
            normalized[str(key)] = max(0.0, float(value or 0.0))
        return normalized

    @classmethod
    def _hydration_timings(cls, stage_timings: dict[str, float]) -> dict[str, float]:
        hydration_keys = {
            key
            for key in stage_timings
            if key.endswith("_lookup")
            or key
            in {
                "coverage_candidate_expansion",
                "context_composition",
                "workspace_rollup_lookup",
            }
        }
        return cls._nonnegative_float_map(
            {key: stage_timings[key] for key in sorted(hydration_keys)}
        )

    def _without_contract_block(
        self, composed_context: ComposedContext
    ) -> ComposedContext:
        contract_tokens = self._context_composer.estimate_tokens(
            composed_context.contract_block
        )
        return composed_context.model_copy(
            update={
                "contract_block": "",
                "total_tokens_estimate": max(
                    0, composed_context.total_tokens_estimate - contract_tokens
                ),
            }
        )
