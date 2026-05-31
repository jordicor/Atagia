"""Adaptive context-cache orchestration above the fresh retrieval path."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import sqlite3
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import aiosqlite

from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import close_connection, open_connection
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.canonical import canonical_json_bytes
from atagia.memory.context_composer import ContextComposer
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.context_staleness import (
    ContextStalenessRequest,
    ContextStalenessScore,
    ContextStalenessScorer,
)
from atagia.memory.policy_manifest import (
    ResolvedRetrievalPolicy,
    compute_effective_policy_hash,
)
from atagia.models.schemas_api import MemorySummary
from atagia.models.schemas_cache import ContextCacheEntry
from atagia.models.schemas_memory import (
    ComposedContext,
    RequestRuntimeDiagnosticsTrace,
    ResolvedOperationalProfile,
    ResponseMode,
    RetrievalTrace,
)
from atagia.memory.lifecycle_runner import cache_generation_key
from atagia.models.schemas_replay import AblationConfig, PipelineResult

if TYPE_CHECKING:
    from atagia.app import AppRuntime
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    answer_support_prompt_payload,
    apply_conversation_policy_overlay,
    build_memory_summaries,
    default_operational_profile_snapshot,
    resolve_retrieval_profile_id,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.services.errors import ConversationNotFoundError
from atagia.services.initial_context_package_signatures import (
    InitialContextPackageCacheInvalidationResult,
    invalidate_initial_context_package_dependency as invalidate_context_package_dependency,
)
from atagia.services.prompt_authority import (
    PromptAuthorityContext,
    effective_allow_private_for_sql_repository,
    normalize_request_authority_context,
    privacy_sql_filters_disabled,
)
from atagia.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

CACHE_GUARD_TTL_SECONDS = 5 * 60
CACHE_GUARD_ACQUIRE_TIMEOUT_SECONDS = 30.0
CACHE_GUARD_INITIAL_DELAY_SECONDS = 0.01
CACHE_GUARD_MAX_DELAY_SECONDS = 0.25
CONTEXT_CACHE_KEY_VERSION = 12
DISCARDABLE_CACHE_SIGNALS = frozenset(
    {
        "assistant_mode_id_mismatch",
        "cache_entry_validation_failed",
        "cached_at_invalid",
        "conversation_id_mismatch",
        "effective_policy_hash_mismatch",
        "message_sequence_rewind",
        "operational_profile_mismatch",
        "policy_prompt_hash_mismatch",
        "user_id_mismatch",
        "workspace_id_mismatch",
    }
)


@dataclass(slots=True)
class AdaptiveContextResolution:
    """Normalized result returned by adaptive context resolution."""

    conversation: dict[str, Any]
    resolved_policy: ResolvedRetrievalPolicy
    resolved_operational_profile: ResolvedOperationalProfile
    composed_context: ComposedContext
    current_contract: dict[str, dict[str, Any]]
    memory_summaries: list[MemorySummary]
    detected_needs: list[str]
    stage_timings: dict[str, float]
    from_cache: bool
    staleness: float
    next_refresh_strategy: Literal["cache", "sync"]
    cache_age_seconds: float | None
    cache_source: Literal["sync", "cache_hit"] | None
    need_detection_skipped: bool
    cache_key: str | None
    source_retrieval_plan: dict[str, Any]
    scored_candidates: list[dict[str, Any]]
    candidate_custody: list[dict[str, Any]]
    retrieval_custody_v2_status: Literal["fresh", "cache_hit_no_candidate_custody"]
    retrieval_sufficiency: dict[str, Any] | None
    sufficiency_diagnostics_v1_status: Literal[
        "fresh", "cache_hit_no_sufficiency_diagnostics"
    ]
    candidate_search_summary: dict[str, Any]
    retrieval_diagnostics_for_guard: dict[str, Any]
    retrieval_trace: dict[str, Any] | None
    pending_cache_entry: ContextCacheEntry | None
    cache_ttl_seconds: int | None
    cache_generation: int = 0


@dataclass(slots=True)
class ContextCacheService:
    """Resolve context from a stable cache entry when it is still safe to reuse."""

    runtime: AppRuntime
    _staleness_scorer: ContextStalenessScorer = field(init=False)

    def __post_init__(self) -> None:
        self._staleness_scorer = ContextStalenessScorer(
            clock=self.runtime.clock,
            llm_client=self.runtime.llm_client,
            settings=self.runtime.settings,
        )

    @asynccontextmanager
    async def user_cache_guard(self, user_id: str) -> AsyncIterator[None]:
        """Serialize interactive cache reads and invalidations for one user."""
        guard_key = self.build_user_guard_key(user_id)
        token = await self._acquire_guard(guard_key)
        try:
            yield
        finally:
            await self.runtime.storage_backend.release_lock(
                guard_key,
                token,
            )

    async def resolve_with_connection(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        assistant_mode_id: str | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
        conversation: dict[str, Any] | None = None,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        ablation: AblationConfig | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
        response_mode: ResponseMode = ResponseMode.NORMAL,
    ) -> AdaptiveContextResolution:
        cache_generation = await self.runtime.storage_backend.get_cache_generation(
            cache_generation_key(self.runtime.database_path, user_id)
        )
        conversations = ConversationRepository(connection, self.runtime.clock)
        messages = MessageRepository(connection, self.runtime.clock)
        active_conversation = conversation or await conversations.get_conversation(
            conversation_id, user_id
        )
        if active_conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")
        authority_context = (
            prompt_authority_context
            or normalize_request_authority_context(
                privacy_enforcement=(
                    ablation.privacy_enforcement
                    if ablation is not None
                    else privacy_enforcement
                ),
                authenticated_user_privilege_level=authenticated_user_privilege_level,
                authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
                user_id=user_id,
                purpose="context_cache",
            )
        )

        resolved_mode_id = resolve_retrieval_profile_id(
            str(active_conversation["assistant_mode_id"]),
            assistant_mode_id,
        )
        resolved_operational_profile = resolve_operational_profile(
            loader=self.runtime.operational_profile_loader,
            settings=self.runtime.settings,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )
        resolved_policy = resolve_policy(
            self.runtime.manifests,
            resolved_mode_id,
            self.runtime.policy_resolver,
            resolved_operational_profile,
        )
        resolved_policy = apply_conversation_policy_overlay(
            resolved_policy,
            active_conversation,
        )
        effective_policy_hash = compute_effective_policy_hash(resolved_policy)
        current_messages = stored_messages
        if current_messages is None:
            current_messages = await messages.get_recent_messages(
                conversation_id,
                user_id,
                limit=RECENT_FETCH_LIMIT,
            )

        cache_key = self.build_cache_key(
            user_id=user_id,
            assistant_mode_id=resolved_mode_id,
            conversation_id=conversation_id,
            workspace_id=active_conversation.get("workspace_id"),
            active_presence_id=active_conversation.get("active_presence_id"),
            active_space_id=active_conversation.get("active_space_id"),
            active_mind_id=active_conversation.get("active_mind_id"),
            mind_topology=active_conversation.get("mind_topology"),
            active_embodiment_id=active_conversation.get("active_embodiment_id"),
            active_realm_id=active_conversation.get("active_realm_id"),
            operational_profile_token=resolved_operational_profile.snapshot.token,
            privacy_enforcement=authority_context.effective_privacy_enforcement,
            authenticated_user_privilege_level=authority_context.normalized_privilege_level,
            authenticated_user_is_atagia_master=authority_context.authenticated_user_is_atagia_master,
            response_mode=response_mode,
        )
        current_message_seq = self._next_message_seq(current_messages)
        cache_lookup_started = perf_counter()
        cache_allowed = (
            self._cache_enabled(ablation)
            and str(active_conversation.get("mind_topology") or "unimind")
            != "ojocentauri"
        )
        raw_entry: dict[str, Any] | None = None
        if cache_allowed:
            raw_entry = await self.runtime.storage_backend.get_context_view(cache_key)
        cache_lookup_elapsed = perf_counter() - cache_lookup_started

        cache_score: ContextStalenessScore | None = None
        cache_age_seconds: float | None = None
        if raw_entry is not None:
            staleness_started = perf_counter()
            cache_score = await self._staleness_scorer.score(
                raw_entry,
                ContextStalenessRequest(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=active_conversation.get("workspace_id"),
                    message_text=message_text,
                    current_message_seq=current_message_seq,
                    cache_enabled=cache_allowed,
                    operational_profile=resolved_operational_profile.snapshot,
                    effective_policy_hash=effective_policy_hash,
                    benchmark_mode=False,
                    replay_mode=False,
                    evaluation_mode=False,
                    mcp_mode=False,
                ),
                resolved_policy,
            )
            staleness_elapsed = perf_counter() - staleness_started
            cache_age_seconds = self._cache_age_seconds(raw_entry)
            if self._should_discard_cache_entry(cache_score):
                await self.runtime.storage_backend.delete_context_view(cache_key)
            if not cache_score.should_refresh:
                entry = ContextCacheEntry.model_validate(raw_entry)
                return AdaptiveContextResolution(
                    conversation=active_conversation,
                    resolved_policy=resolved_policy,
                    resolved_operational_profile=resolved_operational_profile,
                    composed_context=entry.composed_context,
                    current_contract=entry.contract,
                    memory_summaries=entry.memory_summaries,
                    detected_needs=[],
                    stage_timings={
                        "context_cache_lookup": cache_lookup_elapsed,
                        "context_cache_staleness": staleness_elapsed,
                    },
                    from_cache=True,
                    staleness=cache_score.staleness,
                    next_refresh_strategy="cache",
                    cache_age_seconds=cache_age_seconds,
                    cache_source="cache_hit",
                    need_detection_skipped=True,
                    cache_key=cache_key,
                    source_retrieval_plan=dict(entry.source_retrieval_plan),
                    scored_candidates=[],
                    candidate_custody=[],
                    retrieval_custody_v2_status="cache_hit_no_candidate_custody",
                    retrieval_sufficiency=None,
                    sufficiency_diagnostics_v1_status="cache_hit_no_sufficiency_diagnostics",
                    candidate_search_summary={},
                    retrieval_diagnostics_for_guard=dict(
                        entry.retrieval_diagnostics_for_guard
                    ),
                    retrieval_trace=None,
                    pending_cache_entry=None,
                    cache_ttl_seconds=None,
                    cache_generation=cache_generation,
                )
        else:
            staleness_elapsed = 0.0

        retrieval_trace = RetrievalTrace(
            query_text=message_text,
            user_id=user_id,
            conversation_id=conversation_id,
            requested_mode=assistant_mode_id,
            effective_mode=resolved_mode_id,
            timestamp_iso=self.runtime.clock.now().isoformat(),
            privacy_enforcement=authority_context.effective_privacy_enforcement,
        )
        db_diagnostics = _SqliteRequestDiagnostics()
        sqlite_restore = _install_sqlite_request_diagnostics(connection, db_diagnostics)
        try:
            pipeline_result = await RetrievalService(
                self.runtime
            ).retrieve_with_connection(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                message_text=message_text,
                mode=resolved_mode_id,
                ablation=ablation,
                prompt_authority_context=authority_context,
                conversation=active_conversation,
                stored_messages=current_messages,
                operational_profile=resolved_operational_profile,
                trace=retrieval_trace,
            )
        except Exception as exc:
            db_diagnostics.record_exception(exc)
            raise
        finally:
            sqlite_restore.restore()
        memory_summaries = build_memory_summaries(pipeline_result)
        retrieval_diagnostics_for_guard = self._guard_retrieval_diagnostics(
            pipeline_result
        )
        pending_entry: ContextCacheEntry | None = None
        cache_ttl_seconds: int | None = None
        if cache_allowed:
            pending_entry = ContextCacheEntry(
                cache_key=cache_key,
                user_id=user_id,
                conversation_id=conversation_id,
                assistant_mode_id=resolved_mode_id,
                policy_prompt_hash=resolved_policy.prompt_hash,
                effective_policy_hash=effective_policy_hash,
                operational_profile=resolved_operational_profile.snapshot,
                workspace_id=active_conversation.get("workspace_id"),
                composed_context=pipeline_result.composed_context,
                contract=pipeline_result.current_contract,
                memory_summaries=memory_summaries,
                detected_needs=[
                    need.need_type.value for need in pipeline_result.detected_needs
                ],
                source_retrieval_plan=pipeline_result.retrieval_plan.model_dump(
                    mode="json"
                ),
                retrieval_diagnostics_for_guard=retrieval_diagnostics_for_guard,
                selected_memory_ids=list(
                    pipeline_result.composed_context.selected_memory_ids
                ),
                cached_at=self.runtime.clock.now().isoformat(),
                last_retrieval_message_seq=0,
                last_user_message_text=message_text,
                source="sync",
            )
            cache_ttl_seconds = self._cache_ttl_seconds(resolved_policy)

        stage_timings = dict(pipeline_result.stage_timings)
        stage_timings["context_cache_lookup"] = cache_lookup_elapsed
        stage_timings["context_cache_staleness"] = staleness_elapsed
        retrieval_trace_payload = self._retrieval_trace_payload(
            pipeline_result,
            stage_timings=stage_timings,
            db_diagnostics=db_diagnostics,
        )
        return AdaptiveContextResolution(
            conversation=active_conversation,
            resolved_policy=resolved_policy,
            resolved_operational_profile=resolved_operational_profile,
            composed_context=pipeline_result.composed_context,
            current_contract=pipeline_result.current_contract,
            memory_summaries=memory_summaries,
            detected_needs=[
                need.need_type.value for need in pipeline_result.detected_needs
            ],
            stage_timings=stage_timings,
            from_cache=False,
            staleness=cache_score.staleness if cache_score is not None else 1.0,
            next_refresh_strategy="sync",
            cache_age_seconds=cache_age_seconds,
            cache_source="sync",
            need_detection_skipped=False,
            cache_key=cache_key,
            source_retrieval_plan=pipeline_result.retrieval_plan.model_dump(
                mode="json"
            ),
            scored_candidates=[
                candidate.model_dump(mode="json")
                for candidate in pipeline_result.scored_candidates
            ],
            candidate_custody=list(pipeline_result.candidate_custody),
            retrieval_custody_v2_status="fresh",
            retrieval_sufficiency=(
                pipeline_result.retrieval_sufficiency.model_dump(mode="json")
                if pipeline_result.retrieval_sufficiency is not None
                else None
            ),
            sufficiency_diagnostics_v1_status="fresh",
            candidate_search_summary=self._summarize_candidate_channels(
                pipeline_result.raw_candidates,
                top_k=pipeline_result.retrieval_plan.max_candidates,
            ),
            retrieval_diagnostics_for_guard=retrieval_diagnostics_for_guard,
            retrieval_trace=retrieval_trace_payload,
            pending_cache_entry=pending_entry,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_generation=cache_generation,
        )

    async def resolve_fast_with_connection(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        response_mode: ResponseMode,
        assistant_mode_id: str | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
        conversation: dict[str, Any] | None = None,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        ablation: AblationConfig | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
    ) -> AdaptiveContextResolution:
        """Assemble context without the retrieval pipeline (fast / smart_fast).

        Context = prepared initial-context package (read by the caller) + recent
        transcript (built by the caller) + a cheap interaction-contract SQL read.
        For ``smart_fast`` the most recent warmed entry from the smart_fast key
        space is folded in when present (a plain cache read, no staleness call).
        The retrieval pipeline, candidate search, applicability scoring, need
        detection and cache-staleness scoring are never invoked here.
        """
        if response_mode is ResponseMode.NORMAL:
            raise ValueError("resolve_fast_with_connection requires a non-normal mode")
        fast_started = perf_counter()
        cache_generation = await self.runtime.storage_backend.get_cache_generation(
            cache_generation_key(self.runtime.database_path, user_id)
        )
        conversations = ConversationRepository(connection, self.runtime.clock)
        messages = MessageRepository(connection, self.runtime.clock)
        active_conversation = conversation or await conversations.get_conversation(
            conversation_id, user_id
        )
        if active_conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")
        authority_context = (
            prompt_authority_context
            or normalize_request_authority_context(
                privacy_enforcement=(
                    ablation.privacy_enforcement
                    if ablation is not None
                    else privacy_enforcement
                ),
                authenticated_user_privilege_level=authenticated_user_privilege_level,
                authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
                user_id=user_id,
                purpose="context_cache_fast",
            )
        )
        resolved_mode_id = resolve_retrieval_profile_id(
            str(active_conversation["assistant_mode_id"]),
            assistant_mode_id,
        )
        resolved_operational_profile = resolve_operational_profile(
            loader=self.runtime.operational_profile_loader,
            settings=self.runtime.settings,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )
        resolved_policy = resolve_policy(
            self.runtime.manifests,
            resolved_mode_id,
            self.runtime.policy_resolver,
            resolved_operational_profile,
        )
        resolved_policy = apply_conversation_policy_overlay(
            resolved_policy,
            active_conversation,
        )
        if stored_messages is None:
            stored_messages = await messages.get_recent_messages(
                conversation_id,
                user_id,
                limit=RECENT_FETCH_LIMIT,
            )
        cache_key = self.build_cache_key(
            user_id=user_id,
            assistant_mode_id=resolved_mode_id,
            conversation_id=conversation_id,
            workspace_id=active_conversation.get("workspace_id"),
            active_presence_id=active_conversation.get("active_presence_id"),
            active_space_id=active_conversation.get("active_space_id"),
            active_mind_id=active_conversation.get("active_mind_id"),
            mind_topology=active_conversation.get("mind_topology"),
            active_embodiment_id=active_conversation.get("active_embodiment_id"),
            active_realm_id=active_conversation.get("active_realm_id"),
            operational_profile_token=resolved_operational_profile.snapshot.token,
            privacy_enforcement=authority_context.effective_privacy_enforcement,
            authenticated_user_privilege_level=authority_context.normalized_privilege_level,
            authenticated_user_is_atagia_master=authority_context.authenticated_user_is_atagia_master,
            response_mode=response_mode,
        )
        stage_timings: dict[str, float] = {}
        contract_started = perf_counter()
        current_contract = await self._fast_contract_lookup(
            connection,
            conversation=active_conversation,
            resolved_policy=resolved_policy,
            authority_context=authority_context,
            ablation=ablation,
        )
        stage_timings["contract_lookup"] = perf_counter() - contract_started
        contract_block = ContextComposer.render_contract_block(
            current_contract,
            resolved_policy,
        )
        warmed_entry: ContextCacheEntry | None = None
        warmed_memory_summaries: list[MemorySummary] = []
        warmed_composed: ComposedContext | None = None
        if response_mode is ResponseMode.SMART_FAST:
            warm_started = perf_counter()
            raw_warm = await self.runtime.storage_backend.get_context_view(cache_key)
            stage_timings["smart_fast_warm_lookup"] = perf_counter() - warm_started
            if raw_warm is not None:
                try:
                    warmed_entry = ContextCacheEntry.model_validate(raw_warm)
                except Exception:
                    logger.warning(
                        "Discarding malformed smart_fast warm entry for "
                        "user_id=%s conversation_id=%s",
                        user_id,
                        conversation_id,
                    )
                    warmed_entry = None
            if warmed_entry is not None:
                warmed_composed = warmed_entry.composed_context
                warmed_memory_summaries = list(warmed_entry.memory_summaries)
        if warmed_composed is not None:
            composed_context = warmed_composed.model_copy(
                update={"contract_block": contract_block}
            )
        else:
            composed_context = ComposedContext(
                contract_block=contract_block,
                total_tokens_estimate=ContextComposer.estimate_tokens(contract_block),
                budget_tokens=0,
                items_included=0,
                items_dropped=0,
            )
        warm_present = warmed_entry is not None
        source_retrieval_plan: dict[str, Any] = {
            "raw_context_access_mode": "normal",
            "response_mode": response_mode.value,
            "fast_mode": True,
            "smart_fast_warm_entry_present": warm_present,
        }
        retrieval_trace = RetrievalTrace(
            query_text=message_text,
            user_id=user_id,
            conversation_id=conversation_id,
            requested_mode=assistant_mode_id,
            effective_mode=resolved_mode_id,
            response_mode=response_mode.value,
            timestamp_iso=self.runtime.clock.now().isoformat(),
            privacy_enforcement=authority_context.effective_privacy_enforcement,
        )
        stage_timings["context_fast_assembly"] = perf_counter() - fast_started
        retrieval_diagnostics_for_guard = {
            "response_mode": response_mode.value,
            "fast_mode": True,
            "smart_fast_warm_entry_present": warm_present,
            "selected_memory_ids": list(composed_context.selected_memory_ids),
            "selected_memory_count": len(composed_context.selected_memory_ids),
        }
        return AdaptiveContextResolution(
            conversation=active_conversation,
            resolved_policy=resolved_policy,
            resolved_operational_profile=resolved_operational_profile,
            composed_context=composed_context,
            current_contract=current_contract,
            memory_summaries=warmed_memory_summaries,
            detected_needs=[],
            stage_timings=stage_timings,
            from_cache=False,
            staleness=1.0,
            next_refresh_strategy="sync",
            cache_age_seconds=None,
            cache_source=None,
            need_detection_skipped=True,
            cache_key=cache_key,
            source_retrieval_plan=source_retrieval_plan,
            scored_candidates=[],
            candidate_custody=[],
            retrieval_custody_v2_status="cache_hit_no_candidate_custody",
            retrieval_sufficiency=None,
            sufficiency_diagnostics_v1_status="cache_hit_no_sufficiency_diagnostics",
            candidate_search_summary={},
            retrieval_diagnostics_for_guard=retrieval_diagnostics_for_guard,
            retrieval_trace=retrieval_trace.model_dump(mode="json"),
            pending_cache_entry=None,
            cache_ttl_seconds=None,
            cache_generation=cache_generation,
        )

    async def _fast_contract_lookup(
        self,
        connection: aiosqlite.Connection,
        *,
        conversation: dict[str, Any],
        resolved_policy: ResolvedRetrievalPolicy,
        authority_context: PromptAuthorityContext,
        ablation: AblationConfig | None,
    ) -> dict[str, dict[str, Any]]:
        """Run the single cheap interaction-contract SQL read for fast modes."""
        if ablation is not None and ablation.skip_contract_memory:
            return {}
        # Resolve sensitivity gating through the SAME effective ablation and the
        # SAME predicates the full retrieval pipeline uses, so the fast path
        # admits exactly the sensitivity tiers normal retrieval would. Master
        # authority is folded into ``effective_privacy_enforcement`` here, never
        # read again inside the predicates (mirrors retrieval_service's
        # ``retrieval_ablation`` derivation).
        effective_ablation = (
            ablation
            if ablation is not None
            else AblationConfig(
                privacy_enforcement=authority_context.effective_privacy_enforcement
            )
        )
        user_id = str(conversation["user_id"])
        memory_preferences = await UserRepository(
            connection, self.runtime.clock
        ).get_memory_preferences(user_id)
        projector = ContractProjector(
            self.runtime.llm_client,
            self.runtime.clock,
            MessageRepository(connection, self.runtime.clock),
            MemoryObjectRepository(connection, self.runtime.clock),
            ContractDimensionRepository(connection, self.runtime.clock),
            settings=self.runtime.settings,
        )
        character_id = (
            conversation.get("character_id")
            if conversation.get("character_id") is not None
            else conversation.get("workspace_id")
        )
        return await projector.get_current_contract(
            user_id,
            resolved_policy.profile_id.value,
            conversation.get("workspace_id"),
            str(conversation["id"]),
            user_persona_id=conversation.get("user_persona_id"),
            platform_id=str(conversation.get("platform_id") or "default"),
            character_id=character_id,
            incognito=bool(conversation.get("incognito"))
            or bool(conversation.get("isolated_mode")),
            remember_across_chats=bool(memory_preferences["remember_across_chats"]),
            remember_across_devices=bool(memory_preferences["remember_across_devices"]),
            sensitivity_gates_enabled=privacy_sql_filters_disabled(effective_ablation),
            allow_private_sensitivity=effective_allow_private_for_sql_repository(
                resolved_policy,
                effective_ablation,
            ),
            active_space_id=conversation.get("active_space_id"),
            active_space_boundary_mode=conversation.get("active_space_boundary_mode")
            or "focus",
            active_mind_id=conversation.get("active_mind_id"),
            mind_topology=conversation.get("mind_topology") or "unimind",
            active_embodiment_id=conversation.get("active_embodiment_id"),
            active_realm_id=conversation.get("active_realm_id"),
        )

    def schedule_smart_fast_warm(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        assistant_mode_id: str | None,
        operational_profile: str | None,
        operational_signals: Any | None,
        ablation: AblationConfig | None,
        prompt_authority_context: PromptAuthorityContext,
        last_retrieval_message_seq: int,
    ) -> None:
        """Run the normal retrieval in the background and warm the smart_fast key.

        The background task NEVER reuses the request's connection: it opens its
        own connection. The retrieval resolve runs UNGUARDED on that connection
        (concurrent reads alongside foreground requests are the same model the
        stream workers already use, and the resolve writes nothing to SQLite),
        and the per-user cache guard is held ONLY around the smart_fast cache
        write so a rapid same-user follow-up never blocks behind the warm's LLM
        calls. Failures are logged (warning) and never propagate to the
        already-sent answer.
        """
        if getattr(self.runtime, "closed", False):
            return
        self.runtime.spawn_background_task(
            self._run_smart_fast_warm(
                user_id=user_id,
                conversation_id=conversation_id,
                message_text=message_text,
                assistant_mode_id=assistant_mode_id,
                operational_profile=operational_profile,
                operational_signals=operational_signals,
                ablation=ablation,
                prompt_authority_context=prompt_authority_context,
                last_retrieval_message_seq=last_retrieval_message_seq,
            ),
            name="atagia-smart-fast-warm",
        )

    async def _run_smart_fast_warm(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        assistant_mode_id: str | None,
        operational_profile: str | None,
        operational_signals: Any | None,
        ablation: AblationConfig | None,
        prompt_authority_context: PromptAuthorityContext,
        last_retrieval_message_seq: int,
    ) -> None:
        """Resolve the warm context unguarded, then publish under the guard.

        Guard window: the per-user cache guard is acquired ONLY around
        ``publish_pending_cache_entry`` (the single smart_fast cache write). The
        retrieval resolve -- which makes the warm's LLM calls and writes nothing
        to SQLite -- runs outside the guard on its own connection, so it cannot
        starve a foreground same-user turn waiting on the same guard.
        """
        try:
            connection = await open_connection(self.runtime.database_path)
            try:
                resolution = await self.resolve_with_connection(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_text=message_text,
                    assistant_mode_id=assistant_mode_id,
                    operational_profile=operational_profile,
                    operational_signals=operational_signals,
                    ablation=ablation,
                    prompt_authority_context=prompt_authority_context,
                    response_mode=ResponseMode.SMART_FAST,
                )
            finally:
                await close_connection(connection)
            async with self.user_cache_guard(user_id):
                published = await self.publish_pending_cache_entry(
                    resolution,
                    last_retrieval_message_seq=last_retrieval_message_seq,
                )
            logger.info(
                "smart_fast warm completed user_id=%s conversation_id=%s "
                "cache_key=%s published=%s selected_memory_ids=%d from_cache=%s",
                user_id,
                conversation_id,
                resolution.cache_key,
                published,
                len(resolution.composed_context.selected_memory_ids),
                resolution.from_cache,
            )
        except Exception:
            logger.warning(
                "smart_fast warm failed for user_id=%s conversation_id=%s",
                user_id,
                conversation_id,
                exc_info=True,
            )

    async def publish_pending_cache_entry(
        self,
        resolution: AdaptiveContextResolution,
        *,
        last_retrieval_message_seq: int,
    ) -> bool:
        if (
            resolution.pending_cache_entry is None
            or resolution.cache_ttl_seconds is None
        ):
            return False
        current_gen = await self.runtime.storage_backend.get_cache_generation(
            cache_generation_key(
                self.runtime.database_path,
                resolution.pending_cache_entry.user_id,
            )
        )
        if current_gen != resolution.cache_generation:
            logger.debug(
                "Skipping stale cache publish for user %s (gen %d != %d)",
                resolution.pending_cache_entry.user_id,
                resolution.cache_generation,
                current_gen,
            )
            return False
        entry = resolution.pending_cache_entry.model_copy(
            update={
                "cached_at": self.runtime.clock.now().isoformat(),
                "last_retrieval_message_seq": last_retrieval_message_seq,
            }
        )
        return await self.runtime.storage_backend.set_context_view_if_newer(
            entry.cache_key,
            entry.model_dump(mode="json"),
            ttl_seconds=resolution.cache_ttl_seconds,
            monotonic_seq=last_retrieval_message_seq,
        )

    async def invalidate_conversation_cache(
        self,
        *,
        user_id: str,
        assistant_mode_id: str,
        conversation_id: str,
        workspace_id: str | None,
        active_presence_id: str | None = None,
        active_space_id: str | None = None,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
    ) -> str:
        operational_snapshot = default_operational_profile_snapshot(
            loader=self.runtime.operational_profile_loader,
            settings=self.runtime.settings,
        )
        cache_key = self.build_cache_key(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            active_presence_id=active_presence_id,
            active_space_id=active_space_id,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            operational_profile_token=operational_snapshot.token,
        )
        await self.runtime.storage_backend.delete_context_views_for_conversation(
            user_id,
            conversation_id,
        )
        await self.runtime.storage_backend.delete_recent_window_for_conversation(
            user_id,
            conversation_id,
        )
        return cache_key

    async def invalidate_conversation_cache_for_conversation(
        self,
        conversation: dict[str, Any],
    ) -> int:
        user_id = str(conversation["user_id"])
        conversation_id = str(conversation["id"])
        deleted = (
            await self.runtime.storage_backend.delete_context_views_for_conversation(
                user_id,
                conversation_id,
            )
        )
        deleted += (
            await self.runtime.storage_backend.delete_recent_window_for_conversation(
                user_id,
                conversation_id,
            )
        )
        return deleted

    async def invalidate_conversation_cache_by_id(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        deleted = (
            await self.runtime.storage_backend.delete_context_views_for_conversation(
                user_id,
                conversation_id,
            )
        )
        deleted += (
            await self.runtime.storage_backend.delete_recent_window_for_conversation(
                user_id,
                conversation_id,
            )
        )
        return deleted

    async def invalidate_user_cache(self, user_id: str) -> int:
        deleted = await self.runtime.storage_backend.delete_context_views_for_user(
            user_id
        )
        deleted += await self.runtime.storage_backend.delete_recent_windows_for_user(
            user_id
        )
        return deleted

    async def invalidate_initial_context_package_dependency(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str | None = None,
        package_kind: str | None = None,
        retrieval_profile_id: str | None = None,
        commit: bool = True,
    ) -> InitialContextPackageCacheInvalidationResult:
        """Invalidate package rows and dependent context-cache views under guard."""
        async with self.user_cache_guard(user_id):
            return await invalidate_context_package_dependency(
                connection,
                clock=self.runtime.clock,
                storage_backend=self.runtime.storage_backend,
                database_path=self.runtime.database_path,
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=package_kind,
                retrieval_profile_id=retrieval_profile_id,
                commit=commit,
            )

    @staticmethod
    def build_user_guard_key(user_id: str) -> str:
        guard_subject = {
            "v": 1,
            "user_id": user_id,
        }
        return (
            "ctx-guard:user:v1:"
            + hashlib.sha256(canonical_json_bytes(guard_subject)).hexdigest()
        )

    @staticmethod
    def build_cache_key(
        *,
        user_id: str,
        assistant_mode_id: str,
        conversation_id: str,
        workspace_id: str | None,
        operational_profile_token: str,
        active_presence_id: str | None = None,
        active_space_id: str | None = None,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
        response_mode: ResponseMode | str = ResponseMode.NORMAL,
    ) -> str:
        cache_subject = {
            "v": CONTEXT_CACHE_KEY_VERSION,
            "active_embodiment_id": active_embodiment_id,
            "active_mind_id": active_mind_id,
            "active_presence_id": active_presence_id,
            "active_realm_id": active_realm_id,
            "active_space_id": active_space_id,
            "assistant_mode_id": assistant_mode_id,
            "conversation_id": conversation_id,
            "mind_topology": mind_topology or "unimind",
            "operational_profile_token": operational_profile_token,
            "privacy_enforcement": privacy_enforcement,
            "authenticated_user_privilege_level": authenticated_user_privilege_level
            or "standard",
            "authenticated_user_is_atagia_master": bool(
                authenticated_user_is_atagia_master
            ),
            "user_id": user_id,
            "workspace_id": workspace_id,
        }
        # Non-normal modes get their own key space so fast/smart_fast entries
        # never cross-pollute normal-mode reads (and vice versa). Normal-mode
        # keys stay byte-identical to pre-fast-mode keys by omitting the field.
        resolved_response_mode = ResponseMode(response_mode)
        if resolved_response_mode is not ResponseMode.NORMAL:
            cache_subject["response_mode"] = resolved_response_mode.value
        return (
            f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:"
            + hashlib.sha256(canonical_json_bytes(cache_subject)).hexdigest()
        )

    def _cache_enabled(self, ablation: AblationConfig | None) -> bool:
        if not self.runtime.settings.context_cache_enabled:
            return False
        if ablation is not None and ablation.disable_context_cache:
            return False
        if ablation is not None and ablation.composer_strategy not in {
            None,
            "score_first",
        }:
            return False
        if ablation is not None and ablation.override_retrieval_params:
            return False
        if ablation is not None and (
            ablation.context_envelope_budget_tokens is not None
            or ablation.context_envelope_ratios is not None
        ):
            return False
        return True

    def _cache_ttl_seconds(self, resolved_policy: ResolvedRetrievalPolicy) -> int:
        base_ttl = resolved_policy.context_cache_policy.base_ttl_seconds
        return max(
            self.runtime.settings.context_cache_min_ttl_seconds,
            min(self.runtime.settings.context_cache_max_ttl_seconds, base_ttl),
        )

    @staticmethod
    def _next_message_seq(stored_messages: list[dict[str, Any]]) -> int:
        if not stored_messages:
            return 1
        return int(stored_messages[-1]["seq"]) + 1

    def _cache_age_seconds(self, raw_entry: dict[str, Any]) -> float | None:
        try:
            cached_at = datetime.fromisoformat(str(raw_entry.get("cached_at")))
        except ValueError:
            return None
        if cached_at.tzinfo is None:
            return None
        return max(
            0.0,
            (
                self.runtime.clock.now() - cached_at.astimezone(timezone.utc)
            ).total_seconds(),
        )

    @staticmethod
    def _retrieval_trace_payload(
        pipeline_result: PipelineResult,
        *,
        stage_timings: dict[str, float],
        db_diagnostics: "_SqliteRequestDiagnostics",
    ) -> dict[str, Any] | None:
        trace = pipeline_result.trace
        if trace is None:
            return None
        existing_runtime = trace.runtime_diagnostics
        trace.runtime_diagnostics = RequestRuntimeDiagnosticsTrace(
            stage_timings_ms=_nonnegative_float_map(stage_timings),
            db_query_count=db_diagnostics.query_count,
            db_query_count_by_operation=db_diagnostics.query_count_by_operation,
            hydration_timings_ms={
                **existing_runtime.hydration_timings_ms,
                **_hydration_timings(stage_timings),
            },
            lock_wait_count=db_diagnostics.lock_wait_count,
            sqlite_busy_count=db_diagnostics.sqlite_busy_count,
        )
        return trace.model_dump(mode="json")

    @staticmethod
    def _summarize_candidate_channels(
        raw_candidates: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> dict[str, Any]:
        fts_candidates_count = 0
        vec_candidates_count = 0
        consequence_candidates_count = 0
        multi_channel_candidates_count = 0
        top_candidates = raw_candidates[:top_k]
        combination_counts: dict[str, int] = {}

        for candidate in raw_candidates:
            sources = ContextCacheService._candidate_sources(candidate)
            if "fts" in sources:
                fts_candidates_count += 1
            if "embedding" in sources:
                vec_candidates_count += 1
            if "consequence" in sources:
                consequence_candidates_count += 1
            if len(sources) >= 2:
                multi_channel_candidates_count += 1

        for candidate in top_candidates:
            label = (
                "+".join(ContextCacheService._candidate_sources(candidate)) or "unknown"
            )
            combination_counts[label] = combination_counts.get(label, 0) + 1

        top_k_channel_distribution = (
            {
                label: count / len(top_candidates)
                for label, count in combination_counts.items()
            }
            if top_candidates
            else {}
        )

        return {
            "fts_candidates_count": fts_candidates_count,
            "vec_candidates_count": vec_candidates_count,
            "consequence_candidates_count": consequence_candidates_count,
            "multi_channel_candidates_count": multi_channel_candidates_count,
            "top_k_channel_distribution": top_k_channel_distribution,
        }

    @staticmethod
    def _candidate_sources(candidate: dict[str, Any]) -> list[str]:
        raw_sources = candidate.get("retrieval_sources")
        if isinstance(raw_sources, list):
            return [str(source) for source in raw_sources]
        raw_source = candidate.get("retrieval_source")
        if isinstance(raw_source, str) and raw_source:
            return [source for source in raw_source.split("+") if source]
        return []

    @staticmethod
    def _guard_retrieval_diagnostics(
        pipeline_result: PipelineResult,
    ) -> dict[str, Any]:
        retrieval_plan = pipeline_result.retrieval_plan.model_dump(mode="json")
        trace = pipeline_result.trace
        diagnostics: dict[str, Any] = {
            "retrieval_plan": retrieval_plan,
            "retrieval_sufficiency": (
                pipeline_result.retrieval_sufficiency.model_dump(mode="json")
                if pipeline_result.retrieval_sufficiency is not None
                else None
            ),
            "selected_memory_ids": list(
                pipeline_result.composed_context.selected_memory_ids
            ),
            "selected_memory_count": len(
                pipeline_result.composed_context.selected_memory_ids
            ),
            "diagnostic_shape_fallback_used": False,
        }
        answer_evidence = ContextCacheService._answer_evidence_diagnostics(
            pipeline_result.composed_context
        )
        if answer_evidence is not None:
            diagnostics["answer_evidence"] = answer_evidence
        answer_support = answer_support_prompt_payload(
            pipeline_result.composed_context
        )
        if answer_support is not None:
            diagnostics["answer_support"] = answer_support
        if trace is not None:
            diagnostics.update(
                {
                    "need_detection": (
                        trace.need_detection.model_dump(mode="json")
                        if trace.need_detection is not None
                        else None
                    ),
                    "facet_support": (
                        trace.facet_support.model_dump(mode="json")
                        if trace.facet_support is not None
                        else None
                    ),
                    "direct_vs_indirect_provenance": (
                        trace.direct_vs_indirect_provenance.model_dump(mode="json")
                        if trace.direct_vs_indirect_provenance is not None
                        else None
                    ),
                    "cross_conversation_raw_policy": (
                        trace.cross_conversation_raw_policy.model_dump(mode="json")
                        if trace.cross_conversation_raw_policy is not None
                        else None
                    ),
                }
            )
            if diagnostics["retrieval_sufficiency"] is None:
                diagnostics["retrieval_sufficiency"] = (
                    trace.retrieval_sufficiency.model_dump(mode="json")
                    if trace.retrieval_sufficiency is not None
                    else None
                )
        if not isinstance(diagnostics.get("need_detection"), dict):
            diagnostics["need_detection"] = (
                ContextCacheService._need_detection_from_retrieval_plan(retrieval_plan)
            )
            diagnostics["diagnostic_shape_fallback_used"] = True
        selected_evidence_ids = ContextCacheService._selected_evidence_ids(
            diagnostics.get("direct_vs_indirect_provenance"),
            pipeline_result.candidate_custody,
        )
        diagnostics["selected_evidence_ids"] = selected_evidence_ids
        diagnostics["selected_evidence_count"] = len(selected_evidence_ids)
        return {key: value for key, value in diagnostics.items() if value is not None}

    @staticmethod
    def _answer_evidence_diagnostics(
        composed_context: ComposedContext,
    ) -> dict[str, Any] | None:
        sufficiency = composed_context.answer_evidence_sufficiency
        items = composed_context.answer_evidence_items
        if not sufficiency and not items:
            return None

        compact_items: list[dict[str, Any]] = []
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            source_chain = [
                str(line).strip()
                for line in item.get("source_chain") or []
                if str(line).strip()
            ][:8]
            compact_items.append(
                {
                    key: value
                    for key, value in {
                        "memory_id": item.get("memory_id"),
                        "claim": item.get("claim"),
                        "supporting_quote": item.get("supporting_quote"),
                        "quote_source": item.get("quote_source"),
                        "date": item.get("date"),
                        "speaker": item.get("speaker"),
                        "source": item.get("source"),
                        "support_kind": item.get("support_kind"),
                        "why_selected": item.get("why_selected"),
                        "object_type": item.get("object_type"),
                        "source_kind": item.get("source_kind"),
                        "final_score": item.get("final_score"),
                        "selected_for_answer_pack": item.get(
                            "selected_for_answer_pack"
                        ),
                        "source_chain": source_chain,
                    }.items()
                    if value not in (None, "", [])
                }
            )

        direct_ids = [
            str(item.get("memory_id"))
            for item in compact_items
            if str(item.get("memory_id") or "")
            and (
                str(item.get("support_kind") or "")
                in {"direct", "contextual_direct"}
                or str(item.get("quote_source") or "")
                in {
                    "evidence_packet_source",
                    "source_message",
                    "verbatim_evidence_window",
                }
            )
        ]
        return {
            "sufficiency": dict(sufficiency),
            "items": compact_items,
            "item_count": len(items),
            "direct_memory_ids": direct_ids,
        }

    @staticmethod
    def _need_detection_from_retrieval_plan(
        retrieval_plan: dict[str, Any],
    ) -> dict[str, Any]:
        sub_query_plans = retrieval_plan.get("sub_query_plans")
        sub_queries: list[str] = []
        if isinstance(sub_query_plans, list):
            for plan in sub_query_plans:
                if not isinstance(plan, dict):
                    continue
                text = str(plan.get("text") or "").strip()
                if text:
                    sub_queries.append(text)
        return {
            "query_type": str(retrieval_plan.get("query_type") or "default"),
            "exact_recall_needed": bool(retrieval_plan.get("exact_recall_mode")),
            "exact_facets": list(retrieval_plan.get("exact_facets") or []),
            "raw_context_access_mode": str(
                retrieval_plan.get("raw_context_access_mode") or "normal"
            ),
            "retrieval_levels": list(retrieval_plan.get("retrieval_levels") or []),
            "sub_queries": sub_queries,
            "diagnostic_shape_fallback_used": True,
        }

    @staticmethod
    def _selected_evidence_ids(
        provenance: Any,
        candidate_custody: list[dict[str, Any]],
    ) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        if isinstance(provenance, dict):
            evidence = provenance.get("evidence")
            if isinstance(evidence, list):
                for item in evidence:
                    if not isinstance(item, dict) or item.get("selected") is not True:
                        continue
                    memory_id = str(item.get("memory_id") or "").strip()
                    if memory_id and memory_id not in seen:
                        ids.append(memory_id)
                        seen.add(memory_id)
        for record in candidate_custody:
            if record.get("selected") is not True:
                continue
            kind = str(record.get("candidate_kind") or "")
            if kind not in {
                "evidence",
                "verbatim_evidence_search_window",
                "raw_source_span",
            }:
                continue
            candidate_id = str(record.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in seen:
                ids.append(candidate_id)
                seen.add(candidate_id)
        return ids

    async def _acquire_guard(self, guard_key: str) -> str:
        deadline = perf_counter() + CACHE_GUARD_ACQUIRE_TIMEOUT_SECONDS
        delay_seconds = CACHE_GUARD_INITIAL_DELAY_SECONDS
        while True:
            token = await self.runtime.storage_backend.acquire_lock(
                guard_key,
                ttl_seconds=CACHE_GUARD_TTL_SECONDS,
            )
            if token is not None:
                return token
            if perf_counter() >= deadline:
                raise RuntimeError(f"Could not acquire cache guard for {guard_key}")
            await asyncio.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, CACHE_GUARD_MAX_DELAY_SECONDS)

    @staticmethod
    def _should_discard_cache_entry(cache_score: ContextStalenessScore) -> bool:
        return any(
            signal in DISCARDABLE_CACHE_SIGNALS
            for signal in cache_score.matched_signals
        )


@dataclass(slots=True)
class _SqliteRequestDiagnostics:
    """Text-free SQLite statement counters for one retrieval request."""

    query_count: int = 0
    query_count_by_operation_counter: Counter[str] = field(default_factory=Counter)
    lock_wait_count: int = 0
    sqlite_busy_count: int = 0

    @property
    def query_count_by_operation(self) -> dict[str, int]:
        return dict(sorted(self.query_count_by_operation_counter.items()))

    def record_statement(self, statement: Any, *, method_name: str) -> None:
        operation = _sql_operation(statement, method_name=method_name)
        self.query_count += 1
        self.query_count_by_operation_counter[operation] += 1

    def record_exception(self, exc: Exception) -> None:
        if isinstance(exc, sqlite3.OperationalError) and _is_sqlite_busy_error(exc):
            self.sqlite_busy_count += 1


@dataclass(frozen=True, slots=True)
class _SqliteMethodRestore:
    connection: aiosqlite.Connection
    originals: dict[str, Any]

    def restore(self) -> None:
        for name, original in self.originals.items():
            setattr(self.connection, name, original)


def _install_sqlite_request_diagnostics(
    connection: aiosqlite.Connection,
    diagnostics: _SqliteRequestDiagnostics,
) -> _SqliteMethodRestore:
    originals: dict[str, Any] = {}
    for method_name in (
        "execute",
        "execute_fetchall",
        "execute_insert",
        "executemany",
        "executescript",
    ):
        original = getattr(connection, method_name, None)
        if original is None:
            continue
        originals[method_name] = original

        async def wrapper(
            *args: Any,
            __original: Any = original,
            __method_name: str = method_name,
            **kwargs: Any,
        ) -> Any:
            diagnostics.record_statement(
                _statement_from_call(args, kwargs),
                method_name=__method_name,
            )
            try:
                return await __original(*args, **kwargs)
            except Exception as exc:
                diagnostics.record_exception(exc)
                raise

        setattr(connection, method_name, wrapper)
    return _SqliteMethodRestore(connection=connection, originals=originals)


def _statement_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("sql")


def _sql_operation(statement: Any, *, method_name: str) -> str:
    if method_name == "executescript":
        return "SCRIPT"
    if not isinstance(statement, str):
        return "unknown"
    stripped = statement.lstrip()
    if not stripped:
        return "unknown"
    token = stripped.split(maxsplit=1)[0]
    return (
        "".join(character for character in token.upper() if character.isalpha())
        or "unknown"
    )


def _is_sqlite_busy_error(exc: Exception) -> bool:
    error_name = getattr(exc, "sqlite_errorname", None)
    if isinstance(error_name, str) and error_name.startswith("SQLITE_BUSY"):
        return True
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        return (error_code & 0xFF) == sqlite3.SQLITE_BUSY
    message = str(exc).lower()
    return (
        "database is locked" in message
        or "database is busy" in message
        or "sqlite_busy" in message
    )


def _nonnegative_float_map(values: dict[str, float]) -> dict[str, float]:
    return {str(key): max(0.0, float(value or 0.0)) for key, value in values.items()}


def _hydration_timings(stage_timings: dict[str, float]) -> dict[str, float]:
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
    return _nonnegative_float_map(
        {key: stage_timings[key] for key in sorted(hydration_keys)}
    )
