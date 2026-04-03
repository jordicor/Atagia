"""Adaptive context-cache orchestration above the fresh retrieval path."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import aiosqlite

from atagia.core.repositories import ConversationRepository, MessageRepository
from atagia.memory.context_staleness import (
    ContextStalenessRequest,
    ContextStalenessScore,
    ContextStalenessScorer,
)
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_api import MemorySummary
from atagia.models.schemas_cache import ContextCacheEntry
from atagia.models.schemas_memory import ComposedContext
from atagia.models.schemas_replay import AblationConfig

if TYPE_CHECKING:
    from atagia.app import AppRuntime
from atagia.services.chat_support import (
    build_memory_summaries,
    resolve_assistant_mode_id,
    resolve_policy,
)
from atagia.services.errors import ConversationNotFoundError
from atagia.services.retrieval_service import RetrievalService

CACHE_GUARD_TTL_SECONDS = 5 * 60
CACHE_GUARD_ACQUIRE_TIMEOUT_SECONDS = 30.0
CACHE_GUARD_INITIAL_DELAY_SECONDS = 0.01
CACHE_GUARD_MAX_DELAY_SECONDS = 0.25
DISCARDABLE_CACHE_SIGNALS = frozenset(
    {
        "assistant_mode_id_mismatch",
        "cache_entry_validation_failed",
        "cached_at_invalid",
        "conversation_id_mismatch",
        "message_sequence_rewind",
        "policy_prompt_hash_mismatch",
        "user_id_mismatch",
        "workspace_id_mismatch",
    }
)


@dataclass(slots=True)
class AdaptiveContextResolution:
    """Normalized result returned by adaptive context resolution."""

    conversation: dict[str, Any]
    resolved_policy: ResolvedPolicy
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
    pending_cache_entry: ContextCacheEntry | None
    cache_ttl_seconds: int | None


@dataclass(slots=True)
class ContextCacheService:
    """Resolve context from a stable cache entry when it is still safe to reuse."""

    runtime: AppRuntime
    _staleness_scorer: ContextStalenessScorer = field(init=False)

    def __post_init__(self) -> None:
        self._staleness_scorer = ContextStalenessScorer(clock=self.runtime.clock)

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
        ablation: AblationConfig | None = None,
    ) -> AdaptiveContextResolution:
        conversations = ConversationRepository(connection, self.runtime.clock)
        messages = MessageRepository(connection, self.runtime.clock)
        active_conversation = conversation or await conversations.get_conversation(conversation_id, user_id)
        if active_conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")

        resolved_mode_id = resolve_assistant_mode_id(
            str(active_conversation["assistant_mode_id"]),
            assistant_mode_id,
        )
        resolved_policy = resolve_policy(
            self.runtime.manifests,
            resolved_mode_id,
            self.runtime.policy_resolver,
        )
        current_messages = stored_messages
        if current_messages is None:
            current_messages = await messages.get_messages(
                conversation_id,
                user_id,
                limit=500,
                offset=0,
            )

        cache_key = self.build_cache_key(
            user_id=user_id,
            assistant_mode_id=resolved_mode_id,
            conversation_id=conversation_id,
            workspace_id=active_conversation.get("workspace_id"),
        )
        current_message_seq = self._next_message_seq(current_messages)
        cache_lookup_started = perf_counter()
        raw_entry: dict[str, Any] | None = None
        if self._cache_enabled(ablation):
            raw_entry = await self.runtime.storage_backend.get_context_view(cache_key)
        cache_lookup_elapsed = perf_counter() - cache_lookup_started

        cache_score: ContextStalenessScore | None = None
        cache_age_seconds: float | None = None
        if raw_entry is not None:
            staleness_started = perf_counter()
            cache_score = self._staleness_scorer.score(
                raw_entry,
                ContextStalenessRequest(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=active_conversation.get("workspace_id"),
                    message_text=message_text,
                    current_message_seq=current_message_seq,
                    cache_enabled=self._cache_enabled(ablation),
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
                    pending_cache_entry=None,
                    cache_ttl_seconds=None,
                )
        else:
            staleness_elapsed = 0.0

        pipeline_result = await RetrievalService(self.runtime).retrieve_with_connection(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            mode=resolved_mode_id,
            ablation=ablation,
            conversation=active_conversation,
            stored_messages=current_messages,
        )
        memory_summaries = build_memory_summaries(pipeline_result)
        pending_entry: ContextCacheEntry | None = None
        cache_ttl_seconds: int | None = None
        if self._cache_enabled(ablation):
            pending_entry = ContextCacheEntry(
                cache_key=cache_key,
                user_id=user_id,
                conversation_id=conversation_id,
                assistant_mode_id=resolved_mode_id,
                policy_prompt_hash=resolved_policy.prompt_hash,
                workspace_id=active_conversation.get("workspace_id"),
                composed_context=pipeline_result.composed_context,
                contract=pipeline_result.current_contract,
                memory_summaries=memory_summaries,
                detected_needs=[
                    need.need_type.value for need in pipeline_result.detected_needs
                ],
                source_retrieval_plan=pipeline_result.retrieval_plan.model_dump(mode="json"),
                selected_memory_ids=list(pipeline_result.composed_context.selected_memory_ids),
                cached_at=self.runtime.clock.now().isoformat(),
                last_retrieval_message_seq=0,
                last_user_message_text=message_text,
                source="sync",
            )
            cache_ttl_seconds = self._cache_ttl_seconds(resolved_policy)

        stage_timings = dict(pipeline_result.stage_timings)
        stage_timings["context_cache_lookup"] = cache_lookup_elapsed
        stage_timings["context_cache_staleness"] = staleness_elapsed
        return AdaptiveContextResolution(
            conversation=active_conversation,
            resolved_policy=resolved_policy,
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
            source_retrieval_plan=pipeline_result.retrieval_plan.model_dump(mode="json"),
            scored_candidates=[
                candidate.model_dump(mode="json")
                for candidate in pipeline_result.scored_candidates
            ],
            pending_cache_entry=pending_entry,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def publish_pending_cache_entry(
        self,
        resolution: AdaptiveContextResolution,
        *,
        last_retrieval_message_seq: int,
    ) -> bool:
        if resolution.pending_cache_entry is None or resolution.cache_ttl_seconds is None:
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
    ) -> str:
        cache_key = self.build_cache_key(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
        )
        await self.runtime.storage_backend.delete_context_view(cache_key)
        return cache_key

    async def invalidate_conversation_cache_for_conversation(
        self,
        conversation: dict[str, Any],
    ) -> str:
        return await self.invalidate_conversation_cache(
            user_id=str(conversation["user_id"]),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=(
                str(conversation["workspace_id"])
                if conversation.get("workspace_id") is not None
                else None
            ),
        )

    async def invalidate_user_cache(self, user_id: str) -> int:
        return await self.runtime.storage_backend.delete_context_views_for_user(user_id)

    @staticmethod
    def build_user_guard_key(user_id: str) -> str:
        guard_subject = {
            "v": 1,
            "user_id": user_id,
        }
        payload = json.dumps(
            guard_subject,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "ctx-guard:user:v1:" + hashlib.sha256(payload).hexdigest()

    @staticmethod
    def build_cache_key(
        *,
        user_id: str,
        assistant_mode_id: str,
        conversation_id: str,
        workspace_id: str | None,
    ) -> str:
        cache_subject = {
            "v": 1,
            "assistant_mode_id": assistant_mode_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
        }
        payload = json.dumps(
            cache_subject,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "ctx:v1:" + hashlib.sha256(payload).hexdigest()

    def _cache_enabled(self, ablation: AblationConfig | None) -> bool:
        if not self.runtime.settings.context_cache_enabled:
            return False
        if ablation is not None and ablation.disable_context_cache:
            return False
        return True

    def _cache_ttl_seconds(self, resolved_policy: ResolvedPolicy) -> int:
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
            (self.runtime.clock.now() - cached_at.astimezone(timezone.utc)).total_seconds(),
        )

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
        return any(signal in DISCARDABLE_CACHE_SIGNALS for signal in cache_score.matched_signals)
