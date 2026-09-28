"""Chat orchestration service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from atagia.diagnostics.recorder import capture_chat_operation
import logging
from time import perf_counter
from typing import Any

import aiosqlite

from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleRepository,
)
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.llm_output_limits import CHAT_REPLY_MAX_OUTPUT_TOKENS
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.topic_repository import TopicRepository
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.memory.context_envelope import (
    ContextEnvelopeBudget,
    allocate_context_envelope_budget,
    effective_budget_under_envelope,
)
from atagia.memory.lifecycle_runner import request_lifecycle_piggyback
from atagia.core.summary_repository import SummaryRepository
from atagia.core.timestamps import (
    normalize_optional_timestamp,
    resolve_message_occurred_at,
)
from atagia.memory.intimacy_boundary_policy import (
    INTIMACY_FILTER_REASON,
    strongest_intimacy_boundary,
)
from atagia.models.schemas_api import ChatResult
from atagia.models.schemas_memory import (
    ConversationStatus,
    MindTopology,
    ResponseMode,
    TurnSurface,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    RECENT_WINDOW_MESSAGES,
    apply_conversation_policy_overlay,
    build_message_jobs,
    build_response_language_guidance,
    build_system_prompt,
    build_transcript_window,
    build_transcript_window_trace,
    build_turn_telemetry,
    chat_model,
    enqueue_message_jobs,
    filter_topic_working_set_snapshot,
    missing_uncovered_tail_start_seq,
    render_assistant_guidance_block,
    render_answer_support_block,
    render_transcript_window,
    render_topic_working_set_block,
    resolve_retrieval_profile_id,
    resolve_operational_profile,
    resolve_policy,
    summarize_memory_summaries,
    estimate_tokens,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.initial_context_package_prompt import (
    InitialContextPackagePromptAssembly,
    assemble_initial_context_package_prompt,
    drop_initial_context_package_for_overflow,
)
from atagia.services.initial_context_package_refresh_service import (
    InitialContextPackageRefreshEnqueuer,
)
from atagia.services.job_tracking_service import (
    JobTrackingService,
    render_memory_processing_status_block,
)
from atagia.services.worker_control_service import WorkerControlService
from atagia.services.errors import (
    ConversationNotActiveError,
    ConversationNotFoundError,
    LLMUnavailableError,
    UserDeletedError,
)
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMError,
    LLMMessage,
    OutputLimitExceededError,
    known_intimacy_context_metadata,
)
from atagia.services.answer_postcondition import (
    complete_answer_with_postcondition_guard,
)
from atagia.services.model_resolution import resolve_component_model
from atagia.services.identity_hints import validate_optional_identity_hints
from atagia.services.prompt_authority import (
    PromptAuthorityContext,
    resolve_request_authority_context,
)
from atagia.services.sidecar_service import SidecarService

logger = logging.getLogger(__name__)


def _context_envelope_budget(
    settings: Any,
    ablation: AblationConfig | None,
) -> ContextEnvelopeBudget:
    return allocate_context_envelope_budget(
        (
            ablation.context_envelope_budget_tokens
            if ablation is not None
            and ablation.context_envelope_budget_tokens is not None
            else settings.context_envelope_budget_tokens
        ),
        (
            ablation.context_envelope_ratios
            if ablation is not None and ablation.context_envelope_ratios is not None
            else settings.context_envelope_ratios
        ),
    )


def _estimate_llm_input_tokens(messages: list[LLMMessage]) -> int:
    return sum(estimate_tokens(message.content) for message in messages)


def _context_envelope_trace(
    envelope_budget: ContextEnvelopeBudget | None,
    *,
    estimated_input_tokens: int,
    retrieved_context_tokens: int,
    transcript_budget_initial_tokens: int,
    transcript_budget_final_tokens: int,
    transcript_trace: dict[str, Any],
) -> dict[str, Any] | None:
    if envelope_budget is None:
        return None
    budget_payload = envelope_budget.model_dump()
    return {
        "enabled": True,
        **budget_payload,
        "retrieved_context_tokens": retrieved_context_tokens,
        "recent_transcript_budget_initial_tokens": transcript_budget_initial_tokens,
        "recent_transcript_budget_final_tokens": transcript_budget_final_tokens,
        "recent_transcript_used_tokens": int(
            transcript_trace.get("budget_used_tokens") or 0
        ),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_overflow_tokens": max(
            0,
            estimated_input_tokens - envelope_budget.total_budget_tokens,
        ),
        "reserve_tokens": 0,
    }


def _transcript_message_ids(entries: list[Any]) -> set[str]:
    message_ids: set[str] = set()
    for entry in entries:
        message_id = getattr(entry, "message_id", None)
        if message_id is not None:
            message_ids.add(str(message_id))
            continue
        message = getattr(entry, "message", None)
        if isinstance(message, dict) and message.get("id") is not None:
            message_ids.add(str(message["id"]))
    return message_ids


@dataclass(slots=True)
class ChatService:
    """Coordinates end-to-end chat flow."""

    runtime: Any

    @capture_chat_operation
    async def chat_reply(
        self,
        user_id: str,
        conversation_id: str,
        message_text: str,
        assistant_mode_id: str | None = None,
        *,
        ablation: AblationConfig | None = None,
        attachments: list[Any] | None = None,
        message_occurred_at: str | None = None,
        include_thinking: bool = False,
        metadata: dict[str, Any] | None = None,
        debug: bool = False,
        debug_include_sensitive: bool = True,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        mind_id: str | None = None,
        mind_topology: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
        response_mode: ResponseMode | str | None = None,
        adaptive_retrieval: bool | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
    ) -> ChatResult:
        """Run the full retrieval, generation, persistence, and background-job flow."""
        resolved_response_mode = (
            ResponseMode(self.runtime.settings.response_mode)
            if response_mode is None
            else ResponseMode(response_mode)
        )
        resolved_adaptive_retrieval = (
            self.runtime.settings.adaptive_retrieval
            if adaptive_retrieval is None
            else bool(adaptive_retrieval)
        )
        authority_context = resolve_request_authority_context(
            prompt_authority_context,
            privacy_enforcement=privacy_enforcement,
            authenticated_user_privilege_level=authenticated_user_privilege_level,
            authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
            user_id=user_id,
            purpose="chat_reply",
        )
        cache_service = ContextCacheService(self.runtime)
        # TWO clocks, deliberately. The turn's wall clock starts BEFORE the
        # per-user cache guard, which is held for a whole turn and blocks a
        # same-user follow-up for up to CACHE_GUARD_ACQUIRE_TIMEOUT_SECONDS:
        # queueing behind another turn is time the caller waits, so it belongs
        # inside turn_to_event_write_wall_ms, and starting here hid contention
        # entirely. retrieval_duration_ms keeps its own, tighter origin at the
        # resolve call site below, because lock wait is not retrieval work -- one
        # origin for both would report a turn that spent 380ms queueing as 380ms
        # of retrieval, and the difference between the two would stop meaning
        # anything. (The difference is still not "everything after retrieval":
        # it excludes the post-write tail the column's name now announces.)
        turn_started_at = perf_counter()
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            connection = await self.runtime.open_connection()
            chat_result: ChatResult | None = None
            # Per-turn LLM call meter (CS-1.4). Bound to the current context so it
            # counts every synchronous provider round-trip of the turn (retrieval
            # cards through the chat reply), then unbound in the finally below. The
            # metrics dict is filled once the answer is generated (see below).
            turn_call_meter = self.runtime.llm_client.begin_turn_call_meter()
            turn_llm_call_metrics: dict[str, Any] | None = None
            retrieval_duration_ms = 0.0
            try:
                rebuild_repository = TranscriptRebuildRepository(
                    connection,
                    self.runtime.clock,
                )
                await rebuild_repository.require_user_available(user_id)
                conversations = ConversationRepository(connection, self.runtime.clock)
                users = UserRepository(connection, self.runtime.clock)
                messages = MessageRepository(connection, self.runtime.clock)
                memories = MemoryObjectRepository(connection, self.runtime.clock)
                events = RetrievalEventRepository(connection, self.runtime.clock)
                summaries = SummaryRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(
                    connection,
                    self.runtime.clock,
                )
                confirmations = PendingConfirmationService(
                    connection,
                    self.runtime.clock,
                    self.runtime.embedding_index,
                    llm_client=self.runtime.llm_client,
                    settings=self.runtime.settings,
                )
                job_tracking = JobTrackingService(
                    connection,
                    self.runtime.clock,
                    workers_enabled=self.runtime.settings.workers_enabled,
                    settings=self.runtime.settings,
                )

                conversation = await conversations.get_conversation(
                    conversation_id, user_id
                )
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                _validate_optional_identity(
                    conversation,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    active_presence_id=active_presence_id,
                    mind_id=mind_id,
                    mind_topology=mind_topology,
                    embodiment_id=embodiment_id,
                    realm_id=realm_id,
                    space_id=space_id,
                )
                if await users.get_active_user(user_id) is None:
                    raise UserDeletedError("User has been erased or does not exist")
                if str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                    raise ConversationNotActiveError("Conversation is not active")
                sidecar_service = SidecarService(self.runtime)
                conversation = await sidecar_service.ensure_conversation(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=conversation.get("workspace_id"),
                    assistant_mode_id=assistant_mode_id,
                    cross_chat_memory=cross_chat_memory,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    active_presence_id=active_presence_id,
                    mind_id=mind_id,
                    mind_topology=mind_topology,
                    embodiment_id=embodiment_id,
                    realm_id=realm_id,
                    space_id=space_id,
                    mode=mode,
                    incognito=incognito,
                )

                request_snapshot = (
                    await sidecar_service._capture_conversation_request_snapshot(
                        connection,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        source_role="user",
                        workspace_id=conversation.get("workspace_id"),
                        user_persona_id=user_persona_id,
                        platform_id=platform_id,
                        character_id=character_id,
                        active_presence_id=active_presence_id,
                        mind_id=mind_id,
                        mind_topology=mind_topology,
                        embodiment_id=embodiment_id,
                        realm_id=realm_id,
                        space_id=space_id,
                    )
                )
                conversation = request_snapshot.conversation
                active_presence = request_snapshot.active_presence
                user_source_presence = request_snapshot.source_presence
                active_mind = request_snapshot.active_mind
                active_embodiment = request_snapshot.active_embodiment
                active_realm = request_snapshot.active_realm
                active_space = request_snapshot.active_space
                memory_preferences = await users.get_memory_preferences(user_id)
                if memory_preferences is None:
                    raise UserDeletedError("User has been erased or does not exist")

                resolved_mode_id = resolve_retrieval_profile_id(
                    str(conversation["assistant_mode_id"]),
                    mode if mode is not None else assistant_mode_id,
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
                    conversation,
                )
                attachment_bundle = artifacts.prepare_attachments(
                    message_text=message_text,
                    attachments=attachments,
                    user_id=user_id,
                    conversation=conversation,
                )
                prompt_message_text = attachment_bundle.prompt_text
                prior_messages = await messages.get_recent_messages(
                    conversation_id,
                    user_id,
                    limit=RECENT_FETCH_LIMIT,
                )
                conversation_chunks = await summaries.list_all_conversation_chunks(
                    user_id, conversation_id
                )
                missing_tail_start_seq = missing_uncovered_tail_start_seq(
                    prior_messages,
                    conversation_chunks,
                )
                if missing_tail_start_seq is not None and prior_messages:
                    prior_messages = await messages.get_messages_from_seq(
                        conversation_id,
                        user_id,
                        start_seq=missing_tail_start_seq,
                    )
                cold_start = (
                    await memories.count_for_context(
                        user_id,
                        resolved_policy.allowed_scopes,
                        workspace_id=conversation["workspace_id"],
                        conversation_id=conversation_id,
                        assistant_mode_id=resolved_mode_id,
                        user_persona_id=conversation.get("user_persona_id"),
                        platform_id=conversation.get("platform_id") or "default",
                        character_id=conversation.get("character_id")
                        or conversation.get("workspace_id"),
                        incognito=bool(conversation.get("incognito"))
                        or bool(conversation.get("isolated_mode")),
                        remember_across_chats=bool(
                            conversation.get("remember_across_chats", 1)
                        ),
                        remember_across_devices=bool(
                            conversation.get("remember_across_devices", 1)
                        ),
                        active_mind_id=active_mind.mind_id,
                        mind_topology=active_mind.topology,
                        active_embodiment_id=(
                            active_embodiment.embodiment_id
                            if active_embodiment is not None
                            else None
                        ),
                        active_realm_id=(
                            active_realm.realm_id if active_realm is not None else None
                        ),
                    )
                    == 0
                )
                if authority_context.privacy_restrictions_inactive:
                    confirmation_plan = None
                else:
                    confirmation_plan = await confirmations.plan_turn(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_text=prompt_message_text,
                    )
                invalidate_confirmation_cache = (
                    confirmation_plan is not None
                    and confirmation_plan.response_intent is not None
                )
                # Retrieval wall time is measured at the call site rather than
                # read off the pipeline trace: a cache hit never enters the
                # pipeline yet still costs a lookup plus a staleness decision,
                # and that cost has to appear in the persisted number.
                retrieval_started_at = perf_counter()
                if resolved_response_mode is ResponseMode.NORMAL:
                    resolution = await cache_service.resolve_with_connection(
                        connection,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_text=prompt_message_text,
                        assistant_mode_id=resolved_mode_id,
                        stored_messages=prior_messages,
                        conversation=conversation,
                        operational_profile=operational_profile,
                        operational_signals=operational_signals,
                        ablation=ablation,
                        prompt_authority_context=authority_context,
                        adaptive_retrieval=resolved_adaptive_retrieval,
                    )
                else:
                    resolution = await cache_service.resolve_fast_with_connection(
                        connection,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_text=prompt_message_text,
                        response_mode=resolved_response_mode,
                        assistant_mode_id=resolved_mode_id,
                        stored_messages=prior_messages,
                        conversation=conversation,
                        operational_profile=operational_profile,
                        operational_signals=operational_signals,
                        ablation=ablation,
                        prompt_authority_context=authority_context,
                    )
                retrieval_duration_ms = (
                    perf_counter() - retrieval_started_at
                ) * 1000.0
                topic_snapshot = await TopicRepository(
                    connection,
                    self.runtime.clock,
                ).get_topic_snapshot(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    refresh_message_threshold=(
                        self.runtime.settings.topic_working_set_refresh_message_lag
                    ),
                    stale_message_threshold=(
                        self.runtime.settings.topic_working_set_stale_message_lag
                    ),
                    refresh_token_threshold=(
                        self.runtime.settings.topic_working_set_refresh_token_lag
                    ),
                    stale_token_threshold=(
                        self.runtime.settings.topic_working_set_stale_token_lag
                    ),
                )
                visible_topic_snapshot = filter_topic_working_set_snapshot(
                    topic_snapshot,
                    allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    privacy_ceiling=resolution.resolved_policy.privacy_ceiling,
                )
                topic_context_block = render_topic_working_set_block(
                    visible_topic_snapshot,
                    allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    privacy_ceiling=resolution.resolved_policy.privacy_ceiling,
                )
                prompt_memory_processing = await job_tracking.get_status(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                context_envelope_budget = _context_envelope_budget(
                    self.runtime.settings,
                    ablation,
                )
                transcript_budget_tokens = 0
                transcript_budget_initial_tokens = 0
                raw_context_access_mode = str(
                    resolution.source_retrieval_plan.get(
                        "raw_context_access_mode", "normal"
                    )
                )
                if self.runtime.settings.benchmark_disable_raw_recent_transcript:
                    transcript_entries = []
                    transcript_trace = build_transcript_window_trace([], 0)
                else:
                    transcript_budget_tokens = effective_budget_under_envelope(
                        knob="transcript_budget_tokens",
                        policy_budget_tokens=(
                            resolution.resolved_policy.transcript_budget_tokens
                        ),
                        envelope_budget_tokens=(
                            context_envelope_budget.recent_transcript_budget_tokens
                        ),
                        override_budget_tokens=(
                            None
                            if ablation is None
                            else (ablation.override_retrieval_params or {}).get(
                                "transcript_budget_tokens"
                            )
                        ),
                    )
                    transcript_budget_initial_tokens = transcript_budget_tokens
                    transcript_entries = build_transcript_window(
                        prior_messages,
                        conversation_chunks,
                        transcript_budget_tokens,
                        raw_context_access_mode=raw_context_access_mode,
                        allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    )
                    transcript_trace = build_transcript_window_trace(
                        transcript_entries,
                        transcript_budget_tokens,
                    )
                transcript = [
                    *render_transcript_window(transcript_entries),
                    {"role": "user", "text": prompt_message_text},
                ]
                assistant_guidance_block = render_assistant_guidance_block(
                    build_response_language_guidance(
                        resolution.source_retrieval_plan,
                        enabled=self.runtime.settings.assistant_guidance_enabled,
                        from_cache=resolution.from_cache,
                    )
                )
                try:
                    initial_context_package = await assemble_initial_context_package_prompt(
                        connection,
                        self.runtime.clock,
                        enabled=self.runtime.settings.initial_context_package_read_enabled,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        conversation=conversation,
                        resolved_policy=resolution.resolved_policy,
                        authority_context=authority_context,
                        operational_profile=(
                            resolution.resolved_operational_profile.snapshot
                        ),
                        context_envelope_budget=context_envelope_budget,
                        retrieved_context_tokens=(
                            resolution.composed_context.total_tokens_estimate
                        ),
                        selected_memory_ids=(
                            resolution.composed_context.selected_memory_ids
                        ),
                        topic_context_block=topic_context_block,
                        live_contract_block=resolution.composed_context.contract_block,
                        live_state_block=resolution.composed_context.state_block,
                        recent_transcript_message_ids=_transcript_message_ids(
                            transcript_entries
                        ),
                        include_recent_verbatim_seed=(
                            not self.runtime.settings.benchmark_disable_raw_recent_transcript
                        ),
                        prompt_budget_tokens=(
                            self.runtime.settings.initial_context_package_prompt_max_tokens
                        ),
                        refresh_enqueuer=InitialContextPackageRefreshEnqueuer(
                            storage_backend=self.runtime.storage_backend,
                            clock=self.runtime.clock,
                            job_tracking_service=job_tracking,
                            package_repository=InitialContextPackageRepository(
                                connection,
                                self.runtime.clock,
                            ),
                            refresh_enabled=(
                                self.runtime.settings.initial_context_package_refresh_enabled
                            ),
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Initial context package read failed for user_id=%s conversation_id=%s",
                        user_id,
                        conversation_id,
                    )
                    initial_context_package = InitialContextPackagePromptAssembly(
                        "",
                        {
                            "enabled": (
                                self.runtime.settings.initial_context_package_read_enabled
                            ),
                            "rendered": False,
                            "read_ms": 0.0,
                            "packages": [],
                            "error": "read_failed",
                        },
                    )
                system_prompt = build_system_prompt(
                    resolved_mode_id,
                    resolution.resolved_policy,
                    resolution.composed_context.contract_block,
                    resolution.composed_context.workspace_block,
                    resolution.composed_context.memory_block,
                    resolution.composed_context.state_block,
                    answer_support_block=render_answer_support_block(
                        resolution.composed_context
                    ),
                    prepared_initial_context_block=initial_context_package.block,
                    topic_context_block=topic_context_block,
                    memory_processing_block=render_memory_processing_status_block(
                        prompt_memory_processing
                    ),
                    assistant_guidance_block=assistant_guidance_block,
                    answer_stance=self.runtime.settings.answer_stance,
                    answer_stance_prompt_variant=(
                        self.runtime.settings.answer_stance_prompt_variant
                    ),
                    prompt_authority_context=authority_context,
                )
                chat_messages = [
                    LLMMessage(role="system", content=system_prompt),
                    *[
                        LLMMessage(
                            role=str(message["role"]), content=str(message["text"])
                        )
                        for message in transcript
                    ],
                ]
                estimated_input_tokens = _estimate_llm_input_tokens(chat_messages)
                if (
                    estimated_input_tokens > context_envelope_budget.total_budget_tokens
                    and initial_context_package.block
                ):
                    initial_context_package = drop_initial_context_package_for_overflow(
                        initial_context_package
                    )
                    system_prompt = build_system_prompt(
                        resolved_mode_id,
                        resolution.resolved_policy,
                        resolution.composed_context.contract_block,
                        resolution.composed_context.workspace_block,
                        resolution.composed_context.memory_block,
                        resolution.composed_context.state_block,
                        answer_support_block=render_answer_support_block(
                            resolution.composed_context
                        ),
                        prepared_initial_context_block=initial_context_package.block,
                        topic_context_block=topic_context_block,
                        memory_processing_block=render_memory_processing_status_block(
                            prompt_memory_processing
                        ),
                        assistant_guidance_block=assistant_guidance_block,
                        answer_stance=self.runtime.settings.answer_stance,
                        answer_stance_prompt_variant=(
                            self.runtime.settings.answer_stance_prompt_variant
                        ),
                        prompt_authority_context=authority_context,
                    )
                    chat_messages = [
                        LLMMessage(role="system", content=system_prompt),
                        *[
                            LLMMessage(
                                role=str(message["role"]),
                                content=str(message["text"]),
                            )
                            for message in transcript
                        ],
                    ]
                    estimated_input_tokens = _estimate_llm_input_tokens(chat_messages)
                if (
                    estimated_input_tokens > context_envelope_budget.total_budget_tokens
                    and transcript_budget_tokens > 0
                    and transcript_entries
                    and not self.runtime.settings.benchmark_disable_raw_recent_transcript
                ):
                    overflow_tokens = (
                        estimated_input_tokens
                        - context_envelope_budget.total_budget_tokens
                    )
                    transcript_budget_tokens = max(
                        0,
                        transcript_budget_tokens - overflow_tokens,
                    )
                    transcript_entries = build_transcript_window(
                        prior_messages,
                        conversation_chunks,
                        transcript_budget_tokens,
                        raw_context_access_mode=raw_context_access_mode,
                        allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    )
                    transcript_trace = build_transcript_window_trace(
                        transcript_entries,
                        transcript_budget_tokens,
                    )
                    transcript = [
                        *render_transcript_window(transcript_entries),
                        {"role": "user", "text": prompt_message_text},
                    ]
                    chat_messages = [
                        LLMMessage(role="system", content=system_prompt),
                        *[
                            LLMMessage(
                                role=str(message["role"]),
                                content=str(message["text"]),
                            )
                            for message in transcript
                        ],
                    ]
                    estimated_input_tokens = _estimate_llm_input_tokens(chat_messages)
                context_envelope_trace = _context_envelope_trace(
                    context_envelope_budget,
                    estimated_input_tokens=estimated_input_tokens,
                    retrieved_context_tokens=resolution.composed_context.total_tokens_estimate,
                    transcript_budget_initial_tokens=transcript_budget_initial_tokens,
                    transcript_budget_final_tokens=transcript_budget_tokens,
                    transcript_trace=transcript_trace,
                )
                if context_envelope_trace is not None:
                    context_envelope_trace["initial_context_package"] = (
                        initial_context_package.diagnostics
                    )
                chat_request = LLMCompletionRequest(
                    model=chat_model(self.runtime.settings),
                    messages=chat_messages,
                    max_output_tokens=CHAT_REPLY_MAX_OUTPUT_TOKENS,
                    include_thinking=include_thinking,
                    metadata={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "assistant_mode_id": resolved_mode_id,
                        "purpose": "chat_reply",
                        "privacy_enforcement": authority_context.privacy_enforcement,
                        "effective_privacy_enforcement": (
                            authority_context.effective_privacy_enforcement
                        ),
                        "authenticated_privilege_level": (
                            authority_context.normalized_privilege_level
                        ),
                        "authenticated_atagia_master": (
                            authority_context.authenticated_user_is_atagia_master
                        ),
                        "authority_source": authority_context.authority_source,
                        **self._chat_intimacy_metadata(
                            visible_topic_snapshot,
                            allow_intimacy_context=(
                                resolution.resolved_policy.allow_intimacy_context
                            ),
                        ),
                    },
                )
                answer_postcondition_report: dict[str, Any] | None = None
                if self.runtime.settings.answer_postcondition_guard_enabled:
                    guarded_answer = await complete_answer_with_postcondition_guard(
                        llm_client=self.runtime.llm_client,
                        request=chat_request,
                        verifier_model=resolve_component_model(
                            self.runtime.settings,
                            "answer_postcondition",
                        ),
                        original_query=prompt_message_text,
                        composed_context=resolution.composed_context,
                        retrieval_sufficiency=resolution.retrieval_sufficiency,
                        retrieval_diagnostics=resolution.retrieval_diagnostics_for_guard,
                        privacy_enforcement=authority_context.privacy_enforcement,
                        answer_stance=self.runtime.settings.answer_stance,
                        prompt_authority_context=authority_context,
                        retry_max_output_tokens=(
                            self.runtime.settings.answer_postcondition_retry_max_output_tokens
                        ),
                    )
                    llm_response = guarded_answer.response
                    assistant_output_text = guarded_answer.output_text
                    answer_postcondition_report = guarded_answer.report.model_dump(
                        mode="json"
                    )
                else:
                    llm_response = await self.runtime.llm_client.complete(chat_request)
                    assistant_output_text = llm_response.output_text
                recorder = getattr(self.runtime.llm_client, "_diagnostic_recorder", None)
                if recorder is not None:
                    recorder.no_call(
                        "answer_effect",
                        component="chat",
                        user_id=user_id,
                        data={
                            "conversation_id": conversation_id,
                            "messages": recorder.blob(chat_request.messages),
                            "model": chat_request.model,
                            "max_output_tokens": chat_request.max_output_tokens,
                            "final_output": recorder.blob(assistant_output_text),
                            "selected_memory_ids": resolution.composed_context.selected_memory_ids,
                            "postcondition": recorder.blob(answer_postcondition_report) if answer_postcondition_report is not None else None,
                        },
                    )
                # Snapshot the per-turn LLM call metrics here: the chat reply is
                # the last synchronous provider call of the turn, so this captures
                # the full retrieval + answer count. Attaching before the DB write
                # means the persisted retrieval_event trace and the returned debug
                # payload carry the same numbers. Background extraction/contract
                # work runs post-response via workers and is intentionally excluded.
                #
                # The call counters are snapshotted at this instant on purpose,
                # but the turn's wall time is not: it is replaced just before the
                # telemetry write below, so every surface means the same span by
                # turn_to_event_write_wall_ms -- start of turn to the instant the
                # row is written -- instead of stopping here and excluding
                # message persistence, which is real work the caller waits for.
                # The span deliberately stops AT the write and not at the end of
                # the turn: see migration 0072 for the measured tail it leaves
                # out and why closing it costs more than it is worth.
                turn_telemetry = build_turn_telemetry(
                    surface=TurnSurface.CHAT,
                    meter=turn_call_meter,
                    turn_to_event_write_wall_ms=(perf_counter() - turn_started_at) * 1000.0,
                    retrieval_duration_ms=retrieval_duration_ms,
                    stage_timings_ms=resolution.stage_timings,
                )
                # One object feeds both the typed telemetry columns and the trace
                # payload, so a turn cannot persist one count in the columns and
                # a different one in outcome_json.
                turn_llm_call_metrics = turn_telemetry.llm_call_metrics().model_dump(
                    mode="json"
                )
                if resolution.retrieval_trace is not None:
                    resolution.retrieval_trace["llm_call_metrics"] = turn_llm_call_metrics
                response_text = assistant_output_text
                if (
                    confirmation_plan is not None
                    and confirmation_plan.prompt_text is not None
                ):
                    response_text = (
                        f"{confirmation_plan.prompt_text}\n\n{assistant_output_text}"
                    )
                resolved_user_occurred_at = (
                    normalize_optional_timestamp(message_occurred_at)
                    or self.runtime.clock.now().isoformat()
                )
                assistant_occurred_at = self.runtime.clock.now().isoformat()

                await connection.execute("BEGIN IMMEDIATE")
                try:
                    # Re-check exact user and conversation authority after
                    # acquiring the SQLite write lock. The initial read view
                    # cannot exclude a cross-process namespace change while
                    # the model response is being generated.
                    await sidecar_service._require_conversation_request_snapshot(
                        connection,
                        request_snapshot,
                    )
                    user_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="user",
                        seq=None,
                        text=prompt_message_text,
                        token_count=None,
                        metadata=attachment_bundle.message_metadata(metadata),
                        occurred_at=resolved_user_occurred_at,
                        active_presence_id=active_presence.presence_id,
                        source_presence_id=user_source_presence.presence_id,
                        space_id=active_space.space_id
                        if active_space is not None
                        else None,
                        active_mind_id=active_mind.mind_id,
                        source_mind_id=active_mind.mind_id,
                        active_embodiment_id=(
                            active_embodiment.embodiment_id
                            if active_embodiment is not None
                            else None
                        ),
                        active_realm_id=(
                            active_realm.realm_id if active_realm is not None else None
                        ),
                        commit=False,
                    )
                    if attachment_bundle.artifacts:
                        await artifacts.persist_prepared_attachments(
                            bundle=attachment_bundle,
                            message_id=str(user_message["id"]),
                            commit=False,
                        )
                    assistant_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="assistant",
                        seq=None,
                        text=assistant_output_text,
                        token_count=None,
                        metadata={"thinking": llm_response.thinking}
                        if llm_response.thinking
                        else {},
                        occurred_at=assistant_occurred_at,
                        active_presence_id=active_presence.presence_id,
                        source_presence_id=active_presence.presence_id,
                        space_id=active_space.space_id
                        if active_space is not None
                        else None,
                        active_mind_id=active_mind.mind_id,
                        source_mind_id=active_mind.mind_id,
                        active_embodiment_id=(
                            active_embodiment.embodiment_id
                            if active_embodiment is not None
                            else None
                        ),
                        active_realm_id=(
                            active_realm.realm_id if active_realm is not None else None
                        ),
                        commit=False,
                    )
                    composed_context_json = resolution.composed_context.model_dump(
                        mode="json"
                    )
                    retrieval_event = await events.create_event(
                        {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "request_message_id": user_message["id"],
                            "response_message_id": assistant_message["id"],
                            "assistant_mode_id": resolved_mode_id,
                            "user_persona_id": conversation.get("user_persona_id"),
                            "platform_id": conversation.get("platform_id") or "default",
                            "character_id": conversation.get("character_id")
                            or conversation.get("workspace_id"),
                            "mode": conversation.get("mode") or resolved_mode_id,
                            "incognito": bool(conversation.get("incognito"))
                            or bool(conversation.get("isolated_mode")),
                            "remember_across_chats": bool(
                                memory_preferences["remember_across_chats"]
                            ),
                            "remember_across_devices": bool(
                                memory_preferences["remember_across_devices"]
                            ),
                            "memory_privacy_mode": memory_preferences[
                                "memory_privacy_mode"
                            ],
                            "retrieval_plan_json": resolution.source_retrieval_plan,
                            "selected_memory_ids_json": resolution.composed_context.selected_memory_ids,
                            "context_view_json": composed_context_json,
                            "outcome_json": {
                                # Turn-level, not retrieval-level: persisted here
                                # too so cache-hit turns (retrieval_trace is None)
                                # still record their call count -- that fast path
                                # is exactly what later phases widen.
                                "llm_call_metrics": turn_llm_call_metrics,
                                "response_mode": resolved_response_mode.value,
                                "adaptive_retrieval": resolved_adaptive_retrieval,
                                "cold_start": cold_start,
                                "from_cache": resolution.from_cache,
                                "cache_key": resolution.cache_key,
                                "cache_source": resolution.cache_source,
                                "cache_age_seconds": resolution.cache_age_seconds,
                                "staleness": resolution.staleness,
                                "need_detection_skipped": resolution.need_detection_skipped,
                                "detected_needs": resolution.detected_needs,
                                "zero_candidates": not bool(
                                    resolution.composed_context.selected_memory_ids
                                ),
                                "background_tasks_enqueued": False,
                                "scored_candidates": resolution.scored_candidates,
                                "retrieval_custody_v2": resolution.candidate_custody,
                                "retrieval_custody_v2_status": resolution.retrieval_custody_v2_status,
                                "intimacy_boundary_counts": _intimacy_boundary_counts(
                                    resolution.candidate_custody
                                ),
                                "intimacy_policy_filtered_count": _intimacy_policy_filtered_count(
                                    resolution.candidate_custody
                                ),
                                "sufficiency_diagnostics_v1": resolution.retrieval_sufficiency,
                                "sufficiency_diagnostics_v1_status": (
                                    resolution.sufficiency_diagnostics_v1_status
                                ),
                                "retrieval_diagnostics_for_guard": (
                                    resolution.retrieval_diagnostics_for_guard
                                ),
                                "retrieval_trace": resolution.retrieval_trace,
                                "stage_timings_ms": resolution.stage_timings,
                                "transcript_window": transcript_trace,
                                "context_envelope": context_envelope_trace,
                                "initial_context_package": (
                                    initial_context_package.diagnostics
                                ),
                                "operational_profile": (
                                    resolution.resolved_operational_profile.snapshot.model_dump(
                                        mode="json"
                                    )
                                ),
                                "answer_postcondition_guard": answer_postcondition_report,
                                **resolution.candidate_search_summary,
                            },
                        },
                        # Measured here, not at answer-ready: the turn is only
                        # over for the caller once this row exists.
                        telemetry=replace(
                            turn_telemetry,
                            turn_to_event_write_wall_ms=(
                                (perf_counter() - turn_started_at) * 1000.0
                            ),
                        ),
                        commit=False,
                    )
                    confirmation_embedding_upserts = []
                    if confirmation_plan is not None:
                        confirmation_embedding_upserts = (
                            await confirmations.apply_turn_plan(
                                user_id=user_id,
                                plan=confirmation_plan,
                                commit=False,
                            )
                        )
                    turn_jobs = _build_turn_jobs(
                        clock=self.runtime.clock,
                        conversation=conversation,
                        user_message=user_message,
                        assistant_message=assistant_message,
                        prior_messages=prior_messages,
                        prompt_message_text=prompt_message_text,
                        assistant_output_text=assistant_output_text,
                        operational_profile=(
                            resolution.resolved_operational_profile.snapshot
                        ),
                        memory_preferences=memory_preferences,
                        active_presence=active_presence,
                        user_source_presence=user_source_presence,
                        active_space=active_space,
                        active_mind=active_mind,
                        active_embodiment=active_embodiment,
                        active_realm=active_realm,
                    )
                    enqueued_job_ids = await enqueue_message_jobs(
                        storage_backend=self.runtime.storage_backend,
                        jobs=turn_jobs,
                        job_tracking_service=job_tracking,
                        worker_control_service=WorkerControlService(
                            connection,
                            self.runtime.clock,
                        ),
                        initial_context_package_repository=(
                            InitialContextPackageRepository(
                                connection,
                                self.runtime.clock,
                            )
                        ),
                        initial_context_package_refresh_enabled=(
                            self.runtime.settings.initial_context_package_refresh_enabled
                        ),
                        commit=False,
                        dispatch=False,
                    )
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
                # Everything from the terminal commit onward is pure SQL and
                # stream dispatch, so nothing here reopens the LLM call counters
                # snapshotted above -- with ONE dormant exception: this call
                # reaches SQLiteVecBackend.upsert, which goes through
                # llm_client.embed and IS metered. With embedding_backend="none"
                # (the default) it makes no round-trip; enable an embedding
                # backend and these embeddings land on the turn's meter after
                # the telemetry snapshot, so they would be spent but unreported.
                await confirmations.apply_post_commit_embeddings(
                    confirmation_embedding_upserts
                )

                post_commit_errors: list[str] = []
                background_tasks_enqueued = bool(enqueued_job_ids)
                memory_processing = prompt_memory_processing

                try:
                    if invalidate_confirmation_cache:
                        await cache_service.invalidate_conversation_cache_for_conversation(
                            conversation
                        )
                    else:
                        await cache_service.publish_pending_cache_entry(
                            resolution,
                            last_retrieval_message_seq=int(user_message["seq"]),
                        )
                except Exception:
                    if invalidate_confirmation_cache:
                        logger.exception(
                            "Failed to invalidate cache entry after confirmation for retrieval_event_id=%s",
                            retrieval_event["id"],
                        )
                        post_commit_errors.append("cache_invalidation_failed")
                    else:
                        logger.exception(
                            "Failed to publish pending cache entry for retrieval_event_id=%s",
                            retrieval_event["id"],
                        )
                        post_commit_errors.append("cache_publish_failed")

                if resolved_response_mode is ResponseMode.SMART_FAST:
                    # Unbind the turn's meter BEFORE scheduling the warm.
                    # asyncio.create_task copies the current context, so a meter
                    # still bound here would follow the warm and charge its
                    # round-trips to a turn whose row is already written -- calls
                    # that then appear in no persisted telemetry at all. The
                    # sidecar surface already unbinds first; this matches it.
                    # Ending the meter twice is a no-op (removal is by identity),
                    # so the finally below stays correct.
                    self.runtime.llm_client.end_turn_call_meter(turn_call_meter)
                    cache_service.schedule_smart_fast_warm(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_text=prompt_message_text,
                        assistant_mode_id=resolved_mode_id,
                        operational_profile=operational_profile,
                        operational_signals=operational_signals,
                        ablation=ablation,
                        prompt_authority_context=authority_context,
                        last_retrieval_message_seq=int(user_message["seq"]),
                        adaptive_retrieval=resolved_adaptive_retrieval,
                    )

                published_recent_window = False
                try:
                    published_recent_window = await _publish_committed_recent_window(
                        connection,
                        cache_service=cache_service,
                        clock=self.runtime.clock,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to update recent window for conversation_id=%s",
                        conversation_id,
                    )
                except BaseException:
                    # Cancellation leaves EXACTLY the state a failed publish
                    # leaves: this turn's messages are already committed, so the
                    # window still published for the conversation describes an
                    # older transcript. `except Exception` never sees
                    # CancelledError, so the compensating drop below was skipped
                    # and the superseded entry survived the turn. Drop it here,
                    # then let the cancellation continue unchanged.
                    await _drop_superseded_recent_window(
                        cache_service,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    raise
                if not published_recent_window:
                    post_commit_errors.append("recent_window_failed")
                    # This turn committed new messages, so whatever window is
                    # still published for the conversation now describes an
                    # older transcript. The read side refuses any window whose
                    # identity is not the reader's, so a survivor can no longer
                    # be served as current -- but it would sit there until some
                    # later turn overwrote it, and no reader would get the
                    # window it is entitled to. Dropping it is what turns a
                    # failed publish into an honest miss.
                    if not await _drop_superseded_recent_window(
                        cache_service,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ):
                        post_commit_errors.append("recent_window_stale_entry_retained")

                try:
                    await job_tracking.dispatch_pending_jobs(
                        self.runtime.storage_backend
                    )
                    memory_processing = await job_tracking.get_status(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to dispatch durable post-response jobs for retrieval_event_id=%s",
                        retrieval_event["id"],
                    )
                    post_commit_errors.append("job_dispatch_deferred")

                try:
                    await events.update_outcome_fields(
                        str(retrieval_event["id"]),
                        user_id,
                        {
                            "background_tasks_enqueued": background_tasks_enqueued,
                            "post_commit_errors": post_commit_errors,
                        },
                    )
                except Exception:
                    logger.exception(
                        "Failed to update retrieval outcome metadata for retrieval_event_id=%s",
                        retrieval_event["id"],
                    )

                debug_payload: dict[str, Any] | None = None
                if debug:
                    debug_payload = {
                        "response_mode": resolved_response_mode.value,
                        "adaptive_retrieval": resolved_adaptive_retrieval,
                        "cold_start": cold_start,
                        "llm_call_metrics": turn_llm_call_metrics,
                        "detected_needs": list(resolution.detected_needs),
                        "retrieval_plan": dict(resolution.source_retrieval_plan),
                        "selected_memory_ids": resolution.composed_context.selected_memory_ids,
                        "context_view": composed_context_json,
                        "scored_candidates": resolution.scored_candidates,
                        "retrieval_custody_v2": resolution.candidate_custody,
                        "retrieval_sufficiency": resolution.retrieval_sufficiency,
                        "retrieval_diagnostics_for_guard": (
                            resolution.retrieval_diagnostics_for_guard
                        ),
                        "candidate_search_summary": resolution.candidate_search_summary,
                        "retrieval_trace": resolution.retrieval_trace,
                        "context_envelope": context_envelope_trace,
                        "initial_context_package": initial_context_package.diagnostics,
                        "stage_timings": resolution.stage_timings,
                        "cache": {
                            "from_cache": resolution.from_cache,
                            "staleness": resolution.staleness,
                            "next_refresh_strategy": resolution.next_refresh_strategy,
                            "cache_age_seconds": resolution.cache_age_seconds,
                            "cache_source": resolution.cache_source,
                            "need_detection_skipped": resolution.need_detection_skipped,
                            "cache_key": resolution.cache_key,
                        },
                        "enqueued_job_ids": enqueued_job_ids,
                        "post_commit_errors": post_commit_errors,
                        "memory_processing": (
                            None
                            if memory_processing is None
                            else memory_processing.model_dump(mode="json")
                        ),
                        "answer_postcondition_guard": answer_postcondition_report,
                        "topic_working_set": visible_topic_snapshot,
                        "topic_working_set_block": topic_context_block,
                        "authority": {
                            "privacy_enforcement": authority_context.privacy_enforcement,
                            "effective_privacy_enforcement": (
                                authority_context.effective_privacy_enforcement
                            ),
                            "authenticated_privilege_level": (
                                authority_context.normalized_privilege_level
                            ),
                            "authenticated_atagia_master": (
                                authority_context.authenticated_user_is_atagia_master
                            ),
                            "authority_source": authority_context.authority_source,
                            "sensitive_trace": bool(debug_include_sensitive),
                        },
                    }
                    if llm_response.thinking:
                        debug_payload["thinking"] = llm_response.thinking
                    if not debug_include_sensitive:
                        debug_payload = self._redact_debug_payload(debug_payload)

                chat_result = ChatResult(
                    conversation_id=conversation_id,
                    request_message_id=str(user_message["id"]),
                    response_message_id=str(assistant_message["id"]),
                    response_text=response_text,
                    retrieval_event_id=str(retrieval_event["id"]),
                    composed_context=resolution.composed_context,
                    detected_needs=resolution.detected_needs,
                    memories_used=summarize_memory_summaries(
                        resolution.memory_summaries
                    ),
                    memory_processing=memory_processing,
                    debug=debug_payload,
                )
            except OutputLimitExceededError as exc:
                raise LLMUnavailableError("LLM output limit exceeded") from exc
            except LLMError as exc:
                raise LLMUnavailableError("LLM service unavailable") from exc
            finally:
                self.runtime.llm_client.end_turn_call_meter(turn_call_meter)
                await connection.close()
            if self.runtime.settings.lifecycle_lazy_enabled:
                try:
                    request_lifecycle_piggyback(self.runtime, reason="chat_completed")
                except Exception:
                    logger.exception("Failed to spawn lifecycle piggyback task")
            if chat_result is None:
                raise RuntimeError("Chat flow completed without a result")
            return chat_result

    @staticmethod
    def _redact_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Return HTTP-safe observability without raw retrieved context."""
        cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
        postcondition = payload.get("answer_postcondition_guard")
        postcondition_summary = None
        if isinstance(postcondition, dict):
            postcondition_summary = {
                "status": postcondition.get("status"),
                "failure_reasons": postcondition.get("failure_reasons"),
                "retry_count": postcondition.get("retry_count"),
                "output_limit_seen": postcondition.get("output_limit_seen"),
            }
        authority = payload.get("authority")
        if isinstance(authority, dict):
            authority = {**authority, "sensitive_trace": False}
        return {
            "cold_start": payload.get("cold_start"),
            "detected_needs_count": len(payload.get("detected_needs") or []),
            "selected_memory_count": len(payload.get("selected_memory_ids") or []),
            "cache": {
                "from_cache": cache.get("from_cache"),
                "staleness": cache.get("staleness"),
                "next_refresh_strategy": cache.get("next_refresh_strategy"),
                "cache_age_seconds": cache.get("cache_age_seconds"),
                "cache_source": cache.get("cache_source"),
                "need_detection_skipped": cache.get("need_detection_skipped"),
            },
            "post_commit_errors": list(payload.get("post_commit_errors") or []),
            "answer_postcondition_guard": postcondition_summary,
            # Counts, latency, and engine-internal purpose labels only: no user
            # data, so non-admin debug callers keep the cost signal.
            "llm_call_metrics": payload.get("llm_call_metrics"),
            "authority": authority,
        }

    @staticmethod
    def _chat_intimacy_metadata(
        topic_snapshot: dict[str, Any],
        *,
        allow_intimacy_context: bool,
    ) -> dict[str, Any]:
        topics = [
            *(topic_snapshot.get("active_topics") or []),
            *(topic_snapshot.get("parked_topics") or []),
        ]
        intimate_topics = [
            topic
            for topic in topics
            if isinstance(topic, dict)
            and str(topic.get("intimacy_boundary") or "ordinary") != "ordinary"
        ]
        if intimate_topics:
            boundary = strongest_intimacy_boundary(intimate_topics)
            confidence = max(
                (
                    float(topic.get("intimacy_boundary_confidence", 0.0) or 0.0)
                    for topic in intimate_topics
                ),
                default=0.0,
            )
            return known_intimacy_context_metadata(
                reason="topic_working_set_intimacy_boundary",
                boundary=boundary.value,
                confidence=confidence,
            )
        if allow_intimacy_context:
            return known_intimacy_context_metadata(
                reason="resolved_policy_allows_intimacy_context"
            )
        return {}


async def _drop_superseded_recent_window(
    cache_service: ContextCacheService,
    *,
    user_id: str,
    conversation_id: str,
) -> bool:
    """Delete the conversation's published window, cancellation included.

    Deletion is by ``(user_id, conversation_id)``, not by cache identity: the
    entry to remove was published by an EARLIER turn under a different
    identity, so the identity-scoped primitive could not reach it and would
    leave the superseded window in place. The key is derived from the pair, so
    this can only ever touch this conversation's own entry.

    The drop is shielded because the caller runs it on the cancellation path
    too: a plain ``await`` there is itself cancelled at the first suspension,
    which is how a cancelled turn used to leave the superseded entry published.

    Returns whether the conversation is now free of a superseded window, which
    is what the caller reports. Deleting nothing counts: an entry that was not
    there was not left behind.
    """

    task = asyncio.ensure_future(
        cache_service.drop_recent_window(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            logger.exception(
                "Failed to drop the superseded recent window for "
                "conversation_id=%s during cancellation",
                conversation_id,
            )
        raise
    except Exception:
        logger.exception(
            "Failed to drop the superseded recent window for conversation_id=%s",
            conversation_id,
        )
        return False
    return True


async def _publish_committed_recent_window(
    connection: aiosqlite.Connection,
    *,
    cache_service: ContextCacheService,
    clock: Any,
    user_id: str,
    conversation_id: str,
) -> bool:
    """Publish the committed transcript window under live cache coordinates.

    The window content and the coordinates it is stamped with are read in one
    SQLite read transaction, so a published entry always names the exact
    canonical state it describes.

    Reusing the coordinates captured before the model call does not work: any
    derived-state write by a background worker advances
    ``user_lifecycles.cache_revision`` through the ``icp_source_*`` triggers,
    and a turn spans seconds of model latency, so the pre-call snapshot is
    routinely one revision behind by the time the turn commits. The publish
    fence then rejects a window whose transcript is perfectly current.
    """

    await connection.execute("BEGIN")
    try:
        lifecycle_identity = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(user_id)
        conversation_identity = await ConversationLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        window_rows = await MessageRepository(connection, clock).get_recent_messages(
            conversation_id,
            user_id,
            limit=RECENT_WINDOW_MESSAGES,
        )
        await connection.commit()
    except BaseException:
        await connection.rollback()
        raise
    if lifecycle_identity is None or conversation_identity is None:
        return False
    return await cache_service.publish_recent_window(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": str(row["role"]), "content": str(row["text"])}
            for row in window_rows
        ],
        lifecycle_epoch=lifecycle_identity.lifecycle_epoch,
        lifecycle_cleanup_key=lifecycle_identity.lifecycle_cleanup_key,
        cache_revision=lifecycle_identity.cache_revision,
        derivation_revision=lifecycle_identity.derivation_revision,
        conversation_lifecycle_epoch=conversation_identity.lifecycle_epoch,
        conversation_source_revision=conversation_identity.source_revision,
    )


def _build_turn_jobs(
    *,
    clock: Any,
    conversation: dict[str, Any],
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
    prior_messages: list[dict[str, Any]],
    prompt_message_text: str,
    assistant_output_text: str,
    operational_profile: Any,
    memory_preferences: dict[str, Any],
    active_presence: Any,
    user_source_presence: Any,
    active_space: Any,
    active_mind: Any,
    active_embodiment: Any,
    active_realm: Any,
) -> list[tuple[str, Any]]:
    """Build all turn jobs before the caller commits their source messages."""

    common = {
        "clock": clock,
        "conversation": conversation,
        "operational_profile": operational_profile,
        "memory_preferences": memory_preferences,
        "active_presence_id": active_presence.presence_id,
        "active_presence_kind": active_presence.kind.value,
        "active_presence_display_name": active_presence.display_name,
        "active_space_id": active_space.space_id if active_space is not None else None,
        "active_space_boundary_mode": (
            active_space.boundary_mode.value if active_space is not None else None
        ),
        "active_space_display_name": (
            active_space.display_name if active_space is not None else None
        ),
        "active_mind_id": active_mind.mind_id,
        "source_mind_id": active_mind.mind_id,
        "active_mind_display_name": active_mind.display_name,
        "mind_topology": active_mind.topology.value,
        "active_embodiment_id": (
            active_embodiment.embodiment_id if active_embodiment is not None else None
        ),
        "active_embodiment_display_name": (
            active_embodiment.display_name if active_embodiment is not None else None
        ),
        "cross_embodiment_mode": (
            active_embodiment.cross_embodiment_mode.value
            if active_embodiment is not None
            else None
        ),
        "active_realm_id": active_realm.realm_id if active_realm is not None else None,
        "active_realm_display_name": (
            active_realm.display_name if active_realm is not None else None
        ),
        "cross_realm_mode": (
            active_realm.cross_realm_mode.value if active_realm is not None else None
        ),
    }
    user_jobs = build_message_jobs(
        **common,
        message_id=str(user_message["id"]),
        prior_messages=prior_messages,
        message_text=prompt_message_text,
        occurred_at=resolve_message_occurred_at(user_message),
        role="user",
        source_presence_id=user_source_presence.presence_id,
        source_presence_kind=user_source_presence.kind.value,
        source_presence_display_name=user_source_presence.display_name,
    )
    assistant_jobs = build_message_jobs(
        **common,
        message_id=str(assistant_message["id"]),
        prior_messages=[*prior_messages, user_message],
        message_text=assistant_output_text,
        occurred_at=resolve_message_occurred_at(assistant_message),
        role="assistant",
        source_presence_id=active_presence.presence_id,
        source_presence_kind=active_presence.kind.value,
        source_presence_display_name=active_presence.display_name,
    )
    return [*user_jobs, *assistant_jobs]


def _intimacy_boundary_counts(
    candidate_custody: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in candidate_custody:
        boundary = str(record.get("intimacy_boundary") or "ordinary")
        counts[boundary] = counts.get(boundary, 0) + 1
    return counts


def _intimacy_policy_filtered_count(candidate_custody: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in candidate_custody
        if record.get("filter_reason") == INTIMACY_FILTER_REASON
    )


def _validate_optional_identity(
    conversation: dict[str, Any],
    *,
    user_persona_id: str | None,
    platform_id: str | None,
    character_id: str | None,
    active_presence_id: str | None = None,
    mind_id: str | None = None,
    mind_topology: str | None = None,
    embodiment_id: str | None = None,
    realm_id: str | None = None,
    space_id: str | None = None,
) -> None:
    validate_optional_identity_hints(
        conversation,
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
    )
    if active_presence_id is not None:
        actual_presence = conversation.get("active_presence_id")
        actual_presence_text = None if actual_presence is None else str(actual_presence)
        if (
            actual_presence_text is not None
            and actual_presence_text != active_presence_id
        ):
            raise ConversationNotFoundError("Conversation not found for user")
    if mind_id is not None:
        actual_mind = conversation.get("active_mind_id")
        actual_mind_text = None if actual_mind is None else str(actual_mind)
        if actual_mind_text is not None and actual_mind_text != mind_id:
            raise ConversationNotFoundError("Conversation not found for user")
    expected_topology = _normalize_optional_text(mind_topology)
    if expected_topology is not None:
        actual_topology = conversation.get("mind_topology")
        actual_topology_text = None if actual_topology is None else str(actual_topology)
        if (
            actual_topology_text is not None
            and actual_topology_text != expected_topology
        ):
            raise ConversationNotFoundError("Conversation not found for user")
    if space_id is not None:
        actual_space = conversation.get("active_space_id")
        actual_space_text = None if actual_space is None else str(actual_space)
        if actual_space_text is not None and actual_space_text != space_id:
            raise ConversationNotFoundError("Conversation not found for user")
    if embodiment_id is not None:
        actual_embodiment = conversation.get("active_embodiment_id")
        actual_embodiment_text = (
            None if actual_embodiment is None else str(actual_embodiment)
        )
        if (
            actual_embodiment_text is not None
            and actual_embodiment_text != embodiment_id
        ):
            raise ConversationNotFoundError("Conversation not found for user")
    if realm_id is not None:
        actual_realm = conversation.get("active_realm_id")
        actual_realm_text = None if actual_realm is None else str(actual_realm)
        if actual_realm_text is not None and actual_realm_text != realm_id:
            raise ConversationNotFoundError("Conversation not found for user")


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, MindTopology):
        value = value.value
    normalized = str(value).strip()
    return normalized or None
