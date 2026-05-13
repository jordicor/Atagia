"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from atagia.core.llm_output_limits import CHAT_REPLY_MAX_OUTPUT_TOKENS
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.topic_repository import TopicRepository
from atagia.memory.lifecycle_runner import piggyback_lifecycle
from atagia.core.summary_repository import SummaryRepository
from atagia.core.timestamps import normalize_optional_timestamp, resolve_message_occurred_at
from atagia.memory.intimacy_boundary_policy import (
    INTIMACY_FILTER_REASON,
    strongest_intimacy_boundary,
)
from atagia.models.schemas_api import ChatResult
from atagia.models.schemas_memory import ConversationStatus, MindTopology
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import (
    CONTEXT_VIEW_TTL_SECONDS,
    RECENT_FETCH_LIMIT,
    RECENT_WINDOW_MESSAGES,
    apply_conversation_policy_overlay,
    build_message_jobs,
    build_system_prompt,
    build_transcript_window,
    build_transcript_window_trace,
    chat_model,
    enqueue_message_jobs,
    filter_topic_working_set_snapshot,
    missing_uncovered_tail_start_seq,
    render_transcript_window,
    render_topic_working_set_block,
    resolve_retrieval_profile_id,
    resolve_operational_profile,
    resolve_policy,
    summarize_memory_summaries,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.job_tracking_service import (
    JobTrackingService,
    render_memory_processing_status_block,
)
from atagia.services.presence_resolution import (
    ensure_conversation_active_presence,
    resolve_source_presence_for_role,
)
from atagia.services.embodiment_resolution import ensure_conversation_active_embodiment
from atagia.services.mind_resolution import ensure_conversation_active_mind
from atagia.services.realm_resolution import ensure_conversation_active_realm
from atagia.services.space_resolution import ensure_conversation_active_space
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
    known_intimacy_context_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatService:
    """Coordinates end-to-end chat flow."""

    runtime: Any

    async def chat_reply(
        self,
        user_id: str,
        conversation_id: str,
        message_text: str,
        assistant_mode_id: str | None = None,
        *,
        attachments: list[Any] | None = None,
        message_occurred_at: str | None = None,
        include_thinking: bool = False,
        metadata: dict[str, Any] | None = None,
        debug: bool = False,
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
    ) -> ChatResult:
        """Run the full retrieval, generation, persistence, and background-job flow."""
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            connection = await self.runtime.open_connection()
            try:
                conversations = ConversationRepository(connection, self.runtime.clock)
                users = UserRepository(connection, self.runtime.clock)
                messages = MessageRepository(connection, self.runtime.clock)
                memories = MemoryObjectRepository(connection, self.runtime.clock)
                events = RetrievalEventRepository(connection, self.runtime.clock)
                summaries = SummaryRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(
                    connection,
                    self.runtime.clock,
                    blob_store=self.runtime.artifact_blob_store,
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

                conversation = await conversations.get_conversation(conversation_id, user_id)
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
                memory_preferences = await users.get_memory_preferences(user_id)
                if str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                    raise ConversationNotActiveError("Conversation is not active")
                if not cross_chat_memory and not bool(conversation.get("isolated_mode")):
                    updated = await conversations.mark_conversation_isolated(
                        conversation_id,
                        user_id,
                    )
                    if updated is not None:
                        await cache_service.invalidate_conversation_cache_for_conversation(
                            updated
                        )
                        conversation = updated
                if incognito is True and not bool(conversation.get("incognito")):
                    updated = await conversations.mark_conversation_isolated(
                        conversation_id,
                        user_id,
                    )
                    if updated is not None:
                        await cache_service.invalidate_conversation_cache_for_conversation(
                            updated
                        )
                        conversation = updated

                conversation, active_presence = await ensure_conversation_active_presence(
                    connection,
                    self.runtime.clock,
                    conversation=conversation,
                    active_presence_id=active_presence_id,
                    character_id=character_id,
                )
                user_source_presence = await resolve_source_presence_for_role(
                    connection,
                    self.runtime.clock,
                    owner_user_id=user_id,
                    role="user",
                    active_presence=active_presence,
                )
                conversation, active_mind = await ensure_conversation_active_mind(
                    connection,
                    self.runtime.clock,
                    conversation=conversation,
                    mind_id=mind_id,
                    mind_topology=mind_topology,
                    active_presence=active_presence,
                    character_id=character_id,
                )
                conversation, active_embodiment = await ensure_conversation_active_embodiment(
                    connection,
                    self.runtime.clock,
                    conversation=conversation,
                    embodiment_id=embodiment_id,
                )
                conversation, active_realm = await ensure_conversation_active_realm(
                    connection,
                    self.runtime.clock,
                    conversation=conversation,
                    realm_id=realm_id,
                )
                conversation, active_space = await ensure_conversation_active_space(
                    connection,
                    self.runtime.clock,
                    conversation=conversation,
                    space_id=space_id,
                    workspace_id=conversation.get("workspace_id"),
                )

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
                conversation_chunks = await summaries.list_all_conversation_chunks(user_id, conversation_id)
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
                        character_id=conversation.get("character_id") or conversation.get("workspace_id"),
                        incognito=bool(conversation.get("incognito"))
                        or bool(conversation.get("isolated_mode")),
                        remember_across_chats=bool(conversation.get("remember_across_chats", 1)),
                        remember_across_devices=bool(conversation.get("remember_across_devices", 1)),
                        active_mind_id=active_mind.mind_id,
                        mind_topology=active_mind.topology,
                        active_embodiment_id=(
                            active_embodiment.embodiment_id
                            if active_embodiment is not None
                            else None
                        ),
                        active_realm_id=(
                            active_realm.realm_id
                            if active_realm is not None
                            else None
                        ),
                    )
                    == 0
                )
                confirmation_plan = await confirmations.plan_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_text=prompt_message_text,
                )
                invalidate_confirmation_cache = confirmation_plan.response_intent is not None
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
                )
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
                if self.runtime.settings.benchmark_disable_raw_recent_transcript:
                    transcript_entries = []
                    transcript_trace = build_transcript_window_trace([], 0)
                else:
                    transcript_budget_tokens = (
                        self.runtime.settings.effective_recent_transcript_budget_tokens(
                            resolution.resolved_policy.transcript_budget_tokens,
                            hard_cap_tokens=(
                                resolution.resolved_operational_profile.policy_override.transcript_budget_tokens
                            ),
                        )
                    )
                    transcript_entries = build_transcript_window(
                        prior_messages,
                        conversation_chunks,
                        transcript_budget_tokens,
                        raw_context_access_mode=str(
                            resolution.source_retrieval_plan.get("raw_context_access_mode", "normal")
                        ),
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
                llm_response = await self.runtime.llm_client.complete(
                    LLMCompletionRequest(
                        model=chat_model(self.runtime.settings),
                        messages=[
                            LLMMessage(
                                role="system",
                                content=build_system_prompt(
                                    resolved_mode_id,
                                    resolution.resolved_policy,
                                    resolution.composed_context.contract_block,
                                    resolution.composed_context.workspace_block,
                                    resolution.composed_context.memory_block,
                                    resolution.composed_context.state_block,
                                    topic_context_block=topic_context_block,
                                    memory_processing_block=render_memory_processing_status_block(
                                        prompt_memory_processing
                                    ),
                                ),
                            ),
                            *[
                                LLMMessage(role=str(message["role"]), content=str(message["text"]))
                                for message in transcript
                            ],
                        ],
                        temperature=0.0,
                        max_output_tokens=CHAT_REPLY_MAX_OUTPUT_TOKENS,
                        include_thinking=include_thinking,
                        metadata={
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "assistant_mode_id": resolved_mode_id,
                            "purpose": "chat_reply",
                            **self._chat_intimacy_metadata(
                                visible_topic_snapshot,
                                allow_intimacy_context=(
                                    resolution.resolved_policy.allow_intimacy_context
                                ),
                            ),
                        },
                    )
                )
                response_text = llm_response.output_text
                if confirmation_plan.prompt_text is not None:
                    response_text = f"{confirmation_plan.prompt_text}\n\n{llm_response.output_text}"
                resolved_user_occurred_at = (
                    normalize_optional_timestamp(message_occurred_at)
                    or self.runtime.clock.now().isoformat()
                )
                assistant_occurred_at = self.runtime.clock.now().isoformat()

                await connection.execute("BEGIN IMMEDIATE")
                try:
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
                        space_id=active_space.space_id if active_space is not None else None,
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
                        text=llm_response.output_text,
                        token_count=None,
                        metadata={"thinking": llm_response.thinking} if llm_response.thinking else {},
                        occurred_at=assistant_occurred_at,
                        active_presence_id=active_presence.presence_id,
                        source_presence_id=active_presence.presence_id,
                        space_id=active_space.space_id if active_space is not None else None,
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
                    retrieval_event = await events.create_event(
                        {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "request_message_id": user_message["id"],
                            "response_message_id": assistant_message["id"],
                            "assistant_mode_id": resolved_mode_id,
                            "user_persona_id": conversation.get("user_persona_id"),
                            "platform_id": conversation.get("platform_id") or "default",
                            "character_id": conversation.get("character_id") or conversation.get("workspace_id"),
                            "mode": conversation.get("mode") or resolved_mode_id,
                            "incognito": bool(conversation.get("incognito")) or bool(conversation.get("isolated_mode")),
                            "remember_across_chats": bool(memory_preferences["remember_across_chats"]),
                            "remember_across_devices": bool(memory_preferences["remember_across_devices"]),
                            "memory_privacy_mode": memory_preferences["memory_privacy_mode"],
                            "retrieval_plan_json": resolution.source_retrieval_plan,
                            "selected_memory_ids_json": resolution.composed_context.selected_memory_ids,
                            "context_view_json": resolution.composed_context.model_dump(mode="json"),
                            "outcome_json": {
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
                                "stage_timings_ms": resolution.stage_timings,
                                "transcript_window": transcript_trace,
                                "operational_profile": (
                                    resolution.resolved_operational_profile.snapshot.model_dump(mode="json")
                                ),
                                **resolution.candidate_search_summary,
                            },
                        },
                        commit=False,
                    )
                    confirmation_embedding_upserts = await confirmations.apply_turn_plan(
                        user_id=user_id,
                        plan=confirmation_plan,
                        commit=False,
                    )
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
                await confirmations.apply_post_commit_embeddings(confirmation_embedding_upserts)

                post_commit_errors: list[str] = []
                enqueued_job_ids: list[str] = []
                background_tasks_enqueued = False
                memory_processing = prompt_memory_processing

                try:
                    if invalidate_confirmation_cache:
                        await cache_service.invalidate_conversation_cache_for_conversation(conversation)
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

                if self.runtime.settings.lifecycle_lazy_enabled:
                    try:
                        self.runtime.spawn_background_task(
                            piggyback_lifecycle(self.runtime),
                            name="atagia-lifecycle-piggyback",
                        )
                    except Exception:
                        logger.exception("Failed to spawn lifecycle piggyback task")
                        post_commit_errors.append("lifecycle_spawn_failed")

                final_window = [
                    {"role": str(message["role"]), "content": str(message["text"])}
                    for message in [*prior_messages, user_message, assistant_message][-RECENT_WINDOW_MESSAGES:]
                ]
                try:
                    await self.runtime.storage_backend.set_recent_window(
                        f"{user_id}:{conversation_id}",
                        final_window,
                    )
                except Exception:
                    logger.exception(
                        "Failed to update recent window for conversation_id=%s",
                        conversation_id,
                    )
                    post_commit_errors.append("recent_window_failed")

                try:
                    await self.runtime.storage_backend.set_context_view(
                        retrieval_event["id"],
                        resolution.composed_context.model_dump(mode="json"),
                        ttl_seconds=CONTEXT_VIEW_TTL_SECONDS,
                    )
                except Exception:
                    logger.exception(
                        "Failed to store context view for retrieval_event_id=%s",
                        retrieval_event["id"],
                    )
                    post_commit_errors.append("context_view_failed")

                user_jobs = build_message_jobs(
                    clock=self.runtime.clock,
                    conversation=conversation,
                    message_id=str(user_message["id"]),
                    prior_messages=prior_messages,
                    message_text=prompt_message_text,
                    occurred_at=resolve_message_occurred_at(user_message),
                    role="user",
                    operational_profile=resolution.resolved_operational_profile.snapshot,
                    memory_preferences=memory_preferences,
                    active_presence_id=active_presence.presence_id,
                    active_presence_kind=active_presence.kind.value,
                    active_presence_display_name=active_presence.display_name,
                    source_presence_id=user_source_presence.presence_id,
                    source_presence_kind=user_source_presence.kind.value,
                    source_presence_display_name=user_source_presence.display_name,
                    active_space_id=active_space.space_id if active_space is not None else None,
                    active_space_boundary_mode=(
                        active_space.boundary_mode.value if active_space is not None else None
                    ),
                    active_space_display_name=(
                        active_space.display_name if active_space is not None else None
                    ),
                    active_mind_id=active_mind.mind_id,
                    source_mind_id=active_mind.mind_id,
                    active_mind_display_name=active_mind.display_name,
                    mind_topology=active_mind.topology.value,
                    active_embodiment_id=(
                        active_embodiment.embodiment_id
                        if active_embodiment is not None
                        else None
                    ),
                    active_embodiment_display_name=(
                        active_embodiment.display_name
                        if active_embodiment is not None
                        else None
                    ),
                    cross_embodiment_mode=(
                        active_embodiment.cross_embodiment_mode.value
                        if active_embodiment is not None
                        else None
                    ),
                    active_realm_id=(
                        active_realm.realm_id if active_realm is not None else None
                    ),
                    active_realm_display_name=(
                        active_realm.display_name if active_realm is not None else None
                    ),
                    cross_realm_mode=(
                        active_realm.cross_realm_mode.value
                        if active_realm is not None
                        else None
                    ),
                )
                assistant_jobs = build_message_jobs(
                    clock=self.runtime.clock,
                    conversation=conversation,
                    message_id=str(assistant_message["id"]),
                    prior_messages=[*prior_messages, user_message],
                    message_text=llm_response.output_text,
                    occurred_at=resolve_message_occurred_at(assistant_message),
                    role="assistant",
                    operational_profile=resolution.resolved_operational_profile.snapshot,
                    memory_preferences=memory_preferences,
                    active_presence_id=active_presence.presence_id,
                    active_presence_kind=active_presence.kind.value,
                    active_presence_display_name=active_presence.display_name,
                    source_presence_id=active_presence.presence_id,
                    source_presence_kind=active_presence.kind.value,
                    source_presence_display_name=active_presence.display_name,
                    active_space_id=active_space.space_id if active_space is not None else None,
                    active_space_boundary_mode=(
                        active_space.boundary_mode.value if active_space is not None else None
                    ),
                    active_space_display_name=(
                        active_space.display_name if active_space is not None else None
                    ),
                    active_mind_id=active_mind.mind_id,
                    source_mind_id=active_mind.mind_id,
                    active_mind_display_name=active_mind.display_name,
                    mind_topology=active_mind.topology.value,
                    active_embodiment_id=(
                        active_embodiment.embodiment_id
                        if active_embodiment is not None
                        else None
                    ),
                    active_embodiment_display_name=(
                        active_embodiment.display_name
                        if active_embodiment is not None
                        else None
                    ),
                    cross_embodiment_mode=(
                        active_embodiment.cross_embodiment_mode.value
                        if active_embodiment is not None
                        else None
                    ),
                    active_realm_id=(
                        active_realm.realm_id if active_realm is not None else None
                    ),
                    active_realm_display_name=(
                        active_realm.display_name if active_realm is not None else None
                    ),
                    cross_realm_mode=(
                        active_realm.cross_realm_mode.value
                        if active_realm is not None
                        else None
                    ),
                )
                try:
                    enqueued_job_ids = await enqueue_message_jobs(
                        storage_backend=self.runtime.storage_backend,
                        jobs=[*user_jobs, *assistant_jobs],
                        job_tracking_service=job_tracking,
                        worker_control_service=WorkerControlService(
                            connection,
                            self.runtime.clock,
                        ),
                    )
                    memory_processing = await job_tracking.get_status(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    background_tasks_enqueued = bool(enqueued_job_ids)
                except Exception:
                    logger.exception(
                        "Failed to enqueue post-response jobs for retrieval_event_id=%s",
                        retrieval_event["id"],
                    )
                    post_commit_errors.append("job_enqueue_failed")

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
                        "cold_start": cold_start,
                        "detected_needs": list(resolution.detected_needs),
                        "retrieval_plan": dict(resolution.source_retrieval_plan),
                        "selected_memory_ids": resolution.composed_context.selected_memory_ids,
                        "context_view": resolution.composed_context.model_dump(mode="json"),
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
                        "topic_working_set": visible_topic_snapshot,
                        "topic_working_set_block": topic_context_block,
                    }
                    if llm_response.thinking:
                        debug_payload["thinking"] = llm_response.thinking

                return ChatResult(
                    conversation_id=conversation_id,
                    request_message_id=str(user_message["id"]),
                    response_message_id=str(assistant_message["id"]),
                    response_text=response_text,
                    retrieval_event_id=str(retrieval_event["id"]),
                    composed_context=resolution.composed_context,
                    detected_needs=resolution.detected_needs,
                    memories_used=summarize_memory_summaries(resolution.memory_summaries),
                    memory_processing=memory_processing,
                    debug=debug_payload,
                )
            except LLMError as exc:
                raise LLMUnavailableError("LLM service unavailable") from exc
            finally:
                await connection.close()

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


def _intimacy_boundary_counts(candidate_custody: list[dict[str, Any]]) -> dict[str, int]:
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
    for field_name, expected in (
        ("user_persona_id", user_persona_id),
        ("platform_id", platform_id),
        ("character_id", character_id),
    ):
        actual = conversation.get(field_name)
        actual_text = None if actual is None else str(actual)
        if actual_text != expected:
            raise ConversationNotFoundError("Conversation not found for user")
    if active_presence_id is not None:
        actual_presence = conversation.get("active_presence_id")
        actual_presence_text = None if actual_presence is None else str(actual_presence)
        if actual_presence_text is not None and actual_presence_text != active_presence_id:
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
        if actual_topology_text is not None and actual_topology_text != expected_topology:
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
        if actual_embodiment_text is not None and actual_embodiment_text != embodiment_id:
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
