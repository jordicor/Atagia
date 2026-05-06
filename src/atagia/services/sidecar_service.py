"""Sidecar memory operations shared by library mode and REST routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiosqlite

from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.topic_repository import TopicRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import normalize_optional_timestamp, resolve_message_occurred_at
from atagia.memory.lifecycle_runner import cache_generation_key, piggyback_lifecycle
from atagia.models.schemas_api import ContextResult, MemoryProcessingStatus
from atagia.models.schemas_memory import (
    ConfirmationStrategy,
    ConversationStatus,
    IngestOrigin,
    MemoryPrivacyMode,
    resolve_confirmation_strategy,
    resolve_memory_privacy_mode,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import (
    DEFAULT_ASSISTANT_MODE_ID,
    RECENT_FETCH_LIMIT,
    build_recent_transcript_guidance,
    build_recent_transcript_window,
    build_message_jobs,
    build_system_prompt,
    enqueue_message_jobs,
    filter_topic_working_set_snapshot,
    render_assistant_guidance_block,
    render_recent_transcript_json_block,
    render_topic_working_set_block,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.job_tracking_service import (
    JobTrackingService,
    render_memory_processing_status_block,
)
from atagia.services.worker_control_service import WorkerControlService
from atagia.services.errors import (
    ConversationNotActiveError,
    ConversationNotFoundError,
    MessageIdConflictError,
    SourceSequenceConflictError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)

if TYPE_CHECKING:
    from atagia.app import AppRuntime


@dataclass(slots=True)
class SidecarMessageWriteResult:
    """Result of a sidecar message write or idempotent replay."""

    message: dict[str, Any]
    created: bool


@dataclass(slots=True)
class SidecarService:
    """Coordinate sidecar memory operations without performing a chat completion."""

    runtime: AppRuntime

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        ablation: AblationConfig | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> ContextResult:
        """Run retrieval, persist the user message, and return a ready system prompt."""
        resolved_ingest_origin, resolved_confirmation_strategy = self._resolve_ingest_control(
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
        )
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            connection = await self.runtime.open_connection()
            try:
                await self.ensure_user_exists(connection, user_id)
                conversation = await self.ensure_conversation(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=mode,
                    cross_chat_memory=cross_chat_memory,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    mode=mode,
                    incognito=incognito,
                )
                users = UserRepository(connection, self.runtime.clock)
                memory_preferences = await users.get_memory_preferences(user_id)
                resolved_memory_privacy_mode = self._resolve_memory_privacy_mode(
                    memory_privacy_mode,
                    memory_preferences,
                )
                messages = MessageRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(
                    connection,
                    self.runtime.clock,
                    blob_store=self.runtime.artifact_blob_store,
                )
                attachment_bundle = artifacts.prepare_attachments(
                    message_text=message,
                    attachments=attachments,
                    user_id=user_id,
                    conversation=conversation,
                )
                prompt_message_text = attachment_bundle.prompt_text
                resolved_message_id = self._normalize_optional_message_id(message_id)
                resolved_source_seq = self._normalize_optional_source_seq(source_seq)
                existing_user_message = await self._idempotent_message_if_present(
                    messages,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    message_id=resolved_message_id,
                    role="user",
                    text=prompt_message_text,
                    source_seq=resolved_source_seq,
                )
                prior_messages = await self._recent_messages_for_write(
                    messages,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    source_seq=resolved_source_seq,
                    existing_message=existing_user_message,
                )
                resolution = await cache_service.resolve_with_connection(
                    connection,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    message_text=prompt_message_text,
                    assistant_mode_id=mode,
                    stored_messages=prior_messages,
                    conversation=conversation,
                    operational_profile=operational_profile,
                    operational_signals=operational_signals,
                    ablation=ablation,
                )
                topic_snapshot = await self._topic_snapshot(
                    connection,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                )
                visible_topic_snapshot = filter_topic_working_set_snapshot(
                    topic_snapshot,
                    allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    privacy_ceiling=resolution.resolved_policy.privacy_ceiling,
                )
                if existing_user_message is not None:
                    user_message = existing_user_message
                else:
                    await connection.execute("BEGIN")
                    try:
                        await self._ensure_source_seq_available(
                            messages,
                            user_id=user_id,
                            conversation_id=str(conversation["id"]),
                            source_seq=resolved_source_seq,
                            message_id=resolved_message_id,
                        )
                        resolved_user_occurred_at = (
                            normalize_optional_timestamp(occurred_at)
                            or self.runtime.clock.now().isoformat()
                        )
                        user_message = await messages.create_message(
                            message_id=resolved_message_id,
                            conversation_id=str(conversation["id"]),
                            role="user",
                            seq=resolved_source_seq,
                            text=prompt_message_text,
                            token_count=None,
                            metadata=self._message_metadata_with_ingest_control(
                                attachment_bundle.message_metadata(),
                                ingest_origin=resolved_ingest_origin,
                                confirmation_strategy=resolved_confirmation_strategy,
                                memory_privacy_mode=resolved_memory_privacy_mode,
                            ),
                            occurred_at=resolved_user_occurred_at,
                            commit=False,
                        )
                        if attachment_bundle.artifacts:
                            await artifacts.persist_prepared_attachments(
                                bundle=attachment_bundle,
                                message_id=str(user_message["id"]),
                                commit=False,
                            )
                        await connection.commit()
                    except Exception:
                        await connection.rollback()
                        raise
            finally:
                await connection.close()
            await cache_service.publish_pending_cache_entry(
                resolution,
                last_retrieval_message_seq=int(user_message["seq"]),
            )
            if existing_user_message is None:
                memory_processing = await self._enqueue_message_jobs(
                    conversation=conversation,
                    message=user_message,
                    prior_messages=prior_messages,
                    message_text=prompt_message_text,
                    role="user",
                    operational_profile=resolution.resolved_operational_profile.snapshot,
                    ingest_origin=resolved_ingest_origin,
                    confirmation_strategy=resolved_confirmation_strategy,
                    memory_privacy_mode=resolved_memory_privacy_mode,
                )
            else:
                memory_processing = await self._message_processing_status(conversation)
            if self.runtime.settings.lifecycle_lazy_enabled:
                self.runtime.spawn_background_task(
                    piggyback_lifecycle(self.runtime),
                    name="atagia-lifecycle-piggyback",
                )

        recent_transcript_entries = []
        recent_transcript_omissions = []
        recent_transcript_trace = None
        recent_transcript_block = ""
        assistant_guidance = []
        assistant_guidance_block = ""
        if not self.runtime.settings.benchmark_disable_raw_recent_transcript:
            recent_transcript_budget_tokens = (
                self.runtime.settings.effective_recent_transcript_budget_tokens(
                    resolution.resolved_policy.transcript_budget_tokens,
                    hard_cap_tokens=(
                        resolution.resolved_operational_profile.policy_override.transcript_budget_tokens
                    ),
                )
            )
            recent_transcript = build_recent_transcript_window(
                prior_messages,
                recent_transcript_budget_tokens,
                overage_ratio=self.runtime.settings.recent_transcript_overage_ratio,
                raw_context_access_mode=str(
                    resolution.source_retrieval_plan.get("raw_context_access_mode", "normal")
                ),
            )
            recent_transcript_entries = recent_transcript.entries
            recent_transcript_omissions = recent_transcript.omissions
            recent_transcript_trace = recent_transcript.trace
            recent_transcript_block = render_recent_transcript_json_block(
                recent_transcript_entries,
            )
            assistant_guidance = build_recent_transcript_guidance(
                recent_transcript_omissions,
                enabled=self.runtime.settings.assistant_guidance_enabled,
            )
            assistant_guidance_block = render_assistant_guidance_block(
                assistant_guidance,
            )
        return ContextResult(
            system_prompt=build_system_prompt(
                resolution.resolved_policy.profile_id.value,
                resolution.resolved_policy,
                resolution.composed_context.contract_block,
                resolution.composed_context.workspace_block,
                resolution.composed_context.memory_block,
                resolution.composed_context.state_block,
                topic_context_block=render_topic_working_set_block(
                    visible_topic_snapshot,
                    allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                    privacy_ceiling=resolution.resolved_policy.privacy_ceiling,
                ),
                memory_processing_block=render_memory_processing_status_block(memory_processing),
                recent_transcript_block=recent_transcript_block,
                assistant_guidance_block=assistant_guidance_block,
            ),
            topic_working_set=visible_topic_snapshot,
            topic_working_set_block=render_topic_working_set_block(
                visible_topic_snapshot,
                allow_intimacy_context=resolution.resolved_policy.allow_intimacy_context,
                privacy_ceiling=resolution.resolved_policy.privacy_ceiling,
            ),
            recent_transcript=recent_transcript_entries,
            recent_transcript_omissions=recent_transcript_omissions,
            recent_transcript_trace=recent_transcript_trace,
            assistant_guidance=assistant_guidance,
            memories=resolution.memory_summaries,
            contract=resolution.current_contract,
            detected_needs=resolution.detected_needs,
            stage_timings=resolution.stage_timings,
            from_cache=resolution.from_cache,
            staleness=resolution.staleness,
            next_refresh_strategy=resolution.next_refresh_strategy,
            cache_age_seconds=resolution.cache_age_seconds,
            cache_source=resolution.cache_source,
            need_detection_skipped=resolution.need_detection_skipped,
            memory_processing=memory_processing,
            request_message_id=str(user_message["id"]),
        )

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> SidecarMessageWriteResult:
        """Store a message and enqueue extraction without running retrieval."""
        if role not in {"user", "assistant"}:
            raise ValueError("ingest_message role must be 'user' or 'assistant'")
        resolved_ingest_origin, resolved_confirmation_strategy = self._resolve_ingest_control(
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
        )

        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            resolved_operational_profile = resolve_operational_profile(
                loader=self.runtime.operational_profile_loader,
                settings=self.runtime.settings,
                operational_profile=operational_profile,
                operational_signals=operational_signals,
            )
            connection = await self.runtime.open_connection()
            conversation: dict[str, Any] | None = None
            prior_messages: list[dict[str, Any]] = []
            stored_message: dict[str, Any] | None = None
            try:
                await self.ensure_user_exists(connection, user_id)
                conversation = await self.ensure_conversation(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=mode,
                    cross_chat_memory=cross_chat_memory,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    mode=mode,
                    incognito=incognito,
                )
                users = UserRepository(connection, self.runtime.clock)
                memory_preferences = await users.get_memory_preferences(user_id)
                resolved_memory_privacy_mode = self._resolve_memory_privacy_mode(
                    memory_privacy_mode,
                    memory_preferences,
                )
                messages = MessageRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(
                    connection,
                    self.runtime.clock,
                    blob_store=self.runtime.artifact_blob_store,
                )
                attachment_bundle = artifacts.prepare_attachments(
                    message_text=text,
                    attachments=attachments,
                    user_id=user_id,
                    conversation=conversation,
                )
                prompt_message_text = attachment_bundle.prompt_text
                resolved_message_id = self._normalize_optional_message_id(message_id)
                resolved_source_seq = self._normalize_optional_source_seq(source_seq)
                existing_message = await self._idempotent_message_if_present(
                    messages,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    message_id=resolved_message_id,
                    role=role,
                    text=prompt_message_text,
                    source_seq=resolved_source_seq,
                )
                prior_messages = await self._recent_messages_for_write(
                    messages,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    source_seq=resolved_source_seq,
                    existing_message=existing_message,
                )
                if existing_message is not None:
                    stored_message = existing_message
                else:
                    await connection.execute("BEGIN")
                    try:
                        await self._ensure_source_seq_available(
                            messages,
                            user_id=user_id,
                            conversation_id=str(conversation["id"]),
                            source_seq=resolved_source_seq,
                            message_id=resolved_message_id,
                        )
                        resolved_occurred_at = (
                            normalize_optional_timestamp(occurred_at)
                            or self.runtime.clock.now().isoformat()
                        )
                        stored_message = await messages.create_message(
                            message_id=resolved_message_id,
                            conversation_id=str(conversation["id"]),
                            role=role,
                            seq=resolved_source_seq,
                            text=prompt_message_text,
                            token_count=None,
                            metadata=self._message_metadata_with_ingest_control(
                                attachment_bundle.message_metadata(),
                                ingest_origin=resolved_ingest_origin,
                                confirmation_strategy=resolved_confirmation_strategy,
                                memory_privacy_mode=resolved_memory_privacy_mode,
                            ),
                            occurred_at=resolved_occurred_at,
                            commit=False,
                        )
                        if attachment_bundle.artifacts:
                            await artifacts.persist_prepared_attachments(
                                bundle=attachment_bundle,
                                message_id=str(stored_message["id"]),
                                commit=False,
                            )
                        await cache_service.invalidate_conversation_cache_for_conversation(conversation)
                        await connection.commit()
                    except Exception:
                        await connection.rollback()
                        raise
            finally:
                await connection.close()

            if conversation is None or stored_message is None:
                raise RuntimeError("Message ingestion did not persist the message correctly")
            if existing_message is None:
                await self._enqueue_message_jobs(
                    conversation=conversation,
                    message=stored_message,
                    prior_messages=prior_messages,
                    message_text=prompt_message_text,
                    role=role,
                    operational_profile=resolved_operational_profile.snapshot,
                    ingest_origin=resolved_ingest_origin,
                    confirmation_strategy=resolved_confirmation_strategy,
                    memory_privacy_mode=resolved_memory_privacy_mode,
                )
            return SidecarMessageWriteResult(
                message=stored_message,
                created=existing_message is None,
            )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> SidecarMessageWriteResult:
        """Persist an assistant response in the conversation history."""
        resolved_ingest_origin, resolved_confirmation_strategy = self._resolve_ingest_control(
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
        )
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            resolved_operational_profile = resolve_operational_profile(
                loader=self.runtime.operational_profile_loader,
                settings=self.runtime.settings,
                operational_profile=operational_profile,
                operational_signals=operational_signals,
            )
            connection = await self.runtime.open_connection()
            conversation: dict[str, Any] | None = None
            prior_messages: list[dict[str, Any]] = []
            assistant_message: dict[str, Any] | None = None
            try:
                conversations = ConversationRepository(connection, self.runtime.clock)
                users = UserRepository(connection, self.runtime.clock)
                if await users.get_active_user(user_id) is None:
                    raise UserDeletedError("User has been erased or does not exist")
                memory_preferences = await users.get_memory_preferences(user_id)
                resolved_memory_privacy_mode = self._resolve_memory_privacy_mode(
                    memory_privacy_mode,
                    memory_preferences,
                )
                conversation = await conversations.get_conversation(conversation_id, user_id)
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                self._validate_optional_identity(
                    conversation,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                )
                if str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                    raise ConversationNotActiveError("Conversation is not active")
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
                messages = MessageRepository(connection, self.runtime.clock)
                resolved_message_id = self._normalize_optional_message_id(message_id)
                resolved_source_seq = self._normalize_optional_source_seq(source_seq)
                existing_message = await self._idempotent_message_if_present(
                    messages,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_id=resolved_message_id,
                    role="assistant",
                    text=text,
                    source_seq=resolved_source_seq,
                )
                prior_messages = await self._recent_messages_for_write(
                    messages,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    source_seq=resolved_source_seq,
                    existing_message=existing_message,
                )
                if existing_message is not None:
                    assistant_message = existing_message
                else:
                    await connection.execute("BEGIN")
                    try:
                        await self._ensure_source_seq_available(
                            messages,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            source_seq=resolved_source_seq,
                            message_id=resolved_message_id,
                        )
                        resolved_assistant_occurred_at = (
                            normalize_optional_timestamp(occurred_at)
                            or self.runtime.clock.now().isoformat()
                        )
                        assistant_message = await messages.create_message(
                            message_id=resolved_message_id,
                            conversation_id=conversation_id,
                            role="assistant",
                            seq=resolved_source_seq,
                            text=text,
                            token_count=None,
                            metadata=self._message_metadata_with_ingest_control(
                                {},
                                ingest_origin=resolved_ingest_origin,
                                confirmation_strategy=resolved_confirmation_strategy,
                                memory_privacy_mode=resolved_memory_privacy_mode,
                            ),
                            occurred_at=resolved_assistant_occurred_at,
                            commit=False,
                        )
                        await cache_service.invalidate_conversation_cache_for_conversation(conversation)
                        await connection.commit()
                    except Exception:
                        await connection.rollback()
                        raise
            finally:
                await connection.close()
            if conversation is None or assistant_message is None:
                raise RuntimeError("Assistant response did not persist correctly")
            if existing_message is None:
                await self._enqueue_message_jobs(
                    conversation=conversation,
                    message=assistant_message,
                    prior_messages=prior_messages,
                    message_text=text,
                    role="assistant",
                    operational_profile=resolved_operational_profile.snapshot,
                    ingest_origin=resolved_ingest_origin,
                    confirmation_strategy=resolved_confirmation_strategy,
                    memory_privacy_mode=resolved_memory_privacy_mode,
                )
            return SidecarMessageWriteResult(
                message=assistant_message,
                created=existing_message is None,
            )

    async def get_memory_preferences(self, user_id: str) -> dict[str, Any]:
        """Return memory sharing preferences for an active user."""
        connection = await self.runtime.open_connection()
        try:
            users = UserRepository(connection, self.runtime.clock)
            if await users.get_active_user(user_id) is None:
                raise UserDeletedError("User has been erased or does not exist")
            preferences = await users.get_memory_preferences(user_id)
            return {"user_id": user_id, **preferences}
        finally:
            await connection.close()

    async def set_memory_preferences(
        self,
        user_id: str,
        *,
        remember_across_chats: bool | None = None,
        remember_across_devices: bool | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> dict[str, Any]:
        """Update user memory preferences and invalidate stale broad context."""
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            connection = await self.runtime.open_connection()
            try:
                users = UserRepository(connection, self.runtime.clock)
                if await users.get_active_user(user_id) is None:
                    raise UserDeletedError("User has been erased or does not exist")
                preferences = await users.update_memory_preferences(
                    user_id,
                    remember_across_chats=remember_across_chats,
                    remember_across_devices=remember_across_devices,
                    memory_privacy_mode=(
                        resolve_memory_privacy_mode(memory_privacy_mode).value
                        if memory_privacy_mode is not None
                        else None
                    ),
                )
                await cache_service.invalidate_user_cache(user_id)
                await self.runtime.storage_backend.increment_cache_generation(
                    cache_generation_key(self.runtime.database_path, user_id)
                )
                return {"user_id": user_id, **preferences}
            finally:
                await connection.close()

    async def set_conversation_incognito(
        self,
        user_id: str,
        conversation_id: str,
        incognito: bool,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
    ) -> dict[str, Any]:
        """Toggle conversation incognito and invalidate stale context."""
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            connection = await self.runtime.open_connection()
            try:
                conversations = ConversationRepository(connection, self.runtime.clock)
                users = UserRepository(connection, self.runtime.clock)
                if await users.get_active_user(user_id) is None:
                    raise UserDeletedError("User has been erased or does not exist")
                existing = await conversations.get_conversation(conversation_id, user_id)
                if existing is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                self._validate_optional_identity(
                    existing,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                )
                await connection.execute("BEGIN")
                try:
                    updated = await conversations.set_conversation_incognito(
                        conversation_id,
                        user_id,
                        incognito,
                        commit=False,
                    )
                    affected_memory_ids: list[str] = []
                    if incognito:
                        affected_memory_ids = await self._mark_broad_conversation_rows_review_only(
                            connection,
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )
                        await self._delete_retrieval_events_for_memory_ids(
                            connection,
                            user_id=user_id,
                            memory_ids=affected_memory_ids,
                        )
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
                if updated is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                if incognito:
                    await cache_service.invalidate_user_cache(user_id)
                else:
                    await cache_service.invalidate_conversation_cache_for_conversation(updated)
                await self.runtime.storage_backend.increment_cache_generation(
                    cache_generation_key(self.runtime.database_path, user_id)
                )
                return updated
            finally:
                await connection.close()

    async def prepare_save_from_incognito_review(
        self,
        user_id: str,
        conversation_id: str,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Return an explicit review manifest for an incognito rescue request.

        This Phase 4 path deliberately performs no broad memory writes. The
        selected-memory write path lands with extraction/write-policy work, so
        the synchronous endpoint can safely expose reviewable source messages
        without silently promoting incognito-derived data.
        """
        connection = await self.runtime.open_connection()
        try:
            users = UserRepository(connection, self.runtime.clock)
            conversations = ConversationRepository(connection, self.runtime.clock)
            messages = MessageRepository(connection, self.runtime.clock)
            if await users.get_active_user(user_id) is None:
                raise UserDeletedError("User has been erased or does not exist")
            conversation = await conversations.get_conversation(conversation_id, user_id)
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found for user")
            self._validate_optional_identity(
                conversation,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
            )
            if str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                raise ConversationNotActiveError("Conversation is not active")
            if not bool(conversation.get("incognito")):
                raise ValueError("Conversation is not incognito")
            resolved_mode = (
                mode
                or conversation.get("mode")
                or conversation.get("assistant_mode_id")
                or DEFAULT_ASSISTANT_MODE_ID
            )
            resolve_policy(self.runtime.manifests, str(resolved_mode), self.runtime.policy_resolver)
            source_messages = await messages.list_messages_for_conversation(
                conversation_id,
                user_id,
            )
            review_messages = [
                {
                    "message_id": str(message["id"]),
                    "role": str(message["role"]),
                    "seq": int(message["seq"]),
                    "text": str(message["text"]),
                    "occurred_at": message.get("occurred_at"),
                    "content_kind": str(message.get("content_kind") or "text"),
                    "policy_reason": str(message.get("policy_reason") or "normal"),
                    "skip_by_default": bool(message.get("skip_by_default")),
                }
                for message in source_messages
                if str(message["role"]) in {"user", "assistant"}
            ]
            return {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "status": "review_required",
                "review_policy": "non_incognito",
                "source_message_count": len(review_messages),
                "source_messages": review_messages,
                "suggested_memory_count": 0,
                "suggested_memories": [],
                "writes_performed": False,
            }
        finally:
            await connection.close()

    async def ensure_user_exists(self, connection: aiosqlite.Connection, user_id: str) -> None:
        """Create the user if it does not already exist."""
        users = UserRepository(connection, self.runtime.clock)
        user = await users.get_user(user_id)
        if user is not None and user.get("deleted_at") is not None:
            raise UserDeletedError("User has been erased")
        if user is None and await users.has_user_erasure_marker(user_id):
            raise UserDeletedError("User has been erased")
        if user is None:
            await users.create_user(user_id)

    @staticmethod
    def _normalize_optional_message_id(message_id: str | None) -> str | None:
        if message_id is None:
            return None
        normalized = str(message_id).strip()
        return normalized or None

    @staticmethod
    def _normalize_optional_source_seq(source_seq: int | None) -> int | None:
        if source_seq is None:
            return None
        try:
            resolved = int(source_seq)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_seq must be a positive integer") from exc
        if resolved < 1:
            raise ValueError("source_seq must be a positive integer")
        return resolved

    @staticmethod
    def _resolve_ingest_control(
        *,
        ingest_origin: IngestOrigin | str | None,
        confirmation_strategy: ConfirmationStrategy | str | None,
    ) -> tuple[IngestOrigin, ConfirmationStrategy]:
        resolved_origin = IngestOrigin(ingest_origin or IngestOrigin.LIVE_TURN.value)
        resolved_strategy = resolve_confirmation_strategy(
            ingest_origin=resolved_origin,
            confirmation_strategy=confirmation_strategy,
        )
        if (
            resolved_origin is not IngestOrigin.LIVE_TURN
            and resolved_strategy is ConfirmationStrategy.LIVE_PROMPT_ALLOWED
        ):
            raise ValueError("live_prompt_allowed confirmation_strategy requires live_turn origin")
        return resolved_origin, resolved_strategy

    @staticmethod
    def _resolve_memory_privacy_mode(
        memory_privacy_mode: MemoryPrivacyMode | str | None,
        memory_preferences: dict[str, Any] | None,
    ) -> MemoryPrivacyMode:
        if memory_privacy_mode is not None:
            return resolve_memory_privacy_mode(memory_privacy_mode)
        preference_mode = None
        if memory_preferences is not None:
            preference_mode = memory_preferences.get("memory_privacy_mode")
        return resolve_memory_privacy_mode(preference_mode)

    @staticmethod
    def _message_metadata_with_ingest_control(
        metadata: dict[str, Any],
        *,
        ingest_origin: IngestOrigin,
        confirmation_strategy: ConfirmationStrategy,
        memory_privacy_mode: MemoryPrivacyMode,
    ) -> dict[str, Any]:
        normalized = dict(metadata)
        normalized["ingest_origin"] = ingest_origin.value
        normalized["confirmation_strategy"] = confirmation_strategy.value
        normalized["memory_privacy_mode"] = memory_privacy_mode.value
        return normalized

    @staticmethod
    async def _recent_messages_for_write(
        messages: MessageRepository,
        *,
        user_id: str,
        conversation_id: str,
        source_seq: int | None,
        existing_message: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        before_seq = source_seq
        if before_seq is None and existing_message is not None:
            before_seq = int(existing_message["seq"])
        if before_seq is not None:
            return await messages.get_recent_messages_before_seq(
                conversation_id,
                user_id,
                before_seq=before_seq,
                limit=RECENT_FETCH_LIMIT,
            )
        return await messages.get_recent_messages(
            conversation_id,
            user_id,
            limit=RECENT_FETCH_LIMIT,
        )

    @staticmethod
    async def _ensure_source_seq_available(
        messages: MessageRepository,
        *,
        user_id: str,
        conversation_id: str,
        source_seq: int | None,
        message_id: str | None,
    ) -> None:
        if source_seq is None:
            return
        existing = await messages.get_message_by_seq(
            conversation_id,
            user_id,
            source_seq,
        )
        if existing is not None and str(existing["id"]) != str(message_id):
            raise SourceSequenceConflictError(
                "source_seq already exists for a different message in this conversation"
            )

    @staticmethod
    async def _idempotent_message_if_present(
        messages: MessageRepository,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str | None,
        role: str,
        text: str,
        source_seq: int | None,
    ) -> dict[str, Any] | None:
        """Return an existing compatible host message or raise on conflict."""
        if message_id is None:
            return None
        existing = await messages.get_message_for_idempotency(message_id)
        if existing is None:
            return None
        if (
            str(existing.get("_conversation_user_id")) != user_id
            or str(existing.get("conversation_id")) != conversation_id
        ):
            raise MessageIdConflictError(
                "message_id already exists in a different Atagia namespace"
            )
        if str(existing.get("role")) != role or str(existing.get("text")) != text:
            raise MessageIdConflictError(
                "message_id already exists with different role or text"
            )
        if source_seq is not None and int(existing["seq"]) != source_seq:
            raise MessageIdConflictError(
                "message_id already exists with a different source_seq"
            )
        return existing

    async def ensure_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None,
        assistant_mode_id: str | None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        temporary: bool = False,
        temporary_ttl_seconds: int | None = None,
        purge_on_close: bool | None = None,
        cross_chat_memory: bool = True,
        # Namespace redesign identity / privacy fields. Optional during
        # the additive transition; Phase 11 removes the legacy ones.
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> dict[str, Any]:
        """Return an existing conversation or create one with the requested id."""
        conversations = ConversationRepository(connection, self.runtime.clock)
        workspaces = WorkspaceRepository(connection, self.runtime.clock)
        conversation = None
        if conversation_id is not None:
            conversation = await conversations.get_conversation(conversation_id, user_id)
        if conversation is not None:
            if workspace_id is not None and conversation["workspace_id"] != workspace_id:
                raise WorkspaceMismatchError(
                    "Requested workspace does not match the existing conversation workspace"
                )
            self._validate_optional_identity(
                conversation,
                workspace_id=workspace_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
            )
            if str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                raise ConversationNotActiveError("Conversation is not active")
            if not cross_chat_memory and not bool(conversation.get("isolated_mode")):
                updated = await conversations.mark_conversation_isolated(
                    str(conversation["id"]),
                    user_id,
                )
                if updated is not None:
                    await ContextCacheService(self.runtime).invalidate_conversation_cache_for_conversation(
                        updated
                    )
                    conversation = updated
            if incognito is True and not bool(conversation.get("incognito")):
                updated = await conversations.mark_conversation_isolated(
                    str(conversation["id"]),
                    user_id,
                )
                if updated is not None:
                    await ContextCacheService(self.runtime).invalidate_conversation_cache_for_conversation(
                        updated
                    )
                    conversation = updated
            return conversation

        if workspace_id is not None:
            workspace = await workspaces.get_workspace(workspace_id, user_id)
            if workspace is None:
                raise WorkspaceNotFoundError("Workspace not found for user")

        resolved_mode = mode or assistant_mode_id or DEFAULT_ASSISTANT_MODE_ID
        resolve_policy(self.runtime.manifests, resolved_mode, self.runtime.policy_resolver)
        resolved_ttl = (
            temporary_ttl_seconds
            if temporary_ttl_seconds is not None
            else (
                self.runtime.settings.temporary_default_ttl_seconds
                if temporary
                else None
            )
        )
        resolved_purge_on_close = (
            bool(purge_on_close)
            if purge_on_close is not None
            else (
                bool(self.runtime.settings.temporary_default_purge_on_close)
                if temporary
                else False
            )
        )
        # Resolve incognito with strictest-wins between the legacy
        # ``cross_chat_memory`` flag and the new ``incognito`` field.
        # ``incognito`` always wins when explicitly set; otherwise we
        # fall back to ``not cross_chat_memory`` to preserve the current
        # behavior.
        resolved_incognito = (
            bool(incognito) if incognito is not None else (not cross_chat_memory)
        )
        try:
            return await conversations.create_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                workspace_id=workspace_id,
                assistant_mode_id=resolved_mode,
                title=title,
                metadata=metadata or {},
                temporary=temporary,
                temporary_ttl_seconds=resolved_ttl,
                purge_on_close=resolved_purge_on_close,
                isolated_mode=resolved_incognito,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                mode=mode or resolved_mode,
                incognito=resolved_incognito,
            )
        except aiosqlite.IntegrityError as exc:
            raise ConversationNotFoundError("Conversation not found for user") from exc

    @staticmethod
    def _validate_optional_identity(
        conversation: dict[str, Any],
        *,
        workspace_id: str | None = None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
    ) -> None:
        checks = {
            "user_persona_id": user_persona_id,
            "platform_id": platform_id,
            "character_id": character_id if character_id is not None else workspace_id,
        }
        for field_name, expected in checks.items():
            actual = conversation.get(field_name)
            actual_text = None if actual is None else str(actual)
            if actual_text != expected:
                raise ConversationNotFoundError("Conversation not found for user")

    @staticmethod
    async def _mark_broad_conversation_rows_review_only(
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[str]:
        """Hide broad memories whose only recorded support is this conversation.

        Phase 8 will expand this visibility transaction to every derived table.
        For the Phase 4 service toggle, memory objects are the prompt-visible
        surface with the broadest blast radius.
        """

        cursor = await connection.execute(
            """
            SELECT DISTINCT mo.id
            FROM memory_objects AS mo
            LEFT JOIN json_each(
                json_extract(mo.payload_json, '$.source_message_ids')
            ) AS source_ids ON 1 = 1
            WHERE mo.user_id = ?
              AND mo.status = 'active'
              AND COALESCE(mo.scope_canonical, mo.scope) IN ('user', 'character', 'global_user', 'workspace')
              AND (
                  mo.conversation_id = ?
                  OR CAST(source_ids.value AS TEXT) IN (
                      SELECT id
                      FROM messages
                      WHERE conversation_id = ?
                  )
              )
            ORDER BY mo.id ASC
            """,
            (user_id, conversation_id, conversation_id),
        )
        memory_ids = [str(row["id"]) for row in await cursor.fetchall()]
        if not memory_ids:
            return []
        placeholders = ", ".join("?" for _ in memory_ids)
        await connection.execute(
            f"""
            UPDATE memory_objects
            SET status = 'review_required',
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
              AND id IN ({placeholders})
            """,
            (user_id, *memory_ids),
        )
        return memory_ids

    @staticmethod
    async def _delete_retrieval_events_for_memory_ids(
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        await connection.execute(
            f"""
            DELETE FROM retrieval_events
            WHERE user_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM json_each(retrieval_events.selected_memory_ids_json) AS selected_ids
                  WHERE selected_ids.value IN ({placeholders})
              )
            """,
            (user_id, *memory_ids),
        )

    async def _enqueue_message_jobs(
        self,
        *,
        conversation: dict[str, Any],
        message: dict[str, Any],
        prior_messages: list[dict[str, Any]],
        message_text: str,
        role: str,
        operational_profile: Any | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> MemoryProcessingStatus:
        connection = await self.runtime.open_connection()
        try:
            users = UserRepository(connection, self.runtime.clock)
            memory_preferences = await users.get_memory_preferences(str(conversation["user_id"]))
            jobs = build_message_jobs(
                clock=self.runtime.clock,
                conversation=conversation,
                message_id=str(message["id"]),
                prior_messages=prior_messages,
                message_text=message_text,
                occurred_at=resolve_message_occurred_at(message),
                role=role,
                operational_profile=operational_profile,
                memory_preferences=memory_preferences,
                ingest_origin=ingest_origin,
                confirmation_strategy=confirmation_strategy,
                memory_privacy_mode=memory_privacy_mode,
            )
            job_tracking = JobTrackingService(
                connection,
                self.runtime.clock,
                workers_enabled=self.runtime.settings.workers_enabled,
                settings=self.runtime.settings,
            )
            await enqueue_message_jobs(
                storage_backend=self.runtime.storage_backend,
                jobs=jobs,
                job_tracking_service=job_tracking,
                worker_control_service=WorkerControlService(
                    connection,
                    self.runtime.clock,
                ),
            )
            return await job_tracking.get_status(
                user_id=str(conversation["user_id"]),
                conversation_id=str(conversation["id"]),
            )
        finally:
            await connection.close()

    async def _message_processing_status(
        self,
        conversation: dict[str, Any],
    ) -> MemoryProcessingStatus:
        connection = await self.runtime.open_connection()
        try:
            return await JobTrackingService(
                connection,
                self.runtime.clock,
                workers_enabled=self.runtime.settings.workers_enabled,
            ).get_status(
                user_id=str(conversation["user_id"]),
                conversation_id=str(conversation["id"]),
            )
        finally:
            await connection.close()

    async def _topic_snapshot(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        return await TopicRepository(connection, self.runtime.clock).get_topic_snapshot(
            user_id=user_id,
            conversation_id=conversation_id,
            refresh_message_threshold=self.runtime.settings.topic_working_set_refresh_message_lag,
            stale_message_threshold=self.runtime.settings.topic_working_set_stale_message_lag,
            refresh_token_threshold=self.runtime.settings.topic_working_set_refresh_token_lag,
            stale_token_threshold=self.runtime.settings.topic_working_set_stale_token_lag,
        )
