"""Library-mode engine entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings, configured_resource_path
from atagia.core import json_utils
from atagia.core.repositories import MemoryObjectRepository, WorkspaceRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.models.schemas_api import (
    ChatResult,
    ContextResult,
    DeletionReport,
    ErasureReport,
    MemoryPreferencesResponse,
    MemoryProcessingStatus,
    PendingMemoryConfirmationActionResponse,
    PendingMemoryConfirmationListResponse,
    AdminReviewActionResponse,
    AdminReviewMemoryListResponse,
    WorkerControlResponse,
)
from atagia.models.schemas_api import (
    ActivitySnapshotResponse,
    ConversationActivityStats,
    VerbatimPinRecord,
    WarmupConversationResponse,
    WarmupRecommendedConversationsResponse,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryCategory,
    MemoryScope,
    MemoryStatus,
    ResponseMode,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)
from atagia.services.chat_service import ChatService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.conversation_activity_service import ConversationActivityService
from atagia.services.lifecycle_service import (
    HARD_DELETE_MEMORY_CONFIRMATION,
    ConversationLifecycleService,
)
from atagia.services.sidecar_service import SidecarService
from atagia.services.verbatim_pin_service import VerbatimPinService
from atagia.services.errors import RuntimeNotInitializedError
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.worker_control_service import WorkerControlService
from atagia.services.model_resolution import COMPONENTS_BY_ID


async def _worker_control_response(
    service: WorkerControlService,
    *,
    drain_completed: bool | None = None,
) -> WorkerControlResponse:
    state = await service.get_state()
    return WorkerControlResponse(
        mode=state.mode,
        reason=state.reason,
        updated_at=state.updated_at,
        updated_by=state.updated_by,
        new_source_jobs_allowed=await service.allows_new_source_jobs(),
        worker_claims_allowed=await service.allows_worker_claims(),
        periodic_work_allowed=await service.allows_periodic_work(),
        drain_completed=drain_completed,
    )


def _models_after_phase_overrides(
    ambient_models: dict[str, str],
    explicit_models: dict[str, str],
    overridden_categories: set[str],
) -> dict[str, str]:
    models: dict[str, str] = {}
    for component_id, model in ambient_models.items():
        component = COMPONENTS_BY_ID.get(component_id)
        if component is not None and component.category in overridden_categories:
            continue
        models[component_id] = model
    models.update(explicit_models)
    return models


class Atagia:
    """Library interface for retrieval and chat flows."""

    def __init__(
        self,
        db_path: str | Path = "atagia.db",
        redis_url: str | None = None,
        manifests_dir: str | Path | None = None,
        operational_profiles_dir: str | Path | None = None,
        llm_forced_global_model: str | None = None,
        llm_ingest_model: str | None = None,
        llm_retrieval_model: str | None = None,
        llm_chat_model: str | None = None,
        llm_component_models: dict[str, str] | None = None,
        llm_intimacy_ingest_model: str | None = None,
        llm_intimacy_retrieval_model: str | None = None,
        llm_intimacy_component_models: dict[str, str] | None = None,
        llm_intimacy_proactive_routing_enabled: bool | None = None,
        llm_structured_output_retry_attempts: int | None = None,
        llm_structured_output_rescue_enabled: bool | None = None,
        llm_structured_output_rescue_model: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        google_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        embedding_backend: str | None = None,
        embedding_model: str | None = None,
        skip_belief_revision: bool = False,
        skip_compaction: bool = False,
        context_cache_enabled: bool | None = None,
        disable_chunking_extraction: bool | None = None,
        assistant_guidance_enabled: bool | None = None,
        recent_transcript_budget_tokens: int | None = None,
        context_envelope_budget_tokens: int | None = None,
        context_envelope_ratios: dict[str, float] | None = None,
        answer_stance: str | None = None,
        answer_stance_prompt_variant: str | None = None,
        answer_postcondition_guard_enabled: bool | None = None,
    ) -> None:
        self._db_path = (
            str(Path(db_path).expanduser())
            if isinstance(db_path, Path)
            else str(db_path)
        )
        self._redis_url = redis_url
        self._manifests_dir = (
            str(Path(manifests_dir).expanduser())
            if isinstance(manifests_dir, Path)
            else manifests_dir
        )
        self._operational_profiles_dir = (
            str(Path(operational_profiles_dir).expanduser())
            if isinstance(operational_profiles_dir, Path)
            else operational_profiles_dir
        )
        self._llm_forced_global_model = llm_forced_global_model
        self._llm_ingest_model = llm_ingest_model
        self._llm_retrieval_model = llm_retrieval_model
        self._llm_chat_model = llm_chat_model
        self._llm_component_models = dict(llm_component_models or {})
        self._llm_intimacy_ingest_model = llm_intimacy_ingest_model
        self._llm_intimacy_retrieval_model = llm_intimacy_retrieval_model
        self._llm_intimacy_component_models = dict(llm_intimacy_component_models or {})
        self._llm_intimacy_proactive_routing_enabled = (
            llm_intimacy_proactive_routing_enabled
        )
        self._llm_structured_output_retry_attempts = (
            llm_structured_output_retry_attempts
        )
        self._llm_structured_output_rescue_enabled = (
            llm_structured_output_rescue_enabled
        )
        self._llm_structured_output_rescue_model = llm_structured_output_rescue_model
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._google_api_key = google_api_key
        self._openrouter_api_key = openrouter_api_key
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._skip_belief_revision = skip_belief_revision
        self._skip_compaction = skip_compaction
        self._context_cache_enabled = context_cache_enabled
        self._disable_chunking_extraction = disable_chunking_extraction
        self._assistant_guidance_enabled = assistant_guidance_enabled
        self._recent_transcript_budget_tokens = recent_transcript_budget_tokens
        self._context_envelope_budget_tokens = context_envelope_budget_tokens
        self._context_envelope_ratios = (
            dict(context_envelope_ratios)
            if context_envelope_ratios is not None
            else None
        )
        self._answer_stance = answer_stance
        self._answer_stance_prompt_variant = answer_stance_prompt_variant
        self._answer_postcondition_guard_enabled = answer_postcondition_guard_enabled
        self._runtime: AppRuntime | None = None
        self._closed = False

    @property
    def runtime(self) -> AppRuntime | None:
        """Expose the initialized runtime for advanced callers and tests."""
        return self._runtime

    async def setup(self) -> Atagia:
        """Initialize the runtime if it has not been started yet."""
        if self._runtime is None:
            self._runtime = await initialize_runtime(self._build_settings())
        self._closed = False
        return self

    async def __aenter__(self) -> Atagia:
        await self.setup()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
        response_mode: ResponseMode | str | None = None,
        adaptive_retrieval: bool | None = None,
    ) -> ContextResult:
        """Run retrieval, persist the user message, and return a ready system prompt."""
        runtime = await self._require_runtime()
        return await SidecarService(runtime).get_context(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            ablation=ablation,
            attachments=attachments,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
            incognito=incognito,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
            privacy_enforcement=privacy_enforcement,
            authenticated_user_privilege_level=authenticated_user_privilege_level,
            authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
            response_mode=response_mode,
            adaptive_retrieval=adaptive_retrieval,
        )

    async def flush(
        self,
        timeout_seconds: float = 30.0,
        *,
        idle_timeout_seconds: float | None = None,
        progress_interval_seconds: float = 0.0,
        progress_callback: Any | None = None,
    ) -> bool:
        """Wait for pending background work to finish when workers are enabled."""
        runtime = await self._require_runtime()
        if not runtime.settings.workers_enabled:
            return False
        return await runtime.storage_backend.drain(
            timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_interval_seconds=progress_interval_seconds,
            progress_callback=progress_callback,
        )

    async def get_worker_control(self) -> WorkerControlResponse:
        """Return the current background-processing stop-switch state."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = WorkerControlService(connection, runtime.clock)
            return await _worker_control_response(service)
        finally:
            await connection.close()

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> WorkerControlResponse:
        """Set the background-processing stop-switch state."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = WorkerControlService(connection, runtime.clock)
            resolved_mode = WorkerControlMode(mode)
            await service.set_mode(
                resolved_mode,
                reason=reason,
                updated_by="library_admin",
            )
            drain_completed: bool | None = None
            if resolved_mode is WorkerControlMode.DRAIN_AND_PAUSE:
                drain_completed = (
                    await runtime.storage_backend.drain(timeout_seconds)
                    if runtime.settings.workers_enabled
                    else False
                )
            return await _worker_control_response(
                service,
                drain_completed=drain_completed,
            )
        finally:
            await connection.close()

    async def get_processing_status(
        self,
        user_id: str,
        conversation_id: str | None = None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> MemoryProcessingStatus:
        """Return current background memory-processing status."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            return await JobTrackingService(
                connection,
                runtime.clock,
                workers_enabled=runtime.settings.workers_enabled,
            ).get_status(
                user_id=user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id or "default",
                character_id=character_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
            )
        finally:
            await connection.close()

    async def list_pending_memory_confirmations(
        self,
        user_id: str,
        **filters: Any,
    ) -> PendingMemoryConfirmationListResponse:
        """Return safe pending-confirmation records for host user interfaces."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            items = await PendingConfirmationService(
                connection,
                runtime.clock,
            ).list_pending_confirmations(
                user_id=user_id,
                conversation_id=filters.get("conversation_id"),
                platform_id=filters.get("platform_id"),
                user_persona_id=filters.get("user_persona_id"),
                character_id=filters.get("character_id"),
                category=(
                    MemoryCategory(filters["category"])
                    if filters.get("category") is not None
                    else None
                ),
                limit=int(filters.get("limit", 100)),
                offset=int(filters.get("offset", 0)),
            )
            return PendingMemoryConfirmationListResponse.model_validate(
                {"items": items}
            )
        finally:
            await connection.close()

    async def confirm_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        """Confirm one pending memory using Atagia's consent transition."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            memory = await PendingConfirmationService(
                connection,
                runtime.clock,
                embedding_index=runtime.embedding_index,
            ).confirm_pending_memory(user_id=user_id, memory_id=memory_id)
            return PendingMemoryConfirmationActionResponse(
                memory_id=str(memory["id"]),
                status=str(memory["status"]),
            )
        finally:
            await connection.close()

    async def decline_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        """Decline one pending memory using Atagia's consent transition."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            memory = await PendingConfirmationService(
                connection,
                runtime.clock,
            ).decline_pending_memory(user_id=user_id, memory_id=memory_id)
            return PendingMemoryConfirmationActionResponse(
                memory_id=str(memory["id"]),
                status=str(memory["status"]),
            )
        finally:
            await connection.close()

    async def list_review_required_memories(
        self,
        **filters: Any,
    ) -> AdminReviewMemoryListResponse:
        """Return admin-visible review-required memories."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            items = await self._list_review_required_rows(
                connection,
                user_id=filters.get("user_id"),
                platform_id=filters.get("platform_id"),
                user_persona_id=filters.get("user_persona_id"),
                character_id=filters.get("character_id"),
                category=filters.get("category"),
                ingest_origin=filters.get("ingest_origin"),
                limit=int(filters.get("limit", 100)),
                offset=int(filters.get("offset", 0)),
            )
            return AdminReviewMemoryListResponse.model_validate({"items": items})
        finally:
            await connection.close()

    async def archive_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        """Archive one review-required memory."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            memory = await MemoryObjectRepository(
                connection,
                runtime.clock,
            ).get_memory_object(memory_id, user_id)
            if (
                memory is None
                or memory.get("status") != MemoryStatus.REVIEW_REQUIRED.value
            ):
                raise ValueError("Review-required memory not found")
            await ConversationLifecycleService(runtime).delete_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
            )
            return AdminReviewActionResponse(
                memory_id=memory_id,
                status=MemoryStatus.ARCHIVED.value,
            )
        finally:
            await connection.close()

    async def delete_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        """Hard-delete one review-required memory."""

        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            memory = await MemoryObjectRepository(
                connection,
                runtime.clock,
            ).get_memory_object(memory_id, user_id)
            if (
                memory is None
                or memory.get("status") != MemoryStatus.REVIEW_REQUIRED.value
            ):
                raise ValueError("Review-required memory not found")
            await ConversationLifecycleService(runtime).delete_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                hard=True,
                confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
            )
            return AdminReviewActionResponse(
                memory_id=memory_id,
                status=MemoryStatus.DELETED.value,
            )
        finally:
            await connection.close()

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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
    ) -> None:
        """Store a message and enqueue extraction without running retrieval."""
        runtime = await self._require_runtime()
        await SidecarService(runtime).ingest_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            text=text,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            attachments=attachments,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
            incognito=incognito,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
            privacy_enforcement=privacy_enforcement,
            authenticated_user_privilege_level=authenticated_user_privilege_level,
            authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
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
        operational_signals: dict[str, Any] | None = None,
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
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
    ) -> None:
        """Persist an assistant response in the conversation history."""
        runtime = await self._require_runtime()
        await SidecarService(runtime).add_response(
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            occurred_at=occurred_at,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
            privacy_enforcement=privacy_enforcement,
            authenticated_user_privilege_level=authenticated_user_privilege_level,
            authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
        )

    async def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        ablation: AblationConfig | None = None,
        debug: bool = False,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
        incognito: bool | None = None,
        privacy_enforcement: str = "enforce",
        authenticated_user_privilege_level: str | None = None,
        authenticated_user_is_atagia_master: bool = False,
        response_mode: ResponseMode | str | None = None,
        adaptive_retrieval: bool | None = None,
    ) -> ChatResult:
        """Run the full chat flow, including the LLM response generation."""
        runtime = await self._require_runtime()
        sidecar = SidecarService(runtime)
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            await sidecar.ensure_user_exists(connection, user_id)
            await sidecar.ensure_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=mode,
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
        finally:
            await connection.close()
        return await ChatService(runtime).chat_reply(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message,
            assistant_mode_id=mode,
            ablation=ablation,
            attachments=attachments,
            message_occurred_at=occurred_at,
            debug=debug,
            debug_include_sensitive=True,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id if character_id is not None else workspace_id,
            active_presence_id=active_presence_id,
            mind_id=mind_id,
            mind_topology=mind_topology,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            mode=mode,
            incognito=incognito,
            privacy_enforcement=privacy_enforcement,
            authenticated_user_privilege_level=authenticated_user_privilege_level,
            authenticated_user_is_atagia_master=authenticated_user_is_atagia_master,
            response_mode=response_mode,
            adaptive_retrieval=adaptive_retrieval,
        )

    async def get_memory_preferences(self, user_id: str) -> MemoryPreferencesResponse:
        """Return user-level memory sharing preferences."""
        runtime = await self._require_runtime()
        preferences = await SidecarService(runtime).get_memory_preferences(user_id)
        return MemoryPreferencesResponse.model_validate(preferences)

    async def set_memory_preferences(
        self,
        user_id: str,
        *,
        remember_across_chats: bool | None = None,
        remember_across_devices: bool | None = None,
        memory_privacy_mode: str | None = None,
    ) -> MemoryPreferencesResponse:
        """Update user-level memory sharing preferences."""
        runtime = await self._require_runtime()
        preferences = await SidecarService(runtime).set_memory_preferences(
            user_id,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            memory_privacy_mode=memory_privacy_mode,
        )
        return MemoryPreferencesResponse.model_validate(preferences)

    async def set_conversation_incognito(
        self,
        user_id: str,
        conversation_id: str,
        incognito: bool,
    ) -> dict[str, Any]:
        """Set the reversible per-conversation incognito flag."""
        runtime = await self._require_runtime()
        return await SidecarService(runtime).set_conversation_incognito(
            user_id,
            conversation_id,
            incognito,
        )

    async def create_user(self, user_id: str) -> None:
        """Create the user if it does not already exist."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await SidecarService(runtime).ensure_user_exists(connection, user_id)
        finally:
            await connection.close()

    async def create_workspace(
        self, user_id: str, workspace_id: str, name: str
    ) -> None:
        """Create the workspace if it does not already exist."""
        runtime = await self._require_runtime()
        sidecar = SidecarService(runtime)
        connection = await runtime.open_connection()
        try:
            await sidecar.ensure_user_exists(connection, user_id)
            workspaces = WorkspaceRepository(connection, runtime.clock)
            if await workspaces.get_workspace(workspace_id, user_id) is None:
                await workspaces.create_workspace(workspace_id, user_id, name)
        finally:
            await connection.close()

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        *,
        temporary: bool = False,
        temporary_ttl_seconds: int | None = None,
        purge_on_close: bool | None = None,
        cross_chat_memory: bool = True,
        # Public namespace identity fields. Legacy workspace/assistant-mode
        # aliases remain accepted for compatibility with older fixtures.
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
    ) -> str:
        """Create a conversation and return its identifier."""
        runtime = await self._require_runtime()
        sidecar = SidecarService(runtime)
        connection = await runtime.open_connection()
        try:
            await sidecar.ensure_user_exists(connection, user_id)
            conversation = await sidecar.ensure_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                temporary=temporary,
                temporary_ttl_seconds=temporary_ttl_seconds,
                purge_on_close=purge_on_close,
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
            return str(conversation["id"])
        finally:
            await connection.close()

    async def close_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        purge: bool | None = None,
        confirmation: str | None = None,
    ) -> DeletionReport | dict[str, Any]:
        """Close a conversation, optionally purging it when confirmed."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            return await ConversationLifecycleService(runtime).close_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                purge=purge,
                confirmation=confirmation,
            )
        finally:
            await connection.close()

    async def archive_conversation(
        self, user_id: str, conversation_id: str
    ) -> dict[str, Any]:
        """Archive a conversation and hide its derived data from default retrieval."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            return await ConversationLifecycleService(runtime).archive_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        finally:
            await connection.close()

    async def delete_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        confirmation: str,
    ) -> DeletionReport:
        """Hard-delete a conversation cascade after explicit confirmation."""
        runtime = await self._require_runtime()
        cache_service = ContextCacheService(runtime)
        async with cache_service.user_cache_guard(user_id):
            connection = await runtime.open_connection()
            try:
                return await ConversationLifecycleService(runtime).delete_conversation(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    confirmation=confirmation,
                )
            finally:
                await connection.close()

    async def edit_memory(
        self,
        user_id: str,
        memory_id: str,
        new_text: str,
        *,
        edit_source: str = "api",
        edited_by: str = "system",
    ) -> dict[str, Any]:
        """Edit an active evidence memory and preserve the previous text."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            return await ConversationLifecycleService(runtime).edit_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                new_text=new_text,
                edit_source=edit_source,
                edited_by=edited_by,
            )
        finally:
            await connection.close()

    async def delete_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        hard: bool = False,
        confirmation: str | None = None,
    ) -> DeletionReport:
        """Archive or hard-delete a memory object."""
        runtime = await self._require_runtime()
        cache_service = ContextCacheService(runtime)
        async with cache_service.user_cache_guard(user_id):
            connection = await runtime.open_connection()
            try:
                return await ConversationLifecycleService(runtime).delete_memory(
                    connection,
                    user_id=user_id,
                    memory_id=memory_id,
                    hard=hard,
                    confirmation=confirmation,
                )
            finally:
                await connection.close()

    async def erase_user_data(
        self, user_id: str, *, confirmation: str
    ) -> ErasureReport:
        """Erase all user data after explicit right-to-erasure confirmation."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            return await ConversationLifecycleService(runtime).erase_user_data(
                connection,
                user_id=user_id,
                confirmation=confirmation,
            )
        finally:
            await connection.close()

    async def create_verbatim_pin(
        self,
        user_id: str,
        *,
        scope: MemoryScope,
        target_kind: VerbatimPinTargetKind,
        target_id: str,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        reason: str | None = None,
        created_by: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> VerbatimPinRecord:
        """Create a verbatim pin and return the canonical record."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await SidecarService(runtime).ensure_user_exists(connection, user_id)
            created = await VerbatimPinService(runtime).create_verbatim_pin(
                connection,
                user_id=user_id,
                scope=scope,
                target_kind=target_kind,
                target_id=target_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
                canonical_text=canonical_text,
                index_text=index_text,
                target_span_start=target_span_start,
                target_span_end=target_span_end,
                privacy_level=privacy_level,
                intimacy_boundary=intimacy_boundary,
                intimacy_boundary_confidence=intimacy_boundary_confidence,
                reason=reason,
                created_by=created_by,
                expires_at=expires_at,
                payload_json=payload_json,
            )
            return VerbatimPinRecord.model_validate(created)
        finally:
            await connection.close()

    async def get_verbatim_pin(
        self,
        user_id: str,
        pin_id: str,
    ) -> VerbatimPinRecord | None:
        """Return one verbatim pin by id, if it belongs to the user."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            row = await VerbatimPinService(runtime).get_verbatim_pin(
                connection,
                user_id=user_id,
                pin_id=pin_id,
            )
            return None if row is None else VerbatimPinRecord.model_validate(row)
        finally:
            await connection.close()

    async def list_verbatim_pins(
        self,
        user_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        scope_filter: list[MemoryScope] | None = None,
        target_kind_filter: list[VerbatimPinTargetKind] | None = None,
        status_filter: list[VerbatimPinStatus] | None = None,
        target_id: str | None = None,
        include_deleted: bool = False,
        active_only: bool = False,
        as_of: str | None = None,
    ) -> list[VerbatimPinRecord]:
        """Return verbatim pins owned by the user."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            rows = await VerbatimPinService(runtime).list_verbatim_pins(
                connection,
                user_id=user_id,
                limit=limit,
                offset=offset,
                scope_filter=scope_filter,
                target_kind_filter=target_kind_filter,
                status_filter=status_filter,
                target_id=target_id,
                include_deleted=include_deleted,
                active_only=active_only,
                as_of=as_of,
            )
            return [VerbatimPinRecord.model_validate(row) for row in rows]
        finally:
            await connection.close()

    async def update_verbatim_pin(
        self,
        user_id: str,
        pin_id: str,
        *,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int | None = None,
        intimacy_boundary: IntimacyBoundary | None = None,
        intimacy_boundary_confidence: float | None = None,
        status: VerbatimPinStatus | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> VerbatimPinRecord | None:
        """Update a verbatim pin lifecycle or content field."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            updated = await VerbatimPinService(runtime).update_verbatim_pin(
                connection,
                user_id=user_id,
                pin_id=pin_id,
                canonical_text=canonical_text,
                index_text=index_text,
                target_span_start=target_span_start,
                target_span_end=target_span_end,
                privacy_level=privacy_level,
                intimacy_boundary=intimacy_boundary,
                intimacy_boundary_confidence=intimacy_boundary_confidence,
                status=status,
                reason=reason,
                expires_at=expires_at,
                payload_json=payload_json,
            )
            return (
                None if updated is None else VerbatimPinRecord.model_validate(updated)
            )
        finally:
            await connection.close()

    async def delete_verbatim_pin(
        self,
        user_id: str,
        pin_id: str,
    ) -> VerbatimPinRecord | None:
        """Delete a verbatim pin while preserving its audit trail."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            deleted = await VerbatimPinService(runtime).delete_verbatim_pin(
                connection,
                user_id=user_id,
                pin_id=pin_id,
            )
            return (
                None if deleted is None else VerbatimPinRecord.model_validate(deleted)
            )
        finally:
            await connection.close()

    async def get_activity_snapshot(
        self,
        user_id: str,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        as_of: str | None = None,
        refresh: bool = True,
    ) -> ActivitySnapshotResponse:
        """Return a derived activity snapshot for one user or conversation scope."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = ConversationActivityService(runtime)
            snapshot = await service.get_activity_snapshot(
                connection,
                user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                as_of=as_of,
                refresh=refresh,
            )
            return ActivitySnapshotResponse.model_validate(
                {
                    **snapshot,
                    "conversations": [
                        ConversationActivityStats.model_validate(row)
                        for row in snapshot.get("conversations", [])
                    ],
                }
            )
        finally:
            await connection.close()

    async def list_hot_conversations(
        self,
        user_id: str,
        limit: int = 5,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        as_of: str | None = None,
        refresh: bool = True,
    ) -> list[ConversationActivityStats]:
        """Return the hottest active conversations for a user."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = ConversationActivityService(runtime)
            rows = await service.list_hot_conversations(
                connection,
                user_id,
                limit=limit,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                as_of=as_of,
                refresh=refresh,
            )
            return [ConversationActivityStats.model_validate(row) for row in rows]
        finally:
            await connection.close()

    async def warmup_conversation(
        self,
        user_id: str,
        conversation_id: str,
        max_messages: int = 12,
        as_of: str | None = None,
    ) -> WarmupConversationResponse:
        """Warm a single conversation without generating a reply."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = ConversationActivityService(runtime)
            result = await service.warmup_conversation(
                connection,
                user_id,
                conversation_id,
                max_messages=max_messages,
                as_of=as_of,
                refresh_stats=True,
            )
            return WarmupConversationResponse.model_validate(result)
        finally:
            await connection.close()

    async def warmup_recommended_conversations(
        self,
        user_id: str,
        limit: int = 3,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        as_of: str | None = None,
        lead_time_minutes: int | None = None,
        total_message_budget: int = 24,
        per_conversation_message_budget: int = 12,
    ) -> WarmupRecommendedConversationsResponse:
        """Warm the most likely-to-be-used conversations for a user."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            service = ConversationActivityService(runtime)
            result = await service.warmup_recommended_conversations(
                connection,
                user_id,
                limit=limit,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                as_of=as_of,
                lead_time_minutes=lead_time_minutes,
                total_message_budget=total_message_budget,
                per_conversation_message_budget=per_conversation_message_budget,
            )
            return WarmupRecommendedConversationsResponse.model_validate(
                {
                    **result,
                    "hot_conversations": [
                        ConversationActivityStats.model_validate(row)
                        for row in result.get("hot_conversations", [])
                    ],
                    "warmed_conversations": [
                        WarmupConversationResponse.model_validate(row)
                        for row in result.get("warmed_conversations", [])
                    ],
                }
            )
        finally:
            await connection.close()

    async def close(self) -> None:
        """Stop workers and close runtime resources."""
        self._closed = True
        if self._runtime is None:
            return
        await self._runtime.close()
        self._runtime = None

    def _build_settings(self) -> Settings:
        env_settings = Settings.from_env()
        manifests_path = self._manifests_dir or configured_resource_path(
            "manifests",
            os.getenv("ATAGIA_MANIFESTS_PATH"),
        )
        operational_profiles_path = (
            self._operational_profiles_dir
            or configured_resource_path(
                "operational_profiles",
                os.getenv("ATAGIA_OPERATIONAL_PROFILES_PATH"),
            )
        )
        migrations_path = configured_resource_path(
            "migrations",
            os.getenv("ATAGIA_MIGRATIONS_PATH"),
        )
        use_env_redis = (
            self._redis_url is None and env_settings.storage_backend == "redis"
        )
        storage_backend = (
            "redis" if self._redis_url is not None or use_env_redis else "inprocess"
        )
        anthropic_api_key = self._anthropic_api_key or env_settings.anthropic_api_key
        openai_api_key = self._openai_api_key or env_settings.openai_api_key
        google_api_key = self._google_api_key or env_settings.google_api_key
        openrouter_api_key = self._openrouter_api_key or env_settings.openrouter_api_key
        forced_global_model = (
            self._llm_forced_global_model or env_settings.llm_forced_global_model
        )
        overridden_categories = {
            category
            for category, model in (
                ("ingest", self._llm_ingest_model),
                ("retrieval", self._llm_retrieval_model),
                ("chat", self._llm_chat_model),
            )
            if model is not None
        }
        component_models = _models_after_phase_overrides(
            env_settings.llm_component_models,
            self._llm_component_models,
            overridden_categories,
        )
        overridden_intimacy_categories = {
            category
            for category, model in (
                ("ingest", self._llm_intimacy_ingest_model),
                ("retrieval", self._llm_intimacy_retrieval_model),
            )
            if model is not None
        }
        intimacy_component_models = _models_after_phase_overrides(
            env_settings.llm_intimacy_component_models,
            self._llm_intimacy_component_models,
            overridden_intimacy_categories,
        )
        return Settings(
            sqlite_path=self._db_path,
            migrations_path=migrations_path,
            manifests_path=manifests_path,
            operational_profiles_path=operational_profiles_path,
            storage_backend=storage_backend,
            redis_url=self._redis_url or env_settings.redis_url,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            kimi_api_key=env_settings.kimi_api_key,
            minimax_api_key=env_settings.minimax_api_key,
            openrouter_api_key=openrouter_api_key,
            anthropic_base_url=env_settings.anthropic_base_url,
            openai_base_url=env_settings.openai_base_url,
            openai_embedding_base_url=env_settings.openai_embedding_base_url,
            kimi_base_url=env_settings.kimi_base_url,
            minimax_base_url=env_settings.minimax_base_url,
            openrouter_base_url=env_settings.openrouter_base_url,
            openrouter_site_url=env_settings.openrouter_site_url,
            openrouter_app_name=env_settings.openrouter_app_name,
            llm_chat_model=self._llm_chat_model or env_settings.llm_chat_model,
            llm_forced_global_model=forced_global_model,
            llm_ingest_model=self._llm_ingest_model or env_settings.llm_ingest_model,
            llm_retrieval_model=self._llm_retrieval_model
            or env_settings.llm_retrieval_model,
            llm_component_models=component_models,
            llm_intimacy_ingest_model=(
                self._llm_intimacy_ingest_model
                or env_settings.llm_intimacy_ingest_model
            ),
            llm_intimacy_retrieval_model=(
                self._llm_intimacy_retrieval_model
                or env_settings.llm_intimacy_retrieval_model
            ),
            llm_intimacy_component_models=intimacy_component_models,
            llm_intimacy_proactive_routing_enabled=(
                env_settings.llm_intimacy_proactive_routing_enabled
                if self._llm_intimacy_proactive_routing_enabled is None
                else self._llm_intimacy_proactive_routing_enabled
            ),
            llm_structured_output_retry_attempts=(
                env_settings.llm_structured_output_retry_attempts
                if self._llm_structured_output_retry_attempts is None
                else self._llm_structured_output_retry_attempts
            ),
            llm_structured_output_rescue_enabled=(
                env_settings.llm_structured_output_rescue_enabled
                if self._llm_structured_output_rescue_enabled is None
                else self._llm_structured_output_rescue_enabled
            ),
            llm_structured_output_rescue_model=(
                self._llm_structured_output_rescue_model
                or env_settings.llm_structured_output_rescue_model
            ),
            llm_debug_io_enabled=env_settings.llm_debug_io_enabled,
            llm_debug_io_dir=env_settings.llm_debug_io_dir,
            llm_debug_io_purposes=env_settings.llm_debug_io_purposes,
            llm_debug_io_raw=env_settings.llm_debug_io_raw,
            llm_debug_io_max_chars=env_settings.llm_debug_io_max_chars,
            answer_postcondition_guard_enabled=(
                env_settings.answer_postcondition_guard_enabled
                if self._answer_postcondition_guard_enabled is None
                else self._answer_postcondition_guard_enabled
            ),
            answer_postcondition_retry_max_output_tokens=(
                env_settings.answer_postcondition_retry_max_output_tokens
            ),
            answer_stance=(
                env_settings.answer_stance
                if self._answer_stance is None
                else self._answer_stance
            ),
            answer_stance_prompt_variant=(
                env_settings.answer_stance_prompt_variant
                if self._answer_stance_prompt_variant is None
                else self._answer_stance_prompt_variant
            ),
            service_mode=False,
            service_api_key=None,
            admin_api_key=None,
            workers_enabled=True,
            debug=env_settings.debug,
            worker_circuit_breaker_enabled=env_settings.worker_circuit_breaker_enabled,
            worker_circuit_breaker_failure_threshold=(
                env_settings.worker_circuit_breaker_failure_threshold
            ),
            worker_circuit_breaker_window_seconds=(
                env_settings.worker_circuit_breaker_window_seconds
            ),
            worker_circuit_breaker_min_failure_ratio=(
                env_settings.worker_circuit_breaker_min_failure_ratio
            ),
            llm_run_guard_enabled=env_settings.llm_run_guard_enabled,
            llm_run_guard_mode=env_settings.llm_run_guard_mode,
            llm_run_guard_max_total_calls=env_settings.llm_run_guard_max_total_calls,
            llm_run_guard_max_total_failed_calls=(
                env_settings.llm_run_guard_max_total_failed_calls
            ),
            llm_run_guard_max_failed_call_ratio=(
                env_settings.llm_run_guard_max_failed_call_ratio
            ),
            llm_run_guard_failed_ratio_min_calls=(
                env_settings.llm_run_guard_failed_ratio_min_calls
            ),
            llm_run_guard_max_failed_calls_per_purpose=(
                env_settings.llm_run_guard_max_failed_calls_per_purpose
            ),
            llm_run_guard_max_failed_ratio_per_purpose=(
                env_settings.llm_run_guard_max_failed_ratio_per_purpose
            ),
            llm_run_guard_purpose_failure_ratio_min_calls=(
                env_settings.llm_run_guard_purpose_failure_ratio_min_calls
            ),
            llm_run_guard_max_consecutive_failures_per_purpose=(
                env_settings.llm_run_guard_max_consecutive_failures_per_purpose
            ),
            llm_run_guard_max_total_tokens=env_settings.llm_run_guard_max_total_tokens,
            llm_run_guard_max_reported_cost_usd=(
                env_settings.llm_run_guard_max_reported_cost_usd
            ),
            bulk_ingest_llm_run_guard_enabled=(
                env_settings.bulk_ingest_llm_run_guard_enabled
            ),
            bulk_ingest_llm_run_guard_max_total_calls=(
                env_settings.bulk_ingest_llm_run_guard_max_total_calls
            ),
            bulk_ingest_llm_run_guard_max_total_failed_calls=(
                env_settings.bulk_ingest_llm_run_guard_max_total_failed_calls
            ),
            bulk_ingest_llm_run_guard_max_failed_call_ratio=(
                env_settings.bulk_ingest_llm_run_guard_max_failed_call_ratio
            ),
            bulk_ingest_llm_run_guard_failed_ratio_min_calls=(
                env_settings.bulk_ingest_llm_run_guard_failed_ratio_min_calls
            ),
            bulk_ingest_llm_run_guard_max_failed_calls_per_purpose=(
                env_settings.bulk_ingest_llm_run_guard_max_failed_calls_per_purpose
            ),
            bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose=(
                env_settings.bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose
            ),
            bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls=(
                env_settings.bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls
            ),
            bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose=(
                env_settings.bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose
            ),
            bulk_ingest_llm_run_guard_max_total_tokens=(
                env_settings.bulk_ingest_llm_run_guard_max_total_tokens
            ),
            bulk_ingest_llm_run_guard_max_reported_cost_usd=(
                env_settings.bulk_ingest_llm_run_guard_max_reported_cost_usd
            ),
            bulk_ingest_llm_run_guard_max_wall_time_seconds=(
                env_settings.bulk_ingest_llm_run_guard_max_wall_time_seconds
            ),
            allow_insecure_http=True,
            embedding_backend=self._embedding_backend or env_settings.embedding_backend,
            embedding_model=self._embedding_model or env_settings.embedding_model,
            embedding_dimension=env_settings.embedding_dimension,
            embedding_vector_limit_cap=env_settings.embedding_vector_limit_cap,
            embedding_search_overfetch_multiplier=(
                env_settings.embedding_search_overfetch_multiplier
            ),
            memory_fts_canonical_bm25_weight=env_settings.memory_fts_canonical_bm25_weight,
            memory_fts_index_bm25_weight=env_settings.memory_fts_index_bm25_weight,
            lifecycle_decay_days=env_settings.lifecycle_decay_days,
            lifecycle_decay_rate=env_settings.lifecycle_decay_rate,
            lifecycle_archive_vitality=env_settings.lifecycle_archive_vitality,
            lifecycle_archive_confidence=env_settings.lifecycle_archive_confidence,
            ephemeral_scoring_hours=env_settings.ephemeral_scoring_hours,
            lifecycle_ephemeral_ttl_hours=env_settings.lifecycle_ephemeral_ttl_hours,
            lifecycle_review_ttl_days=env_settings.lifecycle_review_ttl_days,
            lifecycle_lazy_enabled=env_settings.lifecycle_lazy_enabled,
            lifecycle_min_interval_seconds=env_settings.lifecycle_min_interval_seconds,
            lifecycle_busy_timeout_ms=env_settings.lifecycle_busy_timeout_ms,
            lifecycle_busy_backoff_seconds=env_settings.lifecycle_busy_backoff_seconds,
            lifecycle_failure_backoff_seconds=env_settings.lifecycle_failure_backoff_seconds,
            lifecycle_worker_enabled=env_settings.lifecycle_worker_enabled,
            lifecycle_worker_interval_seconds=env_settings.lifecycle_worker_interval_seconds,
            retrieval_packets_dry_run_enabled=(
                env_settings.retrieval_packets_dry_run_enabled
            ),
            retrieval_packets_write_enabled=env_settings.retrieval_packets_write_enabled,
            fact_facet_surfaces_enabled=env_settings.fact_facet_surfaces_enabled,
            fact_facet_retrieval_enabled=env_settings.fact_facet_retrieval_enabled,
            fact_facet_structured_only=env_settings.fact_facet_structured_only,
            fact_facet_span_coadmission_enabled=(
                env_settings.fact_facet_span_coadmission_enabled
            ),
            fact_facet_retrieval_limit=env_settings.fact_facet_retrieval_limit,
            fact_facet_retrieval_rrf_weight=(
                env_settings.fact_facet_retrieval_rrf_weight
            ),
            applicability_gate_mode=env_settings.applicability_gate_mode,
            response_mode=env_settings.response_mode,
            adaptive_retrieval=env_settings.adaptive_retrieval,
            promotion_conv_to_ws_min_conversations=env_settings.promotion_conv_to_ws_min_conversations,
            promotion_ws_to_global_min_sessions=env_settings.promotion_ws_to_global_min_sessions,
            promotion_require_mode_consistency=env_settings.promotion_require_mode_consistency,
            skip_belief_revision=self._skip_belief_revision,
            skip_compaction=self._skip_compaction,
            context_cache_enabled=(
                env_settings.context_cache_enabled
                if self._context_cache_enabled is None
                else self._context_cache_enabled
            ),
            context_cache_min_ttl_seconds=env_settings.context_cache_min_ttl_seconds,
            context_cache_max_ttl_seconds=env_settings.context_cache_max_ttl_seconds,
            temporary_default_ttl_seconds=env_settings.temporary_default_ttl_seconds,
            temporary_default_purge_on_close=env_settings.temporary_default_purge_on_close,
            tombstone_retention_days=env_settings.tombstone_retention_days,
            erasure_purge_streams=env_settings.erasure_purge_streams,
            disable_chunking_extraction=(
                env_settings.disable_chunking_extraction
                if self._disable_chunking_extraction is None
                else self._disable_chunking_extraction
            ),
            chunking_extraction_threshold_tokens=(
                env_settings.chunking_extraction_threshold_tokens
            ),
            extraction_watchdog_enabled=env_settings.extraction_watchdog_enabled,
            extraction_watchdog_allow_different_provider=(
                env_settings.extraction_watchdog_allow_different_provider
            ),
            extraction_watchdog_bounded_retry_max_items=(
                env_settings.extraction_watchdog_bounded_retry_max_items
            ),
            extraction_watchdog_bounded_retry_max_output_tokens=(
                env_settings.extraction_watchdog_bounded_retry_max_output_tokens
            ),
            small_corpus_token_threshold_ratio=(
                env_settings.small_corpus_token_threshold_ratio
            ),
            assistant_guidance_enabled=(
                env_settings.assistant_guidance_enabled
                if self._assistant_guidance_enabled is None
                else self._assistant_guidance_enabled
            ),
            recent_transcript_budget_tokens=(
                env_settings.recent_transcript_budget_tokens
                if self._recent_transcript_budget_tokens is None
                else self._recent_transcript_budget_tokens
            ),
            context_envelope_budget_tokens=(
                env_settings.context_envelope_budget_tokens
                if self._context_envelope_budget_tokens is None
                else self._context_envelope_budget_tokens
            ),
            context_envelope_ratios=(
                env_settings.context_envelope_ratios
                if self._context_envelope_ratios is None
                else self._context_envelope_ratios
            ),
            benchmark_disable_raw_recent_transcript=(
                env_settings.benchmark_disable_raw_recent_transcript
            ),
            recent_transcript_overage_ratio=env_settings.recent_transcript_overage_ratio,
            topic_working_set_enabled=env_settings.topic_working_set_enabled,
            topic_working_set_refresh_message_lag=(
                env_settings.topic_working_set_refresh_message_lag
            ),
            topic_working_set_stale_message_lag=(
                env_settings.topic_working_set_stale_message_lag
            ),
            topic_working_set_refresh_token_lag=(
                env_settings.topic_working_set_refresh_token_lag
            ),
            topic_working_set_stale_token_lag=(
                env_settings.topic_working_set_stale_token_lag
            ),
            topic_working_set_refresh_batch_messages=(
                env_settings.topic_working_set_refresh_batch_messages
            ),
            graph_projection_enabled=env_settings.graph_projection_enabled,
            verbatim_evidence_search_enabled=env_settings.verbatim_evidence_search_enabled,
            verbatim_evidence_search_rrf_weight=env_settings.verbatim_evidence_search_rrf_weight,
            verbatim_evidence_search_limit=env_settings.verbatim_evidence_search_limit,
            verbatim_evidence_window_size=env_settings.verbatim_evidence_window_size,
            verbatim_evidence_window_overlap=env_settings.verbatim_evidence_window_overlap,
        )

    async def _require_runtime(self) -> AppRuntime:
        if self._runtime is None:
            await self.setup()
        if self._runtime is None:
            raise RuntimeNotInitializedError("Atagia runtime is not initialized")
        return self._runtime

    @staticmethod
    async def _list_review_required_rows(
        connection: Any,
        *,
        user_id: str | None,
        platform_id: str | None,
        user_persona_id: str | None,
        character_id: str | None,
        category: MemoryCategory | str | None,
        ingest_origin: str | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        clauses = ["status = ?"]
        parameters: list[Any] = [MemoryStatus.REVIEW_REQUIRED.value]
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        if platform_id is not None:
            clauses.append("platform_id = ?")
            parameters.append(platform_id)
        if user_persona_id is not None:
            clauses.append("user_persona_id IS ?")
            parameters.append(user_persona_id)
        if character_id is not None:
            clauses.append("character_id IS ?")
            parameters.append(character_id)
        if category is not None:
            clauses.append("memory_category = ?")
            parameters.append(MemoryCategory(category).value)
        if ingest_origin is not None:
            clauses.append("json_extract(payload_json, '$.ingest_origin') = ?")
            parameters.append(ingest_origin)
        cursor = await connection.execute(
            """
            SELECT *
            FROM memory_objects
            WHERE {clauses}
            ORDER BY created_at ASC, _rowid ASC
            LIMIT ?
            OFFSET ?
            """.format(clauses=" AND ".join(clauses)),
            (*parameters, max(1, min(limit, 500)), max(0, offset)),
        )
        rows = [dict(row) for row in await cursor.fetchall()]
        await cursor.close()
        return [Atagia._review_memory_record(row) for row in rows]

    @staticmethod
    def _review_memory_record(row: dict[str, Any]) -> dict[str, Any]:
        payload = row.get("payload_json")
        if isinstance(payload, str) and payload.strip():
            decoded = json_utils.loads(payload)
            payload = decoded if isinstance(decoded, dict) else {}
        elif not isinstance(payload, dict):
            payload = {}
        source_message_ids = payload.get("source_message_ids")
        if not isinstance(source_message_ids, list):
            source_message_ids = []
        return {
            "memory_id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "conversation_id": row.get("conversation_id"),
            "user_persona_id": row.get("user_persona_id"),
            "platform_id": row.get("platform_id"),
            "character_id": row.get("character_id"),
            "mode": row.get("assistant_mode_id"),
            "object_type": str(row["object_type"]),
            "category": str(row["memory_category"]),
            "scope": str(row["scope"]),
            "scope_canonical": row.get("scope_canonical"),
            "sensitivity": str(row.get("sensitivity") or "unknown"),
            "privacy_level": int(row["privacy_level"]),
            "confidence": float(row["confidence"]),
            "canonical_text": str(row["canonical_text"]),
            "index_text": row.get("index_text"),
            "review_reason": payload.get("review_reason"),
            "ingest_origin": payload.get("ingest_origin"),
            "confirmation_strategy": payload.get("confirmation_strategy"),
            "source_message_ids": [str(item) for item in source_message_ids],
            "payload": payload,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
