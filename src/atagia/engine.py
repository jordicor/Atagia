"""Library-mode engine entry point."""

from __future__ import annotations

import asyncio
from dataclasses import replace as dataclass_replace
import os
from pathlib import Path
import sqlite3
from time import monotonic
from typing import Any

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings, configured_resource_path
from atagia.core.effective_settings import (
    ResolvedPolicyReport,
    build_effective_settings_report,
)
from atagia.core import json_utils
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import MemoryObjectRepository, WorkspaceRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.memory.policy_manifest import resolved_policy_provenance
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


def _resolved_manifest_policies(runtime: AppRuntime) -> dict[str, ResolvedPolicyReport]:
    """Resolve every mode manifest into the retrieval policy a run starts from.

    This is the manifest layer of the effective configuration: the policy as the
    mode files define it, with no workspace, conversation, or operational
    override applied, because those are per-request rather than per-run. Each
    field carries the layer that produced it -- the manifest file, a schema
    default the file left unset, the resolver's own constants, or a value
    computed from the manifest payload.
    """
    return {
        mode_id: ResolvedPolicyReport(
            values=runtime.policy_resolver.resolve(manifest, None, None).model_dump(
                mode="json"
            ),
            provenance=resolved_policy_provenance(manifest),
        )
        for mode_id, manifest in runtime.manifests.items()
    }


def _is_sqlite_busy_or_locked(exc: sqlite3.OperationalError) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int) and (error_code & 0xFF) in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }:
        return True
    message = str(exc).lower()
    return "locked" in message or "busy" in message


# Settings fields that library-mode `_build_settings` writes on top of the
# env-derived base (`Settings.from_env()`). Every other Settings field flows
# straight through from env. This is the single source of truth for the
# override allowlist: the runtime guard in `_build_settings` asserts the
# constructed override dict uses exactly these keys, and the engine tests import
# it to prove no Settings field silently diverges from env in library mode.
#
# Structural allowlist only -- it says which fields `_build_settings` may write,
# NOT which of them the engine actually decided on a given boot. Every entry
# except `_ENGINE_FORCED_SETTINGS_FIELDS` is a conditional merge, so with no
# constructor argument it writes the env-derived value back verbatim. Provenance
# therefore uses `_ENGINE_FORCED_SETTINGS_FIELDS | <caller-supplied fields>`, a
# subset of this set, computed per boot by `_resolve_engine_override_fields`.
_ENGINE_SETTINGS_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    {
        "sqlite_path",
        "manifests_path",
        "operational_profiles_path",
        "storage_backend",
        "redis_url",
        "anthropic_api_key",
        "openai_api_key",
        "google_api_key",
        "openrouter_api_key",
        "inference_access_mode",
        "local_llm_endpoints_file",
        "zero_cost_openrouter_profile",
        "llm_chat_model",
        "llm_forced_global_model",
        "llm_ingest_model",
        "llm_retrieval_model",
        "llm_component_models",
        "llm_intimacy_ingest_model",
        "llm_intimacy_retrieval_model",
        "llm_intimacy_component_models",
        "llm_intimacy_proactive_routing_enabled",
        "llm_structured_output_retry_attempts",
        "llm_structured_output_rescue_enabled",
        "llm_structured_output_rescue_model",
        "answer_postcondition_guard_enabled",
        "answer_stance",
        "answer_stance_prompt_variant",
        "service_mode",
        "service_api_key",
        "admin_api_key",
        "workers_enabled",
        "allow_insecure_http",
        "embedding_backend",
        "embedding_model",
        "skip_belief_revision",
        "skip_compaction",
        "context_cache_enabled",
        "disable_chunking_extraction",
        "assistant_guidance_enabled",
        "context_envelope_budget_tokens",
        "context_envelope_ratios",
    }
)

# Fields library mode pins on its own authority. Their value is a hardcoded
# literal in `_build_settings`, decided whatever the caller passed and whatever
# the environment holds, so the engine is their source on every boot: library
# mode is not a service (`service_mode`, `service_api_key`, `admin_api_key`),
# runs its own workers (`workers_enabled`), and talks over loopback
# (`allow_insecure_http`).
#
# Their environment variables are service-mode configuration read by `app.py`,
# not library-mode configuration. Nothing else belongs here: a field the
# environment is allowed to configure must reach the runtime, so it is a
# sentinel or fallback merge instead (see `_resolve_engine_override_fields`).
_ENGINE_FORCED_SETTINGS_FIELDS: frozenset[str] = frozenset(
    {
        "service_mode",
        "service_api_key",
        "admin_api_key",
        "workers_enabled",
        "allow_insecure_http",
    }
)

# `storage_backend` is neither forced nor a plain merge. Library mode collapses
# everything that is not redis to "inprocess", but WHO asked for redis varies:
# the caller's `redis_url` argument or `ATAGIA_STORAGE_BACKEND=redis`. Reporting
# the environment's own choice as an engine override would credit the wrong
# layer, so `_resolve_engine_override_fields` decides it from the very predicate
# `_build_settings` computes.
_ENV_PREDICATE_MERGE_FIELDS: frozenset[str] = frozenset({"storage_backend"})

# The two mappings `_models_after_phase_overrides` merges: neither a plain
# fallback nor a sentinel-guarded assignment, so `_resolve_engine_override_fields`
# decides them from the merge inputs instead of from a single argument.
_COMPONENT_MODEL_MERGE_FIELDS: frozenset[str] = frozenset(
    {
        "llm_component_models",
        "llm_intimacy_component_models",
    }
)


class Atagia:
    """Library interface for retrieval and chat flows."""

    def __init__(
        self,
        db_path: str | Path | None = None,
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
        inference_access_mode: str | None = None,
        local_llm_endpoints_file: str | Path | None = None,
        zero_cost_openrouter_profile: str | None = None,
        _inference_startup_completion_models: dict[str, str] | None = None,
        embedding_backend: str | None = None,
        embedding_model: str | None = None,
        skip_belief_revision: bool | None = None,
        skip_compaction: bool | None = None,
        context_cache_enabled: bool | None = None,
        disable_chunking_extraction: bool | None = None,
        assistant_guidance_enabled: bool | None = None,
        context_envelope_budget_tokens: int | None = None,
        context_envelope_ratios: dict[str, float] | None = None,
        answer_stance: str | None = None,
        answer_stance_prompt_variant: str | None = None,
        answer_postcondition_guard_enabled: bool | None = None,
    ) -> None:
        # `None` means "the caller did not supply this", which is what lets the
        # environment reach the runtime. Every parameter below whose omission
        # must not silence an env var uses that sentinel, so "not supplied" is
        # distinguishable from "supplied a value that equals the default".
        self._db_path: str | None = (
            str(Path(db_path).expanduser()) if isinstance(db_path, Path) else db_path
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
        self._inference_access_mode = inference_access_mode
        self._local_llm_endpoints_file = (
            str(Path(local_llm_endpoints_file).expanduser())
            if isinstance(local_llm_endpoints_file, Path)
            else local_llm_endpoints_file
        )
        self._zero_cost_openrouter_profile = zero_cost_openrouter_profile
        self._inference_startup_completion_models = dict(
            _inference_startup_completion_models or {}
        )
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._skip_belief_revision = skip_belief_revision
        self._skip_compaction = skip_compaction
        self._context_cache_enabled = context_cache_enabled
        self._disable_chunking_extraction = disable_chunking_extraction
        self._assistant_guidance_enabled = assistant_guidance_enabled
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
        # Populated by `_build_settings` (the environment as it stood when it
        # read its base, and the fields the engine actually decided) and frozen
        # into the effective-settings report by `setup()`.
        self._present_env_names: frozenset[str] = frozenset()
        self._engine_override_fields: frozenset[str] = frozenset()
        self._effective_settings_report: dict[str, Any] | None = None

    @property
    def runtime(self) -> AppRuntime | None:
        """Expose the initialized runtime for advanced callers and tests."""
        return self._runtime

    async def setup(self) -> Atagia:
        """Initialize the runtime if it has not been started yet."""
        if self._runtime is None:
            settings = self._build_settings()
            self._runtime = await initialize_runtime(
                settings,
                additional_inference_completion_models=(
                    self._inference_startup_completion_models
                ),
            )
            self._effective_settings_report = build_effective_settings_report(
                effective_settings=settings,
                present_env_names=self._present_env_names,
                engine_override_fields=self._engine_override_fields,
                resolved_policies=_resolved_manifest_policies(self._runtime),
            )
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
        await self._require_user_memory_available(runtime, user_id)
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
        return await self._drain_runtime(
            runtime,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_interval_seconds=progress_interval_seconds,
            progress_callback=progress_callback,
        )

    async def _drain_runtime(
        self,
        runtime: AppRuntime,
        *,
        timeout_seconds: float,
        idle_timeout_seconds: float | None = None,
        progress_interval_seconds: float = 0.0,
        progress_callback: Any | None = None,
    ) -> bool:
        """Drain durable SQLite work first, then its transient deliveries."""

        timeout = max(0.0, timeout_seconds)
        idle_timeout = (
            None if idle_timeout_seconds is None else max(0.0, idle_timeout_seconds)
        )
        started_at = monotonic()
        deadline = started_at + timeout
        last_progress_at = started_at
        previous_nonterminal: int | None = None
        try:
            connection = await runtime.open_connection(
                busy_timeout_ms=max(1, int(timeout * 1000))
            )
        except sqlite3.OperationalError as exc:
            if _is_sqlite_busy_or_locked(exc):
                return False
            raise
        jobs = JobRunRepository(connection, runtime.clock)
        try:
            while True:
                now = monotonic()
                remaining = max(0.0, deadline - now)
                await connection.execute(
                    f"PRAGMA busy_timeout = {max(1, min(50, int(remaining * 1000)))}"
                )
                try:
                    nonterminal = await jobs.nonterminal_count()
                except sqlite3.OperationalError as exc:
                    if not _is_sqlite_busy_or_locked(exc):
                        raise
                    if connection.in_transaction:
                        await connection.rollback()
                    now = monotonic()
                    if (
                        idle_timeout is not None
                        and now - last_progress_at >= idle_timeout
                    ):
                        return False
                    if now >= deadline:
                        return False
                    await asyncio.sleep(min(0.05, max(0.0, deadline - now)))
                    continue
                if (
                    previous_nonterminal is not None
                    and nonterminal != previous_nonterminal
                ):
                    last_progress_at = now
                previous_nonterminal = nonterminal
                if nonterminal == 0:
                    remaining = max(0.0, deadline - now)
                    backend_idle = await runtime.storage_backend.drain(
                        remaining,
                        idle_timeout_seconds=(
                            None
                            if idle_timeout is None
                            else max(
                                0.0,
                                min(
                                    remaining,
                                    idle_timeout - (now - last_progress_at),
                                ),
                            )
                        ),
                        progress_interval_seconds=progress_interval_seconds,
                        progress_callback=progress_callback,
                    )
                    if not backend_idle:
                        return False
                    if await jobs.nonterminal_count() == 0:
                        return True
                    last_progress_at = monotonic()
                    previous_nonterminal = None
                    continue
                if idle_timeout is not None and now - last_progress_at >= idle_timeout:
                    return False
                if now >= deadline:
                    return False
                await asyncio.sleep(min(0.05, max(0.0, deadline - now)))
        finally:
            await connection.close()

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
                    await self._drain_runtime(
                        runtime,
                        timeout_seconds=timeout_seconds,
                    )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            if filters.get("user_id") is not None:
                await self._require_user_memory_available(
                    runtime,
                    str(filters["user_id"]),
                    connection=connection,
                )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
        await self._require_user_memory_available(runtime, user_id)
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
        await self._require_user_memory_available(runtime, user_id)
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
        await self._require_user_memory_available(runtime, user_id)
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
        await self._require_user_memory_available(runtime, user_id)
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
        await self._require_user_memory_available(runtime, user_id)
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
        connection = await runtime.open_connection()
        try:
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
        connection = await runtime.open_connection()
        try:
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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
            await self._require_user_memory_available(
                runtime,
                user_id,
                connection=connection,
            )
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

    def _resolve_engine_override_fields(
        self,
        *,
        phase_models_overridden: bool,
        intimacy_phase_models_overridden: bool,
        storage_backend_from_env: bool,
    ) -> frozenset[str]:
        """Settings fields whose value THIS boot's engine layer decided.

        `_ENGINE_SETTINGS_OVERRIDE_FIELDS` says which fields `_build_settings`
        may write; this says which of them it actually sourced. The difference
        matters for the effective-settings report: most entries are conditional
        merges, and with no constructor argument the merge is the identity on
        the env-derived value. Tagging those `engine_override` would credit a
        layer that only copied the environment back -- the same misattribution
        the report exists to prevent.

        The rule is per field, mirroring the exact merge `_build_settings`
        performs, never a comparison against the env value:

        * `_ENGINE_FORCED_SETTINGS_FIELDS` -- always the engine (see there).
        * `self._x or env_settings.x` -- the engine when the caller's operand is
          the one `or` selects. An empty string is not selected, so the value
          came from env and is reported as such.
        * `env_settings.x if self._x is None else self._x` -- the engine when the
          argument is not `None`. `None` is the "not supplied" marker for every
          one of these parameters; a falsy-but-supplied value such as `0` or
          `""` IS taken by the engine and is reported as the engine's.
        * the two component-model mappings -- merges rather than fallbacks:
          `_models_after_phase_overrides` drops env-configured components whose
          phase received a constructor-level model, then applies the caller's
          mapping. The engine decided them when the caller's mapping contributed
          at least one entry or a supplied phase model parameterized the filter;
          with neither, the merge returns the env-derived mapping.
        * `storage_backend` -- the engine unless the environment is what asked
          for redis (`storage_backend_from_env`, the same predicate
          `_build_settings` computes). The caller's `redis_url` and the
          collapse-to-inprocess floor are both the engine's decision; a run that
          talks to redis because `ATAGIA_STORAGE_BACKEND=redis` was exported is
          the environment's.
        """
        # `self._x or env_settings.x` merges.
        fallback_arguments: dict[str, Any] = {
            "manifests_path": self._manifests_dir,
            "operational_profiles_path": self._operational_profiles_dir,
            "redis_url": self._redis_url,
            "anthropic_api_key": self._anthropic_api_key,
            "openai_api_key": self._openai_api_key,
            "google_api_key": self._google_api_key,
            "openrouter_api_key": self._openrouter_api_key,
            "llm_forced_global_model": self._llm_forced_global_model,
            "llm_chat_model": self._llm_chat_model,
            "llm_ingest_model": self._llm_ingest_model,
            "llm_retrieval_model": self._llm_retrieval_model,
            "llm_intimacy_ingest_model": self._llm_intimacy_ingest_model,
            "llm_intimacy_retrieval_model": self._llm_intimacy_retrieval_model,
            "llm_structured_output_rescue_model": (
                self._llm_structured_output_rescue_model
            ),
            "embedding_backend": self._embedding_backend,
            "embedding_model": self._embedding_model,
        }
        # `env_settings.x if self._x is None else self._x` merges.
        sentinel_arguments: dict[str, Any] = {
            "sqlite_path": self._db_path,
            "skip_belief_revision": self._skip_belief_revision,
            "skip_compaction": self._skip_compaction,
            "llm_intimacy_proactive_routing_enabled": (
                self._llm_intimacy_proactive_routing_enabled
            ),
            "llm_structured_output_retry_attempts": (
                self._llm_structured_output_retry_attempts
            ),
            "llm_structured_output_rescue_enabled": (
                self._llm_structured_output_rescue_enabled
            ),
            "answer_postcondition_guard_enabled": (
                self._answer_postcondition_guard_enabled
            ),
            "answer_stance": self._answer_stance,
            "answer_stance_prompt_variant": self._answer_stance_prompt_variant,
            "context_cache_enabled": self._context_cache_enabled,
            "disable_chunking_extraction": self._disable_chunking_extraction,
            "assistant_guidance_enabled": self._assistant_guidance_enabled,
            "context_envelope_budget_tokens": self._context_envelope_budget_tokens,
            "context_envelope_ratios": self._context_envelope_ratios,
            "inference_access_mode": self._inference_access_mode,
            "local_llm_endpoints_file": self._local_llm_endpoints_file,
            "zero_cost_openrouter_profile": self._zero_cost_openrouter_profile,
        }
        classified = (
            frozenset(fallback_arguments)
            | frozenset(sentinel_arguments)
            | _COMPONENT_MODEL_MERGE_FIELDS
            | _ENV_PREDICATE_MERGE_FIELDS
            | _ENGINE_FORCED_SETTINGS_FIELDS
        )
        classified_count = (
            len(fallback_arguments)
            + len(sentinel_arguments)
            + len(_COMPONENT_MODEL_MERGE_FIELDS)
            + len(_ENV_PREDICATE_MERGE_FIELDS)
            + len(_ENGINE_FORCED_SETTINGS_FIELDS)
        )
        if (
            classified != _ENGINE_SETTINGS_OVERRIDE_FIELDS
            or classified_count != len(classified)
        ):
            raise RuntimeError(
                "engine provenance classification drifted: every field in "
                "_ENGINE_SETTINGS_OVERRIDE_FIELDS must be classified exactly "
                "once as engine-forced, a fallback merge, a sentinel merge, a "
                "component-model merge, or an env-predicate merge"
            )

        engine_sourced = set(_ENGINE_FORCED_SETTINGS_FIELDS)
        engine_sourced.update(
            field for field, argument in fallback_arguments.items() if argument
        )
        engine_sourced.update(
            field
            for field, argument in sentinel_arguments.items()
            if argument is not None
        )
        if self._llm_component_models or phase_models_overridden:
            engine_sourced.add("llm_component_models")
        if self._llm_intimacy_component_models or intimacy_phase_models_overridden:
            engine_sourced.add("llm_intimacy_component_models")
        if not storage_backend_from_env:
            engine_sourced.add("storage_backend")
        return frozenset(engine_sourced)

    def _build_settings(self) -> Settings:
        env_settings = Settings.from_env()
        # Record the environment as it stood for this build so the
        # effective-settings report describes what ran instead of re-reading a
        # later environment.
        self._present_env_names = frozenset(os.environ)
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
        # `use_env_redis` is also the provenance predicate for `storage_backend`:
        # when it holds, the environment chose redis and the engine only carried
        # the choice through (see `_resolve_engine_override_fields`).
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
        overrides: dict[str, Any] = {
            "sqlite_path": (
                env_settings.sqlite_path if self._db_path is None else self._db_path
            ),
            "manifests_path": manifests_path,
            "operational_profiles_path": operational_profiles_path,
            "storage_backend": storage_backend,
            "redis_url": self._redis_url or env_settings.redis_url,
            "anthropic_api_key": anthropic_api_key,
            "openai_api_key": openai_api_key,
            "google_api_key": google_api_key,
            "openrouter_api_key": openrouter_api_key,
            "inference_access_mode": (
                env_settings.inference_access_mode
                if self._inference_access_mode is None
                else self._inference_access_mode
            ),
            "local_llm_endpoints_file": (
                env_settings.local_llm_endpoints_file
                if self._local_llm_endpoints_file is None
                else self._local_llm_endpoints_file
            ),
            "zero_cost_openrouter_profile": (
                env_settings.zero_cost_openrouter_profile
                if self._zero_cost_openrouter_profile is None
                else self._zero_cost_openrouter_profile
            ),
            "llm_chat_model": self._llm_chat_model or env_settings.llm_chat_model,
            "llm_forced_global_model": forced_global_model,
            "llm_ingest_model": (
                self._llm_ingest_model or env_settings.llm_ingest_model
            ),
            "llm_retrieval_model": (
                self._llm_retrieval_model or env_settings.llm_retrieval_model
            ),
            "llm_component_models": component_models,
            "llm_intimacy_ingest_model": (
                self._llm_intimacy_ingest_model
                or env_settings.llm_intimacy_ingest_model
            ),
            "llm_intimacy_retrieval_model": (
                self._llm_intimacy_retrieval_model
                or env_settings.llm_intimacy_retrieval_model
            ),
            "llm_intimacy_component_models": intimacy_component_models,
            "llm_intimacy_proactive_routing_enabled": (
                env_settings.llm_intimacy_proactive_routing_enabled
                if self._llm_intimacy_proactive_routing_enabled is None
                else self._llm_intimacy_proactive_routing_enabled
            ),
            "llm_structured_output_retry_attempts": (
                env_settings.llm_structured_output_retry_attempts
                if self._llm_structured_output_retry_attempts is None
                else self._llm_structured_output_retry_attempts
            ),
            "llm_structured_output_rescue_enabled": (
                env_settings.llm_structured_output_rescue_enabled
                if self._llm_structured_output_rescue_enabled is None
                else self._llm_structured_output_rescue_enabled
            ),
            "llm_structured_output_rescue_model": (
                self._llm_structured_output_rescue_model
                or env_settings.llm_structured_output_rescue_model
            ),
            "answer_postcondition_guard_enabled": (
                env_settings.answer_postcondition_guard_enabled
                if self._answer_postcondition_guard_enabled is None
                else self._answer_postcondition_guard_enabled
            ),
            "answer_stance": (
                env_settings.answer_stance
                if self._answer_stance is None
                else self._answer_stance
            ),
            "answer_stance_prompt_variant": (
                env_settings.answer_stance_prompt_variant
                if self._answer_stance_prompt_variant is None
                else self._answer_stance_prompt_variant
            ),
            "service_mode": False,
            "service_api_key": None,
            "admin_api_key": None,
            "workers_enabled": True,
            "allow_insecure_http": True,
            "embedding_backend": (
                self._embedding_backend or env_settings.embedding_backend
            ),
            "embedding_model": self._embedding_model or env_settings.embedding_model,
            "skip_belief_revision": (
                env_settings.skip_belief_revision
                if self._skip_belief_revision is None
                else self._skip_belief_revision
            ),
            "skip_compaction": (
                env_settings.skip_compaction
                if self._skip_compaction is None
                else self._skip_compaction
            ),
            "context_cache_enabled": (
                env_settings.context_cache_enabled
                if self._context_cache_enabled is None
                else self._context_cache_enabled
            ),
            "disable_chunking_extraction": (
                env_settings.disable_chunking_extraction
                if self._disable_chunking_extraction is None
                else self._disable_chunking_extraction
            ),
            "assistant_guidance_enabled": (
                env_settings.assistant_guidance_enabled
                if self._assistant_guidance_enabled is None
                else self._assistant_guidance_enabled
            ),
            "context_envelope_budget_tokens": (
                env_settings.context_envelope_budget_tokens
                if self._context_envelope_budget_tokens is None
                else self._context_envelope_budget_tokens
            ),
            "context_envelope_ratios": (
                env_settings.context_envelope_ratios
                if self._context_envelope_ratios is None
                else self._context_envelope_ratios
            ),
        }
        if frozenset(overrides) != _ENGINE_SETTINGS_OVERRIDE_FIELDS:
            raise RuntimeError(
                "engine settings override keys drifted from "
                "_ENGINE_SETTINGS_OVERRIDE_FIELDS; update the allowlist "
                "constant and its test in lockstep"
            )
        # Which of those writes the engine actually sourced, for the
        # effective-settings report `setup()` freezes.
        self._engine_override_fields = self._resolve_engine_override_fields(
            phase_models_overridden=bool(overridden_categories),
            intimacy_phase_models_overridden=bool(overridden_intimacy_categories),
            storage_backend_from_env=use_env_redis,
        )
        return dataclass_replace(env_settings, **overrides)

    def effective_settings_report(self) -> dict[str, Any]:
        """Auditable snapshot of the configuration this engine actually ran with.

        The snapshot is built once, during ``setup()``, from the very Settings
        handed to the runtime and the environment as it stood at that moment; it
        is never re-derived from the live environment, so mutating ``os.environ``
        after setup cannot change what the report claims a run executed with.

        The ``settings`` block carries every ``Settings`` field with a provenance
        tag (``default`` / ``env`` / ``engine_override``) and secret-shaped
        fields redacted mechanically by name; the ``resolved_policy`` block
        carries the retrieval policy resolved from each mode manifest, tagged
        ``manifest``. ``engine_override`` means this engine sourced the value --
        library mode pinned it, or the caller passed it to the constructor -- not
        merely that the field is one library mode is allowed to write. See
        ``_resolve_engine_override_fields``.
        """
        if self._effective_settings_report is None:
            raise RuntimeNotInitializedError(
                "effective_settings_report() reports what a run executed with; "
                "call setup() before reading it"
            )
        return self._effective_settings_report

    @staticmethod
    async def _require_user_memory_available(
        runtime: AppRuntime,
        user_id: str,
        *,
        connection: Any | None = None,
    ) -> None:
        """Reject user-scoped memory access during transcript replacement."""
        if connection is not None:
            await TranscriptRebuildRepository(
                connection,
                runtime.clock,
            ).require_user_available(user_id)
            return

        guard_connection = await runtime.open_connection()
        try:
            await TranscriptRebuildRepository(
                guard_connection,
                runtime.clock,
            ).require_user_available(user_id)
        finally:
            await guard_connection.close()

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
