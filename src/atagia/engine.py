"""Library-mode engine entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import WorkspaceRepository
from atagia.models.schemas_api import ChatResult, ContextResult
from atagia.models.schemas_api import (
    ActivitySnapshotResponse,
    ConversationActivityStats,
    VerbatimPinRecord,
    WarmupConversationResponse,
    WarmupRecommendedConversationsResponse,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.models.schemas_memory import MemoryScope, VerbatimPinStatus, VerbatimPinTargetKind
from atagia.services.chat_service import ChatService
from atagia.services.conversation_activity_service import ConversationActivityService
from atagia.services.sidecar_service import SidecarService
from atagia.services.verbatim_pin_service import VerbatimPinService
from atagia.services.errors import RuntimeNotInitializedError
from atagia.services.model_resolution import provider_qualified_model

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_OPERATIONAL_PROFILES_DIR = _PROJECT_ROOT / "operational_profiles"


class Atagia:
    """Library interface for retrieval and chat flows."""

    def __init__(
        self,
        db_path: str | Path = "atagia.db",
        redis_url: str | None = None,
        manifests_dir: str | Path | None = None,
        operational_profiles_dir: str | Path | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_forced_global_model: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        google_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
        embedding_provider_name: str | None = None,
        skip_belief_revision: bool = False,
        skip_compaction: bool = False,
        context_cache_enabled: bool | None = None,
        chunking_enabled: bool | None = None,
    ) -> None:
        self._db_path = str(Path(db_path).expanduser()) if isinstance(db_path, Path) else str(db_path)
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
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_forced_global_model = llm_forced_global_model
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._google_api_key = google_api_key
        self._openrouter_api_key = openrouter_api_key
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._embedding_provider_name = embedding_provider_name
        self._skip_belief_revision = skip_belief_revision
        self._skip_compaction = skip_compaction
        self._context_cache_enabled = context_cache_enabled
        self._chunking_enabled = chunking_enabled
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )

    async def flush(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for pending background work to finish when workers are enabled."""
        runtime = await self._require_runtime()
        if not runtime.settings.workers_enabled:
            return False
        return await runtime.storage_backend.drain(timeout_seconds)

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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        """Persist an assistant response in the conversation history."""
        runtime = await self._require_runtime()
        await SidecarService(runtime).add_response(
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            occurred_at=occurred_at,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Run the full chat flow, including the LLM response generation."""
        runtime = await self._require_runtime()
        sidecar = SidecarService(runtime)
        connection = await runtime.open_connection()
        try:
            await sidecar.ensure_user_exists(connection, user_id)
            await sidecar.ensure_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=mode,
            )
        finally:
            await connection.close()
        return await ChatService(runtime).chat_reply(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message,
            assistant_mode_id=mode,
            attachments=attachments,
            message_occurred_at=occurred_at,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )

    async def create_user(self, user_id: str) -> None:
        """Create the user if it does not already exist."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await SidecarService(runtime).ensure_user_exists(connection, user_id)
        finally:
            await connection.close()

    async def create_workspace(self, user_id: str, workspace_id: str, name: str) -> None:
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
            )
            return str(conversation["id"])
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
                status=status,
                reason=reason,
                expires_at=expires_at,
                payload_json=payload_json,
            )
            return None if updated is None else VerbatimPinRecord.model_validate(updated)
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
            return None if deleted is None else VerbatimPinRecord.model_validate(deleted)
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
        manifests_path = (
            self._manifests_dir
            or os.getenv("ATAGIA_MANIFESTS_PATH")
            or str(_DEFAULT_MANIFESTS_DIR)
        )
        operational_profiles_path = (
            self._operational_profiles_dir
            or os.getenv("ATAGIA_OPERATIONAL_PROFILES_PATH")
            or str(_DEFAULT_OPERATIONAL_PROFILES_DIR)
        )
        migrations_path = os.getenv("ATAGIA_MIGRATIONS_PATH") or str(_DEFAULT_MIGRATIONS_DIR)
        use_env_redis = self._redis_url is None and env_settings.storage_backend == "redis"
        storage_backend = "redis" if self._redis_url is not None or use_env_redis else "inprocess"
        legacy_provider = (self._llm_provider or env_settings.llm_provider).strip().lower()
        anthropic_api_key = self._anthropic_api_key or env_settings.anthropic_api_key
        openai_api_key = self._openai_api_key or env_settings.openai_api_key
        google_api_key = self._google_api_key or env_settings.google_api_key
        openrouter_api_key = self._openrouter_api_key or env_settings.openrouter_api_key
        if self._llm_api_key:
            if legacy_provider == "anthropic":
                anthropic_api_key = self._llm_api_key
            elif legacy_provider == "openai":
                openai_api_key = self._llm_api_key
            elif legacy_provider in {"gemini", "google"}:
                google_api_key = self._llm_api_key
            elif legacy_provider == "openrouter":
                openrouter_api_key = self._llm_api_key
        forced_global_model = (
            self._llm_forced_global_model
            or provider_qualified_model(legacy_provider, self._llm_model)
            or env_settings.llm_forced_global_model
        )
        return Settings(
            sqlite_path=self._db_path,
            migrations_path=migrations_path,
            manifests_path=manifests_path,
            operational_profiles_path=operational_profiles_path,
            storage_backend=storage_backend,
            redis_url=self._redis_url or env_settings.redis_url,
            llm_provider=legacy_provider,
            llm_api_key=self._llm_api_key or env_settings.llm_api_key,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            llm_base_url=env_settings.llm_base_url,
            anthropic_base_url=env_settings.anthropic_base_url,
            openai_base_url=env_settings.openai_base_url,
            openrouter_base_url=env_settings.openrouter_base_url,
            openrouter_site_url=env_settings.openrouter_site_url,
            openrouter_app_name=env_settings.openrouter_app_name,
            llm_extraction_model=env_settings.llm_extraction_model,
            llm_scoring_model=env_settings.llm_scoring_model,
            llm_classifier_model=env_settings.llm_classifier_model,
            llm_chat_model=env_settings.llm_chat_model,
            llm_forced_global_model=forced_global_model,
            llm_ingest_model=env_settings.llm_ingest_model,
            llm_retrieval_model=env_settings.llm_retrieval_model,
            llm_component_models=dict(env_settings.llm_component_models),
            embedding_provider_name=(
                self._embedding_provider_name or env_settings.embedding_provider_name
            ),
            service_mode=False,
            service_api_key=None,
            admin_api_key=None,
            workers_enabled=True,
            debug=env_settings.debug,
            allow_insecure_http=True,
            embedding_backend=self._embedding_backend,
            embedding_model=self._embedding_model or env_settings.embedding_model,
            embedding_dimension=env_settings.embedding_dimension,
            lifecycle_decay_days=env_settings.lifecycle_decay_days,
            lifecycle_decay_rate=env_settings.lifecycle_decay_rate,
            lifecycle_archive_vitality=env_settings.lifecycle_archive_vitality,
            lifecycle_archive_confidence=env_settings.lifecycle_archive_confidence,
            ephemeral_scoring_hours=env_settings.ephemeral_scoring_hours,
            lifecycle_ephemeral_ttl_hours=env_settings.lifecycle_ephemeral_ttl_hours,
            lifecycle_review_ttl_days=env_settings.lifecycle_review_ttl_days,
            lifecycle_lazy_enabled=env_settings.lifecycle_lazy_enabled,
            lifecycle_min_interval_seconds=env_settings.lifecycle_min_interval_seconds,
            lifecycle_worker_enabled=env_settings.lifecycle_worker_enabled,
            lifecycle_worker_interval_seconds=env_settings.lifecycle_worker_interval_seconds,
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
            chunking_enabled=(
                env_settings.chunking_enabled
                if self._chunking_enabled is None
                else self._chunking_enabled
            ),
            chunking_threshold_tokens=env_settings.chunking_threshold_tokens,
            small_corpus_token_threshold_ratio=(
                env_settings.small_corpus_token_threshold_ratio
            ),
        )

    async def _require_runtime(self) -> AppRuntime:
        if self._runtime is None:
            await self.setup()
        if self._runtime is None:
            raise RuntimeNotInitializedError("Atagia runtime is not initialized")
        return self._runtime
