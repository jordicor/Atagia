"""Library-mode engine entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_api import ChatResult, ContextResult
from atagia.models.schemas_replay import AblationConfig
from atagia.services.chat_service import ChatService
from atagia.services.chat_support import (
    DEFAULT_ASSISTANT_MODE_ID,
    build_message_jobs,
    build_system_prompt,
    enqueue_message_jobs,
    resolve_policy,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    RuntimeNotInitializedError,
    WorkspaceNotFoundError,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"


class Atagia:
    """Library interface for retrieval and chat flows."""

    def __init__(
        self,
        db_path: str | Path = "atagia.db",
        redis_url: str | None = None,
        manifests_dir: str | Path | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
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
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
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
    ) -> ContextResult:
        """Run retrieval, persist the user message, and return a ready system prompt."""
        runtime = await self._require_runtime()
        cache_service = ContextCacheService(runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(runtime)
            connection = await runtime.open_connection()
            try:
                await self._ensure_user_exists(connection, runtime, user_id)
                conversation = await self._ensure_conversation(
                    connection,
                    runtime,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=mode,
                )
                messages = MessageRepository(connection, runtime.clock)
                prior_messages = await messages.get_messages(
                    conversation["id"],
                    user_id,
                    limit=500,
                    offset=0,
                )
                resolution = await cache_service.resolve_with_connection(
                    connection,
                    user_id=user_id,
                    conversation_id=str(conversation["id"]),
                    message_text=message,
                    assistant_mode_id=mode,
                    stored_messages=prior_messages,
                    conversation=conversation,
                    ablation=ablation,
                )
                messages = MessageRepository(connection, runtime.clock)
                await connection.execute("BEGIN")
                try:
                    user_message = await messages.create_message(
                        message_id=None,
                        conversation_id=str(conversation["id"]),
                        role="user",
                        seq=None,
                        text=message,
                        token_count=None,
                        metadata={},
                        occurred_at=occurred_at,
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
            await self._enqueue_message_jobs(
                runtime,
                conversation=conversation,
                message=user_message,
                prior_messages=prior_messages,
                message_text=message,
                role="user",
            )
        return ContextResult(
            system_prompt=build_system_prompt(
                str(conversation["assistant_mode_id"]),
                resolution.resolved_policy,
                resolution.composed_context.contract_block,
                resolution.composed_context.workspace_block,
                resolution.composed_context.memory_block,
                resolution.composed_context.state_block,
            ),
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
    ) -> None:
        """Store a message and enqueue extraction without running retrieval."""
        if role not in {"user", "assistant"}:
            raise ValueError("ingest_message role must be 'user' or 'assistant'")

        runtime = await self._require_runtime()
        cache_service = ContextCacheService(runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(runtime)
            connection = await runtime.open_connection()
            conversation: dict[str, Any] | None = None
            prior_messages: list[dict[str, Any]] = []
            stored_message: dict[str, Any] | None = None
            try:
                await self._ensure_user_exists(connection, runtime, user_id)
                conversation = await self._ensure_conversation(
                    connection,
                    runtime,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=mode,
                )
                messages = MessageRepository(connection, runtime.clock)
                prior_messages = await messages.get_messages(
                    str(conversation["id"]),
                    user_id,
                    limit=500,
                    offset=0,
                )
                await connection.execute("BEGIN")
                try:
                    stored_message = await messages.create_message(
                        message_id=None,
                        conversation_id=str(conversation["id"]),
                        role=role,
                        seq=None,
                        text=text,
                        token_count=None,
                        metadata={},
                        occurred_at=occurred_at,
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
            await self._enqueue_message_jobs(
                runtime,
                conversation=conversation,
                message=stored_message,
                prior_messages=prior_messages,
                message_text=text,
                role=role,
            )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
    ) -> None:
        """Persist an assistant response in the conversation history."""
        runtime = await self._require_runtime()
        cache_service = ContextCacheService(runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(runtime)
            connection = await runtime.open_connection()
            conversation: dict[str, Any] | None = None
            prior_messages: list[dict[str, Any]] = []
            assistant_message: dict[str, Any] | None = None
            try:
                conversations = ConversationRepository(connection, runtime.clock)
                conversation = await conversations.get_conversation(conversation_id, user_id)
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                messages = MessageRepository(connection, runtime.clock)
                prior_messages = await messages.get_messages(
                    conversation_id,
                    user_id,
                    limit=500,
                    offset=0,
                )
                await connection.execute("BEGIN")
                try:
                    assistant_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="assistant",
                        seq=None,
                        text=text,
                        token_count=None,
                        metadata={},
                        occurred_at=occurred_at,
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
            await self._enqueue_message_jobs(
                runtime,
                conversation=conversation,
                message=assistant_message,
                prior_messages=prior_messages,
                message_text=text,
                role="assistant",
            )

    async def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
    ) -> ChatResult:
        """Run the full chat flow, including the LLM response generation."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await self._ensure_user_exists(connection, runtime, user_id)
            await self._ensure_conversation(
                connection,
                runtime,
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
            message_occurred_at=occurred_at,
        )

    async def create_user(self, user_id: str) -> None:
        """Create the user if it does not already exist."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await self._ensure_user_exists(connection, runtime, user_id)
        finally:
            await connection.close()

    async def create_workspace(self, user_id: str, workspace_id: str, name: str) -> None:
        """Create the workspace if it does not already exist."""
        runtime = await self._require_runtime()
        connection = await runtime.open_connection()
        try:
            await self._ensure_user_exists(connection, runtime, user_id)
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
        connection = await runtime.open_connection()
        try:
            await self._ensure_user_exists(connection, runtime, user_id)
            conversation = await self._ensure_conversation(
                connection,
                runtime,
                user_id=user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
            )
            return str(conversation["id"])
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
        migrations_path = os.getenv("ATAGIA_MIGRATIONS_PATH") or str(_DEFAULT_MIGRATIONS_DIR)
        use_env_redis = self._redis_url is None and env_settings.storage_backend == "redis"
        storage_backend = "redis" if self._redis_url is not None or use_env_redis else "inprocess"
        return Settings(
            sqlite_path=self._db_path,
            migrations_path=migrations_path,
            manifests_path=manifests_path,
            storage_backend=storage_backend,
            redis_url=self._redis_url or env_settings.redis_url,
            llm_provider=(self._llm_provider or env_settings.llm_provider).strip().lower(),
            llm_api_key=self._llm_api_key or env_settings.llm_api_key,
            openai_api_key=env_settings.openai_api_key,
            openrouter_api_key=env_settings.openrouter_api_key,
            llm_base_url=env_settings.llm_base_url,
            openrouter_site_url=env_settings.openrouter_site_url,
            openrouter_app_name=env_settings.openrouter_app_name,
            llm_extraction_model=self._llm_model or env_settings.llm_extraction_model,
            llm_scoring_model=self._llm_model or env_settings.llm_scoring_model,
            llm_classifier_model=self._llm_model or env_settings.llm_classifier_model,
            llm_chat_model=self._llm_model or env_settings.llm_chat_model,
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
            lifecycle_ephemeral_ttl_hours=env_settings.lifecycle_ephemeral_ttl_hours,
            lifecycle_review_ttl_days=env_settings.lifecycle_review_ttl_days,
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
        )

    async def _require_runtime(self) -> AppRuntime:
        if self._runtime is None:
            await self.setup()
        if self._runtime is None:
            raise RuntimeNotInitializedError("Atagia runtime is not initialized")
        return self._runtime

    async def _ensure_user_exists(
        self,
        connection: Any,
        runtime: AppRuntime,
        user_id: str,
    ) -> None:
        users = UserRepository(connection, runtime.clock)
        if await users.get_user(user_id) is None:
            await users.create_user(user_id)

    async def _ensure_conversation(
        self,
        connection: Any,
        runtime: AppRuntime,
        *,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None,
        assistant_mode_id: str | None,
    ) -> dict[str, Any]:
        conversations = ConversationRepository(connection, runtime.clock)
        workspaces = WorkspaceRepository(connection, runtime.clock)
        conversation = None
        if conversation_id is not None:
            conversation = await conversations.get_conversation(conversation_id, user_id)
        if conversation is not None:
            if (
                assistant_mode_id is not None
                and assistant_mode_id != conversation["assistant_mode_id"]
            ):
                raise AssistantModeMismatchError(
                    "Requested assistant mode does not match the existing conversation mode"
                )
            if (
                workspace_id is not None
                and conversation["workspace_id"] != workspace_id
            ):
                raise ValueError(
                    "Requested workspace does not match the existing conversation workspace"
                )
            return conversation

        if workspace_id is not None:
            workspace = await workspaces.get_workspace(workspace_id, user_id)
            if workspace is None:
                raise WorkspaceNotFoundError("Workspace not found for user")

        resolved_mode = assistant_mode_id or DEFAULT_ASSISTANT_MODE_ID
        resolve_policy(runtime.manifests, resolved_mode, runtime.policy_resolver)
        return await conversations.create_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            assistant_mode_id=resolved_mode,
            title=None,
            metadata={},
        )

    async def _enqueue_message_jobs(
        self,
        runtime: AppRuntime,
        *,
        conversation: dict[str, Any],
        message: dict[str, Any],
        prior_messages: list[dict[str, Any]],
        message_text: str,
        role: str,
    ) -> None:
        jobs = build_message_jobs(
            clock=runtime.clock,
            conversation=conversation,
            message_id=str(message["id"]),
            prior_messages=prior_messages,
            message_text=message_text,
            occurred_at=resolve_message_occurred_at(message),
            role=role,
        )
        await enqueue_message_jobs(
            storage_backend=runtime.storage_backend,
            jobs=jobs,
        )
