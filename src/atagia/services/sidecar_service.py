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
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import normalize_optional_timestamp, resolve_message_occurred_at
from atagia.memory.lifecycle_runner import piggyback_lifecycle
from atagia.models.schemas_api import ContextResult
from atagia.models.schemas_replay import AblationConfig
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import (
    DEFAULT_ASSISTANT_MODE_ID,
    RECENT_FETCH_LIMIT,
    build_message_jobs,
    build_system_prompt,
    enqueue_message_jobs,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    WorkspaceNotFoundError,
)

if TYPE_CHECKING:
    from atagia.app import AppRuntime


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
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
    ) -> ContextResult:
        """Run retrieval, persist the user message, and return a ready system prompt."""
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
                )
                messages = MessageRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(connection, self.runtime.clock)
                attachment_bundle = artifacts.prepare_attachments(
                    message_text=message,
                    attachments=attachments,
                    user_id=user_id,
                    conversation=conversation,
                )
                prompt_message_text = attachment_bundle.prompt_text
                prior_messages = await messages.get_recent_messages(
                    str(conversation["id"]),
                    user_id,
                    limit=RECENT_FETCH_LIMIT,
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
                await connection.execute("BEGIN")
                try:
                    resolved_user_occurred_at = (
                        normalize_optional_timestamp(occurred_at)
                        or self.runtime.clock.now().isoformat()
                    )
                    user_message = await messages.create_message(
                        message_id=None,
                        conversation_id=str(conversation["id"]),
                        role="user",
                        seq=None,
                        text=prompt_message_text,
                        token_count=None,
                        metadata={
                            "attachments": attachment_bundle.attachments,
                            "attachment_count": len(attachment_bundle.artifacts),
                            "attachment_artifact_ids": [
                                str(prepared.artifact["id"])
                                for prepared in attachment_bundle.artifacts
                            ],
                            "artifact_backed": bool(attachment_bundle.artifacts),
                            "skip_by_default": bool(attachment_bundle.artifacts),
                            "include_raw": not bool(attachment_bundle.artifacts),
                            "requires_explicit_request": bool(attachment_bundle.artifacts),
                            "content_kind": "artifact" if attachment_bundle.artifacts else "text",
                            "context_placeholder": attachment_bundle.context_placeholder,
                        },
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
            await self._enqueue_message_jobs(
                conversation=conversation,
                message=user_message,
                prior_messages=prior_messages,
                message_text=prompt_message_text,
                role="user",
                operational_profile=resolution.resolved_operational_profile.snapshot,
            )
            if self.runtime.settings.lifecycle_lazy_enabled:
                self.runtime.spawn_background_task(
                    piggyback_lifecycle(self.runtime),
                    name="atagia-lifecycle-piggyback",
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
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
    ) -> None:
        """Store a message and enqueue extraction without running retrieval."""
        if role not in {"user", "assistant"}:
            raise ValueError("ingest_message role must be 'user' or 'assistant'")

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
                )
                messages = MessageRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(connection, self.runtime.clock)
                attachment_bundle = artifacts.prepare_attachments(
                    message_text=text,
                    attachments=attachments,
                    user_id=user_id,
                    conversation=conversation,
                )
                prompt_message_text = attachment_bundle.prompt_text
                prior_messages = await messages.get_recent_messages(
                    str(conversation["id"]),
                    user_id,
                    limit=RECENT_FETCH_LIMIT,
                )
                await connection.execute("BEGIN")
                try:
                    resolved_occurred_at = (
                        normalize_optional_timestamp(occurred_at)
                        or self.runtime.clock.now().isoformat()
                    )
                    stored_message = await messages.create_message(
                        message_id=None,
                        conversation_id=str(conversation["id"]),
                        role=role,
                        seq=None,
                        text=prompt_message_text,
                        token_count=None,
                        metadata={
                            "attachments": attachment_bundle.attachments,
                            "attachment_count": len(attachment_bundle.artifacts),
                            "attachment_artifact_ids": [
                                str(prepared.artifact["id"])
                                for prepared in attachment_bundle.artifacts
                            ],
                            "artifact_backed": bool(attachment_bundle.artifacts),
                            "skip_by_default": bool(attachment_bundle.artifacts),
                            "include_raw": not bool(attachment_bundle.artifacts),
                            "requires_explicit_request": bool(attachment_bundle.artifacts),
                            "content_kind": "artifact" if attachment_bundle.artifacts else "text",
                            "context_placeholder": attachment_bundle.context_placeholder,
                        },
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
            await self._enqueue_message_jobs(
                conversation=conversation,
                message=stored_message,
                prior_messages=prior_messages,
                message_text=prompt_message_text,
                role=role,
                operational_profile=resolved_operational_profile.snapshot,
            )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: Any | None = None,
    ) -> None:
        """Persist an assistant response in the conversation history."""
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
                conversation = await conversations.get_conversation(conversation_id, user_id)
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                messages = MessageRepository(connection, self.runtime.clock)
                prior_messages = await messages.get_recent_messages(
                    conversation_id,
                    user_id,
                    limit=RECENT_FETCH_LIMIT,
                )
                await connection.execute("BEGIN")
                try:
                    resolved_assistant_occurred_at = (
                        normalize_optional_timestamp(occurred_at)
                        or self.runtime.clock.now().isoformat()
                    )
                    assistant_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="assistant",
                        seq=None,
                        text=text,
                        token_count=None,
                        metadata={},
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
            await self._enqueue_message_jobs(
                conversation=conversation,
                message=assistant_message,
                prior_messages=prior_messages,
                message_text=text,
                role="assistant",
                operational_profile=resolved_operational_profile.snapshot,
            )

    async def ensure_user_exists(self, connection: aiosqlite.Connection, user_id: str) -> None:
        """Create the user if it does not already exist."""
        users = UserRepository(connection, self.runtime.clock)
        if await users.get_user(user_id) is None:
            await users.create_user(user_id)

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
    ) -> dict[str, Any]:
        """Return an existing conversation or create one with the requested id."""
        conversations = ConversationRepository(connection, self.runtime.clock)
        workspaces = WorkspaceRepository(connection, self.runtime.clock)
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
            if workspace_id is not None and conversation["workspace_id"] != workspace_id:
                raise ValueError(
                    "Requested workspace does not match the existing conversation workspace"
                )
            return conversation

        if workspace_id is not None:
            workspace = await workspaces.get_workspace(workspace_id, user_id)
            if workspace is None:
                raise WorkspaceNotFoundError("Workspace not found for user")

        resolved_mode = assistant_mode_id or DEFAULT_ASSISTANT_MODE_ID
        resolve_policy(self.runtime.manifests, resolved_mode, self.runtime.policy_resolver)
        try:
            return await conversations.create_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                workspace_id=workspace_id,
                assistant_mode_id=resolved_mode,
                title=title,
                metadata=metadata or {},
            )
        except aiosqlite.IntegrityError as exc:
            raise ConversationNotFoundError("Conversation not found for user") from exc

    async def _enqueue_message_jobs(
        self,
        *,
        conversation: dict[str, Any],
        message: dict[str, Any],
        prior_messages: list[dict[str, Any]],
        message_text: str,
        role: str,
        operational_profile: Any | None = None,
    ) -> None:
        jobs = build_message_jobs(
            clock=self.runtime.clock,
            conversation=conversation,
            message_id=str(message["id"]),
            prior_messages=prior_messages,
            message_text=message_text,
            occurred_at=resolve_message_occurred_at(message),
            role=role,
            operational_profile=operational_profile,
        )
        await enqueue_message_jobs(
            storage_backend=self.runtime.storage_backend,
            jobs=jobs,
        )
