"""Retrieval orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.realm_repository import RealmRepository, realm_snapshot
from atagia.core.space_repository import SpaceRepository, space_snapshot
from atagia.core.topic_repository import TopicRepository
from atagia.models.schemas_memory import ExtractionConversationContext, RetrievalTrace
from atagia.models.schemas_memory import ResolvedOperationalProfile, TopicWorkingSetTrace
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    apply_conversation_policy_overlay,
    recent_context,
    resolve_retrieval_profile_id,
    resolve_policy,
)
from atagia.services.errors import ConversationNotFoundError
from atagia.services.retrieval_pipeline import RetrievalPipeline


@dataclass(slots=True)
class RetrievalService:
    """Coordinates retrieval-related components."""

    runtime: Any

    async def retrieve(
        self,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        operational_profile: ResolvedOperationalProfile | None = None,
        ablation: AblationConfig | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        """Execute the retrieval pipeline for the active conversation state."""
        connection = await self.runtime.open_connection()
        try:
            return await self._retrieve_with_connection(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                message_text=message_text,
                mode=mode,
                operational_profile=operational_profile,
                ablation=ablation,
                stored_messages=stored_messages,
                trace=trace,
            )
        finally:
            await connection.close()

    async def _retrieve_with_connection(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        operational_profile: ResolvedOperationalProfile | None = None,
        ablation: AblationConfig | None = None,
        conversation: dict[str, Any] | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        return await self.retrieve_with_connection(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            mode=mode,
            operational_profile=operational_profile,
            ablation=ablation,
            conversation=conversation,
            stored_messages=stored_messages,
            trace=trace,
        )

    async def retrieve_with_connection(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        operational_profile: ResolvedOperationalProfile | None = None,
        ablation: AblationConfig | None = None,
        conversation: dict[str, Any] | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        conversations = ConversationRepository(connection, self.runtime.clock)
        users = UserRepository(connection, self.runtime.clock)
        messages = MessageRepository(connection, self.runtime.clock)
        memories = MemoryObjectRepository(connection, self.runtime.clock)

        active_conversation = conversation or await conversations.get_conversation(conversation_id, user_id)
        if active_conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")
        memory_preferences = await users.get_memory_preferences(user_id)

        assistant_mode_id = resolve_retrieval_profile_id(
            str(active_conversation["assistant_mode_id"]),
            mode,
        )
        if trace is not None:
            trace.requested_mode = mode
            trace.effective_mode = assistant_mode_id
        resolved_policy = resolve_policy(
            self.runtime.manifests,
            assistant_mode_id,
            self.runtime.policy_resolver,
            operational_profile,
        )
        resolved_policy = apply_conversation_policy_overlay(
            resolved_policy,
            active_conversation,
        )
        active_messages = stored_messages or await messages.get_recent_messages(
            conversation_id,
            user_id,
            limit=RECENT_FETCH_LIMIT,
        )
        transcript, prior_messages, source_message_id = self._pipeline_inputs(
            active_messages,
            message_text=message_text,
            conversation_id=conversation_id,
        )
        active_space_id = active_conversation.get("active_space_id")
        active_space_boundary_mode = active_conversation.get("active_space_boundary_mode")
        active_space_display_name = active_conversation.get("active_space_display_name")
        active_mind_id = active_conversation.get("active_mind_id")
        mind_topology = active_conversation.get("mind_topology") or "unimind"
        active_embodiment_id = active_conversation.get("active_embodiment_id")
        active_realm_id = active_conversation.get("active_realm_id")
        active_realm_display_name = active_conversation.get("active_realm_display_name")
        cross_realm_mode = active_conversation.get("cross_realm_mode")
        if active_space_id is not None and active_space_boundary_mode is None:
            space_row = await SpaceRepository(connection, self.runtime.clock).get_space(
                owner_user_id=user_id,
                space_id=str(active_space_id),
            )
            if space_row is not None:
                space = space_snapshot(space_row)
                active_space_boundary_mode = space.boundary_mode.value
                active_space_display_name = space.display_name
        if active_realm_id is not None and cross_realm_mode is None:
            realm_row = await RealmRepository(connection, self.runtime.clock).get_realm(
                owner_user_id=user_id,
                realm_id=str(active_realm_id),
            )
            if realm_row is not None:
                realm = realm_snapshot(realm_row)
                cross_realm_mode = realm.cross_realm_mode.value
                active_realm_display_name = realm.display_name
        conversation_context = ExtractionConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            workspace_id=active_conversation["workspace_id"],
            assistant_mode_id=assistant_mode_id,
            user_persona_id=active_conversation.get("user_persona_id"),
            platform_id=str(active_conversation.get("platform_id") or "default"),
            character_id=(
                active_conversation.get("character_id")
                if active_conversation.get("character_id") is not None
                else active_conversation.get("workspace_id")
            ),
            active_presence_id=active_conversation.get("active_presence_id"),
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode or "focus",
            active_space_display_name=active_space_display_name,
            active_mind_id=active_mind_id,
            source_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            active_realm_display_name=active_realm_display_name,
            cross_realm_mode=cross_realm_mode or "none",
            mode=str(active_conversation.get("mode") or assistant_mode_id),
            recent_messages=recent_context(prior_messages),
            temporary=bool(active_conversation.get("temporary")),
            temporary_ttl_seconds=active_conversation.get("temporary_ttl_seconds"),
            purge_on_close=bool(active_conversation.get("purge_on_close")),
            isolated_mode=bool(active_conversation.get("isolated_mode")),
            incognito=bool(active_conversation.get("incognito")) or bool(active_conversation.get("isolated_mode")),
            remember_across_chats=bool(memory_preferences["remember_across_chats"]),
            remember_across_devices=bool(memory_preferences["remember_across_devices"]),
            memory_privacy_mode=memory_preferences["memory_privacy_mode"],
        )
        cold_start = (
            await memories.count_for_context(
                user_id,
                resolved_policy.allowed_scopes,
                workspace_id=active_conversation["workspace_id"],
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
                user_persona_id=conversation_context.user_persona_id,
                platform_id=conversation_context.platform_id,
                character_id=conversation_context.character_id,
                incognito=conversation_context.incognito,
                remember_across_chats=conversation_context.remember_across_chats,
                remember_across_devices=conversation_context.remember_across_devices,
                active_mind_id=conversation_context.active_mind_id,
                mind_topology=conversation_context.mind_topology,
                active_embodiment_id=conversation_context.active_embodiment_id,
                active_realm_id=conversation_context.active_realm_id,
            )
            == 0
        )
        await self._attach_topic_snapshot(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            trace=trace,
        )
        return await RetrievalPipeline(
            connection=connection,
            llm_client=self.runtime.llm_client,
            embedding_index=self.runtime.embedding_index,
            clock=self.runtime.clock,
            settings=self.runtime.settings,
        ).execute(
            message_text=message_text,
            conversation_context=conversation_context,
            resolved_policy=resolved_policy,
            cold_start=cold_start,
            ablation=ablation,
            conversation_messages=transcript,
            trace=trace,
        )

    async def _attach_topic_snapshot(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        trace: RetrievalTrace | None,
    ) -> None:
        if trace is None:
            return
        settings = getattr(self.runtime, "settings", None)
        freshness_kwargs = (
            {
                "refresh_message_threshold": settings.topic_working_set_refresh_message_lag,
                "stale_message_threshold": settings.topic_working_set_stale_message_lag,
                "refresh_token_threshold": settings.topic_working_set_refresh_token_lag,
                "stale_token_threshold": settings.topic_working_set_stale_token_lag,
            }
            if settings is not None
            else {}
        )
        snapshot = await TopicRepository(connection, self.runtime.clock).get_topic_snapshot(
            user_id=user_id,
            conversation_id=conversation_id,
            **freshness_kwargs,
        )
        trace.topic_snapshot = TopicWorkingSetTrace.model_validate(snapshot)

    @staticmethod
    def _pipeline_inputs(
        stored_messages: list[dict[str, Any]],
        *,
        message_text: str,
        conversation_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        if (
            stored_messages
            and str(stored_messages[-1]["role"]) == "user"
            and str(stored_messages[-1]["text"]) == message_text
        ):
            return stored_messages, stored_messages[:-1], str(stored_messages[-1]["id"])

        synthetic_message = {"role": "user", "text": message_text}
        source_message_id = (
            str(stored_messages[-1]["id"])
            if stored_messages
            else f"pending:{conversation_id}"
        )
        return [*stored_messages, synthetic_message], stored_messages, source_message_id
