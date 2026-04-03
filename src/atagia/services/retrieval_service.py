"""Retrieval orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    recent_context,
    resolve_assistant_mode_id,
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
        ablation: AblationConfig | None = None,
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
                ablation=ablation,
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
        ablation: AblationConfig | None = None,
        conversation: dict[str, Any] | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        return await self.retrieve_with_connection(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            mode=mode,
            ablation=ablation,
            conversation=conversation,
            stored_messages=stored_messages,
        )

    async def retrieve_with_connection(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        ablation: AblationConfig | None = None,
        conversation: dict[str, Any] | None = None,
        stored_messages: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        conversations = ConversationRepository(connection, self.runtime.clock)
        messages = MessageRepository(connection, self.runtime.clock)
        memories = MemoryObjectRepository(connection, self.runtime.clock)

        active_conversation = conversation or await conversations.get_conversation(conversation_id, user_id)
        if active_conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")

        assistant_mode_id = resolve_assistant_mode_id(
            str(active_conversation["assistant_mode_id"]),
            mode,
        )
        resolved_policy = resolve_policy(
            self.runtime.manifests,
            assistant_mode_id,
            self.runtime.policy_resolver,
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
        conversation_context = ExtractionConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            workspace_id=active_conversation["workspace_id"],
            assistant_mode_id=assistant_mode_id,
            recent_messages=recent_context(prior_messages),
        )
        cold_start = (
            await memories.count_for_context(
                user_id,
                resolved_policy.allowed_scopes,
                workspace_id=active_conversation["workspace_id"],
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
            )
            == 0
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
        )

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
