"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_api import ChatResult
from atagia.services.chat_support import (
    CONTEXT_VIEW_TTL_SECONDS,
    RECENT_WINDOW_MESSAGES,
    build_message_jobs,
    build_system_prompt,
    chat_model,
    enqueue_message_jobs,
    resolve_assistant_mode_id,
    resolve_policy,
    summarize_memory_summaries,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import ConversationNotFoundError, LLMUnavailableError
from atagia.services.llm_client import LLMCompletionRequest, LLMError, LLMMessage


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
        message_occurred_at: str | None = None,
        include_thinking: bool = False,
        metadata: dict[str, Any] | None = None,
        debug: bool = False,
    ) -> ChatResult:
        """Run the full retrieval, generation, persistence, and background-job flow."""
        cache_service = ContextCacheService(self.runtime)
        async with cache_service.user_cache_guard(user_id):
            await wait_for_in_memory_worker_quiescence(self.runtime)
            connection = await self.runtime.open_connection()
            try:
                conversations = ConversationRepository(connection, self.runtime.clock)
                messages = MessageRepository(connection, self.runtime.clock)
                memories = MemoryObjectRepository(connection, self.runtime.clock)
                events = RetrievalEventRepository(connection, self.runtime.clock)

                conversation = await conversations.get_conversation(conversation_id, user_id)
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")

                resolved_mode_id = resolve_assistant_mode_id(
                    str(conversation["assistant_mode_id"]),
                    assistant_mode_id,
                )
                resolved_policy = resolve_policy(
                    self.runtime.manifests,
                    resolved_mode_id,
                    self.runtime.policy_resolver,
                )
                prior_messages = await messages.get_messages(
                    conversation_id,
                    user_id,
                    limit=500,
                    offset=0,
                )
                cold_start = (
                    await memories.count_for_context(
                        user_id,
                        resolved_policy.allowed_scopes,
                        workspace_id=conversation["workspace_id"],
                        conversation_id=conversation_id,
                        assistant_mode_id=resolved_mode_id,
                    )
                    == 0
                )
                resolution = await cache_service.resolve_with_connection(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_text=message_text,
                    assistant_mode_id=assistant_mode_id,
                    stored_messages=prior_messages,
                    conversation=conversation,
                )
                transcript = [
                    *prior_messages,
                    {"role": "user", "text": message_text},
                ]
                try:
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
                                    ),
                                ),
                                *[
                                    LLMMessage(role=str(message["role"]), content=str(message["text"]))
                                    for message in transcript
                                ],
                            ],
                            temperature=0.0,
                            include_thinking=include_thinking,
                            metadata={
                                "user_id": user_id,
                                "conversation_id": conversation_id,
                                "assistant_mode_id": resolved_mode_id,
                                "purpose": "chat_reply",
                            },
                        )
                    )
                except LLMError as exc:
                    raise LLMUnavailableError("LLM service unavailable") from exc

                await connection.execute("BEGIN")
                try:
                    user_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="user",
                        seq=None,
                        text=message_text,
                        token_count=None,
                        metadata=metadata or {},
                        occurred_at=message_occurred_at,
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
                        commit=False,
                    )
                    retrieval_event = await events.create_event(
                        {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "request_message_id": user_message["id"],
                            "response_message_id": assistant_message["id"],
                            "assistant_mode_id": resolved_mode_id,
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
                                "background_tasks_enqueued": True,
                                "scored_candidates": resolution.scored_candidates,
                                "stage_timings_ms": resolution.stage_timings,
                            },
                        },
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
            final_window = [
                {"role": str(message["role"]), "content": str(message["text"])}
                for message in [*prior_messages, user_message, assistant_message][-RECENT_WINDOW_MESSAGES:]
            ]
            await self.runtime.storage_backend.set_recent_window(
                f"{user_id}:{conversation_id}",
                final_window,
            )
            await self.runtime.storage_backend.set_context_view(
                retrieval_event["id"],
                resolution.composed_context.model_dump(mode="json"),
                ttl_seconds=CONTEXT_VIEW_TTL_SECONDS,
            )
            user_jobs = build_message_jobs(
                clock=self.runtime.clock,
                conversation=conversation,
                message_id=str(user_message["id"]),
                prior_messages=prior_messages,
                message_text=message_text,
                occurred_at=resolve_message_occurred_at(user_message),
                role="user",
            )
            assistant_jobs = build_message_jobs(
                clock=self.runtime.clock,
                conversation=conversation,
                message_id=str(assistant_message["id"]),
                prior_messages=[*prior_messages, user_message],
                message_text=llm_response.output_text,
                occurred_at=resolve_message_occurred_at(assistant_message),
                role="assistant",
            )
            enqueued_job_ids = await enqueue_message_jobs(
                storage_backend=self.runtime.storage_backend,
                jobs=[*user_jobs, *assistant_jobs],
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
                }
                if llm_response.thinking:
                    debug_payload["thinking"] = llm_response.thinking

            return ChatResult(
                conversation_id=conversation_id,
                request_message_id=str(user_message["id"]),
                response_message_id=str(assistant_message["id"]),
                response_text=llm_response.output_text,
                retrieval_event_id=str(retrieval_event["id"]),
                composed_context=resolution.composed_context,
                detected_needs=resolution.detected_needs,
                memories_used=summarize_memory_summaries(resolution.memory_summaries),
                debug=debug_payload,
            )
