"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.memory.lifecycle_runner import piggyback_lifecycle
from atagia.core.summary_repository import SummaryRepository
from atagia.core.timestamps import normalize_optional_timestamp, resolve_message_occurred_at
from atagia.models.schemas_api import ChatResult
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import (
    CONTEXT_VIEW_TTL_SECONDS,
    RECENT_FETCH_LIMIT,
    RECENT_WINDOW_MESSAGES,
    build_message_jobs,
    build_system_prompt,
    build_transcript_window,
    build_transcript_window_trace,
    chat_model,
    enqueue_message_jobs,
    missing_uncovered_tail_start_seq,
    render_transcript_window,
    resolve_assistant_mode_id,
    resolve_policy,
    summarize_memory_summaries,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.errors import ConversationNotFoundError, LLMUnavailableError
from atagia.services.llm_client import LLMCompletionRequest, LLMError, LLMMessage

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
                summaries = SummaryRepository(connection, self.runtime.clock)
                artifacts = ArtifactService(connection, self.runtime.clock)
                confirmations = PendingConfirmationService(
                    connection,
                    self.runtime.clock,
                    self.runtime.embedding_index,
                    llm_client=self.runtime.llm_client,
                    settings=self.runtime.settings,
                )

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
                    assistant_mode_id=assistant_mode_id,
                    stored_messages=prior_messages,
                    conversation=conversation,
                )
                transcript_entries = build_transcript_window(
                    prior_messages,
                    conversation_chunks,
                    resolution.resolved_policy.transcript_budget_tokens,
                    raw_context_access_mode=str(
                        resolution.source_retrieval_plan.get("raw_context_access_mode", "normal")
                    ),
                )
                transcript_trace = build_transcript_window_trace(
                    transcript_entries,
                    resolution.resolved_policy.transcript_budget_tokens,
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
                response_text = llm_response.output_text
                if confirmation_plan.prompt_text is not None:
                    response_text = f"{confirmation_plan.prompt_text}\n\n{llm_response.output_text}"
                resolved_user_occurred_at = (
                    normalize_optional_timestamp(message_occurred_at)
                    or self.runtime.clock.now().isoformat()
                )
                assistant_occurred_at = self.runtime.clock.now().isoformat()

                await connection.execute("BEGIN")
                try:
                    user_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="user",
                        seq=None,
                        text=prompt_message_text,
                        token_count=None,
                        metadata={
                            **(metadata or {}),
                            "attachments": attachment_bundle.attachments,
                            "attachment_count": len(attachment_bundle.artifacts),
                            "attachment_artifact_ids": [
                                str(prepared.artifact["id"]) for prepared in attachment_bundle.artifacts
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
                    assistant_message = await messages.create_message(
                        message_id=None,
                        conversation_id=conversation_id,
                        role="assistant",
                        seq=None,
                        text=llm_response.output_text,
                        token_count=None,
                        metadata={"thinking": llm_response.thinking} if llm_response.thinking else {},
                        occurred_at=assistant_occurred_at,
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
                                "background_tasks_enqueued": False,
                                "scored_candidates": resolution.scored_candidates,
                                "stage_timings_ms": resolution.stage_timings,
                                "transcript_window": transcript_trace,
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

                try:
                    if invalidate_confirmation_cache:
                        if resolution.cache_key is not None:
                            await self.runtime.storage_backend.delete_context_view(str(resolution.cache_key))
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
                try:
                    enqueued_job_ids = await enqueue_message_jobs(
                        storage_backend=self.runtime.storage_backend,
                        jobs=[*user_jobs, *assistant_jobs],
                    )
                    background_tasks_enqueued = True
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
                    debug=debug_payload,
                )
            except LLMError as exc:
                raise LLMUnavailableError("LLM service unavailable") from exc
            finally:
                await connection.close()
