"""Shared helpers for chat and library mode orchestration."""

from __future__ import annotations

from typing import Any

from atagia.core.ids import new_job_id
from atagia.core.config import Settings
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.memory.policy_manifest import PolicyResolver, ResolvedPolicy
from atagia.models.schemas_api import MemorySummary
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
    MessageJobPayload,
)
from atagia.models.schemas_memory import ExtractionContextMessage, ExtractionConversationContext
from atagia.models.schemas_replay import PipelineResult
from atagia.services.errors import AssistantModeMismatchError, UnknownAssistantModeError

DEFAULT_ASSISTANT_MODE_ID = "general_qa"
DEFAULT_CHAT_MODEL = "claude-sonnet-4-6"
RECENT_CONTEXT_MESSAGES = 6
RECENT_WINDOW_MESSAGES = 12
CONTEXT_VIEW_TTL_SECONDS = 60 * 60
TEXT_PREVIEW_LIMIT = 200


def resolve_assistant_mode_id(
    conversation_mode_id: str,
    requested_mode_id: str | None,
) -> str:
    """Return the active assistant mode, rejecting conflicting overrides."""
    if requested_mode_id is None:
        return conversation_mode_id
    if requested_mode_id != conversation_mode_id:
        raise AssistantModeMismatchError(
            "Requested assistant mode does not match the existing conversation mode"
        )
    return requested_mode_id


def resolve_policy(
    manifests: dict[str, Any],
    assistant_mode_id: str,
    policy_resolver: PolicyResolver,
) -> ResolvedPolicy:
    """Resolve the active assistant mode policy."""
    manifest = manifests.get(assistant_mode_id)
    if manifest is None:
        raise UnknownAssistantModeError(f"Unknown assistant mode: {assistant_mode_id}")
    return policy_resolver.resolve(manifest, None, None)


def recent_context(messages: list[dict[str, Any]]) -> list[ExtractionContextMessage]:
    """Build the short recent-message context used by retrieval and extraction."""
    return [
        ExtractionContextMessage(role=str(message["role"]), content=str(message["text"]))
        for message in messages[-RECENT_CONTEXT_MESSAGES:]
    ]


def chat_model(settings: Settings) -> str:
    """Resolve the chat model used for full reply generation."""
    return settings.llm_chat_model or DEFAULT_CHAT_MODEL


def build_system_prompt(
    assistant_mode_id: str,
    resolved_policy: ResolvedPolicy,
    contract_block: str,
    workspace_block: str,
    memory_block: str,
    state_block: str,
) -> str:
    """Assemble the grounded system prompt passed to the chat model."""
    parts = [
        (
            f"You are the Atagia assistant for mode {assistant_mode_id}. "
            "Use retrieved context only when it is helpful and stay grounded in the active conversation."
        ),
        f"Resolved policy hash: {resolved_policy.prompt_hash}",
    ]
    if contract_block:
        parts.append(contract_block)
    if workspace_block:
        parts.append(workspace_block)
    if memory_block:
        parts.append(memory_block)
    if state_block:
        parts.append(state_block)
    return "\n\n".join(parts)


def build_job_payload(
    *,
    conversation_context: ExtractionConversationContext,
    message_text: str,
    message_occurred_at: str | None = None,
    role: str,
) -> MessageJobPayload:
    """Serialize the message payload used by ingest and contract jobs."""
    return MessageJobPayload(
        message_id=conversation_context.source_message_id,
        message_text=message_text,
        message_occurred_at=normalize_optional_timestamp(message_occurred_at),
        role=role,
        assistant_mode_id=conversation_context.assistant_mode_id,
        workspace_id=conversation_context.workspace_id,
        recent_messages=[
            message.model_dump(mode="json")
            for message in conversation_context.recent_messages
        ],
    )


def build_message_jobs(
    *,
    clock: Any,
    conversation: dict[str, Any],
    message_id: str,
    prior_messages: list[dict[str, Any]],
    message_text: str,
    occurred_at: str | None = None,
    role: str,
    include_contract_projection: bool | None = None,
) -> list[tuple[str, JobEnvelope]]:
    """Build stream jobs for one persisted message."""
    conversation_context = ExtractionConversationContext(
        user_id=str(conversation["user_id"]),
        conversation_id=str(conversation["id"]),
        source_message_id=message_id,
        workspace_id=conversation["workspace_id"],
        assistant_mode_id=str(conversation["assistant_mode_id"]),
        recent_messages=recent_context(prior_messages),
    )
    payload = build_job_payload(
        conversation_context=conversation_context,
        message_text=message_text,
        message_occurred_at=occurred_at,
        role=role,
    ).model_dump(mode="json")
    jobs: list[tuple[str, JobEnvelope]] = [
        (
            EXTRACT_STREAM_NAME,
            JobEnvelope(
                job_id=new_job_id(),
                job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
                user_id=str(conversation["user_id"]),
                conversation_id=str(conversation["id"]),
                message_ids=[message_id],
                payload=payload,
                created_at=clock.now(),
            ),
        )
    ]
    if include_contract_projection is None:
        include_contract_projection = role == "user"
    if include_contract_projection:
        jobs.append(
            (
                CONTRACT_STREAM_NAME,
                JobEnvelope(
                    job_id=new_job_id(),
                    job_type=JobType.PROJECT_CONTRACT,
                    user_id=str(conversation["user_id"]),
                    conversation_id=str(conversation["id"]),
                    message_ids=[message_id],
                    payload=payload,
                    created_at=clock.now(),
                ),
            )
        )
    return jobs


async def enqueue_message_jobs(
    *,
    storage_backend: Any,
    jobs: list[tuple[str, JobEnvelope]],
) -> list[str]:
    """Enqueue message-derived worker jobs and return their job identifiers."""
    job_ids: list[str] = []
    for stream_name, job in jobs:
        await storage_backend.stream_add(stream_name, job.model_dump(mode="json"))
        job_ids.append(job.job_id)
    return job_ids


def summarize_selected_memories(pipeline_result: PipelineResult) -> list[dict[str, Any]]:
    """Return compact selected-memory metadata for debugging and library callers."""
    return summarize_memory_summaries(build_memory_summaries(pipeline_result))


def build_memory_summaries(pipeline_result: PipelineResult) -> list[MemorySummary]:
    """Build typed memory summaries for cache entries and library-mode results."""
    by_id = {
        candidate.memory_id: candidate
        for candidate in pipeline_result.scored_candidates
    }
    summaries: list[MemorySummary] = []
    for memory_id in pipeline_result.composed_context.selected_memory_ids:
        candidate = by_id.get(memory_id)
        if candidate is None:
            continue
        memory_object = candidate.memory_object
        canonical_text = str(memory_object.get("canonical_text", ""))
        summaries.append(
            MemorySummary(
                memory_id=memory_id,
                text=canonical_text,
                object_type=str(memory_object.get("object_type", "")),
                score=candidate.final_score,
                scope=str(memory_object.get("scope", "")),
            )
        )
    return summaries


def summarize_memory_summaries(memory_summaries: list[MemorySummary]) -> list[dict[str, Any]]:
    """Return compact selected-memory metadata from typed memory summaries."""
    return [
        {
            "memory_id": summary.memory_id,
            "score": summary.score,
            "type": summary.object_type,
            "text_preview": summary.text[:TEXT_PREVIEW_LIMIT],
        }
        for summary in memory_summaries
    ]
