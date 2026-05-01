"""MCP server exposing Atagia memory operations as tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from atagia import Atagia
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.ids import new_job_id
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
)
from atagia.models.schemas_memory import ExtractionConversationContext, MemoryStatus
from atagia.models.schemas_replay import AblationConfig
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    build_job_payload,
    recent_context,
    resolve_operational_profile,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    UnknownAssistantModeError,
    WorkspaceNotFoundError,
)

try:
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.session import ServerSession
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "MCP support requires the 'mcp' extra. Install with: pip install 'atagia[mcp]'"
    ) from exc

DEFAULT_DB_PATH = "atagia.db"
DEFAULT_USER_ID = "default_user"
DEFAULT_TRANSPORT = "stdio"
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_LIST_LIMIT = 20


@dataclass(slots=True)
class AtagiaContext:
    """FastMCP lifespan context holding the shared Atagia engine."""

    engine: Atagia
    user_id: str


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[AtagiaContext]:
    """Initialize and close the shared Atagia engine for MCP requests."""
    engine = Atagia(db_path=os.environ.get("ATAGIA_DB_PATH", DEFAULT_DB_PATH))
    user_id = _configured_user_id()
    await engine.setup()
    try:
        yield AtagiaContext(engine=engine, user_id=user_id)
    finally:
        await engine.close()


mcp = FastMCP(
    "atagia-memory",
    instructions=(
        "Atagia is a memory engine for AI assistants. Use these tools to store and "
        "retrieve memories that persist across conversations."
    ),
    lifespan=lifespan,
)


def _configured_user_id() -> str:
    """Return the single-user MCP identity."""
    return os.environ.get("ATAGIA_USER_ID", DEFAULT_USER_ID).strip() or DEFAULT_USER_ID


async def _runtime(engine: Atagia):
    runtime = engine.runtime
    if runtime is None:
        raise RuntimeError("Atagia engine runtime is not initialized")
    return runtime


async def _ensure_conversation_id(
    engine: Atagia,
    user_id: str,
    conversation_id: str | None,
    mode: str | None = None,
) -> str:
    """Create or reuse a conversation and return its identifier."""
    return await engine.create_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
        assistant_mode_id=mode,
    )


async def _get_context_impl(
    engine: Atagia,
    user_id: str,
    message: str,
    conversation_id: str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
) -> str:
    """Retrieve relevant memories for a message as a JSON string."""
    resolved_conversation_id = await _ensure_conversation_id(
        engine,
        user_id,
        conversation_id,
        mode,
    )
    context = await engine.get_context(
        user_id=user_id,
        conversation_id=resolved_conversation_id,
        message=message,
        mode=mode,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
        ablation=AblationConfig(disable_context_cache=True),
    )
    return json.dumps(
        {
            "conversation_id": resolved_conversation_id,
            "system_prompt": context.system_prompt,
            "recent_transcript": [
                entry.model_dump(mode="json") for entry in context.recent_transcript
            ],
            "recent_transcript_omissions": [
                omission.model_dump(mode="json")
                for omission in context.recent_transcript_omissions
            ],
            "recent_transcript_trace": (
                None
                if context.recent_transcript_trace is None
                else context.recent_transcript_trace.model_dump(mode="json")
            ),
            "assistant_guidance": context.assistant_guidance,
            "memories": [memory.model_dump(mode="json") for memory in context.memories],
            "contract": context.contract,
            "detected_needs": context.detected_needs,
            "stage_timings": context.stage_timings,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


async def _add_memory_impl(
    engine: Atagia,
    user_id: str,
    message: str,
    conversation_id: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
) -> str:
    """Store a user message and enqueue extraction jobs."""
    resolved_conversation_id = await _ensure_conversation_id(engine, user_id, conversation_id)
    runtime = await _runtime(engine)
    resolved_operational_profile = resolve_operational_profile(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
    )
    cache_service = ContextCacheService(runtime)
    async with cache_service.user_cache_guard(user_id):
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            conversation = await conversations.get_conversation(resolved_conversation_id, user_id)
            if conversation is None:
                raise ValueError("Conversation not found for user")
            prior_messages = await messages.get_recent_messages(
                resolved_conversation_id,
                user_id,
                limit=RECENT_FETCH_LIMIT,
            )
            await connection.execute("BEGIN")
            try:
                message_occurred_at = runtime.clock.now().isoformat()
                user_message = await messages.create_message(
                    message_id=None,
                    conversation_id=resolved_conversation_id,
                    role="user",
                    seq=None,
                    text=message,
                    token_count=None,
                    metadata={"source": "mcp_add_memory"},
                    occurred_at=message_occurred_at,
                    commit=False,
                )
                payload = build_job_payload(
                    conversation_context=_conversation_context(conversation, user_message["id"], prior_messages),
                    message_text=message,
                    message_occurred_at=resolve_message_occurred_at(user_message),
                    role="user",
                ).model_dump(mode="json")
                await cache_service.invalidate_conversation_cache_for_conversation(conversation)
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        finally:
            await connection.close()

        extract_job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            message_ids=[str(user_message["id"])],
            payload=payload,
            created_at=runtime.clock.now(),
            operational_profile=resolved_operational_profile.snapshot,
        )
        contract_job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.PROJECT_CONTRACT,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            message_ids=[str(user_message["id"])],
            payload=payload,
            created_at=runtime.clock.now(),
            operational_profile=resolved_operational_profile.snapshot,
        )
        await runtime.storage_backend.stream_add(
            EXTRACT_STREAM_NAME,
            extract_job.model_dump(mode="json"),
        )
        await runtime.storage_backend.stream_add(
            CONTRACT_STREAM_NAME,
            contract_job.model_dump(mode="json"),
        )
    return (
        f"Stored memory candidate message {user_message['id']} "
        f"in conversation {resolved_conversation_id}."
    )


async def _search_memories_impl(
    engine: Atagia,
    user_id: str,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> str:
    """Search memories via FTS and return a JSON array string."""
    runtime = await _runtime(engine)
    connection = await runtime.open_connection()
    try:
        rows = await MemoryObjectRepository(connection, runtime.clock).search_memory_objects(
            user_id,
            query,
            limit,
        )
    finally:
        await connection.close()
    filtered = [
        {
            "id": str(row["id"]),
            "text": str(row["canonical_text"]),
            "type": str(row["object_type"]),
            "scope": str(row["scope"]),
            "score": float(row.get("rank", 0.0)),
            "status": str(row.get("status", "")),
        }
        for row in rows
    ]
    return json.dumps(filtered, ensure_ascii=False, sort_keys=True)


async def _list_memories_impl(
    engine: Atagia,
    user_id: str,
    memory_type: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> str:
    """List stored memories as a JSON array string."""
    runtime = await _runtime(engine)
    connection = await runtime.open_connection()
    try:
        rows = await MemoryObjectRepository(connection, runtime.clock).list_for_user(
            user_id,
            # ARCHIVED includes PVG audit-only mirrors. They carry
            # audit_only_mirror=true and their canonical_text is post-refine safe.
            statuses=(MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED),
        )
    finally:
        await connection.close()
    normalized_type = memory_type.strip() if memory_type is not None else None
    filtered_rows = [
        row
        for row in rows
        if normalized_type is None or str(row.get("object_type")) == normalized_type
    ]
    filtered_rows.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))), reverse=True)
    payload = [
        {
            "id": str(row["id"]),
            "text": str(row["canonical_text"]),
            "type": str(row["object_type"]),
            "scope": str(row["scope"]),
            "status": str(row["status"]),
        }
        for row in filtered_rows[:limit]
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _delete_memory_impl(engine: Atagia, user_id: str, memory_id: str) -> str:
    """Archive a memory object and return a confirmation string."""
    runtime = await _runtime(engine)
    cache_service = ContextCacheService(runtime)
    async with cache_service.user_cache_guard(user_id):
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            await connection.execute("BEGIN")
            try:
                if not await memories.archive_memory_object(memory_id, user_id, commit=False):
                    raise ValueError(f"Memory not found for user: {memory_id}")
                await cache_service.invalidate_user_cache(user_id)
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        finally:
            await connection.close()
    try:
        await runtime.embedding_index.delete(memory_id)
    except Exception:
        logger.warning("Embedding cleanup failed for memory_id=%s", memory_id, exc_info=True)
    return f"Archived memory {memory_id}."


def _conversation_context(
    conversation: dict[str, Any],
    source_message_id: str,
    prior_messages: list[dict[str, Any]],
) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id=str(conversation["user_id"]),
        conversation_id=str(conversation["id"]),
        source_message_id=str(source_message_id),
        workspace_id=conversation.get("workspace_id"),
        assistant_mode_id=str(conversation["assistant_mode_id"]),
        recent_messages=recent_context(prior_messages),
    )


def _tool_error(exc: Exception) -> str:
    """Render a stable tool error message."""
    return f"Error: {exc}"


_EXPECTED_TOOL_ERRORS = (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    KeyError,
    OperationalProfileNotAuthorizedError,
    UnknownAssistantModeError,
    UnknownOperationalProfileError,
    ValueError,
    WorkspaceNotFoundError,
)


@mcp.tool()
async def atagia_get_context(
    message: str,
    conversation_id: str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Retrieve relevant memories for a message. Returns enriched context."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _get_context_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            message,
            conversation_id=conversation_id,
            mode=mode,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_add_memory(
    message: str,
    conversation_id: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Store a message and trigger background extraction."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _add_memory_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            message,
            conversation_id=conversation_id,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_search_memories(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Search stored memories with FTS and return matching memory summaries."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _search_memories_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            query,
            limit=limit,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_list_memories(
    memory_type: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """List stored memories, optionally filtered by memory type."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _list_memories_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            memory_type=memory_type,
            limit=limit,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_delete_memory(
    memory_id: str,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Archive a memory object without hard deleting it."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _delete_memory_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            memory_id,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


def main() -> None:
    """Run the MCP server with the configured transport."""
    transport = os.environ.get("ATAGIA_MCP_TRANSPORT", DEFAULT_TRANSPORT)
    mcp.run(transport=transport)


__all__ = [
    "AtagiaContext",
    "lifespan",
    "main",
    "mcp",
]
