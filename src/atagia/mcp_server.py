"""MCP server exposing Atagia memory operations as tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
from typing import Any

from atagia import Atagia
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    conversation_visibility_clause,
)
from atagia.core.ids import new_job_id
from atagia.core.runtime_safety import wait_for_in_memory_worker_quiescence
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
)
from atagia.models.schemas_memory import ExtractionConversationContext, MemoryScope, MemoryStatus
from atagia.models.schemas_replay import AblationConfig
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    build_job_payload,
    enqueue_message_jobs,
    recent_context,
    resolve_operational_profile,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationAlreadyClosedError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    DeletionConfirmationError,
    InvalidConversationTransitionError,
    MemoryNotEditableError,
    MemoryNotFoundError,
    UnknownAssistantModeError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.lifecycle_service import (
    ConversationLifecycleService,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.worker_control_service import WorkerControlService

try:
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.session import ServerSession
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "MCP support requires the 'mcp' extra. Install with: pip install 'atagia[mcp]'"
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "atagia.db"
DEFAULT_TRANSPORT = "stdio"
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_LIST_LIMIT = 20


@dataclass(slots=True)
class AtagiaContext:
    """FastMCP lifespan context holding the shared Atagia engine."""

    engine: Atagia
    user_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    conversation_id: str | None = None
    incognito: bool = False


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[AtagiaContext]:
    """Initialize and close the shared Atagia engine for MCP requests."""
    engine = Atagia(db_path=os.environ.get("ATAGIA_DB_PATH", DEFAULT_DB_PATH))
    user_id = _required_env_text("ATAGIA_USER_ID")
    platform_id = _required_env_text("ATAGIA_PLATFORM_ID")
    user_persona_id = _optional_env_text("ATAGIA_USER_PERSONA_ID")
    character_id = _optional_env_text("ATAGIA_CHARACTER_ID")
    conversation_id = _optional_env_text("ATAGIA_CONVERSATION_ID")
    incognito = _env_bool("ATAGIA_INCOGNITO", default=False)
    await engine.setup()
    try:
        yield AtagiaContext(
            engine=engine,
            user_id=user_id,
            platform_id=platform_id,
            user_persona_id=user_persona_id,
            character_id=character_id,
            conversation_id=conversation_id,
            incognito=incognito,
        )
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


def _required_env_text(name: str) -> str:
    """Return a required nonblank MCP identity value."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} is required for the Atagia MCP server")
    return value.strip()


def _optional_env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


async def _runtime(engine: Atagia):
    runtime = engine.runtime
    if runtime is None:
        raise RuntimeError("Atagia engine runtime is not initialized")
    return runtime


async def _mcp_namespace_kwargs(
    engine: Atagia,
    user_id: str,
    *,
    platform_id: str,
    conversation_id: str | None,
    user_persona_id: str | None,
    character_id: str | None,
    incognito: bool,
    require_active: bool = True,
) -> dict[str, Any]:
    if conversation_id is None:
        raise ValueError(
            "conversation_id is required for MCP namespace mutations; pass it or set "
            "ATAGIA_CONVERSATION_ID"
        )
    runtime = await _runtime(engine)
    connection = await runtime.open_connection()
    try:
        conversation = await ConversationRepository(
            connection,
            runtime.clock,
        ).get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ConversationNotFoundError("Conversation not found for user")
        if require_active and str(conversation.get("status")) != "active":
            raise ConversationNotFoundError("Conversation not found for namespace")
        if (
            conversation.get("platform_id") != platform_id
            or conversation.get("user_persona_id") != user_persona_id
            or conversation.get("character_id") != character_id
            or bool(conversation.get("incognito")) != bool(incognito)
        ):
            raise ConversationNotFoundError("Conversation not found for namespace")
        preferences = await UserRepository(connection, runtime.clock).get_memory_preferences(user_id)
        if preferences is None:
            raise ValueError("User not found")
        return {
            "conversation_id": conversation_id,
            "user_persona_id": user_persona_id,
            "platform_id": platform_id,
            "character_id": character_id,
            "incognito": incognito,
            "remember_across_chats": bool(preferences["remember_across_chats"]),
            "remember_across_devices": bool(preferences["remember_across_devices"]),
        }
    finally:
        await connection.close()


async def _ensure_conversation_id(
    engine: Atagia,
    user_id: str,
    platform_id: str,
    conversation_id: str | None,
    default_conversation_id: str | None = None,
    mode: str | None = None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """Create or reuse a conversation and return its identifier."""
    resolved_conversation_id = conversation_id or default_conversation_id
    if resolved_conversation_id is None:
        raise ValueError(
            "conversation_id is required for MCP tools; pass it or set "
            "ATAGIA_CONVERSATION_ID"
        )
    return await engine.create_conversation(
        user_id=user_id,
        conversation_id=resolved_conversation_id,
        assistant_mode_id=mode,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        mode=mode,
        incognito=incognito,
    )


async def _get_context_impl(
    engine: Atagia,
    user_id: str,
    platform_id: str,
    message: str,
    conversation_id: str | None = None,
    default_conversation_id: str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """Retrieve relevant memories for a message as a JSON string."""
    resolved_conversation_id = await _ensure_conversation_id(
        engine,
        user_id,
        platform_id,
        conversation_id,
        default_conversation_id=default_conversation_id,
        mode=mode,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    context = await engine.get_context(
        user_id=user_id,
        conversation_id=resolved_conversation_id,
        message=message,
        mode=mode,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
        incognito=incognito,
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
            "memory_processing": (
                None
                if context.memory_processing is None
                else context.memory_processing.model_dump(mode="json")
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


async def _add_memory_impl(
    engine: Atagia,
    user_id: str,
    platform_id: str,
    message: str,
    conversation_id: str | None = None,
    default_conversation_id: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """Store a user message and enqueue extraction jobs."""
    resolved_conversation_id = await _ensure_conversation_id(
        engine,
        user_id,
        platform_id,
        conversation_id,
        default_conversation_id=default_conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
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
            users = UserRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            conversation = await conversations.get_conversation(resolved_conversation_id, user_id)
            if conversation is None:
                raise ValueError("Conversation not found for user")
            memory_preferences = await users.get_memory_preferences(user_id)
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
                    conversation_context=_conversation_context(
                        conversation,
                        user_message["id"],
                        prior_messages,
                        memory_preferences=memory_preferences,
                    ),
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
        tracking_connection = await runtime.open_connection()
        try:
            job_tracking = JobTrackingService(
                tracking_connection,
                runtime.clock,
                workers_enabled=runtime.settings.workers_enabled,
                settings=runtime.settings,
            )
            await enqueue_message_jobs(
                storage_backend=runtime.storage_backend,
                jobs=[
                    (EXTRACT_STREAM_NAME, extract_job),
                    (CONTRACT_STREAM_NAME, contract_job),
                ],
                job_tracking_service=job_tracking,
                worker_control_service=WorkerControlService(
                    tracking_connection,
                    runtime.clock,
                ),
            )
        finally:
            await tracking_connection.close()
    return (
        f"Stored memory candidate message {user_message['id']} "
        f"in conversation {resolved_conversation_id}."
    )


async def _processing_status_impl(
    engine: Atagia,
    user_id: str,
    platform_id: str,
    conversation_id: str | None = None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
    remember_across_chats: bool = True,
    remember_across_devices: bool = True,
) -> str:
    """Return current background memory-processing status as JSON."""
    status = await engine.get_processing_status(
        user_id=user_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
        incognito=incognito,
        remember_across_chats=remember_across_chats,
        remember_across_devices=remember_across_devices,
    )
    return status.model_dump_json()


async def _search_memories_impl(
    engine: Atagia,
    user_id: str,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    *,
    conversation_id: str | None = None,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """Search memories via FTS and return a JSON array string."""
    runtime = await _runtime(engine)
    namespace_kwargs = await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    connection = await runtime.open_connection()
    try:
        rows = await _search_visible_mcp_memories(
            connection,
            user_id=user_id,
            query=query,
            limit=limit,
            **namespace_kwargs,
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
    *,
    conversation_id: str | None = None,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """List stored memories as a JSON array string."""
    runtime = await _runtime(engine)
    namespace_kwargs = await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
        require_active=False,
    )
    connection = await runtime.open_connection()
    try:
        rows = await _list_visible_mcp_memories(
            connection,
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
            **namespace_kwargs,
        )
    finally:
        await connection.close()
    payload = [
        {
            "id": str(row["id"]),
            "text": str(row["canonical_text"]),
            "type": str(row["object_type"]),
            "scope": str(row["scope"]),
            "status": str(row["status"]),
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _list_visible_mcp_memories(
    connection: Any,
    *,
    user_id: str,
    memory_type: str | None,
    limit: int,
    conversation_id: str,
    user_persona_id: str | None,
    platform_id: str,
    character_id: str | None,
    incognito: bool,
    remember_across_chats: bool,
    remember_across_devices: bool,
) -> list[dict[str, Any]]:
    visibility_clauses, visibility_parameters = MemoryObjectRepository.namespace_visibility_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
        conversation_id=conversation_id,
        remember_across_chats=remember_across_chats,
        remember_across_devices=remember_across_devices,
        incognito=incognito,
        sensitivity_gates_enabled=False,
        table_alias="mo",
    )
    if not visibility_clauses:
        return []
    clauses = [
        "mo.user_id = ?",
        "mo.status IN (?, ?)",
        "mo.archived_by_conversation_id IS NULL",
        conversation_visibility_clause("mo"),
        *visibility_clauses,
    ]
    parameters: list[Any] = [
        user_id,
        MemoryStatus.ACTIVE.value,
        MemoryStatus.ARCHIVED.value,
        conversation_id,
        *visibility_parameters,
    ]
    normalized_type = memory_type.strip() if memory_type is not None else None
    if normalized_type:
        clauses.append("mo.object_type = ?")
        parameters.append(normalized_type)
    parameters.append(max(1, min(500, int(limit))))
    cursor = await connection.execute(
        """
        SELECT mo.*
        FROM memory_objects AS mo
        WHERE {clauses}
        ORDER BY mo.created_at DESC, mo.id DESC
        LIMIT ?
        """.format(clauses=" AND ".join(clauses)),
        tuple(parameters),
    )
    return [dict(row) for row in await cursor.fetchall()]


async def _search_visible_mcp_memories(
    connection: Any,
    *,
    user_id: str,
    query: str,
    limit: int,
    conversation_id: str,
    user_persona_id: str | None,
    platform_id: str,
    character_id: str | None,
    incognito: bool,
    remember_across_chats: bool,
    remember_across_devices: bool,
) -> list[dict[str, Any]]:
    visibility_clauses, visibility_parameters = MemoryObjectRepository.namespace_visibility_clauses(
        [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
        conversation_id=conversation_id,
        remember_across_chats=remember_across_chats,
        remember_across_devices=remember_across_devices,
        incognito=incognito,
        sensitivity_gates_enabled=False,
        table_alias="mo",
    )
    if not visibility_clauses:
        return []
    clauses = [
        "mo.user_id = ?",
        "memory_objects_fts MATCH ?",
        "mo.status = ?",
        "mo.archived_by_conversation_id IS NULL",
        conversation_visibility_clause("mo"),
        *visibility_clauses,
    ]
    parameters: list[Any] = [
        user_id,
        query,
        MemoryStatus.ACTIVE.value,
        conversation_id,
        *visibility_parameters,
        max(1, min(100, int(limit))),
    ]
    cursor = await connection.execute(
        """
        SELECT
            mo.*,
            bm25(memory_objects_fts) AS rank
        FROM memory_objects_fts
        JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
        WHERE {clauses}
        ORDER BY rank ASC, mo.created_at DESC
        LIMIT ?
        """.format(clauses=" AND ".join(clauses)),
        tuple(parameters),
    )
    return [dict(row) for row in await cursor.fetchall()]


async def _delete_memory_impl(
    engine: Atagia,
    user_id: str,
    memory_id: str,
    *,
    hard: bool = False,
    confirmation: str | None = None,
    conversation_id: str | None = None,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    """Archive or hard-delete a memory object and return a confirmation string."""
    runtime = await _runtime(engine)
    namespace_kwargs = await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    cache_service = ContextCacheService(runtime)
    async with cache_service.user_cache_guard(user_id):
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            await ConversationLifecycleService(runtime).delete_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                hard=hard,
                confirmation=confirmation,
                **namespace_kwargs,
            )
        finally:
            await connection.close()
    action = "Hard-deleted" if hard else "Archived"
    return f"{action} memory {memory_id}."


async def _edit_memory_impl(
    engine: Atagia,
    user_id: str,
    memory_id: str,
    canonical_text: str,
    *,
    conversation_id: str | None = None,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    runtime = await _runtime(engine)
    namespace_kwargs = await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    cache_service = ContextCacheService(runtime)
    async with cache_service.user_cache_guard(user_id):
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            memory = await ConversationLifecycleService(runtime).edit_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                new_text=canonical_text,
                edit_source="mcp",
                **namespace_kwargs,
            )
        finally:
            await connection.close()
    return json.dumps(
        {"id": memory["id"], "canonical_text": memory["canonical_text"]},
        ensure_ascii=False,
        sort_keys=True,
    )


async def _close_conversation_impl(
    engine: Atagia,
    user_id: str,
    conversation_id: str,
    *,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    runtime = await _runtime(engine)
    await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    connection = await runtime.open_connection()
    try:
        result = await ConversationLifecycleService(runtime).close_conversation(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    finally:
        await connection.close()
    return json.dumps(
        result.model_dump(mode="json") if hasattr(result, "model_dump") else result,
        ensure_ascii=False,
        sort_keys=True,
    )


async def _archive_conversation_impl(
    engine: Atagia,
    user_id: str,
    conversation_id: str,
    *,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
) -> str:
    runtime = await _runtime(engine)
    await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
        require_active=False,
    )
    connection = await runtime.open_connection()
    try:
        result = await ConversationLifecycleService(runtime).archive_conversation(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    finally:
        await connection.close()
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


async def _delete_conversation_impl(
    engine: Atagia,
    user_id: str,
    conversation_id: str,
    *,
    platform_id: str,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
    confirmation: str | None = None,
) -> str:
    runtime = await _runtime(engine)
    await _mcp_namespace_kwargs(
        engine,
        user_id,
        platform_id=platform_id,
        conversation_id=conversation_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
        require_active=False,
    )
    cache_service = ContextCacheService(runtime)
    async with cache_service.user_cache_guard(user_id):
        await wait_for_in_memory_worker_quiescence(runtime)
        connection = await runtime.open_connection()
        try:
            report = await ConversationLifecycleService(runtime).delete_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                confirmation=confirmation,
            )
        finally:
            await connection.close()
    return json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _conversation_context(
    conversation: dict[str, Any],
    source_message_id: str,
    prior_messages: list[dict[str, Any]],
    *,
    memory_preferences: dict[str, Any] | None = None,
) -> ExtractionConversationContext:
    preferences = memory_preferences or {}
    return ExtractionConversationContext(
        user_id=str(conversation["user_id"]),
        conversation_id=str(conversation["id"]),
        source_message_id=str(source_message_id),
        workspace_id=conversation.get("workspace_id"),
        assistant_mode_id=str(conversation["assistant_mode_id"]),
        user_persona_id=conversation.get("user_persona_id"),
        platform_id=str(conversation.get("platform_id") or "default"),
        character_id=(
            conversation.get("character_id")
            if conversation.get("character_id") is not None
            else conversation.get("workspace_id")
        ),
        mode=str(conversation.get("mode") or conversation["assistant_mode_id"]),
        recent_messages=recent_context(prior_messages),
        temporary=bool(conversation.get("temporary")),
        temporary_ttl_seconds=conversation.get("temporary_ttl_seconds"),
        purge_on_close=bool(conversation.get("purge_on_close")),
        isolated_mode=bool(conversation.get("isolated_mode")),
        incognito=bool(conversation.get("incognito")) or bool(conversation.get("isolated_mode")),
        remember_across_chats=bool(preferences.get("remember_across_chats", True)),
        remember_across_devices=bool(preferences.get("remember_across_devices", True)),
        memory_privacy_mode=str(preferences.get("memory_privacy_mode") or "balanced"),
    )


def _tool_error(exc: Exception) -> str:
    """Render a stable tool error message."""
    return f"Error: {exc}"


_EXPECTED_TOOL_ERRORS = (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    ConversationNotActiveError,
    ConversationAlreadyClosedError,
    DeletionConfirmationError,
    InvalidConversationTransitionError,
    KeyError,
    MemoryNotEditableError,
    MemoryNotFoundError,
    OperationalProfileNotAuthorizedError,
    UnknownAssistantModeError,
    UnknownOperationalProfileError,
    ValueError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)


@mcp.tool()
async def atagia_get_context(
    message: str,
    conversation_id: str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    incognito: bool = False,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Retrieve relevant memories for a message. Returns enriched context."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _get_context_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            ctx.request_context.lifespan_context.platform_id,
            message,
            conversation_id=conversation_id,
            default_conversation_id=ctx.request_context.lifespan_context.conversation_id,
            mode=mode,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=incognito or ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_add_memory(
    message: str,
    conversation_id: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    incognito: bool = False,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Store a message and trigger background extraction."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _add_memory_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            ctx.request_context.lifespan_context.platform_id,
            message,
            conversation_id=conversation_id,
            default_conversation_id=ctx.request_context.lifespan_context.conversation_id,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=incognito or ctx.request_context.lifespan_context.incognito,
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
            conversation_id=ctx.request_context.lifespan_context.conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_processing_status(
    conversation_id: str | None = None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool = False,
    remember_across_chats: bool = True,
    remember_across_devices: bool = True,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Return current background memory-processing status."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _processing_status_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            ctx.request_context.lifespan_context.platform_id,
            conversation_id=conversation_id,
            user_persona_id=(
                user_persona_id
                if user_persona_id is not None
                else ctx.request_context.lifespan_context.user_persona_id
            ),
            character_id=(
                character_id
                if character_id is not None
                else ctx.request_context.lifespan_context.character_id
            ),
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
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
            conversation_id=ctx.request_context.lifespan_context.conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_delete_memory(
    memory_id: str,
    hard: bool = False,
    confirmation: str | None = None,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Archive a memory object, or hard-delete it when hard=true."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _delete_memory_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            memory_id,
            hard=hard,
            confirmation=confirmation,
            conversation_id=ctx.request_context.lifespan_context.conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_edit_memory(
    memory_id: str,
    canonical_text: str,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Edit the text of an active evidence memory."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _edit_memory_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            memory_id,
            canonical_text,
            conversation_id=ctx.request_context.lifespan_context.conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_close_conversation(
    conversation_id: str,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Close a conversation; temporary purge-on-close conversations are purged."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _close_conversation_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_archive_conversation(
    conversation_id: str,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Archive a non-purge-on-close conversation and archive its conversation memories."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _archive_conversation_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
        )
    except _EXPECTED_TOOL_ERRORS as exc:
        return _tool_error(exc)


@mcp.tool()
async def atagia_delete_conversation(
    conversation_id: str,
    confirmation: str | None = None,
    ctx: Context[ServerSession, AtagiaContext] | None = None,
) -> str:
    """Hard-delete a conversation and conversation-scoped data."""
    try:
        if ctx is None:
            raise RuntimeError("MCP context is required")
        return await _delete_conversation_impl(
            ctx.request_context.lifespan_context.engine,
            ctx.request_context.lifespan_context.user_id,
            conversation_id,
            platform_id=ctx.request_context.lifespan_context.platform_id,
            user_persona_id=ctx.request_context.lifespan_context.user_persona_id,
            character_id=ctx.request_context.lifespan_context.character_id,
            incognito=ctx.request_context.lifespan_context.incognito,
            confirmation=confirmation,
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
