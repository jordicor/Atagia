"""Chat and creation routes."""

from __future__ import annotations

from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request, status

from atagia.api.dependencies import (
    AuthContext,
    ensure_user_access,
    get_auth_context,
    get_clock,
    get_connection,
    get_runtime,
)
from atagia.core.clock import Clock
from atagia.core.repositories import (
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_api import (
    ChatReplyRequest,
    ChatReplyResponse,
    CreateConversationRequest,
    CreateUserRequest,
    CreateWorkspaceRequest,
    ContextResult,
    FlushRequest,
    FlushResponse,
    SidecarAddResponseRequest,
    SidecarContextRequest,
    SidecarIngestMessageRequest,
    SidecarMutationResponse,
)
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_service import ChatService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    LLMUnavailableError,
    UnknownAssistantModeError,
    WorkspaceNotFoundError,
)
from atagia.services.sidecar_service import SidecarService

router = APIRouter(prefix="/v1", tags=["chat"])


async def _ensure_user_exists(users: UserRepository, user_id: str) -> None:
    if await users.get_user(user_id) is None:
        await users.create_user(user_id)


@router.post("/users")
async def create_user(
    payload: CreateUserRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    users = UserRepository(connection, clock)
    existing = await users.get_user(payload.user_id)
    if existing is not None:
        return existing
    return await users.create_user(payload.user_id)


@router.post("/conversations")
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    sidecar = SidecarService(get_runtime(request))
    try:
        await sidecar.ensure_user_exists(connection, payload.user_id)
        return await sidecar.ensure_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            workspace_id=payload.workspace_id,
            assistant_mode_id=payload.assistant_mode_id,
            title=payload.title,
            metadata=payload.metadata,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except WorkspaceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownAssistantModeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AssistantModeMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post("/workspaces")
async def create_workspace(
    payload: CreateWorkspaceRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    await _ensure_user_exists(users, payload.user_id)
    if payload.workspace_id is not None:
        existing = await workspaces.get_workspace(payload.workspace_id, payload.user_id)
        if existing is not None:
            return existing
    return await workspaces.create_workspace(
        workspace_id=payload.workspace_id,
        user_id=payload.user_id,
        name=payload.name,
        metadata=payload.metadata,
    )


@router.post("/chat/{conversation_id}/reply", response_model=ChatReplyResponse)
async def chat_reply(
    conversation_id: str,
    payload: ChatReplyRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ChatReplyResponse:
    ensure_user_access(payload.user_id, auth_context)
    try:
        result = await ChatService(runtime=get_runtime(request)).chat_reply(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            message_text=payload.message_text,
            message_occurred_at=payload.message_occurred_at,
            include_thinking=payload.include_thinking,
            metadata=payload.metadata,
            debug=payload.debug,
            attachments=payload.attachments,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except UnknownAssistantModeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownOperationalProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalProfileNotAuthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except AssistantModeMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service unavailable",
        ) from exc

    return ChatReplyResponse(
        conversation_id=conversation_id,
        request_message_id=result.request_message_id,
        response_message_id=result.response_message_id,
        reply_text=result.response_text,
        retrieval_event_id=result.retrieval_event_id,
        debug=result.debug,
    )


@router.post("/conversations/{conversation_id}/context", response_model=ContextResult)
async def get_sidecar_context(
    conversation_id: str,
    payload: SidecarContextRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ContextResult:
    ensure_user_access(payload.user_id, auth_context)
    try:
        return await SidecarService(get_runtime(request)).get_context(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            message=payload.message_text,
            mode=payload.assistant_mode_id,
            workspace_id=payload.workspace_id,
            occurred_at=payload.message_occurred_at,
            attachments=[
                attachment.model_dump(mode="json") for attachment in payload.attachments
            ],
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except WorkspaceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownAssistantModeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownOperationalProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalProfileNotAuthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except AssistantModeMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post("/conversations/{conversation_id}/messages", response_model=SidecarMutationResponse)
async def ingest_sidecar_message(
    conversation_id: str,
    payload: SidecarIngestMessageRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SidecarMutationResponse:
    ensure_user_access(payload.user_id, auth_context)
    try:
        await SidecarService(get_runtime(request)).ingest_message(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            role=payload.role,
            text=payload.text,
            mode=payload.assistant_mode_id,
            workspace_id=payload.workspace_id,
            occurred_at=payload.occurred_at,
            attachments=[
                attachment.model_dump(mode="json") for attachment in payload.attachments
            ],
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except WorkspaceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownAssistantModeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnknownOperationalProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalProfileNotAuthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except AssistantModeMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return SidecarMutationResponse()


@router.post("/conversations/{conversation_id}/responses", response_model=SidecarMutationResponse)
async def add_sidecar_response(
    conversation_id: str,
    payload: SidecarAddResponseRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SidecarMutationResponse:
    ensure_user_access(payload.user_id, auth_context)
    try:
        await SidecarService(get_runtime(request)).add_response(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            text=payload.text,
            occurred_at=payload.occurred_at,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except UnknownOperationalProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalProfileNotAuthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return SidecarMutationResponse()


@router.post("/flush", response_model=FlushResponse)
async def flush_sidecar_work(
    payload: FlushRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> FlushResponse:
    ensure_user_access(payload.user_id, auth_context)
    runtime = get_runtime(request)
    if not runtime.settings.workers_enabled:
        return FlushResponse(completed=False)
    completed = await runtime.storage_backend.drain(payload.timeout_seconds)
    return FlushResponse(completed=completed)
