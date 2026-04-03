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
    get_manifests,
    get_runtime,
)
from atagia.core.clock import Clock
from atagia.core.repositories import (
    ConversationRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_api import (
    ChatReplyRequest,
    ChatReplyResponse,
    CreateConversationRequest,
    CreateWorkspaceRequest,
)
from atagia.services.chat_service import ChatService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    LLMUnavailableError,
    UnknownAssistantModeError,
)

router = APIRouter(prefix="/v1", tags=["chat"])


async def _ensure_user_exists(users: UserRepository, user_id: str) -> None:
    if await users.get_user(user_id) is None:
        await users.create_user(user_id)


@router.post("/conversations")
async def create_conversation(
    payload: CreateConversationRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    manifests: dict[str, Any] = Depends(get_manifests),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)

    if payload.assistant_mode_id not in manifests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown assistant mode: {payload.assistant_mode_id}",
        )
    await _ensure_user_exists(users, payload.user_id)
    if payload.workspace_id is not None:
        workspace = await workspaces.get_workspace(payload.workspace_id, payload.user_id)
        if workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found for user",
            )
    return await conversations.create_conversation(
        conversation_id=None,
        user_id=payload.user_id,
        workspace_id=payload.workspace_id,
        assistant_mode_id=payload.assistant_mode_id,
        title=payload.title,
        metadata=payload.metadata,
    )


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
    return await workspaces.create_workspace(
        workspace_id=None,
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
