"""Activity ranking and warm-up routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from atagia.api.dependencies import AuthContext, ensure_user_access, get_auth_context, get_runtime
from atagia.models.schemas_api import (
    ActivitySnapshotResponse,
    ConversationActivityStats,
    ConversationWarmupRequest,
    UserWarmupRequest,
    WarmupConversationResponse,
    WarmupRecommendedConversationsResponse,
)
from atagia.services.conversation_activity_service import ConversationActivityService

router = APIRouter(prefix="/v1", tags=["activity"])


def _coerce_stats(row: dict[str, Any] | None) -> ConversationActivityStats | None:
    if row is None:
        return None
    return ConversationActivityStats.model_validate(row)


@router.get("/users/{user_id}/activity/conversations", response_model=ActivitySnapshotResponse)
async def list_hot_conversations(
    user_id: str,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    workspace_id: str | None = None,
    assistant_mode_id: str | None = None,
    limit: int = Query(default=5, ge=1, le=100),
    as_of: str | None = None,
    refresh: bool = True,
) -> ActivitySnapshotResponse:
    ensure_user_access(user_id, auth_context)
    runtime = get_runtime(request)
    connection = await runtime.open_connection()
    try:
        service = ConversationActivityService(runtime)
        conversations = await service.list_hot_conversations(
            connection,
            user_id,
            limit=limit,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            as_of=as_of,
            refresh=refresh,
        )
        return ActivitySnapshotResponse(
            user_id=user_id,
            as_of=service._resolve_as_of(as_of).isoformat(),
            filters={
                "workspace_id": workspace_id,
                "assistant_mode_id": assistant_mode_id,
                "limit": limit,
                "refresh": refresh,
            },
            conversations=[ConversationActivityStats.model_validate(row) for row in conversations],
            conversation_count=len(conversations),
        )
    finally:
        await connection.close()


@router.post("/conversations/{conversation_id}/warmup", response_model=WarmupConversationResponse)
async def warmup_conversation(
    conversation_id: str,
    payload: ConversationWarmupRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> WarmupConversationResponse:
    if payload.user_id is not None:
        ensure_user_access(payload.user_id, auth_context)
    resolved_user_id = auth_context.claimed_user_id or payload.user_id
    if resolved_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required when no authenticated user claim is present",
        )
    runtime = get_runtime(request)
    connection = await runtime.open_connection()
    try:
        service = ConversationActivityService(runtime)
        result = await service.warmup_conversation(
            connection,
            resolved_user_id,
            conversation_id,
            max_messages=payload.max_messages,
            as_of=payload.as_of,
            refresh_stats=True,
        )
        if "conversation_not_found" in result.get("warmup_errors", []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for user",
            )
        return WarmupConversationResponse.model_validate(
            {
                **result,
                "stats": _coerce_stats(result.get("stats")),
            }
        )
    finally:
        await connection.close()


@router.post("/users/{user_id}/warmup", response_model=WarmupRecommendedConversationsResponse)
async def warmup_recommended_conversations(
    user_id: str,
    payload: UserWarmupRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> WarmupRecommendedConversationsResponse:
    ensure_user_access(user_id, auth_context)
    runtime = get_runtime(request)
    connection = await runtime.open_connection()
    try:
        service = ConversationActivityService(runtime)
        result = await service.warmup_recommended_conversations(
            connection,
            user_id,
            limit=payload.limit,
            workspace_id=payload.workspace_id,
            assistant_mode_id=payload.assistant_mode_id,
            as_of=payload.as_of,
            lead_time_minutes=payload.lead_time_minutes,
            total_message_budget=payload.total_message_budget,
            per_conversation_message_budget=payload.per_conversation_message_budget,
        )
        return WarmupRecommendedConversationsResponse.model_validate(
            {
                **result,
                "hot_conversations": [
                    ConversationActivityStats.model_validate(row)
                    for row in result.get("hot_conversations", [])
                ],
                "warmed_conversations": [
                    {
                        **warmup,
                        "stats": _coerce_stats(warmup.get("stats")),
                    }
                    for warmup in result.get("warmed_conversations", [])
                ],
            }
        )
    finally:
        await connection.close()
