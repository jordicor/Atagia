"""Memory inspection and feedback routes."""

from __future__ import annotations

from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status

from atagia.api.dependencies import (
    AuthContext,
    ensure_user_access,
    get_auth_context,
    get_clock,
    get_connection,
    get_llm_client,
    get_settings,
)
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.retrieval_event_repository import (
    MemoryFeedbackMismatchError,
    MemoryFeedbackOwnershipError,
    MemoryFeedbackRepository,
    RetrievalEventRepository,
)
from atagia.memory.contract_projection import ContractProjector
from atagia.models.schemas_api import MemoryFeedbackRequest
from atagia.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["memory"])


@router.post("/memory/feedback")
async def create_memory_feedback(
    payload: MemoryFeedbackRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    events = RetrievalEventRepository(connection, clock)
    event = await events.get_event(payload.retrieval_event_id, payload.user_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval event not found for user",
        )

    feedback_repository = MemoryFeedbackRepository(connection, clock)
    try:
        return await feedback_repository.create_feedback(
            retrieval_event_id=payload.retrieval_event_id,
            memory_id=payload.memory_id,
            user_id=payload.user_id,
            feedback_type=payload.feedback_type.value,
            score=payload.score,
            metadata=payload.metadata,
        )
    except MemoryFeedbackOwnershipError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryFeedbackMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get("/memory/objects/{memory_id}")
async def get_memory_object(
    memory_id: str,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(user_id, auth_context)
    memory = await MemoryObjectRepository(connection, clock).get_memory_object(memory_id, user_id)
    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory object not found for user",
        )
    return memory


@router.get("/users/{user_id}/contract")
async def get_user_contract(
    user_id: str,
    assistant_mode_id: str = Query(...),
    workspace_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[Any] = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, dict[str, Any]]:
    ensure_user_access(user_id, auth_context)
    projector = ContractProjector(
        llm_client=llm_client,
        clock=clock,
        message_repository=MessageRepository(connection, clock),
        memory_repository=MemoryObjectRepository(connection, clock),
        contract_repository=ContractDimensionRepository(connection, clock),
        settings=settings,
    )
    try:
        return await projector.get_current_contract(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/users/{user_id}/state")
async def get_user_state(
    user_id: str,
    assistant_mode_id: str | None = Query(default=None),
    workspace_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(user_id, auth_context)
    return await MemoryObjectRepository(connection, clock).get_state_snapshot(
        user_id,
        assistant_mode_id=assistant_mode_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )
