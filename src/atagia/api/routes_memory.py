"""Memory inspection and feedback routes."""

from __future__ import annotations

from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from atagia.api.dependencies import (
    AuthContext,
    ensure_user_access,
    get_auth_context,
    get_clock,
    get_connection,
    get_llm_client,
    get_runtime,
    get_settings,
)
from atagia.api.namespace_context import (
    RouteNamespaceContext,
    require_current_route_namespace_snapshot,
    require_route_namespace_context,
)
from atagia.api.path_ids import TransportIdRoute
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
from atagia.core.transcript_rebuild_repository import (
    TranscriptRebuildRepository,
    UserAvailabilitySnapshot,
)
from atagia.memory.contract_projection import ContractProjector
from atagia.models.schemas_api import (
    DeleteMemoryRequest,
    DeletionReport,
    EditMemoryRequest,
    MemoryFeedbackRequest,
)
from atagia.services.errors import (
    DeletionConfirmationError,
    MemoryNotEditableError,
    MemoryNotFoundError,
    MemoryProvenanceRepairRequiredError,
    ConversationNotFoundError,
)
from atagia.services.lifecycle_service import ConversationLifecycleService
from atagia.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["memory"], route_class=TransportIdRoute)


async def _require_current_read_authority(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    user_id: str,
    availability: UserAvailabilitySnapshot,
    namespace: RouteNamespaceContext,
) -> None:
    """Validate read authority in one final write-serialized SQLite view."""

    await connection.execute("BEGIN IMMEDIATE")
    try:
        await TranscriptRebuildRepository(
            connection, clock
        ).require_user_availability_snapshot(user_id, availability)
        await require_current_route_namespace_snapshot(
            connection,
            clock,
            namespace.authorization_snapshot,
        )
        await connection.commit()
    except BaseException:
        await connection.rollback()
        raise


def _public_memory_object(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        key: memory.get(key)
        for key in (
            "id",
            "user_id",
            "object_type",
            "scope",
            "scope_canonical",
            "canonical_text",
            "source_kind",
            "confidence",
            "stability",
            "vitality",
            "maya_score",
            "privacy_level",
            "memory_category",
            "sensitivity",
            "status",
            "valid_from",
            "valid_to",
            "created_at",
            "updated_at",
            "user_persona_id",
            "platform_id",
            "character_id",
            "conversation_id",
        )
        if key in memory
    }


@router.post("/memory/feedback")
async def create_memory_feedback(
    payload: MemoryFeedbackRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    await connection.execute("BEGIN IMMEDIATE")
    try:
        await TranscriptRebuildRepository(connection, clock).require_user_available(
            payload.user_id
        )
        namespace = await require_route_namespace_context(
            connection,
            clock,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            platform_id=payload.platform_id,
            user_persona_id=payload.user_persona_id,
            character_id=payload.character_id,
            incognito=payload.incognito,
        )
        events = RetrievalEventRepository(connection, clock)
        event = await events.get_event(payload.retrieval_event_id, payload.user_id)
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Retrieval event not found for user",
            )

        feedback = await MemoryFeedbackRepository(connection, clock).create_feedback(
            retrieval_event_id=payload.retrieval_event_id,
            memory_id=payload.memory_id,
            user_id=payload.user_id,
            feedback_type=payload.feedback_type.value,
            score=payload.score,
            metadata=payload.metadata,
            conversation_id=namespace.conversation_id,
            user_persona_id=namespace.user_persona_id,
            platform_id=namespace.platform_id,
            character_id=namespace.character_id,
            incognito=namespace.incognito,
            remember_across_chats=namespace.remember_across_chats,
            remember_across_devices=namespace.remember_across_devices,
            active_mind_id=namespace.active_mind_id,
            mind_topology=namespace.mind_topology,
            active_embodiment_id=namespace.active_embodiment_id,
            active_realm_id=namespace.active_realm_id,
            mode=namespace.mode or namespace.assistant_mode_id,
            commit=False,
        )
        await connection.commit()
        return feedback
    except MemoryFeedbackOwnershipError as exc:
        await connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryFeedbackMismatchError as exc:
        await connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except BaseException:
        await connection.rollback()
        raise


@router.get("/memory/objects/{memory_id}")
async def get_memory_object(
    memory_id: str,
    user_id: str = Query(...),
    conversation_id: str = Query(...),
    platform_id: str = Query(...),
    user_persona_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    incognito: bool | None = Query(default=None),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(user_id, auth_context)
    rebuilds = TranscriptRebuildRepository(connection, clock)
    availability = await rebuilds.capture_user_availability_snapshot(user_id)
    namespace = await require_route_namespace_context(
        connection,
        clock,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    memory = await MemoryObjectRepository(connection, clock).get_visible_memory_object(
        memory_id,
        user_id,
        **namespace.memory_kwargs(
            include_space=True,
            include_mind=True,
            include_embodiment=True,
            include_realm=True,
        ),
    )
    response = None if memory is None else _public_memory_object(memory)
    await _require_current_read_authority(
        connection,
        clock,
        user_id=user_id,
        availability=availability,
        namespace=namespace,
    )
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory object not found for user",
        )
    return response


@router.patch("/memories/{memory_id}")
async def edit_memory_object(
    memory_id: str,
    payload: EditMemoryRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    await TranscriptRebuildRepository(connection, clock).require_user_available(
        payload.user_id
    )
    namespace = await require_route_namespace_context(
        connection,
        clock,
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        platform_id=payload.platform_id,
        user_persona_id=payload.user_persona_id,
        character_id=payload.character_id,
        incognito=payload.incognito,
    )
    try:
        edited = await ConversationLifecycleService(get_runtime(request)).edit_memory(
            connection,
            user_id=payload.user_id,
            memory_id=memory_id,
            new_text=payload.canonical_text,
            edit_source="api",
            **namespace.memory_kwargs(
                include_space=True,
                include_mind=True,
                include_embodiment=True,
                include_realm=True,
            ),
            namespace_guard=namespace.authorization_snapshot,
        )
        return _public_memory_object(edited)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except MemoryNotEditableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except MemoryProvenanceRepairRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/memories/{memory_id}/delete", response_model=DeletionReport)
async def delete_memory_object(
    memory_id: str,
    payload: DeleteMemoryRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> DeletionReport:
    ensure_user_access(payload.user_id, auth_context)
    await TranscriptRebuildRepository(connection, clock).require_user_available(
        payload.user_id
    )
    runtime = get_runtime(request)
    namespace = await require_route_namespace_context(
        connection,
        clock,
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        platform_id=payload.platform_id,
        user_persona_id=payload.user_persona_id,
        character_id=payload.character_id,
        incognito=payload.incognito,
    )
    try:
        return await ConversationLifecycleService(runtime).delete_memory(
            connection,
            user_id=payload.user_id,
            memory_id=memory_id,
            hard=payload.hard,
            confirmation=payload.confirmation,
            **namespace.memory_kwargs(
                include_space=True,
                include_mind=True,
                include_embodiment=True,
                include_realm=True,
            ),
            namespace_guard=namespace.authorization_snapshot,
        )
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except MemoryProvenanceRepairRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except DeletionConfirmationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get("/users/{user_id}/contract")
async def get_user_contract(
    user_id: str,
    conversation_id: str = Query(...),
    platform_id: str = Query(...),
    user_persona_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    incognito: bool | None = Query(default=None),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    llm_client: LLMClient[Any] = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, dict[str, Any]]:
    ensure_user_access(user_id, auth_context)
    rebuilds = TranscriptRebuildRepository(connection, clock)
    availability = await rebuilds.capture_user_availability_snapshot(user_id)
    namespace = await require_route_namespace_context(
        connection,
        clock,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    projector = ContractProjector(
        llm_client=llm_client,
        clock=clock,
        message_repository=MessageRepository(connection, clock),
        memory_repository=MemoryObjectRepository(connection, clock),
        contract_repository=ContractDimensionRepository(connection, clock),
        settings=settings,
    )
    missing: KeyError | None = None
    try:
        contract = await projector.get_current_contract(
            user_id=user_id,
            assistant_mode_id=namespace.assistant_mode_id
            or namespace.mode
            or "general_qa",
            workspace_id=namespace.workspace_id,
            conversation_id=namespace.conversation_id,
            user_persona_id=namespace.user_persona_id,
            platform_id=namespace.platform_id,
            character_id=namespace.character_id,
            incognito=namespace.incognito,
            remember_across_chats=namespace.remember_across_chats,
            remember_across_devices=namespace.remember_across_devices,
            active_space_id=namespace.active_space_id,
            active_space_boundary_mode=namespace.active_space_boundary_mode,
            active_mind_id=namespace.active_mind_id,
            mind_topology=namespace.mind_topology,
            active_embodiment_id=namespace.active_embodiment_id,
            active_realm_id=namespace.active_realm_id,
        )
    except KeyError as exc:
        missing = exc
        contract = {}
    await _require_current_read_authority(
        connection,
        clock,
        user_id=user_id,
        availability=availability,
        namespace=namespace,
    )
    if missing is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(missing),
        ) from missing
    return contract


@router.get("/users/{user_id}/state")
async def get_user_state(
    user_id: str,
    conversation_id: str = Query(...),
    platform_id: str = Query(...),
    user_persona_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    incognito: bool | None = Query(default=None),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> dict[str, Any]:
    ensure_user_access(user_id, auth_context)
    rebuilds = TranscriptRebuildRepository(connection, clock)
    availability = await rebuilds.capture_user_availability_snapshot(user_id)
    namespace = await require_route_namespace_context(
        connection,
        clock,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        incognito=incognito,
    )
    snapshot = await MemoryObjectRepository(connection, clock).get_state_snapshot(
        user_id,
        assistant_mode_id=namespace.assistant_mode_id,
        workspace_id=namespace.workspace_id,
        conversation_id=namespace.conversation_id,
        user_persona_id=namespace.user_persona_id,
        platform_id=namespace.platform_id,
        character_id=namespace.character_id,
        incognito=namespace.incognito,
        remember_across_chats=namespace.remember_across_chats,
        remember_across_devices=namespace.remember_across_devices,
        active_space_id=namespace.active_space_id,
        active_space_boundary_mode=namespace.active_space_boundary_mode,
        active_mind_id=namespace.active_mind_id,
        mind_topology=namespace.mind_topology,
        active_embodiment_id=namespace.active_embodiment_id,
        active_realm_id=namespace.active_realm_id,
    )
    await _require_current_read_authority(
        connection,
        clock,
        user_id=user_id,
        availability=availability,
        namespace=namespace,
    )
    return snapshot
