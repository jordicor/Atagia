"""Chat and creation routes."""

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
    get_runtime,
)
from atagia.api.namespace_context import require_route_namespace_context
from atagia.core.clock import Clock
from atagia.core.repositories import (
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_api import (
    ChatReplyRequest,
    ChatReplyResponse,
    CloseConversationRequest,
    ConversationIncognitoRequest,
    ConversationLifecycleRequest,
    CreateConversationRequest,
    CreateUserRequest,
    CreateWorkspaceRequest,
    DeleteConversationRequest,
    DeletionReport,
    EraseUserDataRequest,
    ErasureReport,
    ContextResult,
    FlushRequest,
    FlushResponse,
    MemoryPreferencesResponse,
    MemoryProcessingStatus,
    PendingMemoryConfirmationActionResponse,
    PendingMemoryConfirmationListResponse,
    SaveFromIncognitoRequest,
    SaveFromIncognitoResponse,
    SidecarAddResponseRequest,
    SidecarContextRequest,
    SidecarIngestMessageRequest,
    SidecarMutationResponse,
    UpdateMemoryPreferencesRequest,
)
from atagia.models.schemas_memory import MemoryCategory
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_service import ChatService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationAlreadyClosedError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    DeletionConfirmationError,
    InvalidConversationTransitionError,
    LLMUnavailableError,
    MessageIdConflictError,
    SourceSequenceConflictError,
    UnknownAssistantModeError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.lifecycle_service import ConversationLifecycleService
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.sidecar_service import SidecarService
from atagia.transport_ids import decode_path_id

router = APIRouter(prefix="/v1", tags=["chat"])


def _route_id(value: str) -> str:
    try:
        return decode_path_id(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _canonical_mode(legacy_mode: str | None, mode: str | None) -> str | None:
    return mode if mode is not None else legacy_mode


def _require_platform_id_for_service(request: Request, platform_id: str | None) -> None:
    if not get_runtime(request).settings.service_mode:
        return
    if platform_id is None or not platform_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="platform_id is required in service mode",
        )


async def _ensure_user_exists(users: UserRepository, user_id: str) -> None:
    user = await users.get_user(user_id)
    if user is not None and user.get("deleted_at") is not None:
        raise UserDeletedError("User has been erased")
    if user is None and await users.has_user_erasure_marker(user_id):
        raise UserDeletedError("User has been erased")
    if user is None:
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
    try:
        return await users.create_user(payload.user_id)
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc


@router.post("/conversations")
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    sidecar = SidecarService(get_runtime(request))
    try:
        await sidecar.ensure_user_exists(connection, payload.user_id)
        return await sidecar.ensure_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            workspace_id=payload.workspace_id,
            assistant_mode_id=_canonical_mode(payload.assistant_mode_id, payload.mode),
            title=payload.title,
            metadata=payload.metadata,
            cross_chat_memory=payload.cross_chat_memory,
            temporary=payload.temporary,
            temporary_ttl_seconds=payload.temporary_ttl_seconds,
            purge_on_close=payload.purge_on_close,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            mode=payload.mode,
            incognito=payload.incognito,
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
    except WorkspaceMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ConversationNotActiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc


@router.get(
    "/users/{user_id}/memory-preferences",
    response_model=MemoryPreferencesResponse,
)
async def get_memory_preferences(
    user_id: str,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> MemoryPreferencesResponse:
    user_id = _route_id(user_id)
    ensure_user_access(user_id, auth_context)
    try:
        preferences = await SidecarService(get_runtime(request)).get_memory_preferences(
            user_id
        )
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc
    return MemoryPreferencesResponse.model_validate(preferences)


@router.put(
    "/users/{user_id}/memory-preferences",
    response_model=MemoryPreferencesResponse,
)
async def update_memory_preferences(
    user_id: str,
    payload: UpdateMemoryPreferencesRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> MemoryPreferencesResponse:
    user_id = _route_id(user_id)
    ensure_user_access(user_id, auth_context)
    try:
        preferences = await SidecarService(get_runtime(request)).set_memory_preferences(
            user_id,
            remember_across_chats=payload.remember_across_chats,
            remember_across_devices=payload.remember_across_devices,
            memory_privacy_mode=payload.memory_privacy_mode,
        )
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc
    return MemoryPreferencesResponse.model_validate(preferences)


@router.get(
    "/users/{user_id}/memory-confirmations",
    response_model=PendingMemoryConfirmationListResponse,
)
async def list_pending_memory_confirmations(
    user_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
    conversation_id: str | None = Query(default=None),
    platform_id: str | None = Query(default=None),
    user_persona_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    category: MemoryCategory | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PendingMemoryConfirmationListResponse:
    user_id = _route_id(user_id)
    ensure_user_access(user_id, auth_context)
    items = await PendingConfirmationService(connection, clock).list_pending_confirmations(
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        user_persona_id=user_persona_id,
        character_id=character_id,
        category=category,
        limit=limit,
        offset=offset,
    )
    return PendingMemoryConfirmationListResponse.model_validate({"items": items})


@router.post(
    "/users/{user_id}/memory-confirmations/{memory_id}/confirm",
    response_model=PendingMemoryConfirmationActionResponse,
)
async def confirm_pending_memory(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> PendingMemoryConfirmationActionResponse:
    user_id = _route_id(user_id)
    memory_id = _route_id(memory_id)
    ensure_user_access(user_id, auth_context)
    try:
        memory = await PendingConfirmationService(
            connection,
            clock,
        ).confirm_pending_memory(user_id=user_id, memory_id=memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return PendingMemoryConfirmationActionResponse(
        memory_id=str(memory["id"]),
        status=str(memory["status"]),
    )


@router.post(
    "/users/{user_id}/memory-confirmations/{memory_id}/decline",
    response_model=PendingMemoryConfirmationActionResponse,
)
async def decline_pending_memory(
    user_id: str,
    memory_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    clock: Clock = Depends(get_clock),
) -> PendingMemoryConfirmationActionResponse:
    user_id = _route_id(user_id)
    memory_id = _route_id(memory_id)
    ensure_user_access(user_id, auth_context)
    try:
        memory = await PendingConfirmationService(
            connection,
            clock,
        ).decline_pending_memory(user_id=user_id, memory_id=memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return PendingMemoryConfirmationActionResponse(
        memory_id=str(memory["id"]),
        status=str(memory["status"]),
    )


@router.post("/conversations/{conversation_id}/incognito")
async def set_conversation_incognito(
    conversation_id: str,
    payload: ConversationIncognitoRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    try:
        return await SidecarService(get_runtime(request)).set_conversation_incognito(
            payload.user_id,
            conversation_id,
            payload.incognito,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc


@router.post(
    "/conversations/{conversation_id}/save-from-incognito",
    response_model=SaveFromIncognitoResponse,
)
async def save_from_incognito(
    conversation_id: str,
    payload: SaveFromIncognitoRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SaveFromIncognitoResponse:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    try:
        review = await SidecarService(
            get_runtime(request)
        ).prepare_save_from_incognito_review(
            payload.user_id,
            conversation_id,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            mode=payload.mode,
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
    except (ConversationNotActiveError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc
    return SaveFromIncognitoResponse.model_validate(review)


@router.post("/conversations/{conversation_id}/close")
async def close_conversation(
    conversation_id: str,
    payload: CloseConversationRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> dict[str, Any] | DeletionReport:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    await require_route_namespace_context(
        connection,
        get_runtime(request).clock,
        user_id=payload.user_id,
        conversation_id=conversation_id,
        platform_id=payload.platform_id,
        user_persona_id=payload.user_persona_id,
        character_id=payload.character_id,
        incognito=payload.incognito,
    )
    try:
        return await ConversationLifecycleService(get_runtime(request)).close_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=conversation_id,
            purge=payload.purge,
            confirmation=payload.confirmation,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except DeletionConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ConversationAlreadyClosedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InvalidConversationTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    payload: ConversationLifecycleRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> dict[str, Any]:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    await require_route_namespace_context(
        connection,
        get_runtime(request).clock,
        user_id=payload.user_id,
        conversation_id=conversation_id,
        platform_id=payload.platform_id,
        user_persona_id=payload.user_persona_id,
        character_id=payload.character_id,
        incognito=payload.incognito,
        require_active=False,
    )
    try:
        return await ConversationLifecycleService(get_runtime(request)).archive_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except InvalidConversationTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/delete", response_model=DeletionReport)
async def delete_conversation(
    conversation_id: str,
    payload: DeleteConversationRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> DeletionReport:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    runtime = get_runtime(request)
    cache_service = ContextCacheService(runtime)
    await require_route_namespace_context(
        connection,
        runtime.clock,
        user_id=payload.user_id,
        conversation_id=conversation_id,
        platform_id=payload.platform_id,
        user_persona_id=payload.user_persona_id,
        character_id=payload.character_id,
        incognito=payload.incognito,
        require_active=False,
    )
    try:
        async with cache_service.user_cache_guard(payload.user_id):
            return await ConversationLifecycleService(runtime).delete_conversation(
                connection,
                user_id=payload.user_id,
                conversation_id=conversation_id,
                confirmation=payload.confirmation,
            )
    except DeletionConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except InvalidConversationTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/users/{user_id}/erase", response_model=ErasureReport)
async def erase_user_data(
    user_id: str,
    payload: EraseUserDataRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> ErasureReport:
    user_id = _route_id(user_id)
    ensure_user_access(user_id, auth_context)
    if payload.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Path user_id and payload user_id must match",
        )
    try:
        return await ConversationLifecycleService(get_runtime(request)).erase_user_data(
            connection,
            user_id=user_id,
            confirmation=payload.confirmation,
        )
    except DeletionConfirmationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


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
    try:
        await _ensure_user_exists(users, payload.user_id)
    except UserDeletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=str(exc),
        ) from exc
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
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
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
            cross_chat_memory=payload.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            mode=payload.mode,
            incognito=payload.incognito,
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
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE if isinstance(exc, UserDeletedError) else status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return ChatReplyResponse(
        conversation_id=conversation_id,
        request_message_id=result.request_message_id,
        response_message_id=result.response_message_id,
        reply_text=result.response_text,
        retrieval_event_id=result.retrieval_event_id,
        memory_processing=result.memory_processing,
        debug=result.debug,
    )


@router.get(
    "/conversations/{conversation_id}/processing-status",
    response_model=MemoryProcessingStatus,
)
async def get_conversation_processing_status(
    conversation_id: str,
    request: Request,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
) -> MemoryProcessingStatus:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(user_id, auth_context)
    runtime = get_runtime(request)
    connection = await runtime.open_connection()
    try:
        return await JobTrackingService(
            connection,
            runtime.clock,
            workers_enabled=runtime.settings.workers_enabled,
        ).get_status(user_id=user_id, conversation_id=conversation_id)
    finally:
        await connection.close()


@router.get(
    "/users/{user_id}/processing-status",
    response_model=MemoryProcessingStatus,
)
async def get_user_processing_status(
    user_id: str,
    request: Request,
    user_persona_id: str | None = Query(default=None),
    platform_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
    incognito: bool = Query(default=False),
    remember_across_chats: bool = Query(default=True),
    remember_across_devices: bool = Query(default=True),
    auth_context: AuthContext = Depends(get_auth_context),
) -> MemoryProcessingStatus:
    user_id = _route_id(user_id)
    ensure_user_access(user_id, auth_context)
    runtime = get_runtime(request)
    connection = await runtime.open_connection()
    try:
        try:
            return await JobTrackingService(
                connection,
                runtime.clock,
                workers_enabled=runtime.settings.workers_enabled,
            ).get_status(
                user_id=user_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                admin=auth_context.is_admin,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await connection.close()


@router.post("/conversations/{conversation_id}/context", response_model=ContextResult)
async def get_sidecar_context(
    conversation_id: str,
    payload: SidecarContextRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ContextResult:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    try:
        return await SidecarService(get_runtime(request)).get_context(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            message=payload.message_text,
            mode=_canonical_mode(payload.assistant_mode_id, payload.mode),
            workspace_id=payload.workspace_id,
            occurred_at=payload.message_occurred_at,
            attachments=[
                attachment.model_dump(mode="json") for attachment in payload.attachments
            ],
            message_id=payload.message_id,
            source_seq=payload.source_seq,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
            cross_chat_memory=payload.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            incognito=payload.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
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
    except WorkspaceMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE if isinstance(exc, UserDeletedError) else status.HTTP_409_CONFLICT,
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
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    try:
        result = await SidecarService(get_runtime(request)).ingest_message(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            role=payload.role,
            text=payload.text,
            mode=_canonical_mode(payload.assistant_mode_id, payload.mode),
            workspace_id=payload.workspace_id,
            occurred_at=payload.occurred_at,
            attachments=[
                attachment.model_dump(mode="json") for attachment in payload.attachments
            ],
            message_id=payload.message_id,
            source_seq=payload.source_seq,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
            cross_chat_memory=payload.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            incognito=payload.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
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
    except WorkspaceMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE if isinstance(exc, UserDeletedError) else status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return SidecarMutationResponse(
        message_id=str(result.message["id"]),
        seq=int(result.message["seq"]),
        source_seq=payload.source_seq,
        idempotent_replay=not result.created,
    )


@router.post("/conversations/{conversation_id}/responses", response_model=SidecarMutationResponse)
async def add_sidecar_response(
    conversation_id: str,
    payload: SidecarAddResponseRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SidecarMutationResponse:
    conversation_id = _route_id(conversation_id)
    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    try:
        result = await SidecarService(get_runtime(request)).add_response(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            text=payload.text,
            occurred_at=payload.occurred_at,
            message_id=payload.message_id,
            source_seq=payload.source_seq,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            mode=payload.mode,
            incognito=payload.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
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
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE if isinstance(exc, UserDeletedError) else status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return SidecarMutationResponse(
        message_id=str(result.message["id"]),
        seq=int(result.message["seq"]),
        source_seq=payload.source_seq,
        idempotent_replay=not result.created,
    )


@router.post("/flush", response_model=FlushResponse)
async def flush_sidecar_work(
    payload: FlushRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> FlushResponse:
    ensure_user_access(payload.user_id, auth_context)
    conversation_id = _route_id(payload.conversation_id) if payload.conversation_id is not None else None
    runtime = get_runtime(request)
    if not runtime.settings.workers_enabled:
        connection = await runtime.open_connection()
        try:
            try:
                memory_processing = await JobTrackingService(
                    connection,
                    runtime.clock,
                    workers_enabled=runtime.settings.workers_enabled,
                ).get_status(
                    user_id=payload.user_id,
                    conversation_id=conversation_id,
                    user_persona_id=payload.user_persona_id,
                    platform_id=payload.platform_id,
                    character_id=payload.character_id,
                    incognito=payload.incognito,
                    remember_across_chats=payload.remember_across_chats,
                    remember_across_devices=payload.remember_across_devices,
                    admin=auth_context.is_admin,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await connection.close()
        return FlushResponse(completed=False, memory_processing=memory_processing)
    completed = await runtime.storage_backend.drain(payload.timeout_seconds)
    connection = await runtime.open_connection()
    try:
        try:
            memory_processing = await JobTrackingService(
                connection,
                runtime.clock,
                workers_enabled=runtime.settings.workers_enabled,
            ).get_status(
                user_id=payload.user_id,
                conversation_id=conversation_id,
                user_persona_id=payload.user_persona_id,
                platform_id=payload.platform_id,
                character_id=payload.character_id,
                incognito=payload.incognito,
                remember_across_chats=payload.remember_across_chats,
                remember_across_devices=payload.remember_across_devices,
                admin=auth_context.is_admin,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await connection.close()
    return FlushResponse(completed=completed, memory_processing=memory_processing)
