"""Chat and creation routes."""

from __future__ import annotations

from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from starlette.concurrency import run_in_threadpool

from atagia.api.dependencies import (
    AuthContext,
    ensure_user_access,
    get_auth_context,
    get_clock,
    get_connection,
    get_runtime,
    ordinary_http_authority_context,
)
from atagia.api.namespace_context import require_route_namespace_context
from atagia.api.path_ids import TransportIdRoute
from atagia.core.clock import Clock
from atagia.core.mind_repository import MindNotFoundError
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
    ReplaceSelectedTranscriptRequest,
    RetrySelectedTranscriptRequest,
    SaveFromIncognitoRequest,
    SaveFromIncognitoResponse,
    SidecarAddResponseRequest,
    SidecarContextRequest,
    SidecarIngestMessageRequest,
    SidecarMutationResponse,
    SelectedTranscriptRebuildResponse,
    UpdateMemoryPreferencesRequest,
)
from atagia.models.schemas_memory import MemoryCategory
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_service import ChatService
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationAlreadyClosedError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    DeletionConfirmationError,
    InvalidConversationTransitionError,
    LLMUnavailableError,
    MessageIdConflictError,
    MemoryProvenanceRepairRequiredError,
    SourceSequenceConflictError,
    TranscriptRebuildUnavailableError,
    TranscriptSelectionConflictError,
    UnknownAssistantModeError,
    UserDeletedError,
    UserErasureCleanupPendingError,
    UserErasureReconciliationRequiredError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.lifecycle_service import ConversationLifecycleService
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.sidecar_service import SidecarService
from atagia.services.selected_transcript_service import SelectedTranscriptService
from atagia.services.request_controls import (
    ResolvedMemoryScopeControls,
    reject_remote_authority_claims,
    resolve_memory_scope_controls,
)
from atagia.services.request_budgets import (
    RequestBudgetExceededError,
    RequestBudgetLimits,
    RequestPayloadStructureError,
    validate_direct_message_request_budget,
)

router = APIRouter(prefix="/v1", tags=["chat"], route_class=TransportIdRoute)


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


async def _enforce_message_request_budget(
    request: Request,
    *,
    message_text: str,
    attachments: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        await run_in_threadpool(
            validate_direct_message_request_budget,
            message_text=message_text,
            attachments=attachments or [],
            metadata=metadata,
            limits=RequestBudgetLimits.from_settings(get_runtime(request).settings),
        )
    except RequestBudgetExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except RequestPayloadStructureError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


def _ordinary_memory_scope_controls(
    payload: Any,
    request: Request,
    *,
    metadata: dict[str, Any] | None = None,
) -> ResolvedMemoryScopeControls:
    """Validate ordinary request claims and resolve effective memory scope."""

    try:
        reject_remote_authority_claims(
            metadata=metadata,
            headers=request.headers,
        )
        typed_fields = {
            field_name: getattr(payload, field_name)
            for field_name in ("incognito", "cross_chat_memory")
            if field_name in payload.model_fields_set
        }
        return resolve_memory_scope_controls(
            typed_fields=typed_fields,
            metadata=metadata,
            headers=request.headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


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
    memory_scope = _ordinary_memory_scope_controls(
        payload,
        request,
        metadata=payload.metadata,
    )
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
            cross_chat_memory=memory_scope.cross_chat_memory,
            temporary=payload.temporary,
            temporary_ttl_seconds=payload.temporary_ttl_seconds,
            purge_on_close=payload.purge_on_close,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            active_presence_id=payload.active_presence_id,
            mind_id=payload.mind_id,
            mind_topology=payload.mind_topology,
            embodiment_id=payload.embodiment_id,
            realm_id=payload.realm_id,
            space_id=payload.space_id,
            mode=payload.mode,
            incognito=memory_scope.incognito,
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
    except MindNotFoundError as exc:
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
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_410_GONE
                if isinstance(exc, UserDeletedError)
                else status.HTTP_409_CONFLICT
            ),
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
    ensure_user_access(user_id, auth_context)
    items = await PendingConfirmationService(
        connection, clock
    ).list_pending_confirmations(
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
    ensure_user_access(user_id, auth_context)
    try:
        memory = await PendingConfirmationService(
            connection,
            clock,
        ).confirm_pending_memory(user_id=user_id, memory_id=memory_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
    ensure_user_access(user_id, auth_context)
    try:
        memory = await PendingConfirmationService(
            connection,
            clock,
        ).decline_pending_memory(user_id=user_id, memory_id=memory_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
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
    ensure_user_access(payload.user_id, auth_context)
    namespace = await require_route_namespace_context(
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
        return await ConversationLifecycleService(
            get_runtime(request)
        ).close_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=conversation_id,
            purge=payload.purge,
            confirmation=payload.confirmation,
            namespace_guard=namespace.authorization_snapshot,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except DeletionConfirmationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except ConversationAlreadyClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except InvalidConversationTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    payload: ConversationLifecycleRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> dict[str, Any]:
    ensure_user_access(payload.user_id, auth_context)
    namespace = await require_route_namespace_context(
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
        return await ConversationLifecycleService(
            get_runtime(request)
        ).archive_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=conversation_id,
            namespace_guard=namespace.authorization_snapshot,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except InvalidConversationTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except MemoryProvenanceRepairRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/conversations/{conversation_id}/delete", response_model=DeletionReport)
async def delete_conversation(
    conversation_id: str,
    payload: DeleteConversationRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> DeletionReport:
    ensure_user_access(payload.user_id, auth_context)
    runtime = get_runtime(request)
    namespace = await require_route_namespace_context(
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
        return await ConversationLifecycleService(runtime).delete_conversation(
            connection,
            user_id=payload.user_id,
            conversation_id=conversation_id,
            confirmation=payload.confirmation,
            namespace_guard=namespace.authorization_snapshot,
        )
    except DeletionConfirmationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        ) from None
    except InvalidConversationTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/users/{user_id}/erase", response_model=ErasureReport)
async def erase_user_data(
    user_id: str,
    payload: EraseUserDataRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
) -> ErasureReport:
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except UserErasureCleanupPendingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except UserErasureReconciliationRequiredError as exc:
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
    ensure_user_access(payload.user_id, auth_context)
    await _enforce_message_request_budget(
        request,
        message_text=payload.message_text,
        attachments=payload.attachments,
        metadata=payload.metadata,
    )
    _require_platform_id_for_service(request, payload.platform_id)
    memory_scope = _ordinary_memory_scope_controls(
        payload,
        request,
        metadata=payload.metadata,
    )
    authority_context = ordinary_http_authority_context(
        auth_context,
        user_id=payload.user_id,
        purpose="chat_reply",
    )
    try:
        result = await ChatService(runtime=get_runtime(request)).chat_reply(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            message_text=payload.message_text,
            message_occurred_at=payload.message_occurred_at,
            include_thinking=payload.include_thinking,
            metadata=payload.metadata,
            debug=payload.debug,
            debug_include_sensitive=auth_context.is_admin,
            attachments=payload.attachments,
            operational_profile=payload.operational_profile,
            operational_signals=payload.operational_signals,
            cross_chat_memory=memory_scope.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            active_presence_id=payload.active_presence_id,
            mind_id=payload.mind_id,
            mind_topology=payload.mind_topology,
            embodiment_id=payload.embodiment_id,
            realm_id=payload.realm_id,
            space_id=payload.space_id,
            mode=payload.mode,
            incognito=memory_scope.incognito,
            response_mode=payload.response_mode,
            adaptive_retrieval=payload.adaptive_retrieval,
            prompt_authority_context=authority_context,
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
    except MindNotFoundError as exc:
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
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE
            if isinstance(exc, UserDeletedError)
            else status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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


@router.post(
    "/conversations/{conversation_id}/selected-transcript",
    response_model=SelectedTranscriptRebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def replace_selected_transcript(
    conversation_id: str,
    payload: ReplaceSelectedTranscriptRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SelectedTranscriptRebuildResponse:
    """Atomically install a host-selected branch and start its durable rebuild."""

    ensure_user_access(payload.user_id, auth_context)
    _require_platform_id_for_service(request, payload.platform_id)
    for message in payload.messages:
        await _enforce_message_request_budget(request, message_text=message.text)
    try:
        return await SelectedTranscriptService(get_runtime(request)).replace(
            conversation_id=conversation_id,
            request=payload,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_410_GONE
                if isinstance(exc, UserDeletedError)
                else status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc
    except TranscriptSelectionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except TranscriptRebuildUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/conversations/{conversation_id}/selected-transcript/{operation_id}",
    response_model=SelectedTranscriptRebuildResponse,
)
async def get_selected_transcript_status(
    conversation_id: str,
    operation_id: str,
    request: Request,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
) -> SelectedTranscriptRebuildResponse:
    """Poll a selected-transcript rebuild without exposing partial memory state."""

    ensure_user_access(user_id, auth_context)
    try:
        return await SelectedTranscriptService(get_runtime(request)).get_status(
            user_id=user_id,
            conversation_id=conversation_id,
            operation_id=operation_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/conversations/{conversation_id}/selected-transcript/{operation_id}/retry",
    response_model=SelectedTranscriptRebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_selected_transcript_rebuild(
    conversation_id: str,
    operation_id: str,
    payload: RetrySelectedTranscriptRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SelectedTranscriptRebuildResponse:
    """Restart a remediation-required rebuild using canonical selected messages."""

    ensure_user_access(payload.user_id, auth_context)
    try:
        return await SelectedTranscriptService(get_runtime(request)).retry(
            user_id=payload.user_id,
            conversation_id=conversation_id,
            operation_id=operation_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_410_GONE
                if isinstance(exc, UserDeletedError)
                else status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc
    except TranscriptSelectionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except TranscriptRebuildUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


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
    ensure_user_access(payload.user_id, auth_context)
    await _enforce_message_request_budget(
        request,
        message_text=payload.message_text,
        attachments=payload.attachments,
    )
    _require_platform_id_for_service(request, payload.platform_id)
    memory_scope = _ordinary_memory_scope_controls(payload, request)
    authority_context = ordinary_http_authority_context(
        auth_context,
        user_id=payload.user_id,
        purpose="sidecar_context",
    )
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
            cross_chat_memory=memory_scope.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            active_presence_id=payload.active_presence_id,
            mind_id=payload.mind_id,
            mind_topology=payload.mind_topology,
            embodiment_id=payload.embodiment_id,
            realm_id=payload.realm_id,
            space_id=payload.space_id,
            incognito=memory_scope.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
            response_mode=payload.response_mode,
            adaptive_retrieval=payload.adaptive_retrieval,
            prompt_authority_context=authority_context,
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
    except MindNotFoundError as exc:
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
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE
            if isinstance(exc, UserDeletedError)
            else status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/conversations/{conversation_id}/messages", response_model=SidecarMutationResponse
)
async def ingest_sidecar_message(
    conversation_id: str,
    payload: SidecarIngestMessageRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SidecarMutationResponse:
    ensure_user_access(payload.user_id, auth_context)
    await _enforce_message_request_budget(
        request,
        message_text=payload.text,
        attachments=payload.attachments,
    )
    _require_platform_id_for_service(request, payload.platform_id)
    memory_scope = _ordinary_memory_scope_controls(payload, request)
    authority_context = ordinary_http_authority_context(
        auth_context,
        user_id=payload.user_id,
        purpose="sidecar_ingest_message",
    )
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
            cross_chat_memory=memory_scope.cross_chat_memory,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            active_presence_id=payload.active_presence_id,
            mind_id=payload.mind_id,
            mind_topology=payload.mind_topology,
            embodiment_id=payload.embodiment_id,
            realm_id=payload.realm_id,
            space_id=payload.space_id,
            incognito=memory_scope.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
            prompt_authority_context=authority_context,
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
    except MindNotFoundError as exc:
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
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE
            if isinstance(exc, UserDeletedError)
            else status.HTTP_409_CONFLICT,
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


@router.post(
    "/conversations/{conversation_id}/responses", response_model=SidecarMutationResponse
)
async def add_sidecar_response(
    conversation_id: str,
    payload: SidecarAddResponseRequest,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> SidecarMutationResponse:
    ensure_user_access(payload.user_id, auth_context)
    await _enforce_message_request_budget(request, message_text=payload.text)
    _require_platform_id_for_service(request, payload.platform_id)
    memory_scope = _ordinary_memory_scope_controls(payload, request)
    authority_context = ordinary_http_authority_context(
        auth_context,
        user_id=payload.user_id,
        purpose="sidecar_add_response",
    )
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
            active_presence_id=payload.active_presence_id,
            mind_id=payload.mind_id,
            mind_topology=payload.mind_topology,
            embodiment_id=payload.embodiment_id,
            realm_id=payload.realm_id,
            space_id=payload.space_id,
            mode=payload.mode,
            incognito=memory_scope.incognito,
            ingest_origin=payload.ingest_origin,
            confirmation_strategy=payload.confirmation_strategy,
            memory_privacy_mode=payload.memory_privacy_mode,
            prompt_authority_context=authority_context,
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
    except MindNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (MessageIdConflictError, SourceSequenceConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (ConversationNotActiveError, UserDeletedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE
            if isinstance(exc, UserDeletedError)
            else status.HTTP_409_CONFLICT,
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
    conversation_id = payload.conversation_id
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
