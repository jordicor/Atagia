"""OpenAI-compatible memory proxy routes."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from atagia.core.mind_repository import MindNotFoundError
from atagia.api.dependencies import (
    AuthContext,
    get_runtime,
    get_settings,
    ordinary_http_authority_context,
    reject_duplicate_singleton_headers,
)
from atagia.models.schemas_openai_proxy import (
    OpenAIChatCompletionRequest,
    OpenAIModelList,
)
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotFoundError,
    MessageIdConflictError,
    SourceSequenceConflictError,
    ConversationNotActiveError,
    UnknownAssistantModeError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.llm_client import LLMError
from atagia.services.openai_proxy_service import OpenAIProxyService
from atagia.services.openai_proxy_contract import OpenAIProxyProtocolError
from atagia.services.request_controls import reject_remote_authority_claims
from atagia.services.request_budgets import (
    RequestBudgetExceededError,
    RequestPayloadStructureError,
)


router = APIRouter(prefix="/v1", tags=["openai-compatible"])

_PROXY_SINGLETON_HEADERS = (
    "Authorization",
    "X-Atagia-User-Id",
    "X-Atagia-Conversation-Id",
    "X-Atagia-Assistant-Mode",
    "X-Atagia-Mode",
    "X-Atagia-Workspace-Id",
    "X-Atagia-User-Persona-Id",
    "X-Atagia-Platform-Id",
    "X-Atagia-Character-Id",
    "X-Atagia-Active-Presence-Id",
    "X-Atagia-Mind-Id",
    "X-Atagia-Mind-Topology",
    "X-Atagia-Embodiment-Id",
    "X-Atagia-Realm-Id",
    "X-Atagia-Space-Id",
    "X-Atagia-Incognito",
    "X-Atagia-Cross-Chat-Memory",
    "X-Atagia-Message-Id",
    "X-Atagia-Source-Seq",
    "X-Atagia-Response-Message-Id",
    "X-Atagia-Response-Source-Seq",
    "X-Atagia-Ingest-Origin",
    "X-Atagia-Confirmation-Strategy",
    "X-Atagia-Memory-Privacy-Mode",
    "X-Atagia-Response-Mode",
    "X-Atagia-Adaptive-Retrieval",
)


@dataclass(frozen=True, slots=True)
class OpenAIProxyRouteError(Exception):
    """Route-local OpenAI-compatible error response."""

    status_code: int
    message: str
    error_type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


def _openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _route_error_response(exc: OpenAIProxyRouteError) -> JSONResponse:
    return _openai_error_response(
        exc.status_code,
        exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
    )


def _protocol_error_response(exc: OpenAIProxyProtocolError) -> JSONResponse:
    headers = None
    if exc.retry_after_seconds is not None:
        headers = {"Retry-After": str(max(0, int(exc.retry_after_seconds)))}
    return _openai_error_response(
        exc.status_code,
        exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
        headers=headers,
    )


def openai_proxy_validation_error_response(exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc") or ()
    param = ".".join(
        str(part) for part in location if part not in {"body", "query", "header"}
    )
    message = str(first_error.get("msg") or "Invalid request")
    return _openai_error_response(
        422,
        message,
        param=param or None,
        code="validation_error",
    )


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise OpenAIProxyRouteError(
            status.HTTP_401_UNAUTHORIZED,
            "Missing Authorization header",
            error_type="authentication_error",
            code="missing_authorization",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise OpenAIProxyRouteError(
            status.HTTP_401_UNAUTHORIZED,
            "Authorization header must use Bearer <token>",
            error_type="authentication_error",
            code="invalid_authorization",
        )
    return token.strip()


def _authenticate_proxy(
    request: Request,
    authorization: str | None,
    x_atagia_user_id: str | None,
) -> AuthContext:
    settings = get_settings(request)
    if not settings.service_mode:
        return AuthContext(
            service_mode=False,
            is_admin=False,
            actor_id="library_mode",
            claimed_user_id=x_atagia_user_id.strip()
            if x_atagia_user_id and x_atagia_user_id.strip()
            else None,
        )
    if settings.service_api_key is None:
        raise OpenAIProxyRouteError(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "ATAGIA_SERVICE_API_KEY is required in service mode",
            error_type="server_error",
            code="missing_service_api_key",
        )
    token = _bearer_token(authorization)
    if token != settings.service_api_key:
        raise OpenAIProxyRouteError(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid API key",
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return AuthContext(
        service_mode=True,
        is_admin=False,
        actor_id="service_api_key",
        api_key=token,
        claimed_user_id=x_atagia_user_id.strip()
        if x_atagia_user_id and x_atagia_user_id.strip()
        else None,
    )


@router.get("/models", response_model=OpenAIModelList)
async def list_openai_proxy_models(
    request: Request,
    authorization: str | None = Header(default=None),
    x_atagia_user_id: str | None = Header(default=None, alias="X-Atagia-User-Id"),
) -> OpenAIModelList:
    try:
        reject_duplicate_singleton_headers(
            request,
            ("Authorization", "X-Atagia-User-Id"),
        )
        _authenticate_proxy(request, authorization, x_atagia_user_id)
    except ValueError as exc:
        return _openai_error_response(status.HTTP_400_BAD_REQUEST, str(exc))
    except OpenAIProxyRouteError as exc:
        return _route_error_response(exc)
    return OpenAIProxyService(get_runtime(request)).list_models()


@router.post("/chat/completions")
async def create_openai_proxy_chat_completion(
    payload: OpenAIChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_atagia_user_id: str | None = Header(default=None, alias="X-Atagia-User-Id"),
    x_atagia_conversation_id: str | None = Header(
        default=None,
        alias="X-Atagia-Conversation-Id",
    ),
    x_atagia_assistant_mode: str | None = Header(
        default=None,
        alias="X-Atagia-Assistant-Mode",
    ),
    x_atagia_mode: str | None = Header(default=None, alias="X-Atagia-Mode"),
    x_atagia_workspace_id: str | None = Header(
        default=None,
        alias="X-Atagia-Workspace-Id",
    ),
    x_atagia_user_persona_id: str | None = Header(
        default=None,
        alias="X-Atagia-User-Persona-Id",
    ),
    x_atagia_platform_id: str | None = Header(
        default=None,
        alias="X-Atagia-Platform-Id",
    ),
    x_atagia_character_id: str | None = Header(
        default=None,
        alias="X-Atagia-Character-Id",
    ),
    x_atagia_active_presence_id: str | None = Header(
        default=None,
        alias="X-Atagia-Active-Presence-Id",
    ),
    x_atagia_mind_id: str | None = Header(
        default=None,
        alias="X-Atagia-Mind-Id",
    ),
    x_atagia_mind_topology: str | None = Header(
        default=None,
        alias="X-Atagia-Mind-Topology",
    ),
    x_atagia_embodiment_id: str | None = Header(
        default=None,
        alias="X-Atagia-Embodiment-Id",
    ),
    x_atagia_realm_id: str | None = Header(
        default=None,
        alias="X-Atagia-Realm-Id",
    ),
    x_atagia_space_id: str | None = Header(
        default=None,
        alias="X-Atagia-Space-Id",
    ),
    x_atagia_incognito: str | None = Header(
        default=None,
        alias="X-Atagia-Incognito",
    ),
    x_atagia_cross_chat_memory: str | None = Header(
        default=None,
        alias="X-Atagia-Cross-Chat-Memory",
    ),
    x_atagia_message_id: str | None = Header(
        default=None,
        alias="X-Atagia-Message-Id",
    ),
    x_atagia_source_seq: str | None = Header(
        default=None,
        alias="X-Atagia-Source-Seq",
    ),
    x_atagia_response_message_id: str | None = Header(
        default=None,
        alias="X-Atagia-Response-Message-Id",
    ),
    x_atagia_response_source_seq: str | None = Header(
        default=None,
        alias="X-Atagia-Response-Source-Seq",
    ),
    x_atagia_ingest_origin: str | None = Header(
        default=None,
        alias="X-Atagia-Ingest-Origin",
    ),
    x_atagia_confirmation_strategy: str | None = Header(
        default=None,
        alias="X-Atagia-Confirmation-Strategy",
    ),
    x_atagia_memory_privacy_mode: str | None = Header(
        default=None,
        alias="X-Atagia-Memory-Privacy-Mode",
    ),
    x_atagia_response_mode: str | None = Header(
        default=None,
        alias="X-Atagia-Response-Mode",
    ),
    x_atagia_adaptive_retrieval: str | None = Header(
        default=None,
        alias="X-Atagia-Adaptive-Retrieval",
    ),
):
    try:
        reject_duplicate_singleton_headers(request, _PROXY_SINGLETON_HEADERS)
        auth = _authenticate_proxy(request, authorization, x_atagia_user_id)
    except ValueError as exc:
        return _openai_error_response(status.HTTP_400_BAD_REQUEST, str(exc))
    except OpenAIProxyRouteError as exc:
        return _route_error_response(exc)
    try:
        reject_remote_authority_claims(
            metadata=payload.metadata,
            extra_fields=payload.model_extra,
            headers=request.headers,
        )
    except OpenAIProxyProtocolError as exc:
        return _protocol_error_response(exc)
    except ValueError as exc:
        return _openai_error_response(
            status.HTTP_400_BAD_REQUEST,
            str(exc),
        )
    authority_context = ordinary_http_authority_context(
        auth,
        user_id=auth.claimed_user_id,
        purpose="openai_proxy",
    )
    service = OpenAIProxyService(get_runtime(request))
    try:
        if payload.stream:
            stream = await service.stream(
                payload,
                claimed_user_id=auth.claimed_user_id,
                conversation_id_header=x_atagia_conversation_id,
                assistant_mode_header=x_atagia_assistant_mode,
                mode_header=x_atagia_mode,
                workspace_id_header=x_atagia_workspace_id,
                user_persona_id_header=x_atagia_user_persona_id,
                platform_id_header=x_atagia_platform_id,
                character_id_header=x_atagia_character_id,
                active_presence_id_header=x_atagia_active_presence_id,
                mind_id_header=x_atagia_mind_id,
                mind_topology_header=x_atagia_mind_topology,
                embodiment_id_header=x_atagia_embodiment_id,
                realm_id_header=x_atagia_realm_id,
                space_id_header=x_atagia_space_id,
                incognito_header=request.headers.getlist("X-Atagia-Incognito"),
                cross_chat_memory_header=request.headers.getlist(
                    "X-Atagia-Cross-Chat-Memory"
                ),
                message_id_header=x_atagia_message_id,
                source_seq_header=x_atagia_source_seq,
                response_message_id_header=x_atagia_response_message_id,
                response_source_seq_header=x_atagia_response_source_seq,
                ingest_origin_header=x_atagia_ingest_origin,
                confirmation_strategy_header=x_atagia_confirmation_strategy,
                memory_privacy_mode_header=x_atagia_memory_privacy_mode,
                response_mode_header=x_atagia_response_mode,
                adaptive_retrieval_header=x_atagia_adaptive_retrieval,
                prompt_authority_context=authority_context,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return await service.complete(
            payload,
            claimed_user_id=auth.claimed_user_id,
            conversation_id_header=x_atagia_conversation_id,
            assistant_mode_header=x_atagia_assistant_mode,
            mode_header=x_atagia_mode,
            workspace_id_header=x_atagia_workspace_id,
            user_persona_id_header=x_atagia_user_persona_id,
            platform_id_header=x_atagia_platform_id,
            character_id_header=x_atagia_character_id,
            active_presence_id_header=x_atagia_active_presence_id,
            mind_id_header=x_atagia_mind_id,
            mind_topology_header=x_atagia_mind_topology,
            embodiment_id_header=x_atagia_embodiment_id,
            realm_id_header=x_atagia_realm_id,
            space_id_header=x_atagia_space_id,
            incognito_header=request.headers.getlist("X-Atagia-Incognito"),
            cross_chat_memory_header=request.headers.getlist(
                "X-Atagia-Cross-Chat-Memory"
            ),
            message_id_header=x_atagia_message_id,
            source_seq_header=x_atagia_source_seq,
            response_message_id_header=x_atagia_response_message_id,
            response_source_seq_header=x_atagia_response_source_seq,
            ingest_origin_header=x_atagia_ingest_origin,
            confirmation_strategy_header=x_atagia_confirmation_strategy,
            memory_privacy_mode_header=x_atagia_memory_privacy_mode,
            response_mode_header=x_atagia_response_mode,
            adaptive_retrieval_header=x_atagia_adaptive_retrieval,
            prompt_authority_context=authority_context,
        )
    except RequestBudgetExceededError as exc:
        return _openai_error_response(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            str(exc),
            param=exc.field,
            code="request_too_large",
        )
    except RequestPayloadStructureError as exc:
        return _openai_error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            str(exc),
            param=exc.field,
            code="invalid_request_structure",
        )
    except OpenAIProxyProtocolError as exc:
        return _protocol_error_response(exc)
    except MindNotFoundError as exc:
        return _openai_error_response(
            status.HTTP_404_NOT_FOUND,
            str(exc),
            code="mind_not_found",
        )
    except ValueError as exc:
        return _openai_error_response(
            status.HTTP_400_BAD_REQUEST,
            str(exc),
            param="model" if "Unknown model" in str(exc) else None,
            code="model_not_found" if "Unknown model" in str(exc) else None,
        )
    except LLMError:
        return _openai_error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "LLM service unavailable",
            error_type="server_error",
            code="llm_unavailable",
        )
    except ConversationNotFoundError as exc:
        return _openai_error_response(
            status.HTTP_404_NOT_FOUND,
            str(exc),
            code="conversation_not_found",
        )
    except WorkspaceNotFoundError as exc:
        return _openai_error_response(
            status.HTTP_404_NOT_FOUND,
            str(exc),
            code="workspace_not_found",
        )
    except UnknownAssistantModeError as exc:
        return _openai_error_response(
            status.HTTP_404_NOT_FOUND,
            str(exc),
            code="assistant_mode_not_found",
        )
    except UnknownOperationalProfileError as exc:
        return _openai_error_response(
            status.HTTP_404_NOT_FOUND,
            str(exc),
            code="operational_profile_not_found",
        )
    except OperationalProfileNotAuthorizedError as exc:
        return _openai_error_response(
            status.HTTP_403_FORBIDDEN,
            str(exc),
            code="operational_profile_not_authorized",
        )
    except AssistantModeMismatchError as exc:
        return _openai_error_response(
            status.HTTP_409_CONFLICT,
            str(exc),
            code="assistant_mode_conflict",
        )
    except WorkspaceMismatchError as exc:
        return _openai_error_response(
            status.HTTP_409_CONFLICT,
            str(exc),
            code="workspace_conflict",
        )
    except MessageIdConflictError as exc:
        return _openai_error_response(
            status.HTTP_409_CONFLICT,
            str(exc),
            code="message_id_conflict",
        )
    except SourceSequenceConflictError as exc:
        return _openai_error_response(
            status.HTTP_409_CONFLICT,
            str(exc),
            code="source_sequence_conflict",
        )
    except (ConversationNotActiveError, UserDeletedError) as exc:
        return _openai_error_response(
            (
                status.HTTP_410_GONE
                if isinstance(exc, UserDeletedError)
                else status.HTTP_409_CONFLICT
            ),
            str(exc),
        )
