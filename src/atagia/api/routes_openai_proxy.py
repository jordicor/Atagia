"""OpenAI-compatible memory proxy routes."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from atagia.api.dependencies import get_runtime, get_settings
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
    ConversationNotActiveError,
    UnknownAssistantModeError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.llm_client import LLMError
from atagia.services.openai_proxy_service import OpenAIProxyService


router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@dataclass(frozen=True, slots=True)
class OpenAIProxyAuth:
    """Authentication result for OpenAI-compatible proxy requests."""

    claimed_user_id: str | None


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
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
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


def openai_proxy_validation_error_response(exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc") or ()
    param = ".".join(str(part) for part in location if part not in {"body", "query", "header"})
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
) -> OpenAIProxyAuth:
    settings = get_settings(request)
    if not settings.service_mode:
        return OpenAIProxyAuth(
            claimed_user_id=x_atagia_user_id.strip()
            if x_atagia_user_id and x_atagia_user_id.strip()
            else None
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
    return OpenAIProxyAuth(
        claimed_user_id=x_atagia_user_id.strip()
        if x_atagia_user_id and x_atagia_user_id.strip()
        else None
    )


@router.get("/models", response_model=OpenAIModelList)
async def list_openai_proxy_models(
    request: Request,
    authorization: str | None = Header(default=None),
    x_atagia_user_id: str | None = Header(default=None, alias="X-Atagia-User-Id"),
) -> OpenAIModelList:
    try:
        _authenticate_proxy(request, authorization, x_atagia_user_id)
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
):
    try:
        auth = _authenticate_proxy(request, authorization, x_atagia_user_id)
    except OpenAIProxyRouteError as exc:
        return _route_error_response(exc)
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
                incognito_header=x_atagia_incognito,
                cross_chat_memory_header=x_atagia_cross_chat_memory,
                message_id_header=x_atagia_message_id,
                source_seq_header=x_atagia_source_seq,
                response_message_id_header=x_atagia_response_message_id,
                response_source_seq_header=x_atagia_response_source_seq,
                ingest_origin_header=x_atagia_ingest_origin,
                confirmation_strategy_header=x_atagia_confirmation_strategy,
                memory_privacy_mode_header=x_atagia_memory_privacy_mode,
                response_mode_header=x_atagia_response_mode,
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
            incognito_header=x_atagia_incognito,
            cross_chat_memory_header=x_atagia_cross_chat_memory,
            message_id_header=x_atagia_message_id,
            source_seq_header=x_atagia_source_seq,
            response_message_id_header=x_atagia_response_message_id,
            response_source_seq_header=x_atagia_response_source_seq,
            ingest_origin_header=x_atagia_ingest_origin,
            confirmation_strategy_header=x_atagia_confirmation_strategy,
            memory_privacy_mode_header=x_atagia_memory_privacy_mode,
            response_mode_header=x_atagia_response_mode,
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
        return _openai_error_response(status.HTTP_404_NOT_FOUND, str(exc))
    except WorkspaceNotFoundError as exc:
        return _openai_error_response(status.HTTP_404_NOT_FOUND, str(exc))
    except UnknownAssistantModeError as exc:
        return _openai_error_response(status.HTTP_404_NOT_FOUND, str(exc))
    except UnknownOperationalProfileError as exc:
        return _openai_error_response(status.HTTP_404_NOT_FOUND, str(exc))
    except OperationalProfileNotAuthorizedError as exc:
        return _openai_error_response(status.HTTP_403_FORBIDDEN, str(exc))
    except AssistantModeMismatchError as exc:
        return _openai_error_response(status.HTTP_409_CONFLICT, str(exc))
    except WorkspaceMismatchError as exc:
        return _openai_error_response(status.HTTP_409_CONFLICT, str(exc))
    except (ConversationNotActiveError, UserDeletedError) as exc:
        return _openai_error_response(
            (
                status.HTTP_410_GONE
                if isinstance(exc, UserDeletedError)
                else status.HTTP_409_CONFLICT
            ),
            str(exc),
        )
