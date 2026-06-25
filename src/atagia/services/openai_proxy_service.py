"""OpenAI-compatible chat-completion proxy backed by Atagia memory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import logging
import time
import uuid
from typing import Any

from atagia.integrations.message_projection import message_to_text
from atagia.integrations.prompt_injection import build_injection_decision
from atagia.models.schemas_memory import ResponseMode
from atagia.models.schemas_openai_proxy import (
    OpenAIChatCompletionRequest,
    OpenAIModelList,
    OpenAIModelObject,
    OpenAIProxyMessage,
)
from atagia.memory.operational_profile import (
    OperationalProfileNotAuthorizedError,
    UnknownOperationalProfileError,
)
from atagia.services.chat_support import chat_model
from atagia.services.errors import (
    AssistantModeMismatchError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    UnknownAssistantModeError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMMessage,
    LLMStreamEvent,
    LLMToolSpec,
)
from atagia.services.sidecar_service import SidecarService

logger = logging.getLogger(__name__)

_RESPONSE_MODE_VALUES: frozenset[str] = frozenset(mode.value for mode in ResponseMode)

_HARD_CONTEXT_ERRORS = (
    AssistantModeMismatchError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    OperationalProfileNotAuthorizedError,
    UnknownAssistantModeError,
    UnknownOperationalProfileError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)


@dataclass(frozen=True, slots=True)
class OpenAIProxyIdentity:
    """Resolved host identity for an OpenAI-compatible proxy request."""

    user_id: str
    conversation_id: str
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    active_presence_id: str | None = None
    mind_id: str | None = None
    mind_topology: str | None = None
    embodiment_id: str | None = None
    realm_id: str | None = None
    space_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None
    operational_profile: str | None = None
    operational_signals: dict[str, Any] | None = None
    cross_chat_memory: bool = True
    message_id: str | None = None
    source_seq: int | None = None
    response_message_id: str | None = None
    response_source_seq: int | None = None
    ingest_origin: str | None = None
    confirmation_strategy: str | None = None
    memory_privacy_mode: str | None = None
    response_mode: str | None = None
    adaptive_retrieval: bool | None = None


@dataclass(slots=True)
class OpenAIProxyService:
    """Serve OpenAI-compatible chat completions with Atagia context injection."""

    runtime: Any

    def list_models(self) -> OpenAIModelList:
        created = int(time.time())
        return OpenAIModelList(
            data=[
                OpenAIModelObject(
                    id=self.runtime.settings.openai_proxy_model_id,
                    created=created,
                )
            ]
        )

    async def complete(
        self,
        request: OpenAIChatCompletionRequest,
        *,
        claimed_user_id: str | None = None,
        conversation_id_header: str | None = None,
        assistant_mode_header: str | None = None,
        mode_header: str | None = None,
        workspace_id_header: str | None = None,
        user_persona_id_header: str | None = None,
        platform_id_header: str | None = None,
        character_id_header: str | None = None,
        active_presence_id_header: str | None = None,
        mind_id_header: str | None = None,
        mind_topology_header: str | None = None,
        embodiment_id_header: str | None = None,
        realm_id_header: str | None = None,
        space_id_header: str | None = None,
        incognito_header: str | None = None,
        cross_chat_memory_header: str | None = None,
        message_id_header: str | None = None,
        source_seq_header: str | None = None,
        response_message_id_header: str | None = None,
        response_source_seq_header: str | None = None,
        ingest_origin_header: str | None = None,
        confirmation_strategy_header: str | None = None,
        memory_privacy_mode_header: str | None = None,
        response_mode_header: str | None = None,
        adaptive_retrieval_header: str | None = None,
    ) -> dict[str, Any]:
        self._validate_model(request)
        completion_id = _completion_id()
        created = int(time.time())
        identity = self._resolve_identity(
            request,
            claimed_user_id=claimed_user_id,
            conversation_id_header=conversation_id_header,
            assistant_mode_header=assistant_mode_header,
            mode_header=mode_header,
            workspace_id_header=workspace_id_header,
            user_persona_id_header=user_persona_id_header,
            platform_id_header=platform_id_header,
            character_id_header=character_id_header,
            active_presence_id_header=active_presence_id_header,
            mind_id_header=mind_id_header,
            mind_topology_header=mind_topology_header,
            embodiment_id_header=embodiment_id_header,
            realm_id_header=realm_id_header,
            space_id_header=space_id_header,
            incognito_header=incognito_header,
            cross_chat_memory_header=cross_chat_memory_header,
            message_id_header=message_id_header,
            source_seq_header=source_seq_header,
            response_message_id_header=response_message_id_header,
            response_source_seq_header=response_source_seq_header,
            ingest_origin_header=ingest_origin_header,
            confirmation_strategy_header=confirmation_strategy_header,
            memory_privacy_mode_header=memory_privacy_mode_header,
            response_mode_header=response_mode_header,
            adaptive_retrieval_header=adaptive_retrieval_header,
        )
        latest_user_text = _latest_user_text(request.messages)
        context = await self._context_for_turn_fail_open(identity, latest_user_text)
        response = await self.runtime.llm_client.complete(
            self._llm_request(request, context, identity)
        )
        await self._record_response_fail_open(identity, response.output_text)
        return _completion_payload(
            completion_id=completion_id,
            created=created,
            model=request.model,
            response=response,
        )

    async def stream(
        self,
        request: OpenAIChatCompletionRequest,
        *,
        claimed_user_id: str | None = None,
        conversation_id_header: str | None = None,
        assistant_mode_header: str | None = None,
        mode_header: str | None = None,
        workspace_id_header: str | None = None,
        user_persona_id_header: str | None = None,
        platform_id_header: str | None = None,
        character_id_header: str | None = None,
        active_presence_id_header: str | None = None,
        mind_id_header: str | None = None,
        mind_topology_header: str | None = None,
        embodiment_id_header: str | None = None,
        realm_id_header: str | None = None,
        space_id_header: str | None = None,
        incognito_header: str | None = None,
        cross_chat_memory_header: str | None = None,
        message_id_header: str | None = None,
        source_seq_header: str | None = None,
        response_message_id_header: str | None = None,
        response_source_seq_header: str | None = None,
        ingest_origin_header: str | None = None,
        confirmation_strategy_header: str | None = None,
        memory_privacy_mode_header: str | None = None,
        response_mode_header: str | None = None,
        adaptive_retrieval_header: str | None = None,
    ) -> AsyncIterator[str]:
        self._validate_model(request)
        completion_id = _completion_id()
        created = int(time.time())
        identity = self._resolve_identity(
            request,
            claimed_user_id=claimed_user_id,
            conversation_id_header=conversation_id_header,
            assistant_mode_header=assistant_mode_header,
            mode_header=mode_header,
            workspace_id_header=workspace_id_header,
            user_persona_id_header=user_persona_id_header,
            platform_id_header=platform_id_header,
            character_id_header=character_id_header,
            active_presence_id_header=active_presence_id_header,
            mind_id_header=mind_id_header,
            mind_topology_header=mind_topology_header,
            embodiment_id_header=embodiment_id_header,
            realm_id_header=realm_id_header,
            space_id_header=space_id_header,
            incognito_header=incognito_header,
            cross_chat_memory_header=cross_chat_memory_header,
            message_id_header=message_id_header,
            source_seq_header=source_seq_header,
            response_message_id_header=response_message_id_header,
            response_source_seq_header=response_source_seq_header,
            ingest_origin_header=ingest_origin_header,
            confirmation_strategy_header=confirmation_strategy_header,
            memory_privacy_mode_header=memory_privacy_mode_header,
            response_mode_header=response_mode_header,
            adaptive_retrieval_header=adaptive_retrieval_header,
        )
        latest_user_text = _latest_user_text(request.messages)
        context = await self._context_for_turn_fail_open(identity, latest_user_text)
        llm_request = self._llm_request(request, context, identity)
        stream = self.runtime.llm_client.stream(llm_request)
        first_event = await _first_output_stream_event(stream)
        return self._stream_response(
            request=request,
            identity=identity,
            stream=stream,
            first_event=first_event,
            completion_id=completion_id,
            created=created,
        )

    async def _stream_response(
        self,
        *,
        request: OpenAIChatCompletionRequest,
        identity: OpenAIProxyIdentity,
        stream: AsyncIterator[LLMStreamEvent],
        first_event: LLMStreamEvent,
        completion_id: str,
        created: int,
    ) -> AsyncIterator[str]:
        accumulated = ""
        emitted_tool_call = False
        yield _sse(
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                delta={"role": "assistant"},
            )
        )
        try:
            for event in (first_event,):
                chunk, text_delta, tool_call_delta = _stream_event_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    event=event,
                    tool_index=0,
                )
                if chunk is None:
                    continue
                accumulated += text_delta
                emitted_tool_call = emitted_tool_call or tool_call_delta
                yield _sse(chunk)
            tool_index = 1 if emitted_tool_call else 0
            async for event in stream:
                chunk, text_delta, tool_call_delta = _stream_event_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    event=event,
                    tool_index=tool_index,
                )
                if chunk is None:
                    continue
                accumulated += text_delta
                if tool_call_delta:
                    emitted_tool_call = True
                    tool_index += 1
                yield _sse(chunk)
        except Exception:
            logger.exception("OpenAI-compatible proxy stream failed after emission")
            yield _sse_error("Upstream stream failed")
            yield "data: [DONE]\n\n"
            return
        yield _sse(
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                delta={},
                finish_reason="tool_calls" if emitted_tool_call else "stop",
            )
        )
        if accumulated:
            await self._record_response_fail_open(identity, accumulated)
        if _include_stream_usage(request):
            yield _sse(
                _usage_chunk_payload(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    usage=_estimated_usage(accumulated),
                )
            )
        yield "data: [DONE]\n\n"

    def _validate_model(self, request: OpenAIChatCompletionRequest) -> None:
        if request.model != self.runtime.settings.openai_proxy_model_id:
            raise ValueError(
                "Unknown model for Atagia OpenAI-compatible proxy: "
                f"{request.model}"
            )

    async def _context_for_turn_fail_open(
        self,
        identity: OpenAIProxyIdentity,
        latest_user_text: str,
    ) -> Any | None:
        try:
            return await SidecarService(self.runtime).get_context(
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                message=latest_user_text,
                mode=identity.mode or identity.assistant_mode_id,
                workspace_id=identity.workspace_id,
                operational_profile=identity.operational_profile,
                operational_signals=identity.operational_signals,
                cross_chat_memory=identity.cross_chat_memory,
                user_persona_id=identity.user_persona_id,
                platform_id=identity.platform_id,
                character_id=identity.character_id,
                active_presence_id=identity.active_presence_id,
                mind_id=identity.mind_id,
                mind_topology=identity.mind_topology,
                embodiment_id=identity.embodiment_id,
                realm_id=identity.realm_id,
                space_id=identity.space_id,
                incognito=identity.incognito,
                message_id=identity.message_id,
                source_seq=identity.source_seq,
                ingest_origin=identity.ingest_origin,
                confirmation_strategy=identity.confirmation_strategy,
                memory_privacy_mode=identity.memory_privacy_mode,
                response_mode=identity.response_mode,
                adaptive_retrieval=identity.adaptive_retrieval,
            )
        except _HARD_CONTEXT_ERRORS:
            raise
        except Exception:
            logger.warning(
                "OpenAI-compatible proxy memory context failed; continuing without Atagia context",
                exc_info=True,
            )
            return None

    def _resolve_identity(
        self,
        request: OpenAIChatCompletionRequest,
        *,
        claimed_user_id: str | None,
        conversation_id_header: str | None,
        assistant_mode_header: str | None,
        mode_header: str | None,
        workspace_id_header: str | None,
        user_persona_id_header: str | None,
        platform_id_header: str | None,
        character_id_header: str | None,
        active_presence_id_header: str | None,
        mind_id_header: str | None,
        mind_topology_header: str | None,
        embodiment_id_header: str | None,
        realm_id_header: str | None,
        space_id_header: str | None,
        incognito_header: str | None,
        cross_chat_memory_header: str | None,
        message_id_header: str | None,
        source_seq_header: str | None,
        response_message_id_header: str | None,
        response_source_seq_header: str | None,
        ingest_origin_header: str | None,
        confirmation_strategy_header: str | None,
        memory_privacy_mode_header: str | None,
        response_mode_header: str | None,
        adaptive_retrieval_header: str | None,
    ) -> OpenAIProxyIdentity:
        metadata = request.metadata or {}
        user_id = _first_text(
            claimed_user_id,
            metadata.get("atagia_user_id"),
            metadata.get("user_id"),
            request.user,
        )
        if user_id is None:
            raise ValueError(
                "OpenAI proxy requests require X-Atagia-User-Id, "
                "metadata.atagia_user_id, or user"
            )
        conversation_id = _first_text(
            conversation_id_header,
            metadata.get("atagia_conversation_id"),
            metadata.get("conversation_id"),
            metadata.get("chat_id"),
        )
        if conversation_id is None:
            raise ValueError(
                "OpenAI proxy requests require X-Atagia-Conversation-Id "
                "or metadata.atagia_conversation_id"
            )
        assistant_mode_id = _first_text(
            assistant_mode_header,
            metadata.get("atagia_assistant_mode"),
            metadata.get("assistant_mode_id"),
            self.runtime.settings.openai_proxy_default_mode,
        )
        mode = _first_text(
            mode_header,
            metadata.get("atagia_mode"),
            assistant_mode_id,
        )
        workspace_id = _first_text(
            workspace_id_header,
            metadata.get("atagia_workspace_id"),
            metadata.get("workspace_id"),
        )
        user_persona_id = _first_text(
            user_persona_id_header,
            metadata.get("atagia_user_persona_id"),
            metadata.get("user_persona_id"),
        )
        platform_id = _first_text(
            platform_id_header,
            metadata.get("atagia_platform_id"),
            metadata.get("platform_id"),
        )
        if platform_id is None:
            raise ValueError(
                "OpenAI proxy requests require X-Atagia-Platform-Id "
                "or metadata.atagia_platform_id"
            )
        character_id = _first_text(
            character_id_header,
            metadata.get("atagia_character_id"),
            metadata.get("character_id"),
        ) or workspace_id
        active_presence_id = _first_text(
            active_presence_id_header,
            metadata.get("atagia_active_presence_id"),
            metadata.get("active_presence_id"),
        )
        mind_id = _first_text(
            mind_id_header,
            metadata.get("atagia_mind_id"),
            metadata.get("mind_id"),
            metadata.get("active_mind_id"),
        )
        mind_topology = _first_text(
            mind_topology_header,
            metadata.get("atagia_mind_topology"),
            metadata.get("mind_topology"),
        )
        embodiment_id = _first_text(
            embodiment_id_header,
            metadata.get("atagia_embodiment_id"),
            metadata.get("embodiment_id"),
            metadata.get("active_embodiment_id"),
        )
        realm_id = _first_text(
            realm_id_header,
            metadata.get("atagia_realm_id"),
            metadata.get("realm_id"),
            metadata.get("active_realm_id"),
        )
        space_id = _first_text(
            space_id_header,
            metadata.get("atagia_space_id"),
            metadata.get("space_id"),
            metadata.get("active_space_id"),
        )
        operational_profile = _first_text(
            metadata.get("atagia_operational_profile"),
            metadata.get("operational_profile"),
        )
        operational_signals = metadata.get("atagia_operational_signals")
        incognito = _first_bool(
            incognito_header,
            metadata.get("atagia_incognito"),
            metadata.get("incognito"),
        )
        cross_chat_memory = _first_bool(
            cross_chat_memory_header,
            metadata.get("atagia_cross_chat_memory"),
            metadata.get("cross_chat_memory"),
        )
        message_id = _first_text(
            message_id_header,
            metadata.get("atagia_message_id"),
            metadata.get("message_id"),
        )
        source_seq = _first_int(
            source_seq_header,
            metadata.get("atagia_source_seq"),
            metadata.get("source_seq"),
        )
        response_message_id = _first_text(
            response_message_id_header,
            metadata.get("atagia_response_message_id"),
            metadata.get("response_message_id"),
        )
        response_source_seq = _first_int(
            response_source_seq_header,
            metadata.get("atagia_response_source_seq"),
            metadata.get("response_source_seq"),
        )
        ingest_origin = _first_text(
            ingest_origin_header,
            metadata.get("atagia_ingest_origin"),
            metadata.get("ingest_origin"),
        )
        confirmation_strategy = _first_text(
            confirmation_strategy_header,
            metadata.get("atagia_confirmation_strategy"),
            metadata.get("confirmation_strategy"),
        )
        memory_privacy_mode = _first_text(
            memory_privacy_mode_header,
            metadata.get("atagia_memory_privacy_mode"),
            metadata.get("memory_privacy_mode"),
        )
        response_mode = _first_text(
            response_mode_header,
            metadata.get("atagia_response_mode"),
            metadata.get("response_mode"),
            request.response_mode.value if request.response_mode is not None else None,
        )
        if response_mode is not None and response_mode not in _RESPONSE_MODE_VALUES:
            logger.warning(
                "OpenAI-compatible proxy ignoring invalid response mode %r; "
                "falling back to the configured default",
                response_mode,
            )
            response_mode = None
        adaptive_retrieval = _first_bool(
            adaptive_retrieval_header,
            metadata.get("atagia_adaptive_retrieval"),
            metadata.get("adaptive_retrieval"),
            request.adaptive_retrieval,
        )
        resolved_cross_chat_memory = (
            not incognito
            if incognito is not None
            else (True if cross_chat_memory is None else cross_chat_memory)
        )
        return OpenAIProxyIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            mind_id=mind_id,
            mind_topology=mind_topology,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            mode=mode,
            incognito=incognito,
            operational_profile=operational_profile,
            operational_signals=(
                dict(operational_signals)
                if isinstance(operational_signals, dict)
                else None
            ),
            cross_chat_memory=resolved_cross_chat_memory,
            message_id=message_id,
            source_seq=source_seq,
            response_message_id=response_message_id,
            response_source_seq=response_source_seq,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
            response_mode=response_mode,
            adaptive_retrieval=adaptive_retrieval,
        )

    def _llm_request(
        self,
        request: OpenAIChatCompletionRequest,
        context: Any,
        identity: OpenAIProxyIdentity,
    ) -> LLMCompletionRequest:
        upstream_model = (
            self.runtime.settings.openai_proxy_upstream_model
            or chat_model(self.runtime.settings)
        )
        system_prompt, messages = _project_messages(request.messages)
        decision = build_injection_decision(system_prompt, context)
        return LLMCompletionRequest(
            model=upstream_model,
            messages=[
                LLMMessage(role="system", content=decision.full_prompt),
                *messages,
            ],
            temperature=request.temperature,
            max_output_tokens=request.max_completion_tokens or request.max_tokens,
            tools=[] if _tool_choice_none(request.tool_choice) else _llm_tools(request.tools),
            metadata={
                "purpose": "chat_reply",
                "user_id": identity.user_id,
                "conversation_id": identity.conversation_id,
                "assistant_mode_id": identity.assistant_mode_id,
                "mode": identity.mode,
                "user_persona_id": identity.user_persona_id,
                "platform_id": identity.platform_id,
                "character_id": identity.character_id,
                "active_presence_id": identity.active_presence_id,
                "mind_id": identity.mind_id,
                "mind_topology": identity.mind_topology,
                "embodiment_id": identity.embodiment_id,
                "realm_id": identity.realm_id,
                "space_id": identity.space_id,
                "incognito": identity.incognito,
                "cross_chat_memory": identity.cross_chat_memory,
                "message_id": identity.message_id,
                "source_seq": identity.source_seq,
                "response_message_id": identity.response_message_id,
                "response_source_seq": identity.response_source_seq,
                "ingest_origin": identity.ingest_origin,
                "confirmation_strategy": identity.confirmation_strategy,
                "memory_privacy_mode": identity.memory_privacy_mode,
                "atagia_openai_proxy_model": request.model,
                "openai_tool_choice": request.tool_choice,
            },
        )

    async def _record_response_fail_open(
        self,
        identity: OpenAIProxyIdentity,
        text: str,
    ) -> None:
        if not text:
            return
        try:
            await self._record_response(identity, text)
        except Exception:
            logger.warning(
                "OpenAI-compatible proxy response persistence failed",
                exc_info=True,
            )

    async def _record_response(
        self,
        identity: OpenAIProxyIdentity,
        text: str,
    ) -> None:
        await SidecarService(self.runtime).add_response(
            user_id=identity.user_id,
            conversation_id=identity.conversation_id,
            text=text,
            operational_profile=identity.operational_profile,
            operational_signals=identity.operational_signals,
            user_persona_id=identity.user_persona_id,
            platform_id=identity.platform_id,
            character_id=identity.character_id,
            active_presence_id=identity.active_presence_id,
            mind_id=identity.mind_id,
            mind_topology=identity.mind_topology,
            embodiment_id=identity.embodiment_id,
            realm_id=identity.realm_id,
            space_id=identity.space_id,
            mode=identity.mode,
            incognito=identity.incognito,
            message_id=identity.response_message_id,
            source_seq=identity.response_source_seq,
            ingest_origin=identity.ingest_origin,
            confirmation_strategy=identity.confirmation_strategy,
            memory_privacy_mode=identity.memory_privacy_mode,
        )


def _project_messages(
    messages: list[OpenAIProxyMessage],
) -> tuple[str, list[LLMMessage]]:
    system_parts: list[str] = []
    projected: list[LLMMessage] = []
    for message in messages:
        role = message.role.strip().lower()
        text = (
            _tool_content_to_text(message.content)
            if role == "tool"
            else message_to_text(message.content)
        )
        tool_calls = _internal_tool_calls(message.tool_calls)
        if not text and not tool_calls:
            continue
        if role in {"system", "developer"}:
            system_parts.append(text)
            continue
        name = message.name
        if role == "tool":
            name = _first_text(message.tool_call_id, message.name)
        if role not in {"user", "assistant", "tool"}:
            role = "user"
        projected.append(
            LLMMessage(
                role=role,
                content=text,
                name=name,
                tool_calls=tool_calls if role == "assistant" else [],
            )
        )
    return "\n\n".join(system_parts), projected


def _tool_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _llm_tools(tools: list[dict[str, Any]]) -> list[LLMToolSpec]:
    converted: list[LLMToolSpec] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("OpenAI proxy tools must be objects")
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict):
            raise ValueError("OpenAI proxy tool.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("OpenAI proxy function tools require a non-empty name")
        parameters = function.get("parameters", function.get("input_schema", {}))
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError("OpenAI proxy function tool parameters must be an object")
        converted.append(
            LLMToolSpec(
                name=name.strip(),
                description=str(function.get("description") or ""),
                input_schema=parameters,
            )
        )
    return converted


def _tool_choice_none(tool_choice: Any) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"


def _internal_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", "")
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", tool_call.get("input", {}))
        converted.append(
            {
                "id": str(tool_call.get("id") or f"call_atagia_{index}"),
                "type": str(tool_call.get("type") or "function"),
                "name": str(name or "tool"),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            }
        )
    return converted


def _latest_user_text(messages: list[OpenAIProxyMessage]) -> str:
    for message in reversed(messages):
        if message.role.strip().lower() == "user":
            text = message_to_text(message.content).strip()
            if text:
                return text
    raise ValueError("OpenAI proxy requests require at least one non-empty user message")


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ValueError("OpenAI proxy source_seq values must be integers") from exc
    return None


def _completion_id() -> str:
    return f"chatcmpl-atagia-{uuid.uuid4().hex}"


def _completion_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    response: LLMCompletionResponse,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.output_text,
    }
    tool_calls = _openai_tool_calls(response.tool_calls)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            }
        ],
        "usage": response.usage or _estimated_usage(response.output_text),
    }


def _chunk_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }


def _usage_chunk_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    }


def _include_stream_usage(request: OpenAIChatCompletionRequest) -> bool:
    stream_options = request.stream_options
    if not isinstance(stream_options, dict):
        return False
    return _first_bool(stream_options.get("include_usage")) is True


async def _first_output_stream_event(
    stream: AsyncIterator[LLMStreamEvent],
) -> LLMStreamEvent:
    try:
        while True:
            event = await anext(stream)
            if _stream_event_is_output(event):
                return event
    except StopAsyncIteration as exc:
        raise LLMError("LLM stream produced no output events") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError("LLM stream failed before producing output") from exc


def _stream_event_is_output(event: LLMStreamEvent) -> bool:
    return (event.type == "text" and bool(event.content)) or event.type == "tool_call"


def _stream_event_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    event: LLMStreamEvent,
    tool_index: int,
) -> tuple[dict[str, Any] | None, str, bool]:
    if event.type == "text" and event.content:
        return (
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"content": event.content},
            ),
            event.content,
            False,
        )
    if event.type == "tool_call":
        return (
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"tool_calls": _openai_stream_tool_calls(event.payload, tool_index)},
            ),
            "",
            True,
        )
    return None, "", False


def _openai_stream_tool_calls(
    tool_call: dict[str, Any],
    index: int,
) -> list[dict[str, Any]]:
    normalized = _openai_tool_calls([tool_call])
    if not normalized:
        return []
    call = normalized[0]
    call["index"] = index
    return [call]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> str:
    return _sse(
        {
            "error": {
                "message": message,
                "type": "atagia_upstream_stream_error",
            }
        }
    )


def _openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "tool")
        raw_arguments = (
            function.get("arguments")
            if "arguments" in function
            else tool_call.get("arguments", tool_call.get("input", {}))
        )
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, str)
            else json.dumps(raw_arguments, ensure_ascii=False)
        )
        normalized.append(
            {
                "id": str(tool_call.get("id") or f"call_atagia_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized


def _estimated_usage(output_text: str) -> dict[str, int]:
    completion_tokens = max(1, len(output_text) // 4) if output_text else 0
    return {
        "prompt_tokens": 0,
        "completion_tokens": completion_tokens,
        "total_tokens": completion_tokens,
    }
