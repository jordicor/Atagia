"""OpenAI-compatible chat-completion proxy backed by Atagia memory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
import json
import logging
import sqlite3
import time
import uuid
from typing import Any
from starlette.concurrency import run_in_threadpool

from atagia.core.conversation_namespace import (
    ConversationNamespaceSnapshot,
    capture_conversation_namespace_snapshot,
)
from atagia.core.ids import generate_prefixed_id
from atagia.core.proxy_turn_repository import (
    ProxyDurableJobInsert,
    ProxyTurnClaim,
    ProxyTurnRepository,
    ProxyTurnRequestMessage,
    ProxyTurnReservation,
    ProxyTurnResponseMessage,
    proxy_turn_pair_id,
)
from atagia.core.repositories import MessageRepository
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.integrations.message_projection import message_to_text, tool_message_to_text
from atagia.integrations.prompt_injection import build_injection_decision
from atagia.core.mind_repository import MindNotFoundError
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
    MessageIdConflictError,
    ProxyTurnError,
    SourceSequenceConflictError,
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
    normalize_completion_finish_reason,
)
from atagia.services.openai_proxy_contract import (
    OpenAIProxyProtocolError,
    completion_envelope_from_response,
    normalize_openai_tool_calls,
    resolve_proxy_turn_claims,
    validate_proxy_turn_claim_relationships,
)
from atagia.services.prompt_authority import (
    PromptAuthorityContext,
    resolve_request_authority_context,
)
from atagia.services.proxy_transcript import (
    PROXY_TRANSCRIPT_METADATA_KEY,
    ProxyInputProjection,
    ProxyTranscriptError,
    bind_input_metadata,
    build_response_metadata,
    client_request_fingerprint,
    derive_proxy_input,
    final_provider_fingerprint,
    response_retrieval_text,
)
from atagia.services.proxy_turn_service import (
    ProxyTerminalJobPlan,
    dispatch_proxy_terminal_jobs,
    finalize_proxy_turn,
)
from atagia.services.request_controls import resolve_memory_scope_controls
from atagia.services.request_budgets import (
    RequestBudgetLimits,
    validate_openai_proxy_request_budget,
)
from atagia.services.sidecar_service import SidecarService

logger = logging.getLogger(__name__)

_RESPONSE_MODE_VALUES: frozenset[str] = frozenset(mode.value for mode in ResponseMode)

_HARD_CONTEXT_ERRORS = (
    AssistantModeMismatchError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    MessageIdConflictError,
    MindNotFoundError,
    OperationalProfileNotAuthorizedError,
    SourceSequenceConflictError,
    UnknownAssistantModeError,
    UnknownOperationalProfileError,
    UserDeletedError,
    WorkspaceMismatchError,
    WorkspaceNotFoundError,
)

_TRANSIENT_CONTEXT_ERRORS = (
    ConnectionError,
    TimeoutError,
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


@dataclass(frozen=True, slots=True)
class PreparedProxyTurn:
    """One validated and durably reserved proxy turn, or a completed replay."""

    identity: OpenAIProxyIdentity
    authority: PromptAuthorityContext
    input_projection: ProxyInputProjection
    input_metadata: dict[str, Any]
    reservation: ProxyTurnReservation


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
        incognito_header: str | list[str] | None = None,
        cross_chat_memory_header: str | list[str] | None = None,
        message_id_header: str | None = None,
        source_seq_header: str | None = None,
        response_message_id_header: str | None = None,
        response_source_seq_header: str | None = None,
        ingest_origin_header: str | None = None,
        confirmation_strategy_header: str | None = None,
        memory_privacy_mode_header: str | None = None,
        response_mode_header: str | None = None,
        adaptive_retrieval_header: str | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
    ) -> dict[str, Any]:
        self._validate_model(request)
        await self._validate_request_budget(request)
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
        prepared = await self._prepare_turn(
            request,
            identity,
            prompt_authority_context=prompt_authority_context,
        )
        replay = prepared.reservation.replay
        if replay is not None:
            return _completion_payload_from_envelope(
                completion_id=completion_id,
                created=created,
                model=request.model,
                envelope=replay.replay_envelope,
            )
        claim = prepared.reservation.claim
        if claim is None:
            raise RuntimeError(
                "Proxy turn reservation returned neither claim nor replay"
            )

        try:
            async with self._renewing_claim(claim):
                context = await self._context_for_turn_fail_open(
                    prepared.identity,
                    prepared.input_projection,
                    message_metadata=prepared.input_metadata,
                    prompt_authority_context=_scoped_authority_context(
                        prepared.authority,
                        prepared.identity,
                        purpose="sidecar_context",
                    ),
                )
                llm_request = self._llm_request(request, context, prepared.identity)
                claim = await self._establish_final_fingerprint(
                    claim,
                    final_provider_fingerprint(llm_request),
                )
                response = await self.runtime.llm_client.complete(llm_request)
                envelope = completion_envelope_from_response(response).as_dict()
                job_plan, transcript_response = await self._terminal_artifacts(
                    request=request,
                    prepared=prepared,
                    claim=claim,
                    envelope=envelope,
                    structured_tool_calls=response.tool_calls,
                )
                durable_jobs = await finalize_proxy_turn(
                    self.runtime,
                    claim=claim,
                    response=transcript_response,
                    job_plan=job_plan,
                )
        except BaseException as exc:
            await self._abandon_pre_emission_best_effort(claim, exc)
            if isinstance(exc, ProxyTurnError):
                raise _proxy_protocol_error(exc) from exc
            raise

        await self._dispatch_terminal_jobs_best_effort(durable_jobs)
        return _completion_payload_from_envelope(
            completion_id=completion_id,
            created=created,
            model=request.model,
            envelope=envelope,
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
        incognito_header: str | list[str] | None = None,
        cross_chat_memory_header: str | list[str] | None = None,
        message_id_header: str | None = None,
        source_seq_header: str | None = None,
        response_message_id_header: str | None = None,
        response_source_seq_header: str | None = None,
        ingest_origin_header: str | None = None,
        confirmation_strategy_header: str | None = None,
        memory_privacy_mode_header: str | None = None,
        response_mode_header: str | None = None,
        adaptive_retrieval_header: str | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
    ) -> AsyncIterator[str]:
        self._validate_model(request)
        await self._validate_request_budget(request)
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
        prepared = await self._prepare_turn(
            request,
            identity,
            prompt_authority_context=prompt_authority_context,
        )
        replay = prepared.reservation.replay
        if replay is not None:
            return _stream_replay(
                request=request,
                envelope=replay.replay_envelope,
                completion_id=completion_id,
                created=created,
            )
        claim = prepared.reservation.claim
        if claim is None:
            raise RuntimeError(
                "Proxy turn reservation returned neither claim nor replay"
            )
        try:
            async with self._renewing_claim(claim):
                context = await self._context_for_turn_fail_open(
                    prepared.identity,
                    prepared.input_projection,
                    message_metadata=prepared.input_metadata,
                    prompt_authority_context=_scoped_authority_context(
                        prepared.authority,
                        prepared.identity,
                        purpose="sidecar_context",
                    ),
                )
                llm_request = self._llm_request(request, context, prepared.identity)
                claim = await self._establish_final_fingerprint(
                    claim,
                    final_provider_fingerprint(llm_request),
                )
                stream = self.runtime.llm_client.stream(llm_request)
                first_event = await _first_output_or_done_stream_event(stream)
                claim = await self._mark_emission_started(claim)
        except BaseException as exc:
            await self._abandon_pre_emission_best_effort(claim, exc)
            raise
        return self._stream_response(
            request=request,
            prepared=prepared,
            claim=claim,
            stream=stream,
            first_event=first_event,
            completion_id=completion_id,
            created=created,
        )

    async def _stream_response(
        self,
        *,
        request: OpenAIChatCompletionRequest,
        prepared: PreparedProxyTurn,
        claim: ProxyTurnClaim,
        stream: AsyncIterator[LLMStreamEvent],
        first_event: LLMStreamEvent,
        completion_id: str,
        created: int,
    ) -> AsyncIterator[str]:
        async with self._renewing_claim(claim) as ownership_lost:
            async for chunk in self._stream_response_owned(
                request=request,
                prepared=prepared,
                claim=claim,
                stream=stream,
                first_event=first_event,
                completion_id=completion_id,
                created=created,
                ownership_lost=ownership_lost,
            ):
                yield chunk

    async def _stream_response_owned(
        self,
        *,
        request: OpenAIChatCompletionRequest,
        prepared: PreparedProxyTurn,
        claim: ProxyTurnClaim,
        stream: AsyncIterator[LLMStreamEvent],
        first_event: LLMStreamEvent,
        completion_id: str,
        created: int,
        ownership_lost: asyncio.Event,
    ) -> AsyncIterator[str]:
        accumulated = ""
        tool_calls: list[dict[str, Any]] = []
        provider_usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        tool_index = 0
        try:
            yield _sse(
                _chunk_payload(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    delta={"role": "assistant"},
                )
            )
            saw_done = False
            async for event in _prepend_stream_event(first_event, stream):
                if ownership_lost.is_set():
                    raise LLMError("Proxy stream ownership is no longer current")
                if saw_done:
                    raise LLMError("LLM stream produced events after its done event")
                if event.type == "done":
                    saw_done = True
                    event_usage = event.payload.get("usage")
                    if isinstance(event_usage, dict):
                        provider_usage = dict(event_usage)
                    event_finish_reason = event.payload.get("finish_reason")
                    if event_finish_reason is not None:
                        finish_reason = normalize_completion_finish_reason(
                            event_finish_reason,
                            has_tool_calls=bool(tool_calls),
                        )
                    continue
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
                    tool_calls.append(dict(event.payload))
                    tool_index += 1
                yield _sse(chunk)
            if ownership_lost.is_set():
                raise LLMError("Proxy stream ownership is no longer current")
        except BaseException as exc:
            await self._mark_ambiguous_best_effort(claim, exc)
            if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                raise
            logger.exception("OpenAI-compatible proxy stream failed after emission")
            yield _sse_error("Upstream stream failed")
            return
        finish_reason = normalize_completion_finish_reason(
            finish_reason,
            has_tool_calls=bool(tool_calls),
        )
        response = LLMCompletionResponse(
            provider="proxy-stream",
            model=request.model,
            output_text=accumulated,
            tool_calls=tool_calls,
            usage=provider_usage or {},
            finish_reason=finish_reason,
        )
        envelope = completion_envelope_from_response(response).as_dict()
        try:
            job_plan, transcript_response = await self._terminal_artifacts(
                request=request,
                prepared=prepared,
                claim=claim,
                envelope=envelope,
                structured_tool_calls=tool_calls,
            )
            durable_jobs = await finalize_proxy_turn(
                self.runtime,
                claim=claim,
                response=transcript_response,
                job_plan=job_plan,
            )
        except BaseException as exc:
            await self._mark_ambiguous_best_effort(claim, exc)
            if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                raise
            logger.exception("OpenAI-compatible proxy terminal stream commit failed")
            yield _sse_error("Stream completion could not be committed")
            return
        await self._dispatch_terminal_jobs_best_effort(durable_jobs)
        yield _sse(
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                delta={},
                finish_reason=envelope.get("finish_reason"),
            )
        )
        replay_usage = envelope.get("usage")
        if _include_stream_usage(request) and isinstance(replay_usage, dict):
            yield _sse(
                _usage_chunk_payload(
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    usage=replay_usage,
                )
            )
        yield "data: [DONE]\n\n"

    async def _prepare_turn(
        self,
        request: OpenAIChatCompletionRequest,
        identity: OpenAIProxyIdentity,
        *,
        prompt_authority_context: PromptAuthorityContext | None,
    ) -> PreparedProxyTurn:
        if identity.message_id is None and identity.response_message_id is None:
            identity = replace(
                identity,
                message_id=generate_prefixed_id("msg"),
                response_message_id=generate_prefixed_id("msg"),
            )
        if identity.message_id is None or identity.response_message_id is None:
            raise OpenAIProxyProtocolError(
                400,
                "message_id and response_message_id must be supplied together",
                code="incomplete_message_id_pair",
            )
        try:
            input_projection = derive_proxy_input(request.messages)
        except ProxyTranscriptError as exc:
            raise OpenAIProxyProtocolError(
                400,
                str(exc),
                code="invalid_proxy_transcript",
            ) from exc
        resolved_authority = resolve_request_authority_context(
            prompt_authority_context,
            user_id=identity.user_id,
            purpose="openai_proxy",
        )
        authority = _scoped_authority_context(
            resolved_authority,
            identity,
            purpose="openai_proxy",
        )
        if authority is None:
            raise RuntimeError("Proxy authority resolution returned no context")
        fingerprint = client_request_fingerprint(
            request,
            resolved_identity=identity,
            authority=authority,
        )
        pair_id = proxy_turn_pair_id(
            user_id=identity.user_id,
            conversation_id=identity.conversation_id,
            request_message_id=identity.message_id,
            response_message_id=identity.response_message_id,
        )
        lookup_connection = await self.runtime.open_connection()
        try:
            await TranscriptRebuildRepository(
                lookup_connection,
                self.runtime.clock,
            ).require_user_available(identity.user_id)
            existing_run = await ProxyTurnRepository(
                lookup_connection,
                self.runtime.clock,
            ).get_run(pair_id)
        finally:
            await lookup_connection.close()
        expected_namespace_snapshot = None
        if existing_run is None:
            expected_namespace_snapshot = await self._ensure_proxy_namespace(identity)

        connection = await self.runtime.open_connection()
        try:
            repository = ProxyTurnRepository(connection, self.runtime.clock)
            parent_response_message_id: str | None = None
            if input_projection.message_role == "tool":
                parent_response_message_id = await self._tool_parent_for_reservation(
                    connection,
                    repository,
                    pair_id=pair_id,
                    identity=identity,
                    input_projection=input_projection,
                )
            input_metadata = bind_input_metadata(
                input_projection,
                pair_id=pair_id,
                request_message_id=identity.message_id,
                response_message_id=identity.response_message_id,
                client_request_fingerprint=fingerprint,
                parent_response_message_id=parent_response_message_id,
            )
            try:
                reservation = await repository.reserve(
                    request_message=ProxyTurnRequestMessage(
                        message_id=identity.message_id,
                        response_message_id=identity.response_message_id,
                        user_id=identity.user_id,
                        conversation_id=identity.conversation_id,
                        role=input_projection.message_role,  # type: ignore[arg-type]
                        text=input_projection.text,
                        metadata=input_metadata,
                        source_seq=identity.source_seq,
                        response_source_seq=identity.response_source_seq,
                    ),
                    client_request_fingerprint=fingerprint,
                    expected_namespace_snapshot=expected_namespace_snapshot,
                )
            except ProxyTurnError as exc:
                raise _proxy_protocol_error(exc) from exc
        finally:
            await connection.close()
        return PreparedProxyTurn(
            identity=identity,
            authority=authority,
            input_projection=input_projection,
            input_metadata=input_metadata,
            reservation=reservation,
        )

    async def _tool_parent_for_reservation(
        self,
        connection: Any,
        repository: ProxyTurnRepository,
        *,
        pair_id: str,
        identity: OpenAIProxyIdentity,
        input_projection: ProxyInputProjection,
    ) -> str:
        existing_run = await repository.get_run(pair_id)
        if existing_run is not None:
            existing_input = await MessageRepository(
                connection,
                self.runtime.clock,
            ).get_message_for_idempotency(identity.message_id or "")
            metadata = existing_input.get("metadata_json") if existing_input else None
            transcript = (
                metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
                if isinstance(metadata, dict)
                else None
            )
            tool_projection = (
                transcript.get("tool_projection")
                if isinstance(transcript, dict)
                else None
            )
            parent = (
                tool_projection.get("parent_response_message_id")
                if isinstance(tool_projection, dict)
                else None
            )
            if isinstance(parent, str) and parent:
                return parent
        try:
            return await repository.find_parent_tool_response(
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                expected_calls=input_projection.parent_tool_calls,
                response_message_hint=input_projection.parent_response_hint,
            )
        except ProxyTurnError as exc:
            raise _proxy_protocol_error(exc) from exc

    async def _ensure_proxy_namespace(
        self,
        identity: OpenAIProxyIdentity,
    ) -> ConversationNamespaceSnapshot:
        sidecar = SidecarService(self.runtime)
        connection = await self.runtime.open_connection()
        try:
            await sidecar.ensure_user_exists(connection, identity.user_id)
            await sidecar.ensure_conversation(
                connection,
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                workspace_id=identity.workspace_id,
                assistant_mode_id=identity.mode or identity.assistant_mode_id,
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
                mode=identity.mode or identity.assistant_mode_id,
                incognito=identity.incognito,
            )
            await connection.execute("BEGIN")
            snapshot = await capture_conversation_namespace_snapshot(
                connection,
                self.runtime.clock,
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
            )
            if snapshot is None:
                raise ConversationNotFoundError("Conversation not found for user")
            self._validate_proxy_namespace_identity(
                identity=identity,
                snapshot=snapshot,
            )
            await connection.commit()
            return snapshot
        except Exception:
            if connection.in_transaction:
                await connection.rollback()
            raise
        finally:
            await connection.close()

    @staticmethod
    def _validate_proxy_namespace_identity(
        *,
        identity: OpenAIProxyIdentity,
        snapshot: ConversationNamespaceSnapshot,
    ) -> None:
        constrained_text_fields = (
            ("workspace_id", identity.workspace_id, snapshot.workspace_id),
            (
                "user_persona_id",
                identity.user_persona_id,
                snapshot.user_persona_id,
            ),
            ("platform_id", identity.platform_id, snapshot.platform_id),
            ("character_id", identity.character_id, snapshot.character_id),
            (
                "active_presence_id",
                identity.active_presence_id,
                snapshot.active_presence_id,
            ),
            ("active_mind_id", identity.mind_id, snapshot.active_mind_id),
            ("mind_topology", identity.mind_topology, snapshot.mind_topology),
            (
                "active_embodiment_id",
                identity.embodiment_id,
                snapshot.active_embodiment_id,
            ),
            ("active_realm_id", identity.realm_id, snapshot.active_realm_id),
            ("active_space_id", identity.space_id, snapshot.active_space_id),
            (
                "assistant_mode_id",
                identity.assistant_mode_id,
                snapshot.assistant_mode_id,
            ),
            ("mode", identity.mode, snapshot.mode),
        )
        for field_name, expected, actual in constrained_text_fields:
            if expected is not None and str(expected) != actual:
                raise ConversationNotFoundError(
                    f"Conversation {field_name} does not match the proxy request"
                )
        requires_isolation = (
            identity.incognito is True or identity.cross_chat_memory is False
        )
        if requires_isolation and (
            not snapshot.incognito or not snapshot.isolated_mode
        ):
            raise ConversationNotFoundError(
                "Conversation incognito mode does not match the proxy request"
            )
        if (
            identity.incognito is False
            and identity.cross_chat_memory is not False
            and snapshot.incognito
        ):
            raise ConversationNotFoundError(
                "Conversation incognito mode does not match the proxy request"
            )

    async def _establish_final_fingerprint(
        self,
        claim: ProxyTurnClaim,
        fingerprint: str,
    ) -> ProxyTurnClaim:
        connection = await self.runtime.open_connection()
        try:
            try:
                return await ProxyTurnRepository(
                    connection,
                    self.runtime.clock,
                ).establish_final_fingerprint(claim, fingerprint)
            except ProxyTurnError as exc:
                raise _proxy_protocol_error(exc) from exc
        finally:
            await connection.close()

    async def _mark_emission_started(
        self,
        claim: ProxyTurnClaim,
    ) -> ProxyTurnClaim:
        connection = await self.runtime.open_connection()
        try:
            try:
                return await ProxyTurnRepository(
                    connection,
                    self.runtime.clock,
                ).mark_emission_started(claim)
            except ProxyTurnError as exc:
                raise _proxy_protocol_error(exc) from exc
        finally:
            await connection.close()

    async def _renew_claim(self, claim: ProxyTurnClaim) -> bool:
        connection = await self.runtime.open_connection()
        try:
            return await ProxyTurnRepository(
                connection,
                self.runtime.clock,
            ).renew(claim)
        finally:
            await connection.close()

    @asynccontextmanager
    async def _renewing_claim(
        self,
        claim: ProxyTurnClaim,
    ) -> AsyncIterator[asyncio.Event]:
        ownership_lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._claim_heartbeat(claim, ownership_lost))
        try:
            yield ownership_lost
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _claim_heartbeat(
        self,
        claim: ProxyTurnClaim,
        ownership_lost: asyncio.Event,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(10)
                if not await self._renew_claim(claim):
                    ownership_lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            ownership_lost.set()
            logger.warning(
                "Proxy turn lease heartbeat failed; fenced terminal writes remain enforced",
                exc_info=True,
            )

    async def _terminal_artifacts(
        self,
        *,
        request: OpenAIChatCompletionRequest,
        prepared: PreparedProxyTurn,
        claim: ProxyTurnClaim,
        envelope: dict[str, Any],
        structured_tool_calls: list[dict[str, Any]] | None = None,
    ) -> tuple[ProxyTerminalJobPlan, ProxyTurnResponseMessage]:
        content = str(envelope.get("content") or "")
        raw_tool_calls = envelope.get("tool_calls")
        replay_tool_calls = (
            [dict(item) for item in raw_tool_calls if isinstance(item, dict)]
            if isinstance(raw_tool_calls, list)
            else []
        )
        tool_calls = (
            [dict(item) for item in structured_tool_calls]
            if structured_tool_calls is not None
            else replay_tool_calls
        )
        usage = envelope.get("usage")
        response_text = response_retrieval_text(content, tool_calls)
        response_occurred_at = self.runtime.clock.now().isoformat()
        if claim.final_provider_fingerprint is None:
            raise RuntimeError("Proxy terminal artifacts require a final fingerprint")
        metadata = build_response_metadata(
            pair_id=claim.pair_id,
            request_message_id=claim.request_message_id,
            response_message_id=claim.response_message_id,
            client_request_fingerprint=claim.client_request_fingerprint,
            final_provider_fingerprint=claim.final_provider_fingerprint,
            content=content,
            tool_calls=tool_calls,
            replay_tool_calls=replay_tool_calls,
            finish_reason=(
                str(envelope["finish_reason"])
                if envelope.get("finish_reason") is not None
                else None
            ),
            usage=usage if isinstance(usage, dict) else None,
            model=request.model,
        )
        return ProxyTerminalJobPlan(
            response_retrieval_text=response_text,
            response_occurred_at=response_occurred_at,
            operational_profile=prepared.identity.operational_profile,
            operational_signals=prepared.identity.operational_signals,
            ingest_origin=prepared.identity.ingest_origin,
            confirmation_strategy=prepared.identity.confirmation_strategy,
            memory_privacy_mode=prepared.identity.memory_privacy_mode,
            authority=prepared.authority,
        ), ProxyTurnResponseMessage(
            text=response_text,
            metadata=metadata,
            source_seq=claim.response_source_seq,
            occurred_at=response_occurred_at,
        )

    async def _abandon_pre_emission_best_effort(
        self,
        claim: ProxyTurnClaim,
        exc: BaseException,
    ) -> None:
        connection = None
        try:
            connection = await self.runtime.open_connection()
            await ProxyTurnRepository(
                connection,
                self.runtime.clock,
            ).abandon_pre_emission(claim, Exception(str(exc)))
        except Exception:
            logger.warning(
                "Failed to release proxy turn after pre-emission failure", exc_info=True
            )
        finally:
            if connection is not None:
                await connection.close()

    async def _mark_ambiguous_best_effort(
        self,
        claim: ProxyTurnClaim,
        exc: BaseException,
    ) -> None:
        connection = None
        try:
            connection = await self.runtime.open_connection()
            await ProxyTurnRepository(
                connection,
                self.runtime.clock,
            ).mark_ambiguous_exposed(claim, error_message=str(exc))
        except Exception:
            logger.warning(
                "Failed to mark an exposed proxy stream ambiguous", exc_info=True
            )
        finally:
            if connection is not None:
                await connection.close()

    async def _dispatch_terminal_jobs_best_effort(
        self,
        durable_jobs: list[ProxyDurableJobInsert],
    ) -> None:
        try:
            await dispatch_proxy_terminal_jobs(self.runtime, durable_jobs)
        except Exception:
            logger.warning(
                "Committed proxy jobs were not transiently dispatched; durable recovery will retry",
                exc_info=True,
            )

    def _validate_model(self, request: OpenAIChatCompletionRequest) -> None:
        if request.model != self.runtime.settings.openai_proxy_model_id:
            raise ValueError(
                f"Unknown model for Atagia OpenAI-compatible proxy: {request.model}"
            )

    async def _validate_request_budget(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> None:
        await run_in_threadpool(
            validate_openai_proxy_request_budget,
            request,
            limits=RequestBudgetLimits.from_settings(self.runtime.settings),
        )

    async def _context_for_turn_fail_open(
        self,
        identity: OpenAIProxyIdentity,
        input_projection: ProxyInputProjection,
        *,
        message_metadata: dict[str, Any],
        prompt_authority_context: PromptAuthorityContext | None = None,
    ) -> Any | None:
        try:
            return await SidecarService(self.runtime).get_context(
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                message=input_projection.text,
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
                prompt_authority_context=prompt_authority_context,
                message_role=input_projection.message_role,
                message_metadata=message_metadata,
            )
        except _HARD_CONTEXT_ERRORS:
            raise
        except Exception as exc:
            if _is_transient_context_error(exc):
                logger.warning(
                    "OpenAI-compatible proxy memory context failed; continuing without Atagia context",
                    exc_info=True,
                )
                return None
            logger.exception("OpenAI-compatible proxy memory context failed internally")
            raise OpenAIProxyProtocolError(
                500,
                "Internal memory context failure",
                error_type="server_error",
                code="memory_context_internal_error",
            ) from exc

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
        incognito_header: str | list[str] | None,
        cross_chat_memory_header: str | list[str] | None,
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
        character_id = (
            _first_text(
                character_id_header,
                metadata.get("atagia_character_id"),
                metadata.get("character_id"),
            )
            or workspace_id
        )
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
        typed_memory_scope_fields = {
            field_name: getattr(request, field_name)
            for field_name in ("incognito", "cross_chat_memory")
            if field_name in request.model_fields_set
        }
        memory_scope = resolve_memory_scope_controls(
            typed_fields=typed_memory_scope_fields,
            metadata=metadata,
            headers={
                "X-Atagia-Incognito": incognito_header,
                "X-Atagia-Cross-Chat-Memory": cross_chat_memory_header,
            },
        )
        turn_claims = resolve_proxy_turn_claims(
            metadata,
            request_message_id_header=message_id_header,
            request_source_seq_header=source_seq_header,
            response_message_id_header=response_message_id_header,
            response_source_seq_header=response_source_seq_header,
        )
        validate_proxy_turn_claim_relationships(turn_claims, require_id_pair=True)
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
        response_mode = _resolve_response_mode_claim(
            {
                "header": response_mode_header,
                "metadata.atagia_response_mode": metadata.get("atagia_response_mode"),
                "metadata.response_mode": metadata.get("response_mode"),
                "body.response_mode": (
                    request.response_mode.value
                    if request.response_mode is not None
                    else None
                ),
            }
        )
        if response_mode is None:
            response_mode = str(self.runtime.settings.response_mode)
        adaptive_retrieval = _resolve_adaptive_retrieval_claim(
            {
                "header": adaptive_retrieval_header,
                "metadata.atagia_adaptive_retrieval": metadata.get(
                    "atagia_adaptive_retrieval"
                ),
                "metadata.adaptive_retrieval": metadata.get("adaptive_retrieval"),
                "body.adaptive_retrieval": request.adaptive_retrieval,
            }
        )
        if adaptive_retrieval is None:
            adaptive_retrieval = bool(self.runtime.settings.adaptive_retrieval)
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
            incognito=memory_scope.incognito,
            operational_profile=operational_profile,
            operational_signals=(
                dict(operational_signals)
                if isinstance(operational_signals, dict)
                else None
            ),
            cross_chat_memory=memory_scope.cross_chat_memory,
            message_id=turn_claims.request_message_id,
            source_seq=turn_claims.request_source_seq,
            response_message_id=turn_claims.response_message_id,
            response_source_seq=turn_claims.response_source_seq,
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
            max_output_tokens=_effective_external_output_ceiling(
                request,
                server_max_output_tokens=(
                    getattr(
                        self.runtime.settings,
                        "openai_proxy_max_output_tokens",
                        8192,
                    )
                ),
            ),
            tools=[]
            if _tool_choice_none(request.tool_choice)
            else _llm_tools(request.tools),
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
            external_answer=True,
        )


def _project_messages(
    messages: list[OpenAIProxyMessage],
) -> tuple[str, list[LLMMessage]]:
    system_parts: list[str] = []
    projected: list[LLMMessage] = []
    for message in messages:
        role = message.role.strip().lower()
        text = (
            tool_message_to_text(message.content)
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


def _scoped_authority_context(
    context: PromptAuthorityContext | None,
    identity: OpenAIProxyIdentity,
    *,
    purpose: str,
) -> PromptAuthorityContext | None:
    if context is None:
        return None
    if context.user_id is not None and context.user_id != identity.user_id:
        raise ValueError("Prompt authority user_id does not match the resolved user_id")
    return replace(
        context,
        user_id=identity.user_id,
        purpose=purpose,
    )


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
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments),
            }
        )
    return converted


def _resolve_response_mode_claim(claims: Mapping[str, Any]) -> str | None:
    """Resolve one consistent explicit response_mode; invalid or conflicting → 400."""

    supplied: list[str] = []
    for value in claims.values():
        if value is None:
            continue
        normalized = value.strip() if isinstance(value, str) else None
        if not normalized or normalized not in _RESPONSE_MODE_VALUES:
            raise OpenAIProxyProtocolError(
                400,
                "OpenAI proxy response_mode claims must be one of: "
                f"{', '.join(sorted(_RESPONSE_MODE_VALUES))}",
                param="response_mode",
                code="invalid_response_mode",
            )
        supplied.append(normalized)
    if not supplied:
        return None
    canonical = supplied[0]
    if any(value != canonical for value in supplied[1:]):
        raise OpenAIProxyProtocolError(
            400,
            "Conflicting OpenAI proxy response_mode claims",
            param="response_mode",
            code="conflicting_response_mode",
        )
    return canonical


def _resolve_adaptive_retrieval_claim(claims: Mapping[str, Any]) -> bool | None:
    """Resolve one consistent explicit adaptive_retrieval flag; invalid or conflicting → 400."""

    supplied: list[bool] = []
    for value in claims.values():
        if value is None:
            continue
        parsed = _parse_bool_claim(value)
        if parsed is None:
            raise OpenAIProxyProtocolError(
                400,
                "OpenAI proxy adaptive_retrieval claims must be a boolean "
                "(true/false, 1/0, yes/no, on/off)",
                param="adaptive_retrieval",
                code="invalid_adaptive_retrieval",
            )
        supplied.append(parsed)
    if not supplied:
        return None
    canonical = supplied[0]
    if any(value is not canonical for value in supplied[1:]):
        raise OpenAIProxyProtocolError(
            400,
            "Conflicting OpenAI proxy adaptive_retrieval claims",
            param="adaptive_retrieval",
            code="conflicting_adaptive_retrieval",
        )
    return canonical


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_bool_claim(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _completion_id() -> str:
    return f"chatcmpl-atagia-{uuid.uuid4().hex}"


def _completion_payload_from_envelope(
    *,
    completion_id: str,
    created: int,
    model: str,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    content = envelope.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Stored proxy completion content is invalid")
    raw_tool_calls = envelope.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list) or any(
        not isinstance(item, dict) for item in raw_tool_calls
    ):
        raise RuntimeError("Stored proxy completion tool calls are invalid")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if raw_tool_calls:
        message["tool_calls"] = [dict(item) for item in raw_tool_calls]
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": envelope.get("finish_reason"),
                "logprobs": None,
            }
        ],
    }
    usage = envelope.get("usage")
    if isinstance(usage, dict):
        payload["usage"] = dict(usage)
    return payload


async def _stream_replay(
    *,
    request: OpenAIChatCompletionRequest,
    envelope: dict[str, Any],
    completion_id: str,
    created: int,
) -> AsyncIterator[str]:
    content = envelope.get("content")
    raw_tool_calls = envelope.get("tool_calls", [])
    if not isinstance(content, str) or not isinstance(raw_tool_calls, list):
        raise RuntimeError("Stored proxy stream replay envelope is invalid")
    yield _sse(
        _chunk_payload(
            completion_id=completion_id,
            created=created,
            model=request.model,
            delta={"role": "assistant"},
        )
    )
    if content:
        yield _sse(
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                delta={"content": content},
            )
        )
    for index, tool_call in enumerate(raw_tool_calls):
        if not isinstance(tool_call, dict):
            raise RuntimeError("Stored proxy stream tool call is invalid")
        wire_call = dict(tool_call)
        wire_call["index"] = index
        yield _sse(
            _chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                delta={"tool_calls": [wire_call]},
            )
        )
    yield _sse(
        _chunk_payload(
            completion_id=completion_id,
            created=created,
            model=request.model,
            delta={},
            finish_reason=envelope.get("finish_reason"),
        )
    )
    usage = envelope.get("usage")
    if _include_stream_usage(request) and isinstance(usage, dict):
        yield _sse(
            _usage_chunk_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                usage=dict(usage),
            )
        )
    yield "data: [DONE]\n\n"


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
    usage: dict[str, Any],
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
    return _parse_bool_claim(stream_options.get("include_usage")) is True


async def _first_output_or_done_stream_event(
    stream: AsyncIterator[LLMStreamEvent],
) -> LLMStreamEvent:
    try:
        while True:
            event = await anext(stream)
            if _stream_event_is_output(event):
                return event
            if event.type == "done":
                try:
                    await anext(stream)
                except StopAsyncIteration:
                    return event
                raise LLMError("LLM stream produced events after its done event")
    except StopAsyncIteration as exc:
        raise LLMError("LLM stream produced no output events") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError("LLM stream failed before producing output") from exc


async def _prepend_stream_event(
    first_event: LLMStreamEvent,
    stream: AsyncIterator[LLMStreamEvent],
) -> AsyncIterator[LLMStreamEvent]:
    yield first_event
    async for event in stream:
        yield event


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
                delta={
                    "tool_calls": _openai_stream_tool_calls(event.payload, tool_index)
                },
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


def _proxy_protocol_error(exc: ProxyTurnError) -> OpenAIProxyProtocolError:
    return OpenAIProxyProtocolError(
        409,
        str(exc),
        code=exc.code,
        retry_after_seconds=exc.retry_after_seconds,
    )


def _is_transient_context_error(exc: Exception) -> bool:
    if isinstance(exc, _TRANSIENT_CONTEXT_ERRORS):
        return True
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    return isinstance(error_code, int) and (error_code & 0xFF) in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }


def _openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return normalize_openai_tool_calls(tool_calls)


def _effective_external_output_ceiling(
    request: OpenAIChatCompletionRequest,
    *,
    server_max_output_tokens: int,
) -> int:
    ceilings = [server_max_output_tokens]
    if request.max_tokens is not None:
        ceilings.append(request.max_tokens)
    if request.max_completion_tokens is not None:
        ceilings.append(request.max_completion_tokens)
    return min(ceilings)
