"""Transport-agnostic client facade for Atagia integrations."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Literal, Protocol, Self

import httpx

from atagia.engine import Atagia
from atagia.models.schemas_api import (
    AdminReviewActionResponse,
    AdminReviewMemoryListResponse,
    ChatResult,
    ContextResult,
    MemoryProcessingStatus,
    PendingMemoryConfirmationActionResponse,
    PendingMemoryConfirmationListResponse,
    WorkerControlResponse,
)
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.transport_ids import encode_path_id

TransportName = Literal["auto", "local", "http"]


def _path_segment(value: str) -> str:
    return encode_path_id(value)


class AtagiaClient(Protocol):
    """Common async client interface for local and HTTP Atagia transports."""

    async def create_user(self, user_id: str) -> None:
        """Create the user if it does not already exist."""

    async def create_workspace(self, user_id: str, workspace_id: str, name: str) -> None:
        """Create the workspace if it does not already exist."""

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        *,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str:
        """Create or reuse a conversation and return its id."""

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> ContextResult:
        """Return memory context for a host-managed LLM call."""

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        """Persist a host-generated assistant response."""

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        """Persist one historical or sidecar message."""

    async def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
    ) -> ChatResult:
        """Run the full Atagia chat flow."""

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        """Wait for pending background work to finish when supported."""

    async def get_processing_status(
        self,
        user_id: str,
        conversation_id: str | None = None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> MemoryProcessingStatus:
        """Return current background memory-processing status."""

    async def list_pending_memory_confirmations(
        self,
        user_id: str,
        **filters: Any,
    ) -> PendingMemoryConfirmationListResponse:
        """Return pending user memory confirmations."""

    async def confirm_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        """Confirm one pending memory."""

    async def decline_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        """Decline one pending memory."""

    async def list_review_required_memories(
        self,
        **filters: Any,
    ) -> AdminReviewMemoryListResponse:
        """Return admin review-required memories."""

    async def archive_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        """Archive one review-required memory."""

    async def delete_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        """Delete one review-required memory."""

    async def get_worker_control(self) -> WorkerControlResponse:
        """Return current background-processing control state."""

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> WorkerControlResponse:
        """Set the background-processing control state."""

    async def close(self) -> None:
        """Close transport resources."""

    async def __aenter__(self) -> Self:
        """Enter an async context manager."""

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        """Exit an async context manager."""


class LocalAtagiaClient:
    """Transport facade that delegates directly to an in-process Atagia engine."""

    def __init__(self, engine: Atagia) -> None:
        self._engine = engine

    async def setup(self) -> LocalAtagiaClient:
        await self._engine.setup()
        return self

    async def __aenter__(self) -> LocalAtagiaClient:
        await self.setup()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

    async def create_user(self, user_id: str) -> None:
        await self._engine.create_user(user_id)

    async def create_workspace(self, user_id: str, workspace_id: str, name: str) -> None:
        await self._engine.create_workspace(user_id, workspace_id, name)

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        *,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str:
        return await self._engine.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            mode=mode,
            incognito=incognito,
        )

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> ContextResult:
        return await self._engine.get_context(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            attachments=attachments,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            incognito=incognito,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
        )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        await self._engine.add_response(
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            occurred_at=occurred_at,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            mode=mode,
            incognito=incognito,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
        )

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        await self._engine.ingest_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            text=text,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            attachments=attachments,
            message_id=message_id,
            source_seq=source_seq,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            incognito=incognito,
            ingest_origin=ingest_origin,
            confirmation_strategy=confirmation_strategy,
            memory_privacy_mode=memory_privacy_mode,
        )

    async def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
    ) -> ChatResult:
        return await self._engine.chat(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            attachments=attachments,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            incognito=incognito,
        )

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        return await self._engine.flush(timeout_seconds)

    async def get_processing_status(
        self,
        user_id: str,
        conversation_id: str | None = None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> MemoryProcessingStatus:
        return await self._engine.get_processing_status(
            user_id=user_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )

    async def list_pending_memory_confirmations(
        self,
        user_id: str,
        **filters: Any,
    ) -> PendingMemoryConfirmationListResponse:
        return await self._engine.list_pending_memory_confirmations(
            user_id=user_id,
            **filters,
        )

    async def confirm_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        return await self._engine.confirm_pending_memory(
            user_id=user_id,
            memory_id=memory_id,
        )

    async def decline_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        return await self._engine.decline_pending_memory(
            user_id=user_id,
            memory_id=memory_id,
        )

    async def list_review_required_memories(
        self,
        **filters: Any,
    ) -> AdminReviewMemoryListResponse:
        return await self._engine.list_review_required_memories(**filters)

    async def archive_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        return await self._engine.archive_review_required_memory(
            user_id=user_id,
            memory_id=memory_id,
        )

    async def delete_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        return await self._engine.delete_review_required_memory(
            user_id=user_id,
            memory_id=memory_id,
        )

    async def get_worker_control(self) -> WorkerControlResponse:
        return await self._engine.get_worker_control()

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> WorkerControlResponse:
        return await self._engine.set_worker_control(
            mode,
            reason=reason,
            timeout_seconds=timeout_seconds,
        )

    async def close(self) -> None:
        await self._engine.close()


class HttpAtagiaClient:
    """Transport facade that calls an Atagia REST service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        admin_api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required for HTTP transport")
        if not api_key.strip():
            raise ValueError("api_key is required for HTTP transport")
        self._api_key = api_key.strip()
        self._admin_api_key = (
            admin_api_key.strip()
            if admin_api_key is not None and admin_api_key.strip()
            else None
        )
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._owns_client = http_client is None
        self._last_user_id: str | None = None
        self._last_conversation_id: str | None = None

    async def __aenter__(self) -> HttpAtagiaClient:
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

    async def create_user(self, user_id: str) -> None:
        await self._post(
            "/v1/users",
            user_id=user_id,
            json={"user_id": user_id},
        )

    async def create_workspace(self, user_id: str, workspace_id: str, name: str) -> None:
        await self._post(
            "/v1/workspaces",
            user_id=user_id,
            json={
                "user_id": user_id,
                "workspace_id": workspace_id,
                "name": name,
                "metadata": {},
            },
        )

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        *,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str:
        response = await self._post(
            "/v1/conversations",
            user_id=user_id,
            json={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "assistant_mode_id": assistant_mode_id,
                "workspace_id": workspace_id,
                "title": None,
                "metadata": {},
                "cross_chat_memory": cross_chat_memory,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id if character_id is not None else workspace_id,
                "active_presence_id": active_presence_id,
                "embodiment_id": embodiment_id,
                "realm_id": realm_id,
                "space_id": space_id,
                "mode": mode,
                "incognito": incognito,
            },
        )
        if conversation_id is not None:
            return conversation_id
        return str(response.json()["id"])

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> ContextResult:
        response = await self._post(
            f"/v1/conversations/{_path_segment(conversation_id)}/context",
            user_id=user_id,
            json=_omit_none(
                {
                "user_id": user_id,
                "message_text": message,
                "message_id": message_id,
                "source_seq": source_seq,
                "assistant_mode_id": mode,
                "workspace_id": workspace_id,
                "message_occurred_at": occurred_at,
                "attachments": attachments or [],
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "cross_chat_memory": cross_chat_memory,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id if character_id is not None else workspace_id,
                "active_presence_id": active_presence_id,
                "embodiment_id": embodiment_id,
                "realm_id": realm_id,
                "space_id": space_id,
                "mode": mode,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
                }
            ),
        )
        return ContextResult.model_validate(response.json())

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        await self._post(
            f"/v1/conversations/{_path_segment(conversation_id)}/responses",
            user_id=user_id,
            json=_omit_none(
                {
                "user_id": user_id,
                "message_id": message_id,
                "source_seq": source_seq,
                "text": text,
                "occurred_at": occurred_at,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "active_presence_id": active_presence_id,
                "embodiment_id": embodiment_id,
                "realm_id": realm_id,
                "space_id": space_id,
                "mode": mode,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
                }
            ),
        )

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        await self._post(
            f"/v1/conversations/{_path_segment(conversation_id)}/messages",
            user_id=user_id,
            json=_omit_none(
                {
                "user_id": user_id,
                "message_id": message_id,
                "source_seq": source_seq,
                "role": role,
                "text": text,
                "assistant_mode_id": mode,
                "workspace_id": workspace_id,
                "occurred_at": occurred_at,
                "attachments": attachments or [],
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "cross_chat_memory": cross_chat_memory,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id if character_id is not None else workspace_id,
                "active_presence_id": active_presence_id,
                "embodiment_id": embodiment_id,
                "realm_id": realm_id,
                "space_id": space_id,
                "mode": mode,
                "incognito": incognito,
                "ingest_origin": ingest_origin,
                "confirmation_strategy": confirmation_strategy,
                "memory_privacy_mode": memory_privacy_mode,
                }
            ),
        )

    async def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        cross_chat_memory: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        space_id: str | None = None,
        incognito: bool | None = None,
    ) -> ChatResult:
        await self.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=mode,
            cross_chat_memory=cross_chat_memory,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
            space_id=space_id,
            mode=mode,
            incognito=incognito,
        )
        response = await self._post(
            f"/v1/chat/{_path_segment(conversation_id)}/reply",
            user_id=user_id,
            json={
                "user_id": user_id,
                "message_text": message,
                "attachments": attachments or [],
                "message_occurred_at": occurred_at,
                "include_thinking": False,
                "metadata": {},
                "debug": False,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
                "cross_chat_memory": cross_chat_memory,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id if character_id is not None else workspace_id,
                "active_presence_id": active_presence_id,
                "embodiment_id": embodiment_id,
                "realm_id": realm_id,
                "space_id": space_id,
                "mode": mode,
                "incognito": incognito,
            },
        )
        payload = response.json()
        return ChatResult(
            conversation_id=conversation_id,
            request_message_id=str(payload["request_message_id"]),
            response_message_id=str(payload["response_message_id"]),
            response_text=str(payload["reply_text"]),
            retrieval_event_id=payload.get("retrieval_event_id"),
            memory_processing=(
                None
                if payload.get("memory_processing") is None
                else MemoryProcessingStatus.model_validate(payload["memory_processing"])
            ),
            debug=payload.get("debug"),
        )

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        resolved_user_id = user_id or self._last_user_id
        if resolved_user_id is None:
            raise ValueError("user_id is required for the first HTTP flush call")
        payload: dict[str, Any] = {
            "user_id": resolved_user_id,
            "timeout_seconds": timeout_seconds,
        }
        if self._last_conversation_id is not None:
            payload["conversation_id"] = self._last_conversation_id
        response = await self._post(
            "/v1/flush",
            user_id=resolved_user_id,
            json=payload,
        )
        return bool(response.json()["completed"])

    async def get_processing_status(
        self,
        user_id: str,
        conversation_id: str | None = None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> MemoryProcessingStatus:
        if conversation_id is None:
            params = {
                "user_persona_id": user_persona_id,
                "platform_id": platform_id or "default",
                "character_id": character_id,
                "incognito": incognito,
                "remember_across_chats": remember_across_chats,
                "remember_across_devices": remember_across_devices,
            }
            response = await self._get(
                f"/v1/users/{_path_segment(user_id)}/processing-status",
                user_id=user_id,
                params=params,
            )
        else:
            response = await self._get(
                f"/v1/conversations/{_path_segment(conversation_id)}/processing-status",
                user_id=user_id,
                params={"user_id": user_id},
            )
        return MemoryProcessingStatus.model_validate(response.json())

    async def get_worker_control(self) -> WorkerControlResponse:
        response = await self._admin_get("/v1/admin/worker-control")
        return WorkerControlResponse.model_validate(response.json())

    async def list_pending_memory_confirmations(
        self,
        user_id: str,
        **filters: Any,
    ) -> PendingMemoryConfirmationListResponse:
        response = await self._get(
            f"/v1/users/{_path_segment(user_id)}/memory-confirmations",
            user_id=user_id,
            params=filters,
        )
        return PendingMemoryConfirmationListResponse.model_validate(response.json())

    async def confirm_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        response = await self._post(
            f"/v1/users/{_path_segment(user_id)}/memory-confirmations/{_path_segment(memory_id)}/confirm",
            user_id=user_id,
            json={},
        )
        return PendingMemoryConfirmationActionResponse.model_validate(response.json())

    async def decline_pending_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> PendingMemoryConfirmationActionResponse:
        response = await self._post(
            f"/v1/users/{_path_segment(user_id)}/memory-confirmations/{_path_segment(memory_id)}/decline",
            user_id=user_id,
            json={},
        )
        return PendingMemoryConfirmationActionResponse.model_validate(response.json())

    async def list_review_required_memories(
        self,
        **filters: Any,
    ) -> AdminReviewMemoryListResponse:
        response = await self._admin_get("/v1/admin/memory-review", params=filters)
        return AdminReviewMemoryListResponse.model_validate(response.json())

    async def archive_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        response = await self._admin_post(
            f"/v1/admin/memory-review/{_path_segment(user_id)}/{_path_segment(memory_id)}/archive",
            json={},
        )
        return AdminReviewActionResponse.model_validate(response.json())

    async def delete_review_required_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> AdminReviewActionResponse:
        response = await self._admin_post(
            f"/v1/admin/memory-review/{_path_segment(user_id)}/{_path_segment(memory_id)}/delete",
            json={},
        )
        return AdminReviewActionResponse.model_validate(response.json())

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> WorkerControlResponse:
        response = await self._admin_post(
            "/v1/admin/worker-control",
            json={
                "mode": WorkerControlMode(mode).value,
                "reason": reason,
                "timeout_seconds": timeout_seconds,
            },
        )
        return WorkerControlResponse.model_validate(response.json())

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self, user_id: str) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if user_id:
            headers["X-Atagia-User-Id"] = user_id
        return headers

    def _admin_headers(self) -> dict[str, str]:
        if self._admin_api_key is None:
            raise ValueError("admin_api_key is required for HTTP admin routes")
        return {"Authorization": f"Bearer {self._admin_api_key}"}

    async def _post(
        self,
        path: str,
        *,
        user_id: str,
        json: dict[str, Any],
    ) -> httpx.Response:
        if user_id:
            self._last_user_id = user_id
        conversation_id = self._conversation_id_from_payload(json)
        if conversation_id is None:
            conversation_id = self._conversation_id_from_path(path)
        if conversation_id is not None:
            self._last_conversation_id = conversation_id
        response = await self._client.post(
            path,
            json=json,
            headers=self._headers(user_id),
        )
        response.raise_for_status()
        return response

    async def _admin_post(
        self,
        path: str,
        *,
        json: dict[str, Any],
    ) -> httpx.Response:
        response = await self._client.post(
            path,
            json=json,
            headers=self._admin_headers(),
        )
        response.raise_for_status()
        return response

    async def _admin_get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        response = await self._client.get(
            path,
            params=params,
            headers=self._admin_headers(),
        )
        response.raise_for_status()
        return response

    async def _get(
        self,
        path: str,
        *,
        user_id: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if user_id:
            self._last_user_id = user_id
        response = await self._client.get(
            path,
            params=params,
            headers=self._headers(user_id),
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _conversation_id_from_payload(payload: dict[str, Any]) -> str | None:
        for key in ("conversation_id",):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _conversation_id_from_path(path: str) -> str | None:
        parts = [part for part in path.split("/") if part]
        for marker in ("conversations", "chat"):
            if marker in parts:
                index = parts.index(marker)
                if index + 1 < len(parts):
                    value = parts[index + 1]
                    return value or None
        return None


def _omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON payload with absent optional fields omitted."""
    return {key: value for key, value in payload.items() if value is not None}


async def connect_atagia(
    *,
    transport: TransportName = "auto",
    db_path: str | Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    admin_api_key: str | None = None,
    redis_url: str | None = None,
    manifests_dir: str | Path | None = None,
    operational_profiles_dir: str | Path | None = None,
    llm_forced_global_model: str | None = None,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
    context_cache_enabled: bool | None = None,
    disable_chunking_extraction: bool | None = None,
    assistant_guidance_enabled: bool | None = None,
    recent_transcript_budget_tokens: int | None = None,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> AtagiaClient:
    """Create a ready-to-use Atagia client over local or HTTP transport."""
    resolved_base_url = base_url or os.getenv("ATAGIA_BASE_URL")
    resolved_transport = transport
    if resolved_transport == "auto":
        resolved_transport = "http" if resolved_base_url else "local"

    if resolved_transport == "http":
        if resolved_base_url is None:
            raise ValueError("base_url or ATAGIA_BASE_URL is required for HTTP transport")
        resolved_api_key = api_key or os.getenv("ATAGIA_SERVICE_API_KEY")
        if resolved_api_key is None:
            raise ValueError("api_key or ATAGIA_SERVICE_API_KEY is required for HTTP transport")
        return HttpAtagiaClient(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            admin_api_key=admin_api_key or os.getenv("ATAGIA_ADMIN_API_KEY"),
            http_client=http_client,
            timeout=timeout,
        )

    if resolved_transport != "local":
        raise ValueError(f"Unsupported Atagia transport: {transport}")

    resolved_db_path = (
        db_path
        or os.getenv("ATAGIA_DB_PATH")
        or os.getenv("ATAGIA_SQLITE_PATH")
        or "atagia.db"
    )
    client = LocalAtagiaClient(
        Atagia(
            db_path=resolved_db_path,
            redis_url=redis_url,
            manifests_dir=manifests_dir,
            operational_profiles_dir=operational_profiles_dir,
            llm_forced_global_model=llm_forced_global_model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            context_cache_enabled=context_cache_enabled,
            disable_chunking_extraction=disable_chunking_extraction,
            assistant_guidance_enabled=assistant_guidance_enabled,
            recent_transcript_budget_tokens=recent_transcript_budget_tokens,
        )
    )
    await client.setup()
    return client
