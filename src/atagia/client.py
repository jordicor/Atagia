"""Transport-agnostic client facade for Atagia integrations."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Literal, Protocol, Self

import httpx

from atagia.engine import Atagia
from atagia.models.schemas_api import ChatResult, ContextResult

TransportName = Literal["auto", "local", "http"]


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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> ContextResult:
        """Return memory context for a host-managed LLM call."""

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
    ) -> ChatResult:
        """Run the full Atagia chat flow."""

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        """Wait for pending background work to finish when supported."""

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
    ) -> str:
        return await self._engine.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> ContextResult:
        return await self._engine.get_context(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
            attachments=attachments,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
        )

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        await self._engine.add_response(
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            occurred_at=occurred_at,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
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
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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
        )

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        return await self._engine.flush(timeout_seconds)

    async def close(self) -> None:
        await self._engine.close()


class HttpAtagiaClient:
    """Transport facade that calls an Atagia REST service."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required for HTTP transport")
        if not api_key.strip():
            raise ValueError("api_key is required for HTTP transport")
        self._api_key = api_key.strip()
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._owns_client = http_client is None
        self._last_user_id: str | None = None

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
            },
        )
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> ContextResult:
        response = await self._post(
            f"/v1/conversations/{conversation_id}/context",
            user_id=user_id,
            json={
                "user_id": user_id,
                "message_text": message,
                "assistant_mode_id": mode,
                "workspace_id": workspace_id,
                "message_occurred_at": occurred_at,
                "attachments": attachments or [],
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
            },
        )
        return ContextResult.model_validate(response.json())

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        await self._post(
            f"/v1/conversations/{conversation_id}/responses",
            user_id=user_id,
            json={
                "user_id": user_id,
                "text": text,
                "occurred_at": occurred_at,
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
            },
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
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        await self._post(
            f"/v1/conversations/{conversation_id}/messages",
            user_id=user_id,
            json={
                "user_id": user_id,
                "role": role,
                "text": text,
                "assistant_mode_id": mode,
                "workspace_id": workspace_id,
                "occurred_at": occurred_at,
                "attachments": attachments or [],
                "operational_profile": operational_profile,
                "operational_signals": operational_signals,
            },
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
    ) -> ChatResult:
        await self.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=mode,
        )
        response = await self._post(
            f"/v1/chat/{conversation_id}/reply",
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
            },
        )
        payload = response.json()
        return ChatResult(
            conversation_id=str(payload["conversation_id"]),
            request_message_id=str(payload["request_message_id"]),
            response_message_id=str(payload["response_message_id"]),
            response_text=str(payload["reply_text"]),
            retrieval_event_id=payload.get("retrieval_event_id"),
            debug=payload.get("debug"),
        )

    async def flush(self, timeout_seconds: float = 30.0, user_id: str | None = None) -> bool:
        resolved_user_id = user_id or self._last_user_id
        if resolved_user_id is None:
            raise ValueError("user_id is required for the first HTTP flush call")
        response = await self._post(
            "/v1/flush",
            user_id=resolved_user_id,
            json={"user_id": resolved_user_id, "timeout_seconds": timeout_seconds},
        )
        return bool(response.json()["completed"])

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self, user_id: str) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if user_id:
            headers["X-Atagia-User-Id"] = user_id
        return headers

    async def _post(
        self,
        path: str,
        *,
        user_id: str,
        json: dict[str, Any],
    ) -> httpx.Response:
        if user_id:
            self._last_user_id = user_id
        response = await self._client.post(
            path,
            json=json,
            headers=self._headers(user_id),
        )
        response.raise_for_status()
        return response


async def connect_atagia(
    *,
    transport: TransportName = "auto",
    db_path: str | Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    redis_url: str | None = None,
    manifests_dir: str | Path | None = None,
    operational_profiles_dir: str | Path | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_forced_global_model: str | None = None,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    embedding_backend: str = "none",
    embedding_model: str | None = None,
    embedding_provider_name: str | None = None,
    context_cache_enabled: bool | None = None,
    chunking_enabled: bool | None = None,
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
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_forced_global_model=llm_forced_global_model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            embedding_provider_name=embedding_provider_name,
            context_cache_enabled=context_cache_enabled,
            chunking_enabled=chunking_enabled,
        )
    )
    await client.setup()
    return client
