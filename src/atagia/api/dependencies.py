"""FastAPI dependency helpers for runtime state and auth."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiosqlite
from fastapi import Header, HTTPException, Request, status

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.storage_backend import StorageBackend
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.llm_client import LLMClient


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Request-scoped auth information."""

    service_mode: bool
    is_admin: bool
    actor_id: str
    api_key: str | None = None
    claimed_user_id: str | None = None


def get_runtime(request: Request) -> Any:
    """Return the application runtime stored in app.state."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Atagia runtime is not initialized")
    return runtime


def get_settings(request: Request) -> Settings:
    """Return application settings from app.state."""
    return get_runtime(request).settings


async def get_connection(request: Request) -> AsyncIterator[aiosqlite.Connection]:
    """Open one SQLite connection per request-scoped unit of work."""
    connection = await get_runtime(request).open_connection()
    try:
        yield connection
    finally:
        await connection.close()


def get_clock(request: Request) -> Clock:
    """Return the shared clock implementation."""
    return get_runtime(request).clock


def get_storage_backend(request: Request) -> StorageBackend:
    """Return the configured transient storage backend."""
    return get_runtime(request).storage_backend


def get_llm_client(request: Request) -> LLMClient[Any]:
    """Return the configured provider-agnostic LLM client."""
    return get_runtime(request).llm_client


def get_embedding_index(request: Request) -> EmbeddingIndex:
    """Return the configured embedding index."""
    return get_runtime(request).embedding_index


def get_manifest_loader(request: Request) -> ManifestLoader:
    """Return the startup manifest loader."""
    return get_runtime(request).manifest_loader


def get_manifests(request: Request) -> dict[str, Any]:
    """Return the manifest mapping loaded at startup."""
    return get_runtime(request).manifests


def get_policy_resolver(request: Request) -> PolicyResolver:
    """Return the stateless policy resolver."""
    return get_runtime(request).policy_resolver


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer <token>",
        )
    return token.strip()


def get_auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_atagia_user_id: str | None = Header(default=None, alias="X-Atagia-User-Id"),
) -> AuthContext:
    """Authenticate normal user routes."""
    settings = get_settings(request)
    if not settings.service_mode:
        return AuthContext(service_mode=False, is_admin=False, actor_id="library_mode")

    if settings.service_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ATAGIA_SERVICE_API_KEY is required in service mode",
        )

    token = _bearer_token(authorization)
    if token != settings.service_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    if x_atagia_user_id is None or not x_atagia_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Atagia-User-Id header is required in service mode",
        )

    return AuthContext(
        service_mode=True,
        is_admin=False,
        actor_id="service_api_key",
        api_key=token,
        claimed_user_id=x_atagia_user_id.strip(),
    )


def get_admin_auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """Authenticate admin-only routes."""
    settings = get_settings(request)
    if not settings.service_mode:
        return AuthContext(service_mode=False, is_admin=True, actor_id="library_admin")

    if settings.admin_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ATAGIA_ADMIN_API_KEY is required for admin routes in service mode",
        )

    token = _bearer_token(authorization)
    if token != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
    return AuthContext(service_mode=True, is_admin=True, actor_id="admin_api_key", api_key=token)


def ensure_user_access(user_id: str, auth_context: AuthContext) -> None:
    """Reject mismatched authenticated user claims when present."""
    if auth_context.claimed_user_id is None:
        return
    if auth_context.claimed_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated user does not match the requested user_id",
        )
