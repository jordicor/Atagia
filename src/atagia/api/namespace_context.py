"""HTTP helpers for resolving public namespace identity.

Public read/mutation routes that expose memory-derived data must fail closed
unless the caller supplies the active conversation and platform identity. This
keeps one user's personas, platforms, characters, and incognito chats from
bleeding into each other on direct lookup surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite
from fastapi import HTTPException, status

from atagia.core.clock import Clock
from atagia.core.conversation_namespace import (
    ConversationNamespaceSnapshot,
    capture_conversation_namespace_snapshot,
)
from atagia.models.schemas_memory import ConversationStatus


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_required(value: str | None, *, field_name: str) -> str:
    stripped = _clean_optional(value)
    if stripped is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is required",
        )
    return stripped


@dataclass(frozen=True, slots=True)
class RouteNamespaceContext:
    """Resolved identity used by public memory-facing HTTP routes."""

    user_id: str
    conversation_id: str
    platform_id: str
    user_persona_id: str | None
    character_id: str | None
    workspace_id: str | None
    assistant_mode_id: str | None
    mode: str | None
    incognito: bool
    remember_across_chats: bool
    remember_across_devices: bool
    active_space_id: str | None
    active_space_boundary_mode: str | None
    active_mind_id: str | None
    mind_topology: str
    active_embodiment_id: str | None
    active_realm_id: str | None
    authorization_snapshot: ConversationNamespaceSnapshot

    def memory_kwargs(
        self,
        *,
        include_space: bool = False,
        include_mind: bool = False,
        include_embodiment: bool = False,
        include_realm: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "user_persona_id": self.user_persona_id,
            "platform_id": self.platform_id,
            "character_id": self.character_id,
            "incognito": self.incognito,
            "remember_across_chats": self.remember_across_chats,
            "remember_across_devices": self.remember_across_devices,
        }
        if include_space:
            kwargs.update(
                {
                    "active_space_id": self.active_space_id,
                    "active_space_boundary_mode": self.active_space_boundary_mode,
                }
            )
        if include_mind:
            kwargs.update(
                {
                    "active_mind_id": self.active_mind_id,
                    "mind_topology": self.mind_topology,
                }
            )
        if include_embodiment:
            kwargs.update(
                {
                    "active_embodiment_id": self.active_embodiment_id,
                }
            )
        if include_realm:
            kwargs.update(
                {
                    "active_realm_id": self.active_realm_id,
                }
            )
        return kwargs


async def require_route_namespace_context(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    user_id: str,
    conversation_id: str | None,
    platform_id: str | None,
    user_persona_id: str | None = None,
    character_id: str | None = None,
    incognito: bool | None = None,
    require_active: bool = True,
) -> RouteNamespaceContext:
    """Resolve and validate the namespace context for a public route."""

    resolved_conversation_id = _clean_required(
        conversation_id,
        field_name="conversation_id",
    )
    resolved_platform_id = _clean_required(platform_id, field_name="platform_id")
    expected_user_persona_id = _clean_optional(user_persona_id)
    expected_character_id = _clean_optional(character_id)

    snapshot = await capture_conversation_namespace_snapshot(
        connection,
        clock,
        user_id=user_id,
        conversation_id=resolved_conversation_id,
    )
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        )
    if require_active and snapshot.status != ConversationStatus.ACTIVE.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        )

    expected_incognito = bool(incognito) if incognito is not None else False

    if snapshot.platform_id != resolved_platform_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if snapshot.user_persona_id != expected_user_persona_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if snapshot.character_id != expected_character_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if snapshot.incognito != expected_incognito:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )

    return RouteNamespaceContext(
        user_id=user_id,
        conversation_id=resolved_conversation_id,
        platform_id=resolved_platform_id,
        user_persona_id=expected_user_persona_id,
        character_id=expected_character_id,
        workspace_id=snapshot.workspace_id,
        assistant_mode_id=snapshot.assistant_mode_id,
        mode=snapshot.mode,
        incognito=expected_incognito,
        remember_across_chats=snapshot.remember_across_chats,
        remember_across_devices=snapshot.remember_across_devices,
        active_space_id=snapshot.active_space_id,
        active_space_boundary_mode=snapshot.active_space_boundary_mode,
        active_mind_id=snapshot.active_mind_id,
        mind_topology=snapshot.mind_topology,
        active_embodiment_id=snapshot.active_embodiment_id,
        active_realm_id=snapshot.active_realm_id,
        authorization_snapshot=snapshot,
    )


async def require_current_route_namespace_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    snapshot: ConversationNamespaceSnapshot,
) -> None:
    """Fail closed when an authorized conversation namespace has changed."""

    current = await capture_conversation_namespace_snapshot(
        connection,
        clock,
        user_id=snapshot.user_id,
        conversation_id=snapshot.conversation_id,
    )
    if current != snapshot:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation namespace changed while the request was in progress; retry",
        )


__all__ = [
    "RouteNamespaceContext",
    "require_current_route_namespace_snapshot",
    "require_route_namespace_context",
]
