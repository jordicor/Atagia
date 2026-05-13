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
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.space_repository import SpaceRepository, space_snapshot
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


def _row_optional(row: dict[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    return str(value)


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

    conversations = ConversationRepository(connection, clock)
    conversation = await conversations.get_conversation(resolved_conversation_id, user_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        )
    if require_active and str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for user",
        )

    actual_platform_id = _row_optional(conversation, "platform_id")
    actual_user_persona_id = _row_optional(conversation, "user_persona_id")
    actual_character_id = _row_optional(conversation, "character_id")
    actual_incognito = bool(conversation.get("incognito"))
    expected_incognito = bool(incognito) if incognito is not None else False

    if actual_platform_id != resolved_platform_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if actual_user_persona_id != expected_user_persona_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if actual_character_id != expected_character_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )
    if actual_incognito != expected_incognito:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for namespace",
        )

    preferences = await UserRepository(connection, clock).get_memory_preferences(user_id)
    if preferences is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    active_space_id = _row_optional(conversation, "active_space_id")
    active_space_boundary_mode = None
    if active_space_id is not None:
        space_row = await SpaceRepository(connection, clock).get_space(
            owner_user_id=user_id,
            space_id=active_space_id,
        )
        if space_row is not None:
            active_space_boundary_mode = space_snapshot(space_row).boundary_mode.value

    return RouteNamespaceContext(
        user_id=user_id,
        conversation_id=resolved_conversation_id,
        platform_id=resolved_platform_id,
        user_persona_id=expected_user_persona_id,
        character_id=expected_character_id,
        workspace_id=_row_optional(conversation, "workspace_id"),
        assistant_mode_id=_row_optional(conversation, "assistant_mode_id"),
        mode=_row_optional(conversation, "mode"),
        incognito=expected_incognito,
        remember_across_chats=bool(preferences["remember_across_chats"]),
        remember_across_devices=bool(preferences["remember_across_devices"]),
        active_space_id=active_space_id,
        active_space_boundary_mode=active_space_boundary_mode,
        active_mind_id=_row_optional(conversation, "active_mind_id"),
        mind_topology=_row_optional(conversation, "mind_topology") or "unimind",
        active_embodiment_id=_row_optional(conversation, "active_embodiment_id"),
        active_realm_id=_row_optional(conversation, "active_realm_id"),
    )


__all__ = ["RouteNamespaceContext", "require_route_namespace_context"]
