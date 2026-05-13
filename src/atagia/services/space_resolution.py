"""Request-time Space resolution helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.repositories import ConversationRepository
from atagia.core.space_repository import SpaceRepository, SpaceSnapshot, space_snapshot
from atagia.models.schemas_memory import SpaceBoundaryMode


async def resolve_active_space_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    space_id: str | None = None,
    workspace_id: str | None = None,
    boundary_mode: SpaceBoundaryMode | str | None = None,
    display_name: str | None = None,
) -> SpaceSnapshot | None:
    """Resolve the active Space row without mutating a conversation."""

    repository = SpaceRepository(connection, clock)
    resolved_space_id = _optional_text(space_id)
    resolved_workspace_id = _optional_text(workspace_id)
    if resolved_space_id is not None and boundary_mode is None:
        existing = await repository.get_space(
            owner_user_id=owner_user_id,
            space_id=resolved_space_id,
        )
        if existing is not None:
            return space_snapshot(existing)
    if resolved_space_id is None and resolved_workspace_id is not None and boundary_mode is None:
        existing = await repository.get_space(
            owner_user_id=owner_user_id,
            space_id=resolved_workspace_id,
        )
        if existing is not None:
            return space_snapshot(existing)
    resolved_mode = SpaceBoundaryMode(boundary_mode or SpaceBoundaryMode.FOCUS.value)
    row = await repository.resolve_active_space(
        owner_user_id=owner_user_id,
        space_id=resolved_space_id,
        workspace_id=resolved_workspace_id,
        boundary_mode=resolved_mode,
        display_name=display_name,
    )
    if row is None:
        return None
    return space_snapshot(row)


async def ensure_conversation_active_space(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    conversation: dict[str, Any],
    space_id: str | None = None,
    workspace_id: str | None = None,
    boundary_mode: SpaceBoundaryMode | str | None = None,
    display_name: str | None = None,
) -> tuple[dict[str, Any], SpaceSnapshot | None]:
    """Resolve and persist the active Space for a conversation if one applies."""

    owner_user_id = str(conversation["user_id"])
    requested_space_id = _optional_text(space_id) or _optional_text(
        conversation.get("active_space_id")
    )
    effective_workspace_id = workspace_id
    if effective_workspace_id is None:
        effective_workspace_id = _optional_text(conversation.get("workspace_id"))
    snapshot = await resolve_active_space_snapshot(
        connection,
        clock,
        owner_user_id=owner_user_id,
        space_id=requested_space_id,
        workspace_id=effective_workspace_id,
        boundary_mode=boundary_mode,
        display_name=display_name,
    )
    if snapshot is None:
        return conversation, None
    current_space_id = _optional_text(conversation.get("active_space_id"))
    conversation = {
        **conversation,
        "active_space_id": snapshot.space_id,
        "active_space_boundary_mode": snapshot.boundary_mode.value,
        "active_space_display_name": snapshot.display_name,
    }
    if current_space_id is None:
        updated = await ConversationRepository(
            connection,
            clock,
        ).set_active_space(
            str(conversation["id"]),
            owner_user_id,
            snapshot.space_id,
        )
        if updated is not None:
            conversation = {
                **updated,
                "active_space_boundary_mode": snapshot.boundary_mode.value,
                "active_space_display_name": snapshot.display_name,
            }
    elif space_id is not None and current_space_id != snapshot.space_id:
        raise ValueError("space_id does not match the existing conversation")
    return conversation, snapshot


def snapshot_fields(space: SpaceSnapshot | None) -> dict[str, Any]:
    """Return serializable Space fields for contexts and job payloads."""

    return {
        "active_space_id": space.space_id if space is not None else None,
        "active_space_boundary_mode": (
            space.boundary_mode.value if space is not None else SpaceBoundaryMode.FOCUS.value
        ),
        "active_space_display_name": space.display_name if space is not None else None,
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "ensure_conversation_active_space",
    "resolve_active_space_snapshot",
    "snapshot_fields",
]
