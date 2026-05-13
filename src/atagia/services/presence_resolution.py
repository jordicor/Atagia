"""Request-time Presence resolution helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.presence_repository import (
    PresenceRepository,
    PresenceSnapshot,
    presence_snapshot,
)
from atagia.core.repositories import ConversationRepository
from atagia.models.schemas_memory import PresenceKind


async def resolve_active_presence_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    active_presence_id: str | None = None,
    character_id: str | None = None,
) -> PresenceSnapshot:
    """Resolve the active Presence row without mutating a conversation."""

    row = await PresenceRepository(connection, clock).resolve_active_presence(
        owner_user_id=owner_user_id,
        active_presence_id=active_presence_id,
        character_id=character_id,
    )
    return presence_snapshot(row)


async def ensure_conversation_active_presence(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    conversation: dict[str, Any],
    active_presence_id: str | None = None,
    character_id: str | None = None,
) -> tuple[dict[str, Any], PresenceSnapshot]:
    """Resolve and persist the active Presence for a conversation if missing."""

    owner_user_id = str(conversation["user_id"])
    requested_presence_id = active_presence_id or _optional_text(
        conversation.get("active_presence_id")
    )
    effective_character_id = (
        character_id
        if character_id is not None
        else _optional_text(conversation.get("character_id"))
        or _optional_text(conversation.get("workspace_id"))
    )
    snapshot = await resolve_active_presence_snapshot(
        connection,
        clock,
        owner_user_id=owner_user_id,
        active_presence_id=requested_presence_id,
        character_id=effective_character_id,
    )
    current_presence_id = _optional_text(conversation.get("active_presence_id"))
    if current_presence_id is None:
        updated = await ConversationRepository(
            connection,
            clock,
        ).set_active_presence(
            str(conversation["id"]),
            owner_user_id,
            snapshot.presence_id,
        )
        if updated is not None:
            conversation = updated
    elif active_presence_id is not None and current_presence_id != snapshot.presence_id:
        raise ValueError("active_presence_id does not match the existing conversation")
    return conversation, snapshot


async def resolve_source_presence_for_role(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    role: str,
    active_presence: PresenceSnapshot,
) -> PresenceSnapshot:
    """Resolve the source Presence for a stored message role."""

    if role == "user":
        row = await PresenceRepository(
            connection,
            clock,
        ).resolve_human_owner_presence(owner_user_id=owner_user_id)
        return presence_snapshot(row)
    return active_presence


def snapshot_fields(
    *,
    active_presence: PresenceSnapshot | None,
    source_presence: PresenceSnapshot | None = None,
) -> dict[str, Any]:
    """Return serializable Presence fields for contexts and job payloads."""

    source = source_presence or active_presence
    return {
        "active_presence_id": (
            active_presence.presence_id if active_presence is not None else None
        ),
        "active_presence_kind": (
            active_presence.kind.value
            if active_presence is not None
            else PresenceKind.UNKNOWN.value
        ),
        "active_presence_display_name": (
            active_presence.display_name if active_presence is not None else None
        ),
        "source_presence_id": source.presence_id if source is not None else None,
        "source_presence_kind": (
            source.kind.value if source is not None else PresenceKind.UNKNOWN.value
        ),
        "source_presence_display_name": source.display_name if source is not None else None,
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "ensure_conversation_active_presence",
    "resolve_active_presence_snapshot",
    "resolve_source_presence_for_role",
    "snapshot_fields",
]
