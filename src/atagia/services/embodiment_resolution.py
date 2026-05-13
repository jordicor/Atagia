"""Request-time Embodiment resolution helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.embodiment_repository import (
    EmbodimentRepository,
    EmbodimentSnapshot,
    embodiment_snapshot,
)
from atagia.core.repositories import ConversationRepository
from atagia.models.schemas_memory import EmbodimentBoundaryMode


async def resolve_active_embodiment_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    embodiment_id: str | None = None,
    cross_embodiment_mode: EmbodimentBoundaryMode | str | None = None,
    display_name: str | None = None,
) -> EmbodimentSnapshot | None:
    """Resolve the active Embodiment row without mutating a conversation."""

    row = await EmbodimentRepository(connection, clock).resolve_active_embodiment(
        owner_user_id=owner_user_id,
        embodiment_id=_optional_text(embodiment_id),
        cross_embodiment_mode=cross_embodiment_mode,
        display_name=display_name,
    )
    if row is None:
        return None
    return embodiment_snapshot(row)


async def ensure_conversation_active_embodiment(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    conversation: dict[str, Any],
    embodiment_id: str | None = None,
    cross_embodiment_mode: EmbodimentBoundaryMode | str | None = None,
    display_name: str | None = None,
) -> tuple[dict[str, Any], EmbodimentSnapshot | None]:
    """Resolve and persist the active Embodiment for a conversation if present."""

    owner_user_id = str(conversation["user_id"])
    requested_embodiment_id = _optional_text(embodiment_id) or _optional_text(
        conversation.get("active_embodiment_id")
    )
    snapshot = await resolve_active_embodiment_snapshot(
        connection,
        clock,
        owner_user_id=owner_user_id,
        embodiment_id=requested_embodiment_id,
        cross_embodiment_mode=cross_embodiment_mode,
        display_name=display_name,
    )
    if snapshot is None:
        return conversation, None

    current_embodiment_id = _optional_text(conversation.get("active_embodiment_id"))
    conversation = {
        **conversation,
        "active_embodiment_id": snapshot.embodiment_id,
        "active_embodiment_display_name": snapshot.display_name,
        "cross_embodiment_mode": snapshot.cross_embodiment_mode.value,
    }
    if current_embodiment_id is None:
        updated = await ConversationRepository(
            connection,
            clock,
        ).set_active_embodiment(
            str(conversation["id"]),
            owner_user_id,
            snapshot.embodiment_id,
        )
        if updated is not None:
            conversation = {
                **updated,
                "active_embodiment_display_name": snapshot.display_name,
                "cross_embodiment_mode": snapshot.cross_embodiment_mode.value,
            }
    elif embodiment_id is not None and current_embodiment_id != snapshot.embodiment_id:
        raise ValueError("embodiment_id does not match the existing conversation")
    return conversation, snapshot


def snapshot_fields(embodiment: EmbodimentSnapshot | None) -> dict[str, Any]:
    """Return serializable Embodiment fields for contexts and job payloads."""

    return {
        "active_embodiment_id": (
            embodiment.embodiment_id if embodiment is not None else None
        ),
        "active_embodiment_display_name": (
            embodiment.display_name if embodiment is not None else None
        ),
        "cross_embodiment_mode": (
            embodiment.cross_embodiment_mode.value
            if embodiment is not None
            else EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY.value
        ),
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "ensure_conversation_active_embodiment",
    "resolve_active_embodiment_snapshot",
    "snapshot_fields",
]
