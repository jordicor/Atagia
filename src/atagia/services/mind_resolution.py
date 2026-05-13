"""Request-time Mind resolution helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.mind_repository import MindRepository, MindSnapshot, mind_snapshot
from atagia.core.presence_repository import PresenceSnapshot
from atagia.core.repositories import ConversationRepository
from atagia.models.schemas_memory import MindTopology, PresenceKind


async def resolve_active_mind_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    mind_id: str | None = None,
    active_presence: PresenceSnapshot | None = None,
    character_id: str | None = None,
    topology: MindTopology | str | None = None,
) -> MindSnapshot:
    """Resolve the active Mind row without mutating a conversation."""

    resolved_topology = MindTopology(topology or MindTopology.UNIMIND.value)
    row = await MindRepository(connection, clock).resolve_active_mind(
        owner_user_id=owner_user_id,
        mind_id=_optional_text(mind_id),
        active_presence_id=active_presence.presence_id if active_presence is not None else None,
        active_presence_kind=(
            active_presence.kind if active_presence is not None else PresenceKind.UNKNOWN
        ),
        active_presence_display_name=(
            active_presence.display_name if active_presence is not None else None
        ),
        character_id=_optional_text(character_id),
        topology=resolved_topology,
    )
    return mind_snapshot(row, resolved_topology)


async def ensure_conversation_active_mind(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    conversation: dict[str, Any],
    mind_id: str | None = None,
    mind_topology: MindTopology | str | None = None,
    active_presence: PresenceSnapshot | None = None,
    character_id: str | None = None,
) -> tuple[dict[str, Any], MindSnapshot]:
    """Resolve and persist the active Mind for a conversation if missing."""

    owner_user_id = str(conversation["user_id"])
    requested_mind_id = _optional_text(mind_id) or _optional_text(conversation.get("active_mind_id"))
    resolved_topology = MindTopology(
        mind_topology
        or conversation.get("mind_topology")
        or MindTopology.UNIMIND.value
    )
    effective_character_id = (
        character_id
        if character_id is not None
        else _optional_text(conversation.get("character_id"))
        or _optional_text(conversation.get("workspace_id"))
    )
    snapshot = await resolve_active_mind_snapshot(
        connection,
        clock,
        owner_user_id=owner_user_id,
        mind_id=requested_mind_id,
        active_presence=active_presence,
        character_id=effective_character_id,
        topology=resolved_topology,
    )
    current_mind_id = _optional_text(conversation.get("active_mind_id"))
    current_topology = MindTopology(
        conversation.get("mind_topology") or MindTopology.UNIMIND.value
    )
    if current_mind_id is None:
        updated = await ConversationRepository(
            connection,
            clock,
        ).set_active_mind(
            str(conversation["id"]),
            owner_user_id,
            snapshot.mind_id,
            resolved_topology,
        )
        if updated is not None:
            conversation = updated
    elif mind_id is not None and current_mind_id != snapshot.mind_id:
        raise ValueError("mind_id does not match the existing conversation")
    elif mind_topology is not None and current_topology is not resolved_topology:
        raise ValueError("mind_topology does not match the existing conversation")
    return conversation, snapshot


def snapshot_fields(
    *,
    active_mind: MindSnapshot | None,
    source_mind: MindSnapshot | None = None,
) -> dict[str, Any]:
    """Return serializable Mind fields for contexts and job payloads."""

    source = source_mind or active_mind
    return {
        "active_mind_id": active_mind.mind_id if active_mind is not None else None,
        "active_mind_display_name": (
            active_mind.display_name if active_mind is not None else None
        ),
        "source_mind_id": source.mind_id if source is not None else None,
        "mind_topology": (
            active_mind.topology.value
            if active_mind is not None
            else MindTopology.UNIMIND.value
        ),
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "ensure_conversation_active_mind",
    "resolve_active_mind_snapshot",
    "snapshot_fields",
]
