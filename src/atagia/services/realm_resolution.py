"""Request-time Realm resolution helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.realm_repository import (
    RealmRepository,
    RealmSnapshot,
    realm_snapshot,
)
from atagia.core.repositories import ConversationRepository
from atagia.models.schemas_memory import CrossRealmMode


async def resolve_active_realm_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    owner_user_id: str,
    realm_id: str | None = None,
    cross_realm_mode: CrossRealmMode | str | None = None,
    display_name: str | None = None,
) -> RealmSnapshot | None:
    """Resolve the active Realm row without mutating a conversation."""

    row = await RealmRepository(connection, clock).resolve_active_realm(
        owner_user_id=owner_user_id,
        realm_id=_optional_text(realm_id),
        cross_realm_mode=cross_realm_mode,
        display_name=display_name,
    )
    if row is None:
        return None
    return realm_snapshot(row)


async def ensure_conversation_active_realm(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    conversation: dict[str, Any],
    realm_id: str | None = None,
    cross_realm_mode: CrossRealmMode | str | None = None,
    display_name: str | None = None,
) -> tuple[dict[str, Any], RealmSnapshot | None]:
    """Resolve and persist the active Realm for a conversation if present."""

    owner_user_id = str(conversation["user_id"])
    requested_realm_id = _optional_text(realm_id) or _optional_text(
        conversation.get("active_realm_id")
    )
    snapshot = await resolve_active_realm_snapshot(
        connection,
        clock,
        owner_user_id=owner_user_id,
        realm_id=requested_realm_id,
        cross_realm_mode=cross_realm_mode,
        display_name=display_name,
    )
    if snapshot is None:
        return conversation, None

    current_realm_id = _optional_text(conversation.get("active_realm_id"))
    conversation = {
        **conversation,
        "active_realm_id": snapshot.realm_id,
        "active_realm_display_name": snapshot.display_name,
        "cross_realm_mode": snapshot.cross_realm_mode.value,
    }
    if current_realm_id is None:
        updated = await ConversationRepository(
            connection,
            clock,
        ).set_active_realm(
            str(conversation["id"]),
            owner_user_id,
            snapshot.realm_id,
        )
        if updated is not None:
            conversation = {
                **updated,
                "active_realm_display_name": snapshot.display_name,
                "cross_realm_mode": snapshot.cross_realm_mode.value,
            }
    elif realm_id is not None and current_realm_id != snapshot.realm_id:
        raise ValueError("realm_id does not match the existing conversation")
    return conversation, snapshot


def snapshot_fields(realm: RealmSnapshot | None) -> dict[str, Any]:
    """Return serializable Realm fields for contexts and job payloads."""

    return {
        "active_realm_id": realm.realm_id if realm is not None else None,
        "active_realm_display_name": (
            realm.display_name if realm is not None else None
        ),
        "cross_realm_mode": (
            realm.cross_realm_mode.value
            if realm is not None
            else CrossRealmMode.NONE.value
        ),
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "ensure_conversation_active_realm",
    "resolve_active_realm_snapshot",
    "snapshot_fields",
]
