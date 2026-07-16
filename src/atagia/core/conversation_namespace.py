"""Exact conversation namespace snapshots for cross-transaction authorization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleRepository,
)
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.embodiment_repository import EmbodimentRepository
from atagia.core.mind_repository import MindRepository
from atagia.core.presence_repository import PresenceRepository
from atagia.core.realm_repository import RealmRepository
from atagia.core.space_repository import SpaceRepository, space_snapshot
from atagia.core.user_lifecycle_repository import UserLifecycleRepository


@dataclass(frozen=True, slots=True)
class ConversationNamespaceSnapshot:
    """Fields that determine a conversation's memory visibility boundary."""

    user_id: str
    user_lifecycle_epoch: str
    conversation_id: str
    conversation_lifecycle_epoch: str
    status: str
    platform_id: str | None
    user_persona_id: str | None
    character_id: str | None
    workspace_id: str | None
    assistant_mode_id: str | None
    mode: str | None
    incognito: bool
    isolated_mode: bool
    remember_across_chats: bool
    remember_across_devices: bool
    memory_privacy_mode: str
    active_space_id: str | None
    active_space_boundary_mode: str | None
    active_mind_id: str | None
    mind_topology: str
    active_presence_id: str | None
    active_presence_kind: str | None
    active_presence_cluster_id: str | None
    active_mind_kind: str | None
    active_embodiment_id: str | None
    active_embodiment_boundary_mode: str | None
    active_realm_id: str | None
    active_realm_cross_mode: str | None

    def memory_visibility_kwargs(self) -> dict[str, Any]:
        """Return the exact coordinates consumed by memory visibility queries."""

        return {
            "conversation_id": self.conversation_id,
            "user_persona_id": self.user_persona_id,
            "platform_id": self.platform_id,
            "character_id": self.character_id,
            "incognito": self.incognito,
            "remember_across_chats": self.remember_across_chats,
            "remember_across_devices": self.remember_across_devices,
            "active_space_id": self.active_space_id,
            "active_space_boundary_mode": self.active_space_boundary_mode,
            "active_mind_id": self.active_mind_id,
            "mind_topology": self.mind_topology,
            "active_embodiment_id": self.active_embodiment_id,
            "active_realm_id": self.active_realm_id,
        }


_NAMESPACE_REQUIRED_TEXT_FIELDS = {
    "user_id",
    "user_lifecycle_epoch",
    "conversation_id",
    "conversation_lifecycle_epoch",
    "status",
    "mind_topology",
    "memory_privacy_mode",
}
_NAMESPACE_BOOL_FIELDS = {
    "incognito",
    "isolated_mode",
    "remember_across_chats",
    "remember_across_devices",
}


def serialize_conversation_namespace_snapshot(
    snapshot: ConversationNamespaceSnapshot,
) -> dict[str, Any]:
    """Return the canonical versioned payload for durable request fences."""

    return {
        "schema_version": 1,
        "snapshot": asdict(snapshot),
    }


def parse_conversation_namespace_snapshot(
    payload: Any,
) -> ConversationNamespaceSnapshot:
    """Strictly parse a durable namespace snapshot or fail closed."""

    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "snapshot",
    }:
        raise ValueError("Conversation namespace snapshot envelope is invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Conversation namespace snapshot version is unsupported")
    values = payload["snapshot"]
    expected_fields = {field.name for field in fields(ConversationNamespaceSnapshot)}
    if not isinstance(values, dict) or set(values) != expected_fields:
        raise ValueError("Conversation namespace snapshot fields are invalid")
    for field_name in _NAMESPACE_REQUIRED_TEXT_FIELDS:
        if not isinstance(values[field_name], str):
            raise ValueError(f"Conversation namespace field {field_name} must be text")
    for field_name in _NAMESPACE_BOOL_FIELDS:
        if type(values[field_name]) is not bool:
            raise ValueError(
                f"Conversation namespace field {field_name} must be boolean"
            )
    optional_text_fields = (
        expected_fields - _NAMESPACE_REQUIRED_TEXT_FIELDS - _NAMESPACE_BOOL_FIELDS
    )
    for field_name in optional_text_fields:
        if values[field_name] is not None and not isinstance(values[field_name], str):
            raise ValueError(
                f"Conversation namespace field {field_name} must be text or null"
            )
    return ConversationNamespaceSnapshot(**values)


async def capture_conversation_namespace_snapshot(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    user_id: str,
    conversation_id: str,
) -> ConversationNamespaceSnapshot | None:
    """Read all canonical fields that control one conversation namespace."""

    conversation = await ConversationRepository(
        connection,
        clock,
    ).get_conversation(conversation_id, user_id)
    if conversation is None:
        return None
    preferences = await UserRepository(
        connection,
        clock,
    ).get_memory_preferences(user_id)
    if preferences is None:
        return None
    user_lifecycle = await UserLifecycleRepository(
        connection,
        clock,
    ).get_active_identity(user_id)
    if user_lifecycle is None:
        return None
    lifecycle = await ConversationLifecycleRepository(
        connection,
        clock,
    ).get_identity(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if lifecycle is None:
        return None

    active_space_id = conversation.get("active_space_id")
    active_space_boundary_mode = None
    if active_space_id is not None:
        space = await SpaceRepository(connection, clock).get_space(
            owner_user_id=user_id,
            space_id=str(active_space_id),
        )
        if space is not None:
            active_space_boundary_mode = space_snapshot(space).boundary_mode.value

    active_presence_id = conversation.get("active_presence_id")
    presence = (
        await PresenceRepository(connection, clock).get_presence(
            owner_user_id=user_id,
            presence_id=str(active_presence_id),
        )
        if active_presence_id is not None
        else None
    )
    active_mind_id = conversation.get("active_mind_id")
    mind = (
        await MindRepository(connection, clock).get_mind(
            owner_user_id=user_id,
            mind_id=str(active_mind_id),
        )
        if active_mind_id is not None
        else None
    )
    active_embodiment_id = conversation.get("active_embodiment_id")
    embodiment = (
        await EmbodimentRepository(connection, clock).get_embodiment(
            owner_user_id=user_id,
            embodiment_id=str(active_embodiment_id),
        )
        if active_embodiment_id is not None
        else None
    )
    active_realm_id = conversation.get("active_realm_id")
    realm = (
        await RealmRepository(connection, clock).get_realm(
            owner_user_id=user_id,
            realm_id=str(active_realm_id),
        )
        if active_realm_id is not None
        else None
    )

    return ConversationNamespaceSnapshot(
        user_id=user_id,
        user_lifecycle_epoch=user_lifecycle.lifecycle_epoch,
        conversation_id=conversation_id,
        conversation_lifecycle_epoch=lifecycle.lifecycle_epoch,
        status=str(conversation.get("status") or ""),
        platform_id=conversation.get("platform_id"),
        user_persona_id=conversation.get("user_persona_id"),
        character_id=conversation.get("character_id"),
        workspace_id=conversation.get("workspace_id"),
        assistant_mode_id=conversation.get("assistant_mode_id"),
        mode=conversation.get("mode"),
        incognito=bool(conversation.get("incognito")),
        isolated_mode=bool(conversation.get("isolated_mode")),
        remember_across_chats=bool(preferences["remember_across_chats"]),
        remember_across_devices=bool(preferences["remember_across_devices"]),
        memory_privacy_mode=str(preferences["memory_privacy_mode"]),
        active_space_id=(str(active_space_id) if active_space_id is not None else None),
        active_space_boundary_mode=active_space_boundary_mode,
        active_mind_id=(str(active_mind_id) if active_mind_id is not None else None),
        mind_topology=str(conversation.get("mind_topology") or "unimind"),
        active_presence_id=(
            str(active_presence_id) if active_presence_id is not None else None
        ),
        active_presence_kind=(str(presence["kind"]) if presence is not None else None),
        active_presence_cluster_id=(
            str(presence["presence_cluster_id"])
            if presence is not None and presence.get("presence_cluster_id") is not None
            else None
        ),
        active_mind_kind=(str(mind["kind"]) if mind is not None else None),
        active_embodiment_id=(
            str(active_embodiment_id) if active_embodiment_id is not None else None
        ),
        active_embodiment_boundary_mode=(
            str(embodiment["cross_embodiment_mode"]) if embodiment is not None else None
        ),
        active_realm_id=(str(active_realm_id) if active_realm_id is not None else None),
        active_realm_cross_mode=(
            str(realm["cross_realm_mode"]) if realm is not None else None
        ),
    )
