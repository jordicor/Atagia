"""Helpers for mapping memory scopes to canonical identifier ownership."""

from __future__ import annotations

from atagia.models.schemas_memory import MemoryScope


def resolve_scope_identifiers(
    scope: MemoryScope,
    *,
    assistant_mode_id: str | None,
    workspace_id: str | None,
    conversation_id: str | None,
) -> dict[str, str | None] | None:
    """Return the only identifiers that should be persisted for a given scope."""
    if scope is MemoryScope.GLOBAL_USER:
        return {
            "assistant_mode_id": None,
            "workspace_id": None,
            "conversation_id": None,
        }
    if scope is MemoryScope.ASSISTANT_MODE:
        if assistant_mode_id is None:
            return None
        return {
            "assistant_mode_id": assistant_mode_id,
            "workspace_id": None,
            "conversation_id": None,
        }
    if scope is MemoryScope.WORKSPACE:
        if assistant_mode_id is None or workspace_id is None:
            return None
        return {
            "assistant_mode_id": assistant_mode_id,
            "workspace_id": workspace_id,
            "conversation_id": None,
        }
    if assistant_mode_id is None or conversation_id is None:
        return None
    return {
        "assistant_mode_id": assistant_mode_id,
        "workspace_id": workspace_id,
        "conversation_id": conversation_id,
    }
