"""Helpers for mapping memory scopes to canonical identifier ownership.

The legacy ``resolve_scope_identifiers`` returns the
(``assistant_mode_id``, ``workspace_id``, ``conversation_id``) triple that
the pre-redesign schema expects. Phase 2 adds new helpers that resolve the
post-redesign identifiers (``user_persona_id``, ``character_id``,
``conversation_id``) for the canonical scopes (``chat`` / ``character`` /
``user``). Both flavors coexist during the additive phase; Phase 11 will
remove the legacy resolver once all callers move to the namespace-aware
helpers.
"""

from __future__ import annotations

from typing import Any

from atagia.memory.namespace import MemoryNamespaceContext, tagged_key
from atagia.models.schemas_memory import MemoryScope


def resolve_scope_identifiers(
    scope: MemoryScope,
    *,
    assistant_mode_id: str | None,
    workspace_id: str | None,
    conversation_id: str | None,
) -> dict[str, str | None] | None:
    """Return the only identifiers that should be persisted for a given scope.

    This helper preserves the pre-redesign mapping so existing repository
    writes continue to insert the legacy columns until they are migrated to
    the namespace context. Phase 11 removes it; until then, it accepts only
    legacy scopes.
    """
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
    if scope in (MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION):
        if assistant_mode_id is None or conversation_id is None:
            return None
        return {
            "assistant_mode_id": assistant_mode_id,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
        }
    # Canonical scopes are not addressable through the legacy resolver. The
    # repositories migrated in Phase 3 use ``resolve_namespace_identifiers``.
    return None


def resolve_namespace_identifiers(
    scope: MemoryScope,
    context: MemoryNamespaceContext,
) -> dict[str, str | None] | None:
    """Return canonical identifiers for the chat/character/user scopes.

    The chat scope binds to the active ``conversation_id``; the character
    scope binds to the active ``character_id`` and is invalid without one
    (rule from the plan: a ``character`` write with no ``character_id`` is
    refused, never silently promoted to ``user``); the user scope binds only
    to ``user_persona_id`` (NULL means base identity).

    All three scopes are persona-bound by ``user_persona_id``; that field
    isolates an alter-ego from the base identity even when both share the
    same ``user_id``.
    """

    if scope is MemoryScope.CHAT:
        return {
            "user_persona_id": context.user_persona_id,
            "character_id": context.character_id,
            "conversation_id": context.conversation_id,
        }
    if scope is MemoryScope.CHARACTER:
        if context.character_id is None:
            return None
        return {
            "user_persona_id": context.user_persona_id,
            "character_id": context.character_id,
            "conversation_id": None,
        }
    if scope is MemoryScope.USER:
        return {
            "user_persona_id": context.user_persona_id,
            "character_id": None,
            "conversation_id": None,
        }
    return None


def scope_hash_seed(
    scope: MemoryScope,
    context: MemoryNamespaceContext,
    *,
    platform_locked: bool = False,
) -> tuple[str, ...]:
    """Stable hash inputs for extraction dedupe and cache keying.

    The plan requires the extraction hash scope seed to include the user,
    persona, character (for character scope), conversation (for chat scope)
    and platform-lock metadata so a fact stated under different namespaces
    cannot collide across them.
    """

    parts: list[str] = [
        f"u:{context.user_id}",
        f"p:{tagged_key(context.user_persona_id)}",
        f"s:{scope.value}",
    ]
    if scope is MemoryScope.CHARACTER:
        parts.append(f"c:{tagged_key(context.character_id)}")
    if scope is MemoryScope.CHAT:
        parts.append(f"v:{tagged_key(context.conversation_id)}")
    if platform_locked:
        parts.append(f"pl:{tagged_key(context.platform_id)}")
    return tuple(parts)


def platform_filter_clause(
    *,
    platform_locked_column: str,
    platform_id_lock_column: str,
    context: MemoryNamespaceContext,
    table_alias: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the SQL fragment + parameters that enforce platform policy.

    Two rules combine here:

    * platform-locked memories are visible only on their locked platform;
    * when ``remember_across_devices=False`` the caller wants to hide rows
      that originated on other platforms even if they are not locked.

    The clause assumes the caller already restricts by ``user_id``. It can
    be appended with ``AND`` to whatever scope/identity filter the
    repository builds.
    """

    prefix = f"{table_alias}." if table_alias else ""
    locked_col = f"{prefix}{platform_locked_column}"
    lock_value_col = f"{prefix}{platform_id_lock_column}"

    params: dict[str, Any] = {}
    fragments: list[str] = []

    if context.cross_device_allowed():
        # Locked rows still must match the active platform; unlocked rows
        # are eligible across devices.
        fragments.append(
            f"({locked_col} = 0 OR {lock_value_col} = :ctx_platform_id)"
        )
    else:
        # Cross-device disabled: keep only memory that originated on or is
        # locked to the current platform.
        fragments.append(
            f"({lock_value_col} = :ctx_platform_id "
            f"OR (({locked_col} = 0) AND :ctx_platform_id IS NOT NULL "
            f"AND {prefix}platform_id = :ctx_platform_id))"
        )
    params["ctx_platform_id"] = context.platform_id
    return " AND ".join(fragments), params


__all__ = [
    "platform_filter_clause",
    "resolve_namespace_identifiers",
    "resolve_scope_identifiers",
    "scope_hash_seed",
]
