"""Tests for the post-redesign namespace context and scope helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atagia.memory.namespace import (
    MemoryNamespaceContext,
    MemoryWritePolicy,
    tagged_key,
)
from atagia.memory.scope_utils import (
    platform_filter_clause,
    resolve_namespace_identifiers,
    resolve_scope_identifiers,
    scope_hash_seed,
)
from atagia.models.schemas_memory import (
    CANONICAL_SCOPES,
    LEGACY_SCOPES,
    MemoryScope,
    MemorySensitivity,
)


def test_tagged_key_handles_none_and_collisions() -> None:
    assert tagged_key(None) == "n"
    assert tagged_key("") == "v:0:"
    # A caller-supplied id of 'n' must not collide with the null marker.
    assert tagged_key("n") == "v:1:n"
    # Two values that share a prefix but different lengths cannot collide.
    assert tagged_key("abc") == "v:3:abc"
    assert tagged_key("abcd") == "v:4:abcd"


def test_namespace_context_strips_required_fields() -> None:
    ctx = MemoryNamespaceContext(
        user_id="  u1  ",
        platform_id="  p1  ",
        conversation_id="c1",
    )
    assert ctx.user_id == "u1"
    assert ctx.platform_id == "p1"
    assert ctx.user_persona_key == "n"
    assert ctx.character_key == "n"
    assert ctx.conversation_key == "v:2:c1"


def test_namespace_context_rejects_blank_required_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryNamespaceContext(user_id=" ", platform_id="p", conversation_id="c")
    with pytest.raises(ValidationError):
        MemoryNamespaceContext(user_id="u", platform_id="", conversation_id="c")
    with pytest.raises(ValidationError):
        MemoryNamespaceContext(user_id="u", platform_id="p", conversation_id="")


def test_namespace_context_rejects_blank_optional_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryNamespaceContext(
            user_id="u",
            platform_id="p",
            conversation_id="c",
            user_persona_id="   ",
        )


def test_cross_chat_and_cross_device_flags_match_plan_rules() -> None:
    ctx = MemoryNamespaceContext(
        user_id="u",
        platform_id="p",
        conversation_id="c",
    )
    assert ctx.cross_chat_allowed() is True
    assert ctx.cross_device_allowed() is True

    incognito = ctx.with_overrides(incognito=True)
    assert incognito.cross_chat_allowed() is False

    no_cross_chat = ctx.with_overrides(remember_across_chats=False)
    assert no_cross_chat.cross_chat_allowed() is False

    no_cross_device = ctx.with_overrides(remember_across_devices=False)
    assert no_cross_device.cross_device_allowed() is False


def test_effective_policy_hash_excludes_mode_but_includes_privacy_axes() -> None:
    base = MemoryNamespaceContext(
        user_id="u", platform_id="p", conversation_id="c", mode="companion"
    )
    other_mode = base.with_overrides(mode="brainstorm")
    assert base.effective_policy_hash_inputs() == other_mode.effective_policy_hash_inputs()

    incognito = base.with_overrides(incognito=True)
    assert base.effective_policy_hash_inputs() != incognito.effective_policy_hash_inputs()


def test_resolve_scope_identifiers_only_handles_legacy_scopes() -> None:
    assert resolve_scope_identifiers(
        MemoryScope.GLOBAL_USER,
        assistant_mode_id="m",
        workspace_id="w",
        conversation_id="c",
    ) == {"assistant_mode_id": None, "workspace_id": None, "conversation_id": None}

    assert resolve_scope_identifiers(
        MemoryScope.WORKSPACE,
        assistant_mode_id="m",
        workspace_id=None,
        conversation_id=None,
    ) is None

    # Canonical scopes must not be addressable through the legacy resolver.
    for scope in (MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER):
        assert resolve_scope_identifiers(
            scope,
            assistant_mode_id="m",
            workspace_id="w",
            conversation_id="c",
        ) is None


def test_resolve_namespace_identifiers_for_canonical_scopes() -> None:
    ctx = MemoryNamespaceContext(
        user_id="u",
        platform_id="p",
        conversation_id="c",
        user_persona_id="alter",
        character_id="ch",
    )
    assert resolve_namespace_identifiers(MemoryScope.CHAT, ctx) == {
        "user_persona_id": "alter",
        "character_id": "ch",
        "conversation_id": "c",
    }
    assert resolve_namespace_identifiers(MemoryScope.CHARACTER, ctx) == {
        "user_persona_id": "alter",
        "character_id": "ch",
        "conversation_id": None,
    }
    assert resolve_namespace_identifiers(MemoryScope.USER, ctx) == {
        "user_persona_id": "alter",
        "character_id": None,
        "conversation_id": None,
    }


def test_character_scope_requires_character_id() -> None:
    ctx = MemoryNamespaceContext(user_id="u", platform_id="p", conversation_id="c")
    assert resolve_namespace_identifiers(MemoryScope.CHARACTER, ctx) is None


def test_scope_hash_seed_segregates_namespaces() -> None:
    ctx = MemoryNamespaceContext(
        user_id="u",
        platform_id="p1",
        conversation_id="c1",
        character_id="ch1",
    )
    ctx_other_platform = ctx.with_overrides(platform_id="p2")
    ctx_other_persona = ctx.with_overrides(user_persona_id="alter")
    ctx_other_chat = ctx.with_overrides(conversation_id="c2")
    ctx_other_character = ctx.with_overrides(character_id="ch2")

    chat_seed = scope_hash_seed(MemoryScope.CHAT, ctx)
    assert scope_hash_seed(MemoryScope.CHAT, ctx_other_chat) != chat_seed
    # Platform only enters the seed when the row is platform-locked.
    assert scope_hash_seed(MemoryScope.CHAT, ctx_other_platform) == chat_seed
    assert scope_hash_seed(
        MemoryScope.CHAT, ctx_other_platform, platform_locked=True
    ) != scope_hash_seed(MemoryScope.CHAT, ctx, platform_locked=True)

    character_seed = scope_hash_seed(MemoryScope.CHARACTER, ctx)
    assert scope_hash_seed(MemoryScope.CHARACTER, ctx_other_character) != character_seed

    user_seed = scope_hash_seed(MemoryScope.USER, ctx)
    assert scope_hash_seed(MemoryScope.USER, ctx_other_persona) != user_seed
    assert user_seed != character_seed
    assert chat_seed != user_seed


def test_platform_filter_clause_respects_cross_device_setting() -> None:
    ctx = MemoryNamespaceContext(
        user_id="u", platform_id="p1", conversation_id="c"
    )
    clause_on, params_on = platform_filter_clause(
        platform_locked_column="platform_locked",
        platform_id_lock_column="platform_id_lock",
        context=ctx,
    )
    assert "platform_locked = 0 OR platform_id_lock = :ctx_platform_id" in clause_on
    assert params_on == {"ctx_platform_id": "p1"}

    ctx_off = ctx.with_overrides(remember_across_devices=False)
    clause_off, params_off = platform_filter_clause(
        platform_locked_column="platform_locked",
        platform_id_lock_column="platform_id_lock",
        context=ctx_off,
    )
    assert "platform_id = :ctx_platform_id" in clause_off
    assert params_off == {"ctx_platform_id": "p1"}


def test_canonical_and_legacy_scope_sets_are_disjoint() -> None:
    assert CANONICAL_SCOPES.isdisjoint(LEGACY_SCOPES)
    assert CANONICAL_SCOPES == {
        MemoryScope.CHAT,
        MemoryScope.CHARACTER,
        MemoryScope.USER,
    }


def test_memory_write_policy_only_accepts_canonical_scopes() -> None:
    MemoryWritePolicy(scope=MemoryScope.CHAT, sensitivity=MemorySensitivity.PUBLIC)
    for legacy in LEGACY_SCOPES:
        with pytest.raises(ValidationError):
            MemoryWritePolicy(scope=legacy)
