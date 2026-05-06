"""Runtime identity context introduced by the memory namespace redesign.

This module defines the explicit identity axes the system uses after the
namespace cut: ``user_id`` + optional ``user_persona_id`` + ``platform_id``
+ optional ``character_id`` + ``conversation_id``, plus the soft retrieval
hints (``mode``) and the privacy controls (``incognito`` and the user-level
``remember_across_chats`` / ``remember_across_devices`` preferences).

The context lives in code rather than the database. Callers build a
:class:`MemoryNamespaceContext` once per request/turn and pass it through
repositories, retrieval, extraction and workers. SQL filters in Phase 3+
read identity directly from this object instead of reconstructing it from
``assistant_mode_id`` / ``workspace_id`` like the legacy paths do.

The ``user_persona_id``, ``platform_id`` and ``character_id`` fields are
explicit caller inputs. They must never be inferred from prompt text, model
names, or integration metadata; that is part of the security contract
documented in the plan.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atagia.models.schemas_memory import MemoryScope, MemorySensitivity


def tagged_key(value: str | None) -> str:
    """Encode an optional id as a SQLite-safe collision-free key.

    The plan specifies the rule:

    * ``None`` -> ``'n'``
    * non-``None`` -> ``'v:<length>:<raw value>'``

    Wrapping every raw id with the ``v:<length>:`` prefix means a caller-
    supplied id like ``'n'`` or ``'v:3:abc'`` cannot collide with the null
    marker or with another raw value of a different length. This is used as
    the upsert/uniqueness key for nullable identity columns such as
    ``user_persona_id``, ``character_id`` and ``conversation_id``.

    The same rule is implemented in SQL via ``GENERATED ALWAYS AS`` virtual
    columns on the affected tables (see migration 0031). This Python helper
    exists for non-DB code paths (cache keys, in-memory dedupe, tests).
    """

    if value is None:
        return "n"
    return f"v:{len(value)}:{value}"


class MemoryNamespaceContext(BaseModel):
    """Resolved identity + privacy context for a single request/turn.

    All fields are explicit caller input. ``mode`` is a retrieval/profile
    hint only and never participates in scope SQL or sensitivity gates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    user_persona_id: str | None = None
    platform_id: str = Field(min_length=1)
    character_id: str | None = None
    conversation_id: str = Field(min_length=1)
    mode: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True

    @field_validator("user_id", "platform_id", "conversation_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Identity field must be a non-blank string")
        return stripped

    @field_validator("user_persona_id", "character_id", "mode")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Optional identity fields must be None or non-blank")
        return stripped

    @property
    def user_persona_key(self) -> str:
        return tagged_key(self.user_persona_id)

    @property
    def character_key(self) -> str:
        return tagged_key(self.character_id)

    @property
    def conversation_key(self) -> str:
        return tagged_key(self.conversation_id)

    def cross_chat_allowed(self) -> bool:
        """True when broad (character/user) memory is eligible this turn.

        Incognito and the user preference to remember across chats both
        narrow this to chat scope, regardless of the active ``character_id``.
        """

        return (not self.incognito) and self.remember_across_chats

    def cross_device_allowed(self) -> bool:
        """True when memory created on other platforms is eligible.

        ``remember_across_devices=False`` keeps cross-chat memory on the
        active platform only; this is independent of incognito.
        """

        return self.remember_across_devices

    def effective_policy_hash_inputs(self) -> tuple[str, ...]:
        """Stable inputs for context cache keying / effective-policy hashing.

        Includes every value that can change retrieval visibility: identity,
        incognito, and the two user-level memory preferences. Mode is a
        ranking hint only and is intentionally excluded so cache entries
        survive a mode switch within the same conversation.
        """

        return (
            self.user_id,
            self.user_persona_key,
            self.platform_id,
            self.character_key,
            self.conversation_id,
            "incognito=1" if self.incognito else "incognito=0",
            "rac=1" if self.remember_across_chats else "rac=0",
            "rad=1" if self.remember_across_devices else "rad=0",
        )

    def with_overrides(self, **changes: Any) -> "MemoryNamespaceContext":
        """Return a copy with the given fields overridden.

        The model is frozen, so callers that need to derive a new context
        for a worker job or admin operation use this helper rather than
        mutating the existing instance.
        """

        return self.model_copy(update=changes)


class MemoryWritePolicy(BaseModel):
    """Resolved write-time policy applied by extraction and worker writes.

    The fields here come from a strictest-wins merge across (a) the source
    turn snapshot, (b) the current conversation/user preferences, and (c)
    any sensitivity overrides imposed by intimacy, high-risk policy, or the
    extractor itself. Workers re-evaluate this object before activating
    delayed writes; current state may *narrow* a job but never *broaden* a
    snapshot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: MemoryScope
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    themes: tuple[str, ...] = ()
    auto_expires: bool = False
    platform_locked: bool = False
    platform_id_lock: str | None = None

    @field_validator("scope")
    @classmethod
    def _scope_must_be_canonical(cls, value: MemoryScope) -> MemoryScope:
        if value not in {MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER}:
            raise ValueError(
                "MemoryWritePolicy only accepts canonical scopes "
                "(chat / character / user); legacy scopes are storage-only."
            )
        return value


__all__ = [
    "MemoryNamespaceContext",
    "MemoryWritePolicy",
    "tagged_key",
]
