"""Aurvek naming conventions for Atagia sidecar integrations.

This module intentionally does not import Aurvek. It only centralizes the
stable IDs Aurvek-style hosts should pass into the generic sidecar bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

AURVEK_PLATFORM_ID = "aurvek"


@dataclass(frozen=True, slots=True)
class AurvekNamespace:
    """Resolved Atagia namespace values for one Aurvek turn."""

    user_id: str
    conversation_id: str
    platform_id: str = AURVEK_PLATFORM_ID
    character_id: str | None = None

    @classmethod
    def from_ids(
        cls,
        *,
        user_id: int | str,
        conversation_id: int | str,
        prompt_id: int | str | None = None,
    ) -> "AurvekNamespace":
        return cls(
            user_id=aurvek_user_id(user_id),
            conversation_id=aurvek_conversation_id(conversation_id),
            character_id=(
                aurvek_prompt_character_id(prompt_id)
                if prompt_id is not None
                else None
            ),
        )


def aurvek_user_id(value: int | str) -> str:
    """Return the canonical Atagia user id for an Aurvek user."""
    return f"aurvek:user:{_id_part(value)}"


def aurvek_conversation_id(value: int | str) -> str:
    """Return the canonical Atagia conversation id for an Aurvek chat."""
    return f"aurvek:conv:{_id_part(value)}"


def aurvek_message_id(value: int | str) -> str:
    """Return the canonical Atagia message id for an Aurvek message."""
    return f"aurvek:msg:{_id_part(value)}"


def aurvek_prompt_character_id(prompt_id: int | str) -> str:
    """Return the Atagia character id for an Aurvek prompt/persona."""
    return f"prompt:{_id_part(prompt_id)}"


def _id_part(value: int | str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Aurvek IDs must be non-empty")
    return text


__all__ = [
    "AURVEK_PLATFORM_ID",
    "AurvekNamespace",
    "aurvek_conversation_id",
    "aurvek_message_id",
    "aurvek_prompt_character_id",
    "aurvek_user_id",
]
