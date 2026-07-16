"""Shared comparison rules for optional conversation identity hints."""

from __future__ import annotations

from typing import Any, Mapping

from atagia.services.errors import ConversationNotFoundError


def validate_optional_identity_hints(
    conversation: Mapping[str, Any],
    *,
    user_persona_id: str | None,
    platform_id: str | None,
    character_id: str | None,
) -> None:
    """Require supplied hints to match without constraining omitted hints.

    ``None`` is the transport-neutral representation of an omitted or explicit
    JSON-null hint. Every non-null value, including the empty string, is an
    explicit claim and therefore participates in the comparison.
    """

    for field_name, expected in (
        ("user_persona_id", user_persona_id),
        ("platform_id", platform_id),
        ("character_id", character_id),
    ):
        if expected is None:
            continue
        actual = conversation.get(field_name)
        actual_text = None if actual is None else str(actual)
        if actual_text != expected:
            raise ConversationNotFoundError("Conversation not found for user")
