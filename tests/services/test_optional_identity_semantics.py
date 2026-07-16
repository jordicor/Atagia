"""Shared optional-identity semantics at chat and sidecar service boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from atagia.services.chat_service import (
    _validate_optional_identity as validate_chat_identity,
)
from atagia.services.errors import ConversationNotFoundError
from atagia.services.sidecar_service import SidecarService


CONVERSATION = {
    "user_persona_id": "persona-a",
    "platform_id": "platform-a",
    "character_id": "character-a",
}


def _validate_chat(conversation: dict[str, Any], hints: dict[str, str | None]) -> None:
    validate_chat_identity(conversation, **hints)


def _validate_sidecar(
    conversation: dict[str, Any],
    hints: dict[str, str | None],
) -> None:
    SidecarService._validate_optional_identity(
        conversation,
        workspace_id=None,
        **hints,
    )


@pytest.mark.parametrize(
    "validator",
    [_validate_chat, _validate_sidecar],
    ids=["chat-service", "sidecar-service"],
)
@pytest.mark.parametrize(
    ("field_name", "persisted_value"),
    [
        ("user_persona_id", "persona-a"),
        ("platform_id", "platform-a"),
        ("character_id", "character-a"),
    ],
)
@pytest.mark.parametrize("claim", ["omitted", "matching", "conflicting"])
def test_optional_identity_hint_matrix(
    validator: Callable[[dict[str, Any], dict[str, str | None]], None],
    field_name: str,
    persisted_value: str,
    claim: str,
) -> None:
    hints: dict[str, str | None] = {
        "user_persona_id": None,
        "platform_id": None,
        "character_id": None,
    }
    if claim == "matching":
        hints[field_name] = persisted_value
    elif claim == "conflicting":
        hints[field_name] = "different-value"

    if claim == "conflicting":
        with pytest.raises(
            ConversationNotFoundError,
            match="Conversation not found for user",
        ):
            validator(CONVERSATION, hints)
    else:
        validator(CONVERSATION, hints)


@pytest.mark.parametrize("validator", [_validate_chat, _validate_sidecar])
@pytest.mark.parametrize("field_name", CONVERSATION)
def test_empty_string_is_a_supplied_identity_claim(
    validator: Callable[[dict[str, Any], dict[str, str | None]], None],
    field_name: str,
) -> None:
    hints: dict[str, str | None] = {
        "user_persona_id": None,
        "platform_id": None,
        "character_id": None,
    }
    hints[field_name] = ""

    with pytest.raises(ConversationNotFoundError):
        validator(CONVERSATION, hints)
