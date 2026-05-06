from __future__ import annotations

import pytest

from atagia.integrations import (
    AURVEK_PLATFORM_ID,
    AurvekNamespace,
    aurvek_conversation_id,
    aurvek_message_id,
    aurvek_prompt_character_id,
    aurvek_user_id,
)


def test_aurvek_helpers_build_namespaced_ids() -> None:
    assert AURVEK_PLATFORM_ID == "aurvek"
    assert aurvek_user_id(12) == "aurvek:user:12"
    assert aurvek_conversation_id("34") == "aurvek:conv:34"
    assert aurvek_message_id(56) == "aurvek:msg:56"
    assert aurvek_prompt_character_id("78") == "prompt:78"


def test_aurvek_namespace_resolves_prompt_character_id() -> None:
    namespace = AurvekNamespace.from_ids(
        user_id=12,
        conversation_id=34,
        prompt_id=78,
    )

    assert namespace.user_id == "aurvek:user:12"
    assert namespace.conversation_id == "aurvek:conv:34"
    assert namespace.platform_id == "aurvek"
    assert namespace.character_id == "prompt:78"


def test_aurvek_helpers_reject_empty_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aurvek_message_id(" ")
