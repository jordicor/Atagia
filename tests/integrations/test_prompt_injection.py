from __future__ import annotations

from dataclasses import dataclass

from atagia.integrations.prompt_injection import (
    ATAGIA_CONTEXT_HEADER,
    build_injection_decision,
    context_messages_for_provider,
    extract_context_message_id,
    extract_context_system_prompt,
)
from atagia.models.schemas_api import ContextResult


@dataclass(slots=True)
class ContextObject:
    system_prompt: str
    request_message_id: str


def test_build_injection_decision_appends_context_and_suppresses_history() -> None:
    decision = build_injection_decision(
        "Base prompt",
        {"system_prompt": "Memory says: prefers short answers.", "message_id": "msg_1"},
    )

    assert decision.active is True
    assert decision.reason == "active"
    assert ATAGIA_CONTEXT_HEADER in decision.full_prompt
    assert "Memory says: prefers short answers." in decision.full_prompt
    assert decision.atagia_user_message_id == "msg_1"
    assert context_messages_for_provider([{"role": "user"}], decision) == []


def test_build_injection_decision_keeps_prompt_when_context_missing() -> None:
    decision = build_injection_decision("Base prompt", None)

    assert decision.active is False
    assert decision.reason == "no_context"
    assert decision.full_prompt == "Base prompt"


def test_extract_context_message_id_accepts_object_context() -> None:
    context = ContextObject(system_prompt="Memory", request_message_id="msg_2")

    assert extract_context_message_id(context) == "msg_2"


def test_context_result_exposes_prompt_and_request_message_id() -> None:
    context = ContextResult(
        system_prompt="Memory",
        request_message_id="aurvek:msg:1",
    )

    decision = build_injection_decision("Base prompt", context)

    assert extract_context_system_prompt(context) == "Memory"
    assert decision.active is True
    assert decision.atagia_user_message_id == "aurvek:msg:1"
