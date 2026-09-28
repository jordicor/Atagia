"""Tests for minimal host injection in the core prompt_injection helper."""

from __future__ import annotations

from atagia.integrations.prompt_injection import (
    ATAGIA_CONTEXT_FOOTER,
    ATAGIA_CONTEXT_HEADER,
    MINIMAL_MEMORY_INSTRUCTION,
    append_context_to_prompt,
    build_injection_decision,
    extract_prompt_data_sections,
    minimal_memory_payload,
)

COMPOSED_PROMPT = (
    "You are the Atagia assistant for mode general_qa. Answer with care.\n"
    "Follow every factual grounding rule listed above.\n"
    "Resolved policy hash: deadbeef\n"
    "\n"
    "<retrieved_memory>\n"
    "1. (evidence, date: 2026-03-08) The user practices pottery on weekends.\n"
    "</retrieved_memory>\n"
    "\n"
    "<current_user_state>\n"
    "- active project: a ceramic vase\n"
    "</current_user_state>\n"
    "\n"
    "<prepared_initial_context>\n"
    "[Prepared Initial Context]\n"
    "- The user prefers morning meetings.\n"
    "</prepared_initial_context>"
)

COMPOSED_PROMPT_WITHOUT_MEMORIES = (
    "You are the Atagia assistant for mode general_qa. Answer with care.\n"
    "Resolved policy hash: deadbeef"
)


def test_minimal_payload_extracts_memory_sections_from_composed_prompt() -> None:
    payload = minimal_memory_payload(COMPOSED_PROMPT)

    assert "The user practices pottery on weekends." in payload
    assert "active project: a ceramic vase" in payload
    assert "The user prefers morning meetings." in payload
    assert "Resolved policy hash" not in payload
    assert "factual grounding rule" not in payload


def test_minimal_payload_returns_empty_for_composed_prompt_without_sections() -> None:
    assert minimal_memory_payload(COMPOSED_PROMPT_WITHOUT_MEMORIES) == ""


def test_minimal_payload_passes_foreign_text_verbatim() -> None:
    foreign = "Memory says: prefers short answers."

    assert minimal_memory_payload(foreign) == foreign


def test_minimal_payload_is_idempotent_over_already_minimal_text() -> None:
    once = minimal_memory_payload(COMPOSED_PROMPT)

    assert minimal_memory_payload(f"{MINIMAL_MEMORY_INSTRUCTION}\n\n{once}") == once


def test_extract_prompt_data_sections_returns_sections_in_tag_order() -> None:
    sections = extract_prompt_data_sections(COMPOSED_PROMPT)

    assert len(sections) == 3
    assert sections[0].startswith("<retrieved_memory>")
    assert sections[0].endswith("</retrieved_memory>")
    assert sections[1].startswith("<current_user_state>")
    assert sections[2].startswith("<prepared_initial_context>")


def test_append_context_to_prompt_wraps_minimal_payload() -> None:
    augmented = append_context_to_prompt(
        "Base prompt",
        {"system_prompt": COMPOSED_PROMPT},
    )

    assert augmented.startswith("Base prompt")
    assert ATAGIA_CONTEXT_HEADER in augmented
    assert "- INTERNAL" not in augmented
    assert MINIMAL_MEMORY_INSTRUCTION in augmented
    assert "The user practices pottery on weekends." in augmented
    assert "Resolved policy hash" not in augmented
    assert augmented.endswith(ATAGIA_CONTEXT_FOOTER)


def test_build_injection_decision_skips_composed_prompt_without_memories() -> None:
    decision = build_injection_decision(
        "Base prompt",
        {"system_prompt": COMPOSED_PROMPT_WITHOUT_MEMORIES},
    )

    assert decision.active is False
    assert decision.reason == "empty_context"
    assert decision.full_prompt == "Base prompt"


def test_build_injection_decision_injects_prepared_context_without_memories() -> None:
    prepared_only_prompt = (
        "You are the Atagia assistant for mode companion. Answer with care.\n"
        "Resolved policy hash: deadbeef\n"
        "\n"
        "<prepared_initial_context>\n"
        "[Prepared Initial Context]\n"
        "- The user prefers morning meetings.\n"
        "</prepared_initial_context>"
    )

    decision = build_injection_decision(
        "Base prompt",
        {"system_prompt": prepared_only_prompt},
    )

    # New conversations have no retrieved memories yet; the prepared initial
    # context is the continuity seed and must still reach the host.
    assert decision.active is True
    assert "morning meetings" in decision.full_prompt
    assert "Resolved policy hash" not in decision.full_prompt
