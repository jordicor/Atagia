"""Tests for the card-examples on/off toggle (compose_card_prompt + resolver)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atagia.core.config import Settings
from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.consequence_detector import (
    _CARD_PURPOSES as CONSEQUENCE_CARD_PURPOSES,
    _card_task as consequence_card_task,
)
from atagia.memory.need_detector import _CARD_NAMES, _card_task as need_card_task
from atagia.services.model_resolution import (
    ModelResolutionError,
    component_env_examples_from_env,
    examples_enabled_for_component,
)


def test_compose_card_prompt_includes_and_omits_examples() -> None:
    instruction = "Read the user message. Answer yes or no."
    examples = "Worked? -> yes\nBroke? -> no"
    with_examples = compose_card_prompt(instruction, examples, include_examples=True)
    without_examples = compose_card_prompt(instruction, examples, include_examples=False)
    assert with_examples == f"{instruction}\n\nExamples:\n{examples}"
    assert without_examples == instruction
    # Missing or empty example blocks never add a header.
    assert compose_card_prompt(instruction, None, include_examples=True) == instruction
    assert compose_card_prompt(instruction, "  ", include_examples=True) == instruction


def test_examples_enabled_resolution_layers() -> None:
    global_on = SimpleNamespace(card_examples_enabled=True, llm_component_examples={})
    assert examples_enabled_for_component(global_on, "extractor") is True

    global_off = SimpleNamespace(card_examples_enabled=False, llm_component_examples={})
    assert examples_enabled_for_component(global_off, "extractor") is False

    override = SimpleNamespace(
        card_examples_enabled=True,
        llm_component_examples={"extractor": False, "need_detector_needs": True},
    )
    assert examples_enabled_for_component(override, "extractor") is False
    assert examples_enabled_for_component(override, "need_detector_needs") is True
    # Unset components fall back to the global default.
    assert examples_enabled_for_component(override, "topic_working_set") is True

    with pytest.raises(ModelResolutionError):
        examples_enabled_for_component(global_on, "does_not_exist")


def test_examples_env_override_parsing() -> None:
    parsed = component_env_examples_from_env(
        {
            "ATAGIA_LLM_EXAMPLES__EXTRACTOR": "false",
            "ATAGIA_LLM_EXAMPLES__APPLICABILITY_SCORER": "true",
            "ATAGIA_LLM_EXAMPLES__UNRELATED": "false",
        }
    )
    assert parsed == {"extractor": False, "applicability_scorer": True}


def test_settings_default_keeps_examples_on() -> None:
    settings = Settings.from_env()
    assert settings.card_examples_enabled is True
    assert examples_enabled_for_component(settings, "extractor") is True


def test_need_detector_card_task_gates_examples() -> None:
    for card_name in _CARD_NAMES:
        instruction, examples, _ = need_card_task(card_name)
        # The demonstration block lives in `examples`, never baked into the
        # instruction (an inline value hint like "Examples: en, es" is not a
        # demonstration block and may remain in the instruction).
        assert examples and examples.strip()
        on = compose_card_prompt(instruction, examples, include_examples=True)
        off = compose_card_prompt(instruction, examples, include_examples=False)
        assert on.endswith(examples.strip())
        assert on.count("Examples:\n") == 1
        assert off == instruction.rstrip()
        assert examples.strip() not in off


def test_consequence_card_task_gates_examples() -> None:
    for card_name in CONSEQUENCE_CARD_PURPOSES:
        on = consequence_card_task(card_name, include_examples=True)
        off = consequence_card_task(card_name, include_examples=False)
        assert "Examples:" in on
        assert "Examples:" not in off
        assert len(off) < len(on)
