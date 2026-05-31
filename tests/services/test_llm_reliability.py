"""Tests for mechanical LLM reliability helpers."""

from __future__ import annotations

import json

from atagia.services.llm_reliability import (
    LLMTechnicalRecoveryConfig,
    analyze_runaway_signals,
    is_mechanical_runaway,
)


def test_runaway_analysis_ignores_repeated_json_structure() -> None:
    payload = {
        "evidences": [
            {
                "canonical_text": (
                    f"alpha{index} bravo{index} charlie{index} delta{index} "
                    f"echo{index} foxtrot{index}"
                ),
                "index_text": f"lookup{index} surface{index} detail{index}",
                "scope": "conversation",
                "source_kind": "extracted",
            }
            for index in range(32)
        ]
    }

    signals = analyze_runaway_signals(
        json.dumps(payload, indent=2),
        source_input_tokens=10_000,
    )

    assert is_mechanical_runaway(
        signals,
        LLMTechnicalRecoveryConfig.default_enabled(),
        hard_abort=False,
    ) is False
    assert all("canonical_text" not in phrase.text for phrase in signals.repeated_phrases)
    assert all("source_kind" not in phrase.text for phrase in signals.repeated_phrases)


def test_runaway_analysis_detects_repeated_out_of_json_tail() -> None:
    payload = {
        "evidences": [
            {
                "canonical_text": "A distinct supported memory value lives in the JSON.",
                "index_text": "distinct supported lookup text",
            }
        ]
    }
    output = json.dumps(payload) + "\n" + ("hope this helps " * 80)

    signals = analyze_runaway_signals(output, source_input_tokens=10_000)

    assert is_mechanical_runaway(
        signals,
        LLMTechnicalRecoveryConfig.default_enabled(),
        hard_abort=False,
    ) is True
    assert any("hope this helps" in phrase.text for phrase in signals.repeated_phrases)


def test_runaway_analysis_detects_loop_inside_long_string_value() -> None:
    repeated = "alpha beta gamma delta epsilon zeta eta theta " * 80
    payload = {
        "evidences": [
            {
                "canonical_text": repeated,
                "index_text": "distinct lookup text",
            }
        ]
    }

    signals = analyze_runaway_signals(json.dumps(payload), source_input_tokens=10_000)

    assert is_mechanical_runaway(
        signals,
        LLMTechnicalRecoveryConfig.default_enabled(),
        hard_abort=False,
    ) is True
    assert any("alpha beta gamma" in phrase.text for phrase in signals.repeated_phrases)


def test_runaway_analysis_allows_low_ratio_phrase_repeated_across_values() -> None:
    shared_phrase = "shared phrase appears in quoted context"
    payload = {
        "evidences": [
            {
                "canonical_text": (
                    " ".join(f"unique{index}_{token}" for token in range(180))
                    + f" {shared_phrase}"
                ),
                "index_text": f"lookup{index}",
            }
            for index in range(3)
        ]
    }

    signals = analyze_runaway_signals(json.dumps(payload), source_input_tokens=10_000)

    assert signals.max_repeat_count >= 3
    assert signals.max_repeat_ratio_tokens < 0.12
    assert is_mechanical_runaway(
        signals,
        LLMTechnicalRecoveryConfig.default_enabled(),
        hard_abort=False,
    ) is False
