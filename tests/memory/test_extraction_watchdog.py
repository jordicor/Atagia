"""Tests for extraction watchdog mechanical signals."""

from __future__ import annotations

import json

import pytest

from atagia.memory.extraction_watchdog import (
    analyze_repetition_signals,
    validate_watchdog_provider_policy,
)


def test_repetition_signal_ignores_json_keys() -> None:
    payload = {
        "evidences": [
            {
                "canonical_text": f"unique durable memory content {index}",
                "index_text": f"lookup surface {index}",
                "scope": "conversation",
                "source_kind": "extracted",
            }
            for index in range(12)
        ]
    }

    signals = analyze_repetition_signals(json.dumps(payload), source_input_tokens=100)

    assert signals.max_repeat_count == 0
    assert all("canonical_text" not in phrase.text for phrase in signals.repeated_phrases)


def test_repetition_signal_detects_repeated_content_values() -> None:
    repeated = "alpha beta gamma delta epsilon zeta eta theta"
    payload = {
        "evidences": [
            {
                "canonical_text": repeated,
                "index_text": f"{repeated} lookup {index}",
                "scope": "conversation",
                "source_kind": "extracted",
            }
            for index in range(4)
        ]
    }

    signals = analyze_repetition_signals(json.dumps(payload), source_input_tokens=20)

    assert signals.max_repeat_count >= 4
    assert signals.max_repeat_ratio_tokens > 0
    assert any("alpha beta gamma delta epsilon" in phrase.text for phrase in signals.repeated_phrases)


def test_watchdog_provider_policy_rejects_different_provider_without_opt_in() -> None:
    with pytest.raises(ValueError, match="different provider"):
        validate_watchdog_provider_policy(
            extractor_model="openai/gpt-5-mini",
            watchdog_model="openrouter/google/gemini-3.1-flash-lite-preview",
            allow_different_provider=False,
        )


def test_watchdog_provider_policy_allows_matching_provider() -> None:
    validate_watchdog_provider_policy(
        extractor_model="openai/gpt-5-mini",
        watchdog_model="openai/gpt-5-nano",
        allow_different_provider=False,
    )


def test_watchdog_provider_policy_allows_different_provider_with_opt_in() -> None:
    validate_watchdog_provider_policy(
        extractor_model="openai/gpt-5-mini",
        watchdog_model="openrouter/google/gemini-3.1-flash-lite-preview",
        allow_different_provider=True,
    )
