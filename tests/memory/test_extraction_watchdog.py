"""Tests for extraction watchdog mechanical signals."""

from __future__ import annotations

import json

import pytest

from atagia.memory.extraction_watchdog import (
    ExtractionWatchdogConfig,
    ExtractionWatchdogObserver,
    ExtractionWatchdogRetry,
    analyze_repetition_signals,
    evaluate_mechanical_hard_abort,
    validate_watchdog_provider_policy,
)
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.run_counters import (
    RunCounterAccumulator,
    use_run_counter_accumulator,
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


def test_repetition_signal_detects_unclosed_content_value_runaway() -> None:
    repeated = "thank you please let me know if you need anything else "
    partial_payload = '{"evidences":[{"canonical_text":"' + repeated * 20

    signals = analyze_repetition_signals(partial_payload, source_input_tokens=30)

    assert signals.max_repeat_count >= 3
    assert signals.max_repeat_ratio_tokens > 0
    assert any("thank you please let me know" in phrase.text for phrase in signals.repeated_phrases)


def test_mechanical_hard_abort_allows_late_high_ratio_output() -> None:
    text = " ".join(f"tok{index}" for index in range(5000))
    signals = analyze_repetition_signals(text, source_input_tokens=1000)

    decision = evaluate_mechanical_hard_abort(signals)

    assert decision.allowed is True
    assert decision.policy == "allowed_mechanical_hard_runaway_growth"
    assert "late_runaway_growth_ratio" in decision.mechanical_evidence


@pytest.mark.asyncio
async def test_watchdog_observer_mechanical_hard_abort_does_not_call_llm() -> None:
    observer = _observer()
    output = " ".join(f"tok{index}" for index in range(5000))
    run_counters = RunCounterAccumulator()

    with use_run_counter_accumulator(run_counters):
        with pytest.raises(ExtractionWatchdogRetry) as exc_info:
            await observer.on_text(output, output, _request())

    assert exc_info.value.abort_policy.policy == "allowed_mechanical_hard_runaway_growth"
    assert exc_info.value.telemetry.gate_trigger == "mechanical_hard_abort"
    assert run_counters.snapshot()["labeled_counts"][
        "mechanical_runaway_abort_count"
    ] == {"layer=extraction_watchdog|mode=hard_abort": 1}


def test_watchdog_provider_policy_rejects_different_provider_without_opt_in() -> None:
    with pytest.raises(ValueError, match="different provider"):
        validate_watchdog_provider_policy(
            extractor_model="openai/gpt-5-mini",
            watchdog_model="openrouter/google/gemini-3.1-flash-lite",
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
        watchdog_model="openrouter/google/gemini-3.1-flash-lite",
        allow_different_provider=True,
    )


def _observer() -> ExtractionWatchdogObserver:
    config = ExtractionWatchdogConfig(
        enabled=True,
        allow_different_provider=False,
        bounded_retry_max_items=8,
        bounded_retry_max_output_tokens=8192,
    )
    return ExtractionWatchdogObserver(
        config=config,
        source_input_tokens=100,
    )


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[LLMMessage(role="user", content="extract")],
    )
