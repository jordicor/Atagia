"""Tests for answer-time context envelope budget allocation."""

from __future__ import annotations

import pytest

from atagia.memory.context_envelope import allocate_context_envelope_budget


def test_context_envelope_allocates_full_budget_with_default_ratios() -> None:
    budget = allocate_context_envelope_budget(32_768)

    assert (
        budget.instructions_budget_tokens
        + budget.current_turn_budget_tokens
        + budget.retrieved_context_budget_tokens
        + budget.recent_transcript_budget_tokens
        == 32_768
    )
    assert budget.retrieved_context_budget_tokens == 21_954
    assert budget.recent_transcript_budget_tokens == 6_554


def test_context_envelope_normalizes_custom_ratios() -> None:
    budget = allocate_context_envelope_budget(
        10_000,
        {
            "retrieved_context": 3.0,
            "recent_transcript": 1.0,
            "instructions": 1.0,
            "current_turn": 0.0,
        },
    )

    assert budget.retrieved_context_budget_tokens == 6_000
    assert budget.recent_transcript_budget_tokens == 2_000
    assert budget.instructions_budget_tokens == 2_000
    assert budget.current_turn_budget_tokens == 0


def test_context_envelope_rejects_unknown_ratio_keys() -> None:
    with pytest.raises(ValueError, match="Unknown context envelope ratio keys"):
        allocate_context_envelope_budget(10_000, {"junk": 1.0})
