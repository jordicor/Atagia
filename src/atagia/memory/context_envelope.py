"""Token budget allocation for answer-time context envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any, Mapping


CONTEXT_ENVELOPE_DEFAULT_RATIOS: dict[str, float] = {
    "instructions": 0.10,
    "current_turn": 0.03,
    "retrieved_context": 0.67,
    "recent_transcript": 0.20,
}

_CONTEXT_ENVELOPE_KEYS = tuple(CONTEXT_ENVELOPE_DEFAULT_RATIOS)


@dataclass(frozen=True, slots=True)
class ContextEnvelopeBudget:
    """Concrete token budgets derived from one global answer-input budget."""

    total_budget_tokens: int
    instructions_budget_tokens: int
    current_turn_budget_tokens: int
    retrieved_context_budget_tokens: int
    recent_transcript_budget_tokens: int
    ratios: dict[str, float]

    def model_dump(self) -> dict[str, Any]:
        return {
            "total_budget_tokens": self.total_budget_tokens,
            "instructions_budget_tokens": self.instructions_budget_tokens,
            "current_turn_budget_tokens": self.current_turn_budget_tokens,
            "retrieved_context_budget_tokens": self.retrieved_context_budget_tokens,
            "recent_transcript_budget_tokens": self.recent_transcript_budget_tokens,
            "ratios": dict(self.ratios),
        }


def allocate_context_envelope_budget(
    total_budget_tokens: int,
    ratios: Mapping[str, float] | None = None,
) -> ContextEnvelopeBudget:
    """Allocate one global prompt budget across context sections.

    The full budget is assigned across named sections. This does not force the
    prompt builder to invent or include low-value context; it only prevents a
    silent reserve from hiding available budget from useful context.
    """

    normalized_total = int(total_budget_tokens)
    if normalized_total <= 0:
        raise ValueError("total_budget_tokens must be positive")

    normalized_ratios = _normalize_context_envelope_ratios(ratios)
    raw_budgets = {
        key: normalized_total * normalized_ratios[key] for key in _CONTEXT_ENVELOPE_KEYS
    }
    budgets = {key: floor(value) for key, value in raw_budgets.items()}
    remainder = normalized_total - sum(budgets.values())
    fractional_order = sorted(
        _CONTEXT_ENVELOPE_KEYS,
        key=lambda key: (raw_budgets[key] - budgets[key], key),
        reverse=True,
    )
    for key in fractional_order[:remainder]:
        budgets[key] += 1

    retrieved_context_budget = budgets["retrieved_context"]
    if retrieved_context_budget <= 0:
        raise ValueError("retrieved_context ratio leaves no prompt budget")
    return ContextEnvelopeBudget(
        total_budget_tokens=normalized_total,
        instructions_budget_tokens=budgets["instructions"],
        current_turn_budget_tokens=budgets["current_turn"],
        retrieved_context_budget_tokens=retrieved_context_budget,
        recent_transcript_budget_tokens=budgets["recent_transcript"],
        ratios=normalized_ratios,
    )


def _normalize_context_envelope_ratios(
    ratios: Mapping[str, float] | None,
) -> dict[str, float]:
    supplied = dict(ratios or {})
    unknown = sorted(set(supplied) - set(_CONTEXT_ENVELOPE_KEYS))
    if unknown:
        raise ValueError("Unknown context envelope ratio keys: " + ", ".join(unknown))
    merged = dict(CONTEXT_ENVELOPE_DEFAULT_RATIOS)
    merged.update({key: float(value) for key, value in supplied.items()})
    if any(value < 0.0 for value in merged.values()):
        raise ValueError("Context envelope ratios must be non-negative")
    ratio_total = sum(merged.values())
    if ratio_total <= 0.0:
        raise ValueError("At least one context envelope ratio must be positive")
    return {key: value / ratio_total for key, value in merged.items()}
