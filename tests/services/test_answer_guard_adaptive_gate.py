"""D10: the answer postcondition guard treats gate-skipped turns like fast mode.

A gate-skipped (or fast-mode) turn answers from contract + transcript + prepared
context with no retrieved evidence, so the guard must not place evidence
obligations on it. These unit tests exercise the explicit decision predicates
keyed on the gate status, not inferred from absent diagnostics.
"""

from __future__ import annotations

from atagia.services.answer_postcondition import (
    _retrieval_is_fast_mode_equivalent,
    _should_attempt_supported_answer_repair,
    _should_evaluate_abstention_legitimacy,
)


def _slot_fill_diagnostics(adaptive_gate: dict[str, object] | None) -> dict[str, object]:
    # A slot_fill query with genuine base-joined obligation support would
    # normally make the guard evaluate abstention legitimacy / attempt a
    # supported repair. The shapes here match the predicates in
    # answer_postcondition (_has_base_joined_obligation_support /
    # _eligible_evidence_ids).
    diagnostics: dict[str, object] = {
        "need_detection": {
            "query_type": "slot_fill",
            "exact_recall_needed": True,
        },
        "facet_support": {
            "obligations": [
                {
                    "id": "f1",
                    "description": "the slot value",
                    "status": "covered",
                    "support_verdict": "supported",
                    "selected_memory_ids": ["mem_1"],
                    "composed_memory_ids": ["mem_1"],
                }
            ],
        },
        "direct_vs_indirect_provenance": {
            "evidence": [
                {
                    "memory_id": "mem_1",
                    "proof_source": "base_canonical",
                    "selected": True,
                    "joined_to_base": True,
                }
            ],
        },
        "selected_evidence_ids": ["mem_1"],
        "selected_evidence_count": 1,
    }
    if adaptive_gate is not None:
        diagnostics["adaptive_gate"] = adaptive_gate
    return diagnostics


def test_gate_skipped_turn_is_fast_mode_equivalent() -> None:
    diagnostics = {
        "adaptive_gate": {
            "status": "skipped",
            "classification": "world",
            "skipped": True,
            "fast_mode_equivalent": True,
        }
    }
    assert _retrieval_is_fast_mode_equivalent(diagnostics) is True


def test_fast_mode_turn_is_fast_mode_equivalent() -> None:
    assert _retrieval_is_fast_mode_equivalent({"fast_mode": True}) is True


def test_retrieved_turn_is_not_fast_mode_equivalent() -> None:
    diagnostics = {
        "adaptive_gate": {
            "status": "retrieved",
            "classification": "personal",
            "skipped": False,
            "fast_mode_equivalent": False,
        }
    }
    assert _retrieval_is_fast_mode_equivalent(diagnostics) is False


def test_not_applicable_gate_is_not_fast_mode_equivalent_without_fast_flag() -> None:
    # A normal full-retrieval turn whose gate did not run (shadow) must still be
    # subject to evidence obligations.
    diagnostics = {
        "adaptive_gate": {
            "status": "shadow",
            "classification": "personal",
            "skipped": False,
            "fast_mode_equivalent": False,
        }
    }
    assert _retrieval_is_fast_mode_equivalent(diagnostics) is False


def test_gate_skipped_turn_skips_abstention_legitimacy_evaluation() -> None:
    skipped = _slot_fill_diagnostics(
        {
            "status": "skipped",
            "classification": "world",
            "skipped": True,
            "fast_mode_equivalent": True,
        }
    )
    assert (
        _should_evaluate_abstention_legitimacy(
            retrieval_sufficiency=None,
            retrieval_diagnostics=skipped,
        )
        is False
    )


def test_retrieved_turn_still_evaluates_abstention_legitimacy() -> None:
    retrieved = _slot_fill_diagnostics(
        {
            "status": "retrieved",
            "classification": "personal",
            "skipped": False,
            "fast_mode_equivalent": False,
        }
    )
    assert (
        _should_evaluate_abstention_legitimacy(
            retrieval_sufficiency=None,
            retrieval_diagnostics=retrieved,
        )
        is True
    )


def test_gate_skipped_turn_skips_supported_answer_repair() -> None:
    skipped = _slot_fill_diagnostics(
        {
            "status": "skipped",
            "classification": "conversation",
            "skipped": True,
            "fast_mode_equivalent": True,
        }
    )
    assert (
        _should_attempt_supported_answer_repair(
            retrieval_sufficiency=None,
            retrieval_diagnostics=skipped,
        )
        is False
    )
