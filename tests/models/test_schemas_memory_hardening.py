"""Structured-output normalization hardening tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atagia.memory.belief_reviser import RevisionDecision
from atagia.models.schemas_memory import (
    ContractProjectionResult,
    ExactFacet,
    ExtractionResult,
    MemoryScope,
    QueryIntelligenceResult,
)


def test_contract_projection_normalizes_non_empty_signals_to_not_durable_false() -> None:
    result = ContractProjectionResult.model_validate(
        {
            "signals": [
                {
                    "canonical_text": "The user prefers concise answers.",
                    "dimension_name": "directness",
                    "value_json": {"label": "concise"},
                    "confidence": 0.8,
                    "scope": MemoryScope.CONVERSATION.value,
                    "privacy_level": 1,
                }
            ],
            "nothing_durable": True,
        }
    )

    assert result.nothing_durable is False
    assert len(result.signals) == 1


def test_extraction_result_normalizes_non_empty_lists_to_not_durable_false() -> None:
    result = ExtractionResult.model_validate(
        {
            "evidences": [
                {
                    "canonical_text": "The user is testing bulk ingest.",
                    "scope": MemoryScope.CONVERSATION.value,
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "language_codes": ["en"],
                }
            ],
            "nothing_durable": True,
        }
    )

    assert result.nothing_durable is False
    assert len(result.evidences) == 1


def test_extraction_defaults_temporal_confidence_when_temporal_fields_are_explicit() -> None:
    result = ExtractionResult.model_validate(
        {
            "evidences": [
                {
                    "canonical_text": "The user is available until Friday.",
                    "scope": MemoryScope.CONVERSATION.value,
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "language_codes": ["en"],
                    "temporal_type": "bounded",
                    "valid_to_iso": "2026-04-24T17:00:00+00:00",
                }
            ],
            "nothing_durable": False,
        }
    )

    assert result.evidences[0].temporal_confidence == pytest.approx(0.6)


def test_query_intelligence_exact_facets_enable_exact_recall() -> None:
    result = QueryIntelligenceResult.model_validate(
        {
            "needs": [],
            "sub_queries": ["When did I meet Alex?"],
            "exact_recall_needed": False,
            "exact_facets": [ExactFacet.DATE.value, ExactFacet.PERSON_NAME.value],
        }
    )

    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.DATE, ExactFacet.PERSON_NAME]


def test_revision_decision_requires_successor_text_for_successor_actions() -> None:
    with pytest.raises(ValidationError):
        RevisionDecision.model_validate(
            {
                "action": "SUPERSEDE",
                "explanation": "The new evidence replaces the old belief.",
                "successor_canonical_text": None,
            }
        )
