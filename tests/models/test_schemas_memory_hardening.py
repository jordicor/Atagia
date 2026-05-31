"""Structured-output normalization hardening tests."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from atagia.memory.belief_reviser import RevisionDecision
from atagia.models.schemas_memory import (
    ContractProjectionResult,
    ExactFacet,
    ExtractionResult,
    MemoryStatus,
    MemoryScope,
    PlannedSubQuery,
    QueryIntelligenceResult,
    QueryPlanCore,
    RetrievalPlan,
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


def test_extraction_result_drops_invalid_content_language_codes_before_persistence() -> None:
    result = ExtractionResult.model_validate(
        {
            "evidences": [
                {
                    "canonical_text": "Parlem de memoria en catala.",
                    "scope": MemoryScope.CONVERSATION.value,
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "language_codes": ["jp", "CA", "zz"],
                }
            ],
            "nothing_durable": False,
        }
    )

    assert result.evidences[0].language_codes == ["ca"]

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(
            {
                "evidences": [
                    {
                        "canonical_text": "This language metadata is invalid.",
                        "scope": MemoryScope.CONVERSATION.value,
                        "confidence": 0.9,
                        "source_kind": "extracted",
                        "privacy_level": 0,
                        "language_codes": ["jp", "zz"],
                    }
                ],
                "nothing_durable": False,
            }
        )


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


def test_query_plan_core_schema_is_much_smaller_than_rich_result() -> None:
    core_schema = json.dumps(TypeAdapter(QueryPlanCore).json_schema())
    rich_schema = json.dumps(TypeAdapter(QueryIntelligenceResult).json_schema())

    # The lean schema must stay well under the cap and a wide margin below the
    # rich schema, because the medium-model JSON failures came from the rich
    # contract (structured anchors + cross-field validators).
    assert len(core_schema) < 3500
    assert len(core_schema) < len(rich_schema) - 1500


def test_query_plan_core_enforces_per_field_validation() -> None:
    # Empty sub_queries is a shape error and must still raise.
    with pytest.raises(ValidationError):
        QueryPlanCore.model_validate({"sub_queries": []})

    # Duplicate exact_facets is a shape error and must still raise.
    with pytest.raises(ValidationError):
        QueryPlanCore.model_validate(
            {
                "sub_queries": ["x"],
                "exact_facets": [ExactFacet.DATE.value, ExactFacet.DATE.value],
            }
        )

    # Out-of-range retrieval level is a shape error and must still raise.
    with pytest.raises(ValidationError):
        QueryPlanCore.model_validate(
            {"sub_queries": ["x"], "retrieval_levels": [9]}
        )


def test_query_plan_core_does_not_raise_on_broken_hint_linkage() -> None:
    # The rich result raises when a hint references a non-existent sub_query;
    # the lean core must accept it so server-side repair can re-link it.
    core = QueryPlanCore.model_validate(
        {
            "query_type": "slot_fill",
            "sub_queries": ["real sub query"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "does not exist",
                    "fts_phrase": "foo bar",
                    "must_keep_terms": ["foo"],
                }
            ],
            "exact_recall_needed": True,
            "exact_facets": [ExactFacet.DATE.value],
        }
    )

    assert core.sparse_query_hints[0].sub_query_text == "does not exist"
    assert core.exact_recall_needed is True


def test_query_plan_core_normalizes_optional_language_codes() -> None:
    core = QueryPlanCore.model_validate(
        {
            "sub_queries": ["que idioma uso"],
            "query_language": " ES ",
            "answer_language": "CA",
        }
    )

    assert core.query_language == "es"
    assert core.answer_language == "ca"


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


def test_query_intelligence_normalizes_and_drops_invalid_optional_language_guidance() -> None:
    result = QueryIntelligenceResult.model_validate(
        {
            "needs": [],
            "sub_queries": ["Que idioma uso?"],
            "query_language": " ES ",
            "answer_language": "CA",
        }
    )

    assert result.query_language == "es"
    assert result.answer_language == "ca"

    invalid = QueryIntelligenceResult.model_validate(
        {
            "needs": [],
            "sub_queries": ["Que idioma uso?"],
            "query_language": ["jp"],
            "answer_language": {"language": "spanish"},
        }
    )
    assert invalid.query_language is None
    assert invalid.answer_language is None


def test_retrieval_plan_normalizes_and_drops_invalid_optional_language_guidance() -> None:
    plan = RetrievalPlan(
        original_query="Que idioma uso?",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(text="Que idioma uso?", fts_queries=["idioma"])
        ],
        query_language=" ES ",
        answer_language="CA",
        scope_filter=[MemoryScope.CHARACTER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
    )

    assert plan.query_language == "es"
    assert plan.answer_language == "ca"

    invalid = RetrievalPlan(
        original_query="Que idioma uso?",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(text="Que idioma uso?", fts_queries=["idioma"])
        ],
        query_language=["zz"],
        answer_language={"language": "spanish"},
        scope_filter=[MemoryScope.CHARACTER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
    )
    assert invalid.query_language is None
    assert invalid.answer_language is None


def test_revision_decision_requires_successor_text_for_successor_actions() -> None:
    with pytest.raises(ValidationError):
        RevisionDecision.model_validate(
            {
                "action": "SUPERSEDE",
                "explanation": "The new evidence replaces the old belief.",
                "successor_canonical_text": None,
            }
        )
