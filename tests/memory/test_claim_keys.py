"""Belief claim-key boundaries and literal memory preservation."""

import pytest
from pydantic import ValidationError

from atagia.core.source_references import SourceReferenceCatalog
from atagia.memory.claim_keys import validate_claim_key
from atagia.memory.extraction_cards import (
    CardResult,
    CandidateDraft,
    assemble_card_result,
    parse_belief_key_output,
    parse_belief_value_output,
)
from atagia.models.schemas_memory import LeanExtractionCandidate


@pytest.mark.parametrize(
    "key",
    [
        "workflow.debugging_style",
        "preferences.answer_format.v2",
        "status.is_active",
        "status.is_not_active",
    ],
)
def test_claim_key_accepts_canonical_syntax_without_rewriting(key: str) -> None:
    assert validate_claim_key(key) == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        " Current.status",
        "Current.status",
        "current..status",
        "current.status.",
        "current-status",
        "current.status!",
        "2current.status",
        "status.és_actiu",
        "status.активен",
    ],
)
def test_claim_key_rejects_noncanonical_syntax_without_transliteration(key: str) -> None:
    with pytest.raises(ValueError, match="claim_key"):
        validate_claim_key(key)


def test_belief_cards_preserve_distinct_polarities_and_literal_values() -> None:
    assert parse_belief_key_output("status.is_active") == "status.is_active"
    assert parse_belief_key_output("status.is_not_active") == "status.is_not_active"
    assert parse_belief_value_output("sí") == "sí"
    assert parse_belief_value_output("  No le gusta el Café  Sol  ") == "No le gusta el Café  Sol"


def test_belief_card_rejects_invalid_key_instead_of_repairing_it() -> None:
    with pytest.raises(ValueError, match="claim_key"):
        parse_belief_key_output("no_és_actiu")


@pytest.mark.parametrize("output", ["", "none", "null", "n/a", "sí\nno"])
def test_belief_card_rejects_missing_or_multiple_values(output: str) -> None:
    with pytest.raises(ValueError, match="claim_value"):
        parse_belief_value_output(output)


@pytest.mark.parametrize(
    "value",
    [
        "Candela prefiere revisar los cambios",
        "candles help with concentration",
        "cand_001 sí",
    ],
)
def test_belief_value_keeps_free_text_without_candidate_id_normalization(value: str) -> None:
    assert parse_belief_value_output(value) == value


def test_multilingual_belief_keeps_source_text_and_quote() -> None:
    source = "Prefiero que me respondas en catalán: «demà al matí»."
    catalog = SourceReferenceCatalog(source)
    quote = "demà al matí"
    quote_start = source.index(quote)
    quote_end = quote_start + len(quote)
    anchors = [
        anchor for anchor in catalog.anchors
        if anchor.char_end > quote_start and anchor.char_start < quote_end
    ]
    result, repairs = assemble_card_result(
        (CandidateDraft("cand_001", source),),
        [
            CardResult("memory_kind", "", {"cand_001": "belief"}),
            CardResult("memory_scope", "", {"cand_001": "user"}),
            CardResult("memory_confidence", "", {"cand_001": 0.9}),
            CardResult(
                "evidence",
                "",
                {"cand_001": {
                    "start_ref": anchors[0].reference_id,
                    "end_ref": anchors[-1].reference_id,
                    "language_codes": ("es", "ca"),
                    "support_kind": "direct",
                    "preserve_verbatim": False,
                }},
            ),
            CardResult("temporal", "", {"cand_001": None}),
            CardResult("coverage_members", "", {"cand_001": []}),
            CardResult("belief", "", {"cand_001": {"claim_key": "reply.language.preference", "claim_value": "catalán"}}),
        ],
        source_catalog=catalog,
    )
    assert repairs == []
    candidate = result.candidates[0]
    assert candidate.claim_key == "reply.language.preference"
    assert candidate.claim_value == "catalán"
    assert candidate.canonical_text == source
    assert candidate.source_span == "demà al matí"
    assert candidate.language_codes == ["ca", "es"]


def test_belief_without_claim_fields_is_not_reclassified_as_evidence() -> None:
    source = "No quiero cambios automáticos."
    catalog = SourceReferenceCatalog(source)
    with pytest.raises(ValueError, match="belief requires claim_key and claim_value"):
        assemble_card_result(
            (CandidateDraft("cand_001", source),),
            [
                CardResult("memory_kind", "", {"cand_001": "belief"}),
                CardResult("memory_scope", "", {"cand_001": "user"}),
                CardResult("memory_confidence", "", {"cand_001": 0.9}),
                CardResult("evidence", "", {"cand_001": {
                    "start_ref": catalog.anchors[0].reference_id,
                    "end_ref": catalog.anchors[-1].reference_id,
                    "language_codes": ("es",),
                    "support_kind": "direct",
                    "preserve_verbatim": False,
                }}),
                CardResult("temporal", "", {"cand_001": None}),
                CardResult("coverage_members", "", {"cand_001": []}),
                CardResult("belief", "", {}),
            ],
            source_catalog=catalog,
        )


def test_structured_boundary_rejects_noncanonical_belief_key() -> None:
    with pytest.raises(ValidationError, match="claim_key"):
        LeanExtractionCandidate.model_validate(
            {
                "canonical_text": "El usuario prefiere respuestas breves.",
                "kind": "belief",
                "subject_scope": "user",
                "confidence": 0.9,
                "language_codes": ["es"],
                "claim_key": "estilo.respuéstas_breves",
                "claim_value": "breve",
            }
        )
