"""Tests for the lean extraction schema and the lean->rich mapper."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from atagia.memory.extraction_mapping import lean_result_to_extraction_result
from atagia.models.schemas_memory import (
    ExtractionResult,
    LeanExtractionCandidate,
    LeanExtractionResult,
    LeanTemporalStatus,
    MemoryCategory,
    MemoryEvidencePolarity,
    MemoryEvidenceSpeakerRelation,
    MemoryEvidenceSupportKind,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
)

# Mirrors the extractor's persistence gate (TEMPORAL_CONFIDENCE_THRESHOLD).
_TEMPORAL_CONFIDENCE_THRESHOLD = 0.6


def _candidate(**overrides: object) -> LeanExtractionCandidate:
    base: dict[str, object] = {
        "canonical_text": "User prefers concise answers.",
        "kind": "evidence",
        "subject_scope": "user",
        "confidence": 0.9,
        "language_codes": ["en"],
    }
    base.update(overrides)
    return LeanExtractionCandidate(**base)


# --------------------------------------------------------------------------- #
# Lean schema validators
# --------------------------------------------------------------------------- #
def test_lean_candidate_requires_non_empty_language_codes() -> None:
    with pytest.raises(ValidationError):
        _candidate(language_codes=[])


def test_lean_candidate_rejects_only_invalid_language_codes() -> None:
    with pytest.raises(ValidationError):
        _candidate(language_codes=["zz", "??"])


def test_lean_candidate_normalizes_language_codes() -> None:
    candidate = _candidate(language_codes=["ES", "es", "en"])
    assert candidate.language_codes == ["en", "es"]


def test_lean_belief_requires_claim_key_and_value() -> None:
    with pytest.raises(ValidationError):
        _candidate(kind="belief")
    with pytest.raises(ValidationError):
        _candidate(kind="belief", claim_key="response_style.verbosity")


def test_lean_belief_accepts_claim_key_and_value() -> None:
    candidate = _candidate(
        kind="belief",
        claim_key="response_style.verbosity",
        claim_value="concise",
    )
    assert candidate.claim_key == "response_style.verbosity"
    assert candidate.claim_value == "concise"


def test_lean_temporal_status_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        LeanTemporalStatus(
            type="bounded",
            valid_from_iso="2024-01-02T00:00:00",
            valid_to_iso="2024-01-01T00:00:00",
        )


def test_lean_temporal_status_accepts_ordered_bounds() -> None:
    status = LeanTemporalStatus(
        type="bounded",
        valid_from_iso="2024-01-01T00:00:00",
        valid_to_iso="2024-01-02T00:00:00",
    )
    assert status.valid_from_iso == "2024-01-01T00:00:00"


def test_lean_result_nothing_durable_consistency() -> None:
    with pytest.raises(ValidationError):
        LeanExtractionResult(nothing_durable=True, candidates=[_candidate()])
    # Consistent: nothing_durable with no candidates is fine.
    assert LeanExtractionResult(nothing_durable=True).candidates == []


# --------------------------------------------------------------------------- #
# Mapper: one candidate per kind lands in the right bucket with exact defaults
# --------------------------------------------------------------------------- #
def test_mapper_routes_each_kind_to_its_bucket() -> None:
    lean = LeanExtractionResult(
        candidates=[
            _candidate(kind="evidence"),
            _candidate(
                kind="belief",
                claim_key="response_style.verbosity",
                claim_value="concise",
            ),
            _candidate(kind="contract_signal"),
            _candidate(kind="state_update"),
        ]
    )
    result = lean_result_to_extraction_result(lean)
    assert len(result.evidences) == 1
    assert len(result.beliefs) == 1
    assert len(result.contract_signals) == 1
    assert len(result.state_updates) == 1
    assert result.nothing_durable is False


def test_mapper_applies_server_side_defaults() -> None:
    lean = LeanExtractionResult(candidates=[_candidate(source_span="  exact  source  ")])
    evidence = lean_result_to_extraction_result(lean).evidences[0]

    assert evidence.source_kind is MemorySourceKind.EXTRACTED
    assert evidence.support_kind is MemoryEvidenceSupportKind.DIRECT
    assert evidence.evidence_polarity is MemoryEvidencePolarity.SUPPORTS
    assert evidence.speaker_relation_to_subject is MemoryEvidenceSpeakerRelation.UNKNOWN
    assert evidence.privacy_level == 0
    assert evidence.sensitivity is MemorySensitivity.UNKNOWN
    assert evidence.themes == []
    assert evidence.auto_expires is False
    assert evidence.platform_locked is False
    assert evidence.memory_category is MemoryCategory.UNKNOWN
    assert evidence.informational_mention is None
    assert evidence.subject_presence_ids == []
    # coverage_members is always emitted as the key-presence processed marker;
    # an evidence candidate with no enumerable members defaults to [].
    assert evidence.payload == {"coverage_members": []}
    assert evidence.trigger_message_ids == []
    assert evidence.trigger_quote is None
    assert evidence.support_rationale is None
    assert evidence.confidence_details == {}
    # source_span maps to source_quote (whitespace-normalized by the rich model).
    assert evidence.source_quote == "exact source"
    assert evidence.scope is MemoryScope.USER


def test_mapper_preserves_preserve_verbatim_and_support_kind() -> None:
    lean = LeanExtractionResult(
        candidates=[
            _candidate(
                preserve_verbatim=True,
                support_kind=MemoryEvidenceSupportKind.CONTEXTUAL_DIRECT,
            )
        ]
    )
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    assert evidence.preserve_verbatim is True
    assert evidence.support_kind is MemoryEvidenceSupportKind.CONTEXTUAL_DIRECT


def test_mapper_maps_belief_claim_fields() -> None:
    lean = LeanExtractionResult(
        candidates=[
            _candidate(
                kind="belief",
                claim_key="coding.language.primary",
                claim_value="python",
            )
        ]
    )
    belief = lean_result_to_extraction_result(lean).beliefs[0]
    assert belief.claim_key == "coding.language.primary"
    assert belief.claim_value == "python"


# --------------------------------------------------------------------------- #
# Mapper: temporal_confidence rule
# --------------------------------------------------------------------------- #
def test_mapper_temporal_status_with_type_and_bounds_sets_confidence_high() -> None:
    lean = LeanExtractionResult(
        candidates=[
            _candidate(
                temporal_status=LeanTemporalStatus(
                    type="bounded",
                    valid_from_iso="2024-01-01T00:00:00",
                    valid_to_iso="2024-01-02T00:00:00",
                )
            )
        ]
    )
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    assert evidence.temporal_confidence == 0.8
    assert evidence.temporal_confidence >= _TEMPORAL_CONFIDENCE_THRESHOLD
    assert evidence.temporal_type == "bounded"
    assert evidence.valid_from_iso == "2024-01-01T00:00:00"
    assert evidence.valid_to_iso == "2024-01-02T00:00:00"


def test_mapper_temporal_status_unknown_without_bounds_sets_confidence_zero() -> None:
    lean = LeanExtractionResult(
        candidates=[_candidate(temporal_status=LeanTemporalStatus(type="unknown"))]
    )
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    assert evidence.temporal_confidence == 0.0
    assert evidence.temporal_type == "unknown"
    assert evidence.valid_from_iso is None
    assert evidence.valid_to_iso is None


def test_mapper_no_temporal_status_sets_confidence_zero() -> None:
    lean = LeanExtractionResult(candidates=[_candidate()])
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    assert evidence.temporal_confidence == 0.0
    assert evidence.temporal_type == "unknown"


def test_mapper_temporal_status_unknown_type_with_bounds_keeps_bounds() -> None:
    # A bound present even with an unknown type counts as a usable status.
    lean = LeanExtractionResult(
        candidates=[
            _candidate(
                temporal_status=LeanTemporalStatus(
                    type="unknown",
                    valid_from_iso="2024-03-01T00:00:00",
                )
            )
        ]
    )
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    assert evidence.temporal_confidence == 0.8
    assert evidence.valid_from_iso == "2024-03-01T00:00:00"


def test_mapped_temporal_confidence_survives_extractor_persistence_gate() -> None:
    """The 0.8 the mapper assigns must clear the extractor's drop threshold."""

    from atagia.memory.extractor import TEMPORAL_CONFIDENCE_THRESHOLD, MemoryExtractor

    assert TEMPORAL_CONFIDENCE_THRESHOLD == _TEMPORAL_CONFIDENCE_THRESHOLD
    lean = LeanExtractionResult(
        candidates=[
            _candidate(
                temporal_status=LeanTemporalStatus(
                    type="bounded",
                    valid_from_iso="2024-01-01T00:00:00",
                    valid_to_iso="2024-01-02T00:00:00",
                )
            )
        ]
    )
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    valid_from, valid_to, temporal_type = MemoryExtractor._resolved_temporal_fields(
        MemoryExtractor.__new__(MemoryExtractor),
        evidence,
        occurred_at=None,
    )
    assert temporal_type == "bounded"
    assert valid_from == "2024-01-01T00:00:00"
    assert valid_to == "2024-01-02T00:00:00"


def test_mapped_zero_temporal_confidence_drops_bounds_in_persistence_gate() -> None:
    from atagia.memory.extractor import MemoryExtractor

    lean = LeanExtractionResult(candidates=[_candidate()])
    evidence = lean_result_to_extraction_result(lean).evidences[0]
    valid_from, valid_to, temporal_type = MemoryExtractor._resolved_temporal_fields(
        MemoryExtractor.__new__(MemoryExtractor),
        evidence,
        occurred_at=None,
    )
    assert (valid_from, valid_to, temporal_type) == (None, None, "unknown")


# --------------------------------------------------------------------------- #
# Schema size: the model-facing contract is small
# --------------------------------------------------------------------------- #
def test_lean_schema_is_small() -> None:
    serialized = json.dumps(TypeAdapter(LeanExtractionResult).json_schema())
    assert len(serialized) < 4000


def test_lean_schema_is_far_smaller_than_rich_extraction_result() -> None:
    lean_size = len(json.dumps(TypeAdapter(LeanExtractionResult).json_schema()))
    rich_size = len(json.dumps(ExtractionResult.model_json_schema()))
    # The rich schema is the bloated contract we removed from the model path.
    assert rich_size > 12000
    assert lean_size < rich_size / 3


def test_lean_schema_has_no_nullable_anyof_explosion() -> None:
    schema = TypeAdapter(LeanExtractionResult).json_schema()
    defs = schema.get("$defs", {})
    # Far fewer $defs than the 12 the rich ExtractionResult carries.
    assert len(defs) <= 4
    assert "LeanExtractionCandidate" in defs
