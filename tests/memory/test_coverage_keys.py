"""Tests for the shared coverage member-key resolver (single source of truth)."""

from __future__ import annotations

from typing import Any

from atagia.memory.carrier_dedupe import collapse_duplicate_carriers
from atagia.memory.context_composer import ContextComposer
from atagia.memory.coverage_keys import (
    normalize_coverage_key,
    resolve_member_keys,
)
from atagia.models.schemas_memory import ScoredCandidate


def _scored_candidate(payload_json: Any) -> ScoredCandidate:
    return ScoredCandidate(
        memory_id="mem_1",
        memory_object={
            "id": "mem_1",
            "object_type": "evidence",
            "source_kind": "extracted",
            "canonical_text": "PERSON_A owns a pottery workshop.",
            "payload_json": payload_json,
        },
        llm_applicability=0.5,
        retrieval_score=0.5,
        vitality_boost=0.0,
        confirmation_boost=0.0,
        need_boost=0.0,
        penalty=0.0,
        final_score=0.5,
    )


class TestNormalizeCoverageKey:
    def test_casefold_and_whitespace_collapse(self) -> None:
        assert normalize_coverage_key("  Dance   Studio ") == "dance studio"
        assert normalize_coverage_key("POTTERY") == "pottery"


class TestResolveMemberKeys:
    def test_modern_coverage_members(self) -> None:
        payload = {
            "coverage_members": [
                {"member_key": "Dance Studio", "display_text": "dance studio"},
                {"member_key": "pottery", "display_text": "pottery"},
            ]
        }
        assert resolve_member_keys(payload) == frozenset({"dance studio", "pottery"})

    def test_present_but_empty_members_means_no_keys(self) -> None:
        """Processed-with-no-members must NOT fall through to the legacy ladder."""
        payload = {"coverage_members": [], "value_norm_key": "pottery"}
        assert resolve_member_keys(payload) == frozenset()

    def test_legacy_value_ladder(self) -> None:
        assert resolve_member_keys({"value_norm_key": "Pottery Workshop"}) == (
            frozenset({"pottery workshop"})
        )
        assert resolve_member_keys({"value_text": "pottery"}) == frozenset({"pottery"})
        assert resolve_member_keys({"subject_surface": "PERSON_A"}) == (
            frozenset({"person_a"})
        )

    def test_ladder_order_prefers_value_norm_key(self) -> None:
        payload = {"value_norm_key": "alpha", "value_text": "beta"}
        assert resolve_member_keys(payload) == frozenset({"alpha"})

    def test_empty_and_non_dict_payloads(self) -> None:
        assert resolve_member_keys({}) == frozenset()
        assert resolve_member_keys(None) == frozenset()
        assert resolve_member_keys("not a dict") == frozenset()
        assert resolve_member_keys({"coverage_members": "corrupt"}) == frozenset()

    def test_blank_values_are_unkeyed(self) -> None:
        assert resolve_member_keys({"value_norm_key": "   "}) == frozenset()
        assert resolve_member_keys(
            {"coverage_members": [{"member_key": "  ", "display_text": "x"}]}
        ) == frozenset()


class TestComposerDedupeAgreement:
    """The composer and the carrier dedupe must resolve identical member keys.

    This is the MEDIUM-1 review case: a legacy row carrying only a
    ``value_norm_key`` used to be UNKEYED for the dedupe while the composer
    resolved a member — a collapse could shrink the composer's member
    universe (latent B3 bypass on mixed-vintage data).
    """

    def test_legacy_payload_resolves_identically(self) -> None:
        legacy_payload = {"value_norm_key": "Dance Studio"}
        composer_keys = ContextComposer._coverage_member_keys(
            _scored_candidate(legacy_payload)
        )
        shared_keys = resolve_member_keys(legacy_payload)
        assert composer_keys == shared_keys == frozenset({"dance studio"})

    def test_modern_payload_resolves_identically(self) -> None:
        modern_payload = {
            "coverage_members": [{"member_key": "pottery", "display_text": "pottery"}]
        }
        composer_keys = ContextComposer._coverage_member_keys(
            _scored_candidate(modern_payload)
        )
        assert composer_keys == resolve_member_keys(modern_payload) == (
            frozenset({"pottery"})
        )

    def test_legacy_member_blocks_collapse_into_foreign_representative(self) -> None:
        """A legacy carrier whose ladder-resolved member is not carried by the
        representative must be released, exactly as a modern carrier would be."""
        candidates = [
            {
                "id": "mem_rep",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.9,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "coverage_members": [
                        {"member_key": "workshop", "display_text": "workshop"}
                    ],
                },
            },
            {
                # Legacy row: identical text + shared span (R-TEXT join) but a
                # ladder-resolved member key the representative does not carry.
                "id": "mem_legacy",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.4,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "value_norm_key": "pottery workshop",
                },
            },
        ]
        result = collapse_duplicate_carriers(candidates)
        assert [str(c["id"]) for c in result.candidates] == ["mem_rep", "mem_legacy"]
        assert result.collapsed_into == {}

    def test_legacy_member_collapses_when_representative_carries_it(self) -> None:
        """Collapse is allowed only when the representative carries BOTH the
        ladder-resolved member and the same structured value identity (the
        value-key guard treats a value-bearing carrier vs a value-less
        representative as a mismatch — conservative under-collapse)."""
        candidates = [
            {
                "id": "mem_rep",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.9,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "value_norm_key": "Pottery Workshop",
                    "coverage_members": [
                        {"member_key": "Pottery Workshop", "display_text": "x"}
                    ],
                },
            },
            {
                "id": "mem_legacy",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.4,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "value_norm_key": "pottery workshop",
                },
            },
        ]
        result = collapse_duplicate_carriers(candidates)
        assert [str(c["id"]) for c in result.candidates] == ["mem_rep"]
        assert result.collapsed_into == {"mem_legacy": "mem_rep"}

    def test_value_bearing_legacy_row_never_collapses_into_valueless_rep(
        self,
    ) -> None:
        """The value-key guard releases a value-bearing carrier when the
        representative has no structured value identity at all."""
        candidates = [
            {
                "id": "mem_rep",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.9,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "coverage_members": [
                        {"member_key": "pottery workshop", "display_text": "x"}
                    ],
                },
            },
            {
                "id": "mem_legacy",
                "object_type": "evidence",
                "source_kind": "extracted",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "rrf_score": 0.4,
                "payload_json": {
                    "source_message_ids": ["msg_1"],
                    "value_norm_key": "pottery workshop",
                },
            },
        ]
        result = collapse_duplicate_carriers(candidates)
        assert [str(c["id"]) for c in result.candidates] == ["mem_rep", "mem_legacy"]
        assert result.collapsed_into == {}
