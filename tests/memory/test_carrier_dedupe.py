"""Tests for the mechanical duplicate-carrier collapse (CS-2.3)."""

from __future__ import annotations

from typing import Any

from atagia.memory.carrier_dedupe import collapse_duplicate_carriers


def _candidate(
    candidate_id: str,
    *,
    canonical_text: str = "",
    rrf_score: float = 0.5,
    object_type: str = "evidence",
    source_kind: str = "extracted",
    source_message_ids: list[str] | None = None,
    source_object_ids: list[str] | None = None,
    coverage_members: list[dict[str, str]] | None = None,
    value_norm_key: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if source_message_ids is not None:
        payload["source_message_ids"] = source_message_ids
    if source_object_ids is not None:
        payload["source_object_ids"] = source_object_ids
    if coverage_members is not None:
        payload["coverage_members"] = coverage_members
    if value_norm_key is not None:
        payload["value_norm_key"] = value_norm_key
    return {
        "id": candidate_id,
        "canonical_text": canonical_text,
        "object_type": object_type,
        "source_kind": source_kind,
        "rrf_score": rrf_score,
        "payload_json": payload,
        **extra,
    }


def _ids(candidates: list[dict[str, Any]]) -> list[str]:
    return [str(candidate["id"]) for candidate in candidates]


class TestContentTokenJoin:
    def test_identical_text_with_shared_span_collapses(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_1", "msg_2"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a"]
        assert result.collapsed_into == {"mem_b": "mem_a"}
        assert result.candidates[0]["deduped_carrier_ids"] == ["mem_b"]

    def test_token_normalization_ignores_case_and_punctuation(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="person_a  owns a pottery workshop",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a"]

    def test_identical_text_disjoint_spans_stays_separate(self) -> None:
        """Repeated statements in different messages are distinct events."""
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                source_message_ids=["msg_2"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]
        assert result.collapsed_into == {}

    def test_different_text_shared_span_stays_separate(self) -> None:
        """Two distinct facts extracted from one message never merge."""
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A lost a banking job.",
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A is opening a pottery workshop.",
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]
        assert result.collapsed_into == {}

    def test_chained_span_overlap_is_transitive(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.3,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1", "msg_2"],
            ),
            _candidate(
                "mem_c",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.5,
                source_message_ids=["msg_2"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        # mem_b wins on rrf and takes the group's earliest position.
        assert _ids(result.candidates) == ["mem_b"]
        assert result.collapsed_into == {"mem_a": "mem_b", "mem_c": "mem_b"}
        assert result.candidates[0]["deduped_carrier_ids"] == ["mem_a", "mem_c"]

    def test_empty_text_never_joins(self) -> None:
        candidates = [
            _candidate("mem_a", canonical_text="", source_message_ids=["msg_1"]),
            _candidate("mem_b", canonical_text="", source_message_ids=["msg_1"]),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]


class TestMemberKeyJoin:
    def test_equal_member_sets_collapse_without_span_overlap(self) -> None:
        """Under exhaustive coverage, the engine-minted member key is shared
        provenance across sessions."""
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="The assistant is opening a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
                coverage_members=[
                    {"member_key": "pottery workshop", "display_text": "pottery workshop"}
                ],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.8,
                source_message_ids=["msg_2"],
                coverage_members=[
                    {"member_key": "Pottery Workshop", "display_text": "pottery workshop"}
                ],
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["mem_b"]
        assert result.collapsed_into == {"mem_a": "mem_b"}

    def test_subset_member_sets_stay_separate(self) -> None:
        """A carrier enumerating extra members is NOT a duplicate (B3)."""
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A plays chess.",
                source_message_ids=["msg_1"],
                coverage_members=[{"member_key": "chess", "display_text": "chess"}],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A plays chess and go.",
                source_message_ids=["msg_2"],
                coverage_members=[
                    {"member_key": "chess", "display_text": "chess"},
                    {"member_key": "go", "display_text": "go"},
                ],
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]
        assert result.collapsed_into == {}

    def test_empty_member_lists_never_join(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A likes tea.",
                source_message_ids=["msg_1"],
                coverage_members=[],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A likes coffee.",
                source_message_ids=["msg_2"],
                coverage_members=[],
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]

    def test_multi_member_equal_sets_collapse(self) -> None:
        members = [
            {"member_key": "alpha", "display_text": "alpha"},
            {"member_key": "beta", "display_text": "beta"},
        ]
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A uses alpha and beta.",
                rrf_score=0.7,
                source_message_ids=["msg_1"],
                coverage_members=list(members),
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_B recommended alpha and beta to PERSON_A.",
                rrf_score=0.5,
                source_message_ids=["msg_2"],
                coverage_members=list(reversed(members)),
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["mem_a"]
        assert result.collapsed_into == {"mem_b": "mem_a"}


    def test_member_join_is_off_outside_exhaustive_coverage(self) -> None:
        """Same member key alone is NOT enough provenance outside exhaustive
        known-set coverage: carriers may hold distinct secondary facts."""
        members = [{"member_key": "toby", "display_text": "Toby"}]
        candidates = [
            _candidate(
                "mem_plain",
                canonical_text="PERSON_A has a pet named Toby.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
                coverage_members=list(members),
            ),
            _candidate(
                "mem_dated",
                canonical_text="PERSON_A adopted Toby last month.",
                rrf_score=0.4,
                source_message_ids=["msg_2"],
                coverage_members=list(members),
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_plain", "mem_dated"]
        assert result.collapsed_into == {}


class TestValueNormKeyJoin:
    def test_equal_value_key_with_shared_span_collapses(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A's favorite color is blue.",
                rrf_score=0.6,
                source_message_ids=["msg_1"],
                value_norm_key="blue",
            ),
            _candidate(
                "mem_b",
                canonical_text="Favorite color of PERSON_A: blue.",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
                value_norm_key="Blue",
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a"]
        assert result.collapsed_into == {"mem_b": "mem_a"}

    def test_equal_value_key_disjoint_spans_stays_separate(self) -> None:
        """Same value about possibly different subjects must not merge."""
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A's favorite color is blue.",
                source_message_ids=["msg_1"],
                value_norm_key="blue",
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_B's favorite color is blue.",
                source_message_ids=["msg_2"],
                value_norm_key="blue",
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]


class TestGuards:
    def test_cross_class_carriers_never_join(self) -> None:
        """A summary can never absorb a source-backed evidence row."""
        members = [{"member_key": "pottery workshop", "display_text": "x"}]
        candidates = [
            _candidate(
                "sum_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                object_type="summary_view",
                source_kind="summarized",
                source_message_ids=["msg_1"],
                coverage_members=list(members),
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
                coverage_members=list(members),
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["sum_a", "mem_b"]
        assert result.collapsed_into == {}

    def test_channel_kind_flags_split_classes(self) -> None:
        candidates = [
            _candidate(
                "win_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                source_kind="verbatim",
                source_message_ids=["msg_1"],
                is_verbatim_evidence_window=True,
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                source_kind="verbatim",
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["win_a", "mem_b"]

    def test_representative_must_carry_collapsed_member_keys(self) -> None:
        """Cross-rule bridging may not take a member key out of the pool.

        mem_a and mem_b join by identical text + shared span; mem_b and mem_c
        join by equal member sets. mem_a wins on rrf but carries no member
        keys, so mem_b/mem_c (member carriers) are released, not collapsed.
        """
        members = [{"member_key": "pottery workshop", "display_text": "x"}]
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.5,
                source_message_ids=["msg_1"],
                coverage_members=list(members),
            ),
            _candidate(
                "mem_c",
                canonical_text="The assistant runs a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_9"],
                coverage_members=list(members),
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert set(_ids(result.candidates)) == {"mem_a", "mem_b", "mem_c"}
        assert result.collapsed_into == {}

    def test_value_key_conflict_releases_carrier(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="Budget noted: exact figure recorded.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
                value_norm_key="2800",
            ),
            _candidate(
                "mem_b",
                canonical_text="Budget noted: exact figure recorded.",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
                value_norm_key="3100",
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        assert _ids(result.candidates) == ["mem_a", "mem_b"]
        assert result.collapsed_into == {}


class TestMechanics:
    def test_empty_and_single_inputs_pass_through(self) -> None:
        assert collapse_duplicate_carriers([]).candidates == []
        single = [_candidate("mem_a", canonical_text="x", source_message_ids=["m1"])]
        result = collapse_duplicate_carriers(single)
        assert _ids(result.candidates) == ["mem_a"]
        assert result.collapsed_into == {}

    def test_input_candidates_are_not_mutated(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_1"],
            ),
        ]
        collapse_duplicate_carriers(candidates)
        assert "deduped_carrier_ids" not in candidates[0]
        assert "deduped_carrier_ids" not in candidates[1]

    def test_representative_takes_group_earliest_position(self) -> None:
        candidates = [
            _candidate("mem_x", canonical_text="Unrelated fact one.", rrf_score=0.9),
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.2,
                source_message_ids=["msg_1"],
            ),
            _candidate("mem_y", canonical_text="Unrelated fact two.", rrf_score=0.8),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.7,
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        # mem_b (best rrf in its group) is emitted at mem_a's earlier slot.
        assert _ids(result.candidates) == ["mem_x", "mem_b", "mem_y"]

    def test_deterministic_tie_break_prefers_earlier_position(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.5,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.5,
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_a"]
        assert result.collapsed_into == {"mem_b": "mem_a"}

    def test_missing_rrf_score_treated_as_zero(self) -> None:
        candidates = [
            {
                "id": "mem_a",
                "canonical_text": "PERSON_A owns a pottery workshop.",
                "object_type": "evidence",
                "source_kind": "extracted",
                "payload_json": {"source_message_ids": ["msg_1"]},
            },
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.3,
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert _ids(result.candidates) == ["mem_b"]

    def test_collapsed_count_property(self) -> None:
        candidates = [
            _candidate(
                "mem_a",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
            ),
            _candidate(
                "mem_b",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.1,
                source_message_ids=["msg_1"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        assert result.collapsed_count == 1
        assert result.collapsed_ids_by_representative == {"mem_a": ["mem_b"]}


class TestSpanUnion:
    def test_representative_gains_collapsed_source_spans(self) -> None:
        """Quote funding must still reach every utterance of the fact."""
        members = [{"member_key": "toby", "display_text": "Toby"}]
        candidates = [
            _candidate(
                "mem_rep",
                canonical_text="PERSON_A has a pet named Toby.",
                rrf_score=0.9,
                source_message_ids=["msg_plain"],
                coverage_members=list(members),
            ),
            _candidate(
                "mem_dated",
                canonical_text="PERSON_A has a puppy named Toby.",
                rrf_score=0.4,
                source_message_ids=["msg_with_date"],
                coverage_members=list(members),
            ),
        ]
        result = collapse_duplicate_carriers(candidates, member_key_collapse=True)
        representative = result.candidates[0]
        assert representative["id"] == "mem_rep"
        assert representative["payload_json"]["source_message_ids"] == [
            "msg_plain",
            "msg_with_date",
        ]
        # The input dict and its payload are never mutated.
        assert candidates[0]["payload_json"]["source_message_ids"] == ["msg_plain"]

    def test_source_object_ids_also_union(self) -> None:
        candidates = [
            _candidate(
                "mem_rep",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.9,
                source_message_ids=["msg_1"],
                source_object_ids=["mem_l0_a"],
            ),
            _candidate(
                "mem_dup",
                canonical_text="PERSON_A owns a pottery workshop.",
                rrf_score=0.4,
                source_message_ids=["msg_1", "msg_2"],
                source_object_ids=["mem_l0_b"],
            ),
        ]
        result = collapse_duplicate_carriers(candidates)
        representative = result.candidates[0]
        assert representative["payload_json"]["source_message_ids"] == [
            "msg_1",
            "msg_2",
        ]
        assert representative["payload_json"]["source_object_ids"] == [
            "mem_l0_a",
            "mem_l0_b",
        ]
