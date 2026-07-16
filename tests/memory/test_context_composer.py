"""Tests for final context composition."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest import mock

from atagia.core.clock import FrozenClock
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import ScoredCandidate
from atagia.services.answer_postcondition import _verification_prompt
from atagia.services.chat_support import answer_support_prompt_payload

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"


def _resolved_policy(context_budget_tokens: int = 5300):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()["coding_debug"]
    resolved = PolicyResolver().resolve(manifest, None, None)
    return resolved.model_copy(update={"context_budget_tokens": context_budget_tokens})


def _policy_with_final_context_items(
    context_budget_tokens: int,
    final_context_items: int,
):
    policy = _resolved_policy(context_budget_tokens)
    return policy.model_copy(
        update={
            "retrieval_params": policy.retrieval_params.model_copy(
                update={"final_context_items": final_context_items}
            )
        }
    )


def _candidate(
    memory_id: str,
    *,
    final_score: float,
    canonical_text: str,
    object_type: str = "evidence",
    confidence: float = 0.8,
    scope: str = "conversation",
    payload_json: dict | None = None,
    updated_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    temporal_type: str = "unknown",
    resolved_date: str | None = None,
    evidence_packets: list[dict] | None = None,
    llm_applicability: float = 0.7,
    retrieval_score: float = 0.6,
) -> ScoredCandidate:
    memory_object = {
        "id": memory_id,
        "object_type": object_type,
        "confidence": confidence,
        "scope": scope,
        "canonical_text": canonical_text,
        "payload_json": payload_json or {},
        "updated_at": updated_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "temporal_type": temporal_type,
    }
    if evidence_packets is not None:
        memory_object["evidence_packets"] = evidence_packets
    return ScoredCandidate(
        memory_id=memory_id,
        memory_object=memory_object,
        llm_applicability=llm_applicability,
        retrieval_score=retrieval_score,
        vitality_boost=0.2,
        confirmation_boost=0.0,
        need_boost=0.0,
        penalty=0.0,
        final_score=final_score,
        resolved_date=resolved_date,
    )


def _contract() -> dict[str, dict]:
    return {
        "depth": {"label": "detailed explanations preferred", "score": 0.72},
        "directness": {"label": "high", "score": 0.85},
    }


def _composer() -> ContextComposer:
    return ContextComposer(
        FrozenClock(datetime(2026, 3, 30, 22, 0, tzinfo=timezone.utc))
    )


def test_normal_composition_includes_contract_and_memories_within_budget() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_2",
                final_score=0.74,
                canonical_text="FastAPI and SQLite are the current stack.",
            ),
            _candidate(
                "mem_1",
                final_score=0.91,
                canonical_text="User prefers patch-style debugging help.",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert context.contract_block.startswith("[Interaction Contract]")
    assert (
        "depth: detailed explanations preferred (confidence: 0.72)"
        in context.contract_block
    )
    assert context.memory_block.startswith("[Retrieved Memories]")
    assert "User prefers patch-style debugging help." in context.memory_block
    assert "FastAPI and SQLite are the current stack." in context.memory_block
    assert context.selected_memory_ids == ["mem_1", "mem_2"]
    assert context.items_included == 2
    assert context.items_dropped == 0
    assert context.total_tokens_estimate <= context.budget_tokens
    assert context.state_block == ""


def test_open_domain_composition_does_not_populate_answer_support_metadata() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_vibes",
                final_score=0.88,
                canonical_text="Caroline said Paris was energizing.",
                payload_json={
                    "source_message_ids": ["msg_paris"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[],
    )

    assert context.coverage_state == "unknown"
    assert context.allowed_values == []
    assert context.missing_slots == []
    assert context.support_map == {}


def test_source_quote_dedupe_allows_superset_and_suppresses_subset() -> None:
    short_quote = "Caroline said she lived in Paris"
    long_quote = f"{short_quote} before moving to Rome for work"
    short_key = ContextComposer._normalize_quote_for_compare(short_quote)
    long_key = ContextComposer._normalize_quote_for_compare(long_quote)

    assert not ContextComposer._source_quote_is_suppressed(
        long_quote,
        frozenset({short_key}),
    )
    assert ContextComposer._source_quote_is_suppressed(
        short_quote,
        frozenset({short_key}),
    )
    assert ContextComposer._source_quote_is_suppressed(
        short_quote,
        frozenset({long_key}),
    )


def test_source_quote_superset_renders_after_shorter_quote() -> None:
    composer = _composer()
    short_quote = "Caroline said she lived in Paris"
    long_quote = f"{short_quote} before moving to Rome for work"

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_short",
                final_score=0.96,
                canonical_text="Caroline lived in Paris.",
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": short_quote,
                            }
                        ],
                    }
                ],
            ),
            _candidate(
                "mem_long",
                final_score=0.95,
                canonical_text="Caroline lived in Paris and Rome.",
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": long_quote,
                            }
                        ],
                    }
                ],
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="Where did Caroline live?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert short_quote in context.memory_block
    assert long_quote in context.memory_block


def test_slot_fill_composition_adds_final_answer_evidence_pack() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_prism",
                final_score=0.88,
                canonical_text="The prism is stored in locker NOVA-417.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "I stored the prism in locker NOVA-417.",
                                "occurred_at": "2026-02-11T09:15:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "user"},
                            }
                        ],
                    }
                ],
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="Which locker code did I use for the prism?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.memory_block.startswith("[Final Answer Evidence Pack]")
    assert context.answer_evidence_memory_ids == ["mem_prism"]
    assert context.answer_evidence_items[0]["supporting_quote"].startswith(
        "user @ "
    )
    assert "NOVA-417" in context.answer_evidence_block
    assert "NOVA-417" in context.answer_evidence_block
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_items[0]["selected_for_answer_pack"] is True
    assert context.answer_evidence_items[0]["normalization"]["speaker_role"] == "user"
    assert (
        context.answer_evidence_items[0]["normalization"]["evidence_occurred_at"]
        == "2026-02-11T09:15:00+00:00"
    )


def test_answer_evidence_diagnostic_is_populated_without_rendering_pack() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_prism",
                final_score=0.88,
                canonical_text="The prism is stored in locker NOVA-417.",
                valid_from="2026-02-11T09:15:00+00:00",
                temporal_type="event_triggered",
                resolved_date="2026-02-11T09:15:00+00:00",
                payload_json={
                    "source_message_ids": ["msg_2"],
                    "source_message_window_start_occurred_at": (
                        "2026-02-11T09:15:00+00:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2026-02-11T09:15:00+00:00"
                    ),
                },
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "message_id": "msg_2",
                                "quote_text": "I stored the prism in locker NOVA-417.",
                                "occurred_at": "2026-02-11T09:15:00+00:00",
                                "seq": 2,
                                "metadata_json": {"message_role": "user"},
                            },
                            {
                                "span_role": "trigger",
                                "message_id": "msg_1",
                                "quote_text": "Which locker should hold the calibrated prism?",
                                "occurred_at": "2026-02-11T09:15:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "assistant"},
                            },
                        ],
                    }
                ],
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="Which locker code did I use for the prism?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=False,
    )

    assert context.answer_evidence_block == ""
    assert context.answer_evidence_memory_ids == []
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_sufficiency["rendered"] is False
    assert context.answer_evidence_items[0]["selected_for_answer_pack"] is False
    normalization = context.answer_evidence_items[0]["normalization"]
    assert normalization["resolved_date"] == "2026-02-11T09:15:00+00:00"
    assert normalization["source_message_ids"] == ["msg_2"]
    assert normalization["evidence_packet_message_ids"] == ["msg_2", "msg_1"]
    assert (
        normalization["source_window_start"] == "2026-02-11T09:15:00+00:00"
    )
    assert "Which locker should hold the calibrated prism?" in normalization["trigger_quote"]


def test_answer_evidence_pack_does_not_promote_low_score_quote() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_high_no_quote",
                final_score=0.95,
                canonical_text="The gallery closes at 18:00.",
            ),
            _candidate(
                "mem_low_literal",
                final_score=0.31,
                canonical_text="The prism is stored in locker NOVA-417.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "I stored the prism in locker NOVA-417.",
                                "occurred_at": "2026-02-11T09:15:00+00:00",
                                "seq": 2,
                                "metadata_json": {"message_role": "user"},
                            }
                        ],
                    }
                ],
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="Which locker code did I use for the prism?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_block == ""
    assert context.answer_evidence_memory_ids == []
    assert context.answer_evidence_items[0]["memory_id"] == "mem_low_literal"
    assert context.answer_evidence_sufficiency["state"] == "weak_low_applicability"
    assert "[Final Answer Evidence Pack]" not in context.memory_block


def test_answer_evidence_prefers_query_relevant_source_quote_over_first_packet_span() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_pavilion_opening",
                final_score=0.89,
                canonical_text=(
                    "Lio described Amara's wind-harp pavilion and praised it."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "source_message_ids": ["msg_1", "msg_2"],
                },
                evidence_packets=[
                    {
                        "support_kind": "inferred",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "message_id": "msg_1",
                                "quote_text": "Theo sorted invoices last week.",
                                "occurred_at": "2023-06-19T10:04:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "assistant"},
                            }
                        ],
                    }
                ],
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "assistant",
                "seq": 1,
                "text": "Theo sorted invoices last week.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_2",
                "role": "user",
                "seq": 2,
                "text": "Lio: The wind-harp pavilion sounds wonderfully clear.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="How does Lio describe the pavilion Amara built?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["quote_source"] == "source_message"
    assert "sounds wonderfully clear" in context.answer_evidence_items[0]["supporting_quote"]
    assert "sorted invoices" not in context.answer_evidence_items[0]["supporting_quote"]


def test_answer_evidence_keeps_named_speaker_prefix_for_quote_relevance() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_tidal_gauge",
                final_score=0.99,
                canonical_text=(
                    "[user] Sela Nori: The tidal gauge installation begins on November 6.\n"
                    "[assistant] Engineer: I will note the installation date.\n"
                    "[user] Sela Nori: The dock inspection starts at 14:20.\n"
                    "[assistant] Engineer: Recorded the dock inspection time."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": [
                        "msg_278",
                        "msg_279",
                        "msg_280",
                        "msg_281",
                    ],
                },
                llm_applicability=0.9,
                retrieval_score=0.72,
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_278",
                "role": "user",
                "seq": 278,
                "text": "Sela Nori: The tidal gauge installation begins on November 6.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": "Engineer: I will note the installation date.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Sela Nori: The dock inspection starts at 14:20.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Engineer: Recorded the dock inspection time.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
        ],
        query_text="When does the tidal gauge installation begin?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_memory_ids == ["vew_tidal_gauge"]
    assert "Sela Nori" in context.answer_evidence_items[0]["supporting_quote"]
    assert "November 6" in context.answer_evidence_items[0]["supporting_quote"]


def test_answer_evidence_ranks_query_relevant_quote_before_higher_score_distractor() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_pavilion_flooring",
                final_score=0.95,
                canonical_text="Lio liked the pavilion flooring.",
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "The cork flooring dampens footsteps well.",
                                "occurred_at": "2023-01-29T14:32:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "user"},
                            }
                        ],
                    }
                ],
            ),
            _candidate(
                "sum_pavilion_opening",
                final_score=0.72,
                canonical_text="Lio described Amara's wind-harp pavilion.",
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "source_message_ids": ["msg_1"],
                },
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "seq": 2,
                "text": "Lio: The wind-harp pavilion sounds wonderfully clear.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            }
        ],
        query_text="How does Lio describe the pavilion Amara built?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["memory_id"] == "sum_pavilion_opening"
    assert "sounds wonderfully clear" in context.answer_evidence_items[0]["supporting_quote"]


def test_broad_list_answer_evidence_prefers_material_applicability_over_ir_noise() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_northglass",
                final_score=0.86,
                canonical_text="[assistant] Rhea: I inspected Northglass Station yesterday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_northglass"],
                },
                llm_applicability=1.0,
                retrieval_score=0.75,
            ),
            _candidate(
                "vew_ir_noise",
                final_score=0.35,
                canonical_text="[assistant] Rhea: The archive shelves need labels.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_noise"],
                },
                llm_applicability=0.0,
                retrieval_score=0.98,
            ),
            _candidate(
                "sum_ember_shoal",
                final_score=0.20,
                canonical_text=(
                    "Rhea mentioned a short survey trip to Ember Shoal Station."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_ember_shoal"],
                    "source_message_window_start_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                },
                llm_applicability=0.2,
                retrieval_score=0.44,
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1600),
        conversation_messages=[
            {
                "id": "msg_northglass",
                "role": "assistant",
                "seq": 32,
                "text": "Rhea: I inspected Northglass Station yesterday.",
                "occurred_at": "2023-01-28T14:32:00",
            },
            {
                "id": "msg_noise",
                "role": "assistant",
                "seq": 34,
                "text": "Rhea: The archive shelves need labels.",
                "occurred_at": "2023-01-29T14:32:00",
            },
            {
                "id": "msg_ember_shoal",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Rhea: I made a short survey trip last week to Ember Shoal "
                    "Station."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
        ],
        query_text="Which field stations has Rhea inspected?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=True,
    )

    assert [item["memory_id"] for item in context.answer_evidence_items[:2]] == [
        "vew_northglass",
        "sum_ember_shoal",
    ]
    assert "Ember Shoal" in context.answer_evidence_block
    assert "vew_ir_noise" not in context.answer_evidence_memory_ids


def test_broad_list_evidence_obligation_reserves_applicable_source_linked_summary() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_northglass",
                final_score=0.86,
                canonical_text="[assistant] Rhea: I inspected Northglass Station yesterday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_northglass"],
                },
                llm_applicability=1.0,
            ),
            _candidate(
                "mem_distractor",
                final_score=0.82,
                canonical_text="Rhea discussed archive shelf labels.",
                llm_applicability=0.1,
            ),
            _candidate(
                "sum_unrelated",
                final_score=0.81,
                canonical_text="Rhea discussed unrelated archive logistics.",
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 1,
                    "source_object_ids": ["vew_unrelated"],
                },
                llm_applicability=0.1,
            ),
            _candidate(
                "vew_unrelated",
                final_score=0.70,
                canonical_text="[assistant] Rhea: Acid-free folders seem practical.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_unrelated"],
                },
                llm_applicability=0.1,
            ),
            _candidate(
                "sum_ember_shoal",
                final_score=0.20,
                canonical_text=(
                    "Rhea mentioned a short survey trip to Ember Shoal Station."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_ember_shoal"],
                    "source_message_window_start_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                },
                llm_applicability=0.2,
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_policy_with_final_context_items(1300, 2),
        conversation_messages=[
            {
                "id": "msg_northglass",
                "role": "assistant",
                "seq": 32,
                "text": "Rhea: I inspected Northglass Station yesterday.",
                "occurred_at": "2023-01-28T14:32:00",
            },
            {
                "id": "msg_ember_shoal",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Rhea: I made a short survey trip last week to Ember Shoal "
                    "Station."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
            {
                "id": "msg_unrelated",
                "role": "assistant",
                "seq": 36,
                "text": "Rhea: Acid-free folders seem practical.",
                "occurred_at": "2023-01-29T14:32:00",
            },
        ],
        query_text="Which field stations has Rhea inspected?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["vew_northglass", "sum_ember_shoal"]
    assert "Ember Shoal" in context.memory_block
    assert "archive shelf labels" not in context.memory_block


def test_broad_list_answer_evidence_renders_material_direct_quote_below_score_floor() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_high_ir_noise",
                final_score=0.92,
                canonical_text="Rhea discussed archive logistics.",
                llm_applicability=0.0,
                retrieval_score=0.95,
            ),
            _candidate(
                "sum_ember_shoal",
                final_score=0.20,
                canonical_text=(
                    "Rhea mentioned a short survey trip to Ember Shoal Station."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_ember_shoal"],
                    "source_message_window_start_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2023-06-19T10:04:00"
                    ),
                },
                llm_applicability=0.2,
                retrieval_score=0.3,
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1600),
        conversation_messages=[
            {
                "id": "msg_ember_shoal",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Rhea: I made a short survey trip last week to Ember Shoal "
                    "Station."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
        ],
        query_text="Which field stations has Rhea inspected?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_memory_ids == ["sum_ember_shoal"]
    assert "Ember Shoal" in context.answer_evidence_block


def test_memory_entry_adds_query_relevant_source_quote_when_packet_span_is_weak() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_pavilion_opening",
                final_score=0.89,
                canonical_text="Lio described Amara's wind-harp pavilion and praised it.",
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "source_message_ids": ["msg_1", "msg_2"],
                },
                evidence_packets=[
                    {
                        "support_kind": "inferred",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "message_id": "msg_1",
                                "quote_text": "Theo sorted invoices last week.",
                                "occurred_at": "2023-06-19T10:04:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "assistant"},
                            }
                        ],
                    }
                ],
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "assistant",
                "seq": 1,
                "text": "Theo sorted invoices last week.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_2",
                "role": "user",
                "seq": 2,
                "text": "Lio: The wind-harp pavilion sounds wonderfully clear.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="How does Lio describe the pavilion Amara built?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=False,
    )

    assert "evidence_packet: support: inferred" in context.memory_block
    assert "source_quote: user @ 2023-06-19T10:04:00+00:00 seq 2:" in context.memory_block
    assert "sounds wonderfully clear" in context.memory_block


def test_summary_memory_entry_renders_short_source_chain_from_first_query_match() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_clock_opening",
                final_score=0.88,
                canonical_text=(
                    "Museum planning summary: the kinetic clock exhibit opens "
                    "on October 12 and the preview tour starts at 09:30."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "source_message_ids": [
                        "msg_277",
                        "msg_278",
                        "msg_279",
                        "msg_280",
                        "msg_281",
                        "msg_282",
                        "msg_283",
                    ],
                },
                evidence_packets=[
                    {
                        "support_kind": "inferred",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "message_id": "msg_277",
                                "quote_text": (
                                    "Curator: Which exhibit date belongs on the calendar?"
                                ),
                                "occurred_at": "2026-03-08T10:00:00+00:00",
                                "seq": 277,
                                "metadata_json": {"message_role": "assistant"},
                            }
                        ],
                    }
                ],
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1600),
        conversation_messages=[
            {
                "id": "msg_277",
                "role": "assistant",
                "seq": 277,
                "text": "Curator: Which exhibit date belongs on the calendar?",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_278",
                "role": "user",
                "seq": 278,
                "text": "Rin Vale: The kinetic clock exhibit opens on October 12.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": "Curator: I will note the public opening date.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Rin Vale: The preview tour starts at 09:30.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Curator: Recorded the preview tour time.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_282",
                "role": "user",
                "seq": 282,
                "text": "Rin Vale: The brass pendulum arrives two days earlier.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_283",
                "role": "assistant",
                "seq": 283,
                "text": "Curator: I will keep the delivery date with the notes.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
        ],
        query_text="When does the kinetic clock exhibit open?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert "source_chain:" in context.memory_block
    assert "October 12" in context.memory_block
    assert any(
        "October 12" in line
        for line in context.answer_evidence_items[0]["source_chain"]
    )


def test_answer_evidence_uses_verbatim_window_text_when_source_messages_are_absent() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_launch_41_43",
                final_score=0.91,
                canonical_text=(
                    "[user] Pause and record the prototype launch.\n"
                    "[assistant] I will record a voice note before packing up."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_41", "msg_42", "msg_43"],
                },
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="What will Niko do after the prototype launch?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["quote_source"] == "verbatim_evidence_window"
    assert "record a voice note" in context.answer_evidence_items[0]["supporting_quote"]
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"


def test_summary_source_window_answer_evidence_renders_source_chain() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_clock_opening_277_278",
                final_score=0.98,
                canonical_text=(
                    "[assistant] Curator: Which exhibit date belongs on the calendar?\n"
                    "[user] Rin Vale: The kinetic clock exhibit opens on October 12."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_277", "msg_278"],
                },
            ),
            _candidate(
                "ssw_sum_clock_opening_277_283",
                final_score=0.93,
                canonical_text=(
                    "[assistant] Curator: Which exhibit date belongs on the calendar?\n"
                    "[user] Rin Vale: The kinetic clock exhibit opens on October 12.\n"
                    "[assistant] Curator: I will note the public opening date.\n"
                    "[user] Rin Vale: The preview tour starts at 09:30.\n"
                    "[assistant] Curator: Recorded the preview tour time.\n"
                    "[user] Rin Vale: The brass pendulum arrives two days earlier.\n"
                    "[assistant] Curator: I will keep the delivery date with the notes."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "summary_source_window",
                    "source_message_ids": [
                        "msg_277",
                        "msg_278",
                        "msg_279",
                        "msg_280",
                        "msg_281",
                        "msg_282",
                        "msg_283",
                    ],
                    "source_message_window_start_occurred_at": (
                        "2026-03-08T10:00:00+00:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2026-03-08T10:00:00+00:00"
                    ),
                },
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1600),
        conversation_messages=[
            {
                "id": "msg_277",
                "role": "assistant",
                "seq": 277,
                "text": "Curator: Which exhibit date belongs on the calendar?",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_278",
                "role": "user",
                "seq": 278,
                "text": "Rin Vale: The kinetic clock exhibit opens on October 12.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": "Curator: I will note the public opening date.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Rin Vale: The preview tour starts at 09:30.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Curator: Recorded the preview tour time.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_282",
                "role": "user",
                "seq": 282,
                "text": "Rin Vale: The brass pendulum arrives two days earlier.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
            {
                "id": "msg_283",
                "role": "assistant",
                "seq": 283,
                "text": "Curator: I will keep the delivery date with the notes.",
                "occurred_at": "2026-03-08T10:00:00+00:00",
            },
        ],
        query_text="When does the kinetic clock exhibit open?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    source_chain = context.answer_evidence_items[0]["source_chain"]
    assert context.answer_evidence_memory_ids == ["ssw_sum_clock_opening_277_283"]
    assert context.memory_block.startswith("[Final Answer Evidence Pack]")
    assert any("October 12" in line for line in source_chain)
    assert "October 12" in context.answer_evidence_block


def test_literal_evidence_is_selected_before_higher_scoring_summary() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_summary",
                final_score=0.99,
                canonical_text="The prism is somewhere in the observatory storage area.",
                object_type="summary_view",
                payload_json={"hierarchy_level": 1, "source_message_ids": ["msg_1"]},
            ),
            _candidate(
                "mem_literal",
                final_score=0.62,
                canonical_text="The prism is stored in locker NOVA-417.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "I stored the prism in locker NOVA-417.",
                                "metadata_json": {"message_role": "user"},
                            }
                        ],
                    }
                ],
            ),
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_policy_with_final_context_items(1000, 1),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "text": "I stored the prism in locker NOVA-417.",
            }
        ],
        query_text="Which locker code did I use for the prism?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids == ["mem_literal"]
    assert "I stored the prism in locker NOVA-417." in context.memory_block


def test_cross_presence_memory_is_rendered_with_attribution() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_cross",
        final_score=0.91,
        canonical_text="Character Beta prefers terse implementation notes.",
        payload_json={
            "presence_attribution": {
                "active": {
                    "presence_id": "character_beta",
                    "kind": "owned_facet",
                    "display_name": "Character Beta",
                },
                "source": {
                    "presence_id": "human_owner",
                    "kind": "human",
                    "display_name": "User",
                },
            }
        },
    )
    candidate.memory_object.update(
        {
            "active_presence_id": "character_beta",
            "source_presence_id": "human_owner",
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_presence_id="character_alpha",
    )

    assert context.selected_memory_ids == ["mem_cross"]
    assert "presence: active=Character Beta [owned_facet]; source=User [human]" in (
        context.memory_block
    )


def test_space_scoped_memory_is_rendered_with_space_label() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_space",
        final_score=0.91,
        canonical_text="Alpha launch checklist lives in the vault.",
        payload_json={
            "space_boundary": {
                "active_space_id": "space_vault",
                "boundary_mode": "privacy_vault",
                "display_name": "Alpha Vault",
            }
        },
    )
    candidate.memory_object["space_id"] = "space_vault"
    candidate.memory_object["space_boundary_mode"] = "privacy_vault"

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
    )

    assert "space: Alpha Vault [privacy_vault]" in context.memory_block


def test_cross_realm_memory_is_rendered_with_attribution() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_realm",
        final_score=0.91,
        canonical_text="The raid leader prefers teleport crystals in Aincrad.",
        payload_json={
            "realm": {
                "active_realm_id": "realm_aincrad",
                "display_name": "Aincrad",
                "cross_realm_mode": "attributed",
            }
        },
    )
    candidate.memory_object["realm_id"] = "realm_aincrad"
    candidate.memory_object["realm_bridge_mode"] = "attributed"

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_realm_id="realm_real",
    )

    assert context.selected_memory_ids == ["mem_realm"]
    assert (
        "realm: in Realm Aincrad [cross_realm: attributed; active=realm_real]"
        in context.memory_block
    )


def test_same_realm_memory_is_rendered_with_same_realm_label() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_realm",
        final_score=0.91,
        canonical_text="The desktop environment uses the real printer queue.",
    )
    candidate.memory_object["realm_id"] = "realm_real"

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_realm_id="realm_real",
    )

    assert "realm: in Realm realm_real [same]" in context.memory_block


def test_cross_realm_contract_value_keeps_realm_provenance() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract={
            "tone": {
                "label": "use story-world tone",
                "score": 0.8,
                "realm": {
                    "active_realm_id": "realm_aincrad",
                    "active_request_realm_id": "realm_real",
                    "cross_realm_mode": "applicable",
                    "display_name": "Aincrad",
                },
            }
        },
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_realm_id="realm_real",
    )

    assert (
        "realm: in Realm Aincrad [cross_realm: applicable; active=realm_real]"
        in context.contract_block
    )


def test_cross_realm_state_value_keeps_realm_provenance() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract={},
        user_state={
            "current_user_state": {
                "value": "aincrad state",
                "realm": {
                    "active_realm_id": "realm_aincrad",
                    "active_request_realm_id": "realm_real",
                    "cross_realm_mode": "applicable",
                    "display_name": "Aincrad",
                },
            }
        },
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_realm_id="realm_real",
    )

    assert (
        "current_user_state: aincrad state "
        "[realm: in Realm Aincrad [cross_realm: applicable; active=realm_real]]"
        in context.state_block
    )


def test_unknown_cross_presence_memory_is_filtered_closed() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_unknown",
        final_score=0.91,
        canonical_text="An unattributed actor prefers terse implementation notes.",
        payload_json={
            "presence_attribution": {
                "active": {
                    "presence_id": "mystery_presence",
                    "kind": "unknown",
                    "display_name": None,
                }
            }
        },
    )
    candidate.memory_object["active_presence_id"] = "mystery_presence"

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
        active_presence_id="character_alpha",
    )

    assert context.selected_memory_ids == []
    assert "unattributed actor" not in context.memory_block


def test_memory_block_redacts_high_risk_secret_literals() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_secret",
        final_score=0.91,
        canonical_text="The account PIN is 1234.",
    )
    candidate.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert "privacy_level: 3" in context.memory_block
    assert "memory_category: pin_or_password" in context.memory_block
    assert "preserve_verbatim: true" in context.memory_block
    assert "disclosure_action: withhold_secret_literal" in context.memory_block
    assert "raw value withheld" in context.memory_block
    assert "1234" not in context.memory_block


def test_answer_evidence_and_verifier_prompt_omit_withheld_secret_literals() -> None:
    composer = _composer()
    secret_literal = "fixture-secret-Q7X9"
    candidate = _candidate(
        "mem_secret",
        final_score=0.98,
        canonical_text=f"The production jump host password is {secret_literal}.",
        payload_json={"source_message_ids": ["msg_secret"]},
        evidence_packets=[
            {
                "support_kind": "direct",
                "spans": [
                    {
                        "span_role": "source",
                        "message_id": "msg_secret",
                        "quote_text": f"The production jump host password is {secret_literal}.",
                    }
                ],
            }
        ],
    )
    candidate.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_secret",
                "role": "user",
                "seq": 1,
                "text": f"The production jump host password is {secret_literal}.",
                "occurred_at": "2026-03-30T12:00:00+00:00",
            }
        ],
        query_text="What is the production jump host password?",
        query_type="slot_fill",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=True,
    )
    prompt = _verification_prompt(
        original_query="What is the production jump host password?",
        answer_text="I cannot disclose that secret in chat.",
        composed_context=context,
        retrieval_sufficiency=None,
        privacy_enforcement="enforce",
        answer_stance="reactive",
    )
    serialized_context = context.model_dump_json()

    assert context.answer_evidence_items == []
    assert secret_literal not in context.memory_block
    assert secret_literal not in serialized_context
    assert secret_literal not in prompt


def test_coverage_metadata_redacts_secret_literals_without_hiding_gap() -> None:
    composer = _composer()
    selected_secret = _candidate(
        "mem_jump_host_secret",
        final_score=0.96,
        canonical_text="The production jump host password is fixture-secret-Q7X9.",
        payload_json={
            "source_message_ids": ["msg_jump_host"],
            "value_norm_key": "fixture-secret-Q7X9",
            "value_text": "fixture-secret-Q7X9",
        },
    )
    selected_secret.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )
    missing_secret = _candidate(
        "mem_vault_secret",
        final_score=0.2,
        canonical_text="The vault backup code is VaultReset-9911.",
        payload_json={
            "source_message_ids": ["msg_vault"],
            "value_norm_key": "VaultReset-9911",
            "value_text": "VaultReset-9911",
        },
    )
    missing_secret.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_paris",
                final_score=0.97,
                canonical_text="Caroline mentioned Paris.",
                payload_json={
                    "source_message_ids": ["msg_paris"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            ),
            selected_secret,
            missing_secret,
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(900, 2),
        conversation_messages=[],
        query_text="Which values did Caroline mention?",
        query_type="broad_list",
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert context.coverage_state == "partial"
    assert context.missing_slots == []
    assert "Protected high-risk memory present; raw value withheld." in context.memory_block
    assert "fixture-secret-Q7X9" not in context.memory_block
    assert "VaultReset-9911" not in context.memory_block
    serialized_support = json.dumps(
        {
            "allowed_values": context.allowed_values,
            "missing_slots": context.missing_slots,
            "support_map": context.support_map,
            "answer_support": answer_support_prompt_payload(context),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert "fixture-secret-Q7X9" not in serialized_support
    assert "VaultReset-9911" not in serialized_support
    assert "Protected high-risk memory present" not in serialized_support
    assert "withheld|high_risk_secret_literal" not in serialized_support


def test_privacy_off_can_render_high_risk_secret_literals() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_secret",
        final_score=0.91,
        canonical_text="The account PIN is 1234.",
    )
    candidate.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
        redact_high_risk_secret_literals=False,
    )

    assert (
        "privacy_restrictions_inactive: high_risk_secret_literal_unredacted"
        in context.memory_block
    )
    assert "privacy_classification_non_blocking: level_3" in context.memory_block
    assert "memory_category_non_blocking: pin_or_password" in context.memory_block
    assert "privacy_level: 3" not in context.memory_block
    assert "memory_category: pin_or_password" not in context.memory_block
    assert "The account PIN is 1234." in context.memory_block
    assert "raw value withheld" not in context.memory_block
    assert "disclosure_action: withhold_secret_literal" not in context.memory_block


def test_privacy_off_renders_source_quote_for_high_risk_secret() -> None:
    composer = _composer()
    secret_message_text = "The production jump host password is fixture-secret-Q7X9."
    candidate = _candidate(
        "mem_secret_quote",
        final_score=0.93,
        canonical_text="The production jump host password is fixture-secret-Q7X9.",
        payload_json={"source_message_ids": ["msg_secret"]},
    )
    candidate.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[
            {
                "id": "msg_secret",
                "role": "user",
                "seq": 1,
                "text": secret_message_text,
                "occurred_at": "2026-03-30T12:00:00+00:00",
            }
        ],
        query_text="What is the production jump host password?",
        query_type="slot_fill",
        exact_recall_mode=True,
        redact_high_risk_secret_literals=False,
    )

    assert "source_quote:" in context.memory_block
    assert secret_message_text in context.memory_block


def test_privacy_off_renders_source_time_private_text_as_non_blocking_context() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_private",
        final_score=0.91,
        canonical_text=(
            "Ben is seeing Dr. Reeves for anxiety. He asked at the time "
            "not to use that information in other contexts."
        ),
    )
    candidate.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "interaction_contract",
        }
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
        redact_high_risk_secret_literals=False,
    )

    assert "privacy_classification_non_blocking: level_3" in context.memory_block
    assert "memory_category_non_blocking: interaction_contract" in context.memory_block
    assert "privacy_level: 3" not in context.memory_block
    assert "Ben is seeing Dr. Reeves for anxiety." in context.memory_block


def test_budget_exhaustion_keeps_only_top_candidates_that_fit() -> None:
    composer = _composer()
    first = _candidate(
        "mem_1", final_score=0.95, canonical_text="Short top-priority memory."
    )
    contract_block = ContextComposer.render_contract_block(_contract(), _resolved_policy(400))
    tight_budget = (
        composer.estimate_tokens(contract_block)
        + composer.estimate_tokens("[Retrieved Memories]\n")
        + composer.estimate_tokens(composer._format_memory_entry(1, first))
    )

    context = composer.compose(
        scored_candidates=[
            first,
            _candidate(
                "mem_2",
                final_score=0.70,
                canonical_text="Second memory that should not fit once the first one has consumed the remaining budget.",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(tight_budget),
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_1"]
    assert context.items_included == 1
    assert context.items_dropped == 1


def test_oversized_candidate_does_not_block_smaller_later_memory() -> None:
    composer = _composer()
    later_fit = _candidate(
        "mem_fit", final_score=0.6, canonical_text="Short memory that fits."
    )
    contract_block = ContextComposer.render_contract_block(_contract(), _resolved_policy(400))
    tight_budget = (
        composer.estimate_tokens(contract_block)
        + composer.estimate_tokens("[Retrieved Memories]\n")
        + composer.estimate_tokens(composer._format_memory_entry(1, later_fit))
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_large",
                final_score=0.95,
                canonical_text=(
                    "This higher-scored memory is intentionally long enough to exceed the remaining "
                    "budget and should be skipped instead of blocking the rest of the shortlist."
                ),
            ),
            later_fit,
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(tight_budget),
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_fit"]
    assert "Short memory that fits." in context.memory_block
    assert "1. (evidence" in context.memory_block


def test_contract_is_always_included_even_when_budget_is_tight() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_1",
                final_score=0.95,
                canonical_text="A memory that definitely cannot fit under a tiny budget.",
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(1),
        conversation_messages=[],
    )

    assert context.contract_block
    assert context.memory_block == ""
    assert context.selected_memory_ids == []
    assert context.items_included == 0
    assert context.items_dropped == 1
    assert context.total_tokens_estimate <= context.budget_tokens


def test_empty_candidates_returns_contract_only_for_cold_start() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract={
            "implementation_first": {"label": "default", "source": "manifest_default"},
            "depth": {"label": "default", "source": "manifest_default"},
        },
        user_state=None,
        resolved_policy=_resolved_policy(120),
        conversation_messages=[],
    )

    assert context.contract_block.startswith("[Interaction Contract]")
    assert context.memory_block == ""
    assert context.selected_memory_ids == []
    assert context.items_included == 0
    assert context.items_dropped == 0


def test_oversized_contract_is_truncated_to_fit_budget() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract={
            f"dimension_{index}": {"label": "x" * 80, "score": 0.9}
            for index in range(10)
        },
        user_state=None,
        resolved_policy=_resolved_policy(20),
        conversation_messages=[],
    )

    assert context.contract_block
    assert composer.estimate_tokens(context.contract_block) <= context.budget_tokens
    assert context.total_tokens_estimate <= context.budget_tokens


def test_priority_ordering_prefers_higher_scored_candidates_first() -> None:
    composer = _composer()
    first = _candidate(
        "mem_high", final_score=0.95, canonical_text="High-priority memory."
    )
    contract_block = ContextComposer.render_contract_block(_contract(), _resolved_policy(300))
    tight_budget = (
        composer.estimate_tokens(contract_block)
        + composer.estimate_tokens("[Retrieved Memories]\n")
        + composer.estimate_tokens(composer._format_memory_entry(1, first))
    )

    context = composer.compose(
        scored_candidates=[
            first,
            _candidate(
                "mem_low",
                final_score=0.20,
                canonical_text="Lower-priority memory that should be dropped.",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(tight_budget),
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_high"]
    assert "High-priority memory." in context.memory_block
    assert "Lower-priority memory" not in context.memory_block


def test_budgeted_marginal_strategy_prefers_higher_value_per_token_set() -> None:
    composer = _composer()
    long_candidate = _candidate(
        "mem_long",
        final_score=0.95,
        canonical_text=(
            "This memory has a high scalar score but contains a long operational narrative "
            "with background details, repeated caveats, and enough extra explanation to make "
            "it a poor use of a tight memory budget for this specific composition test."
        ),
    )
    short_a = _candidate(
        "mem_short_a", final_score=0.72, canonical_text="Short high-value fact A."
    )
    short_b = _candidate(
        "mem_short_b", final_score=0.70, canonical_text="Short high-value fact B."
    )
    contract_block = ContextComposer.render_contract_block(_contract(), _resolved_policy(500))
    memory_header_tokens = composer.estimate_tokens("[Retrieved Memories]\n")
    budget = (
        composer.estimate_tokens(contract_block)
        + memory_header_tokens
        + composer.estimate_tokens(composer._format_memory_entry(1, long_candidate))
    )
    policy = _resolved_policy(budget).model_copy(
        update={
            "retrieval_params": _resolved_policy(budget).retrieval_params.model_copy(
                update={"final_context_items": 2}
            )
        }
    )

    score_first = composer.compose(
        scored_candidates=[long_candidate, short_a, short_b],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
    )
    budgeted = composer.compose(
        scored_candidates=[long_candidate, short_a, short_b],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        composer_strategy="budgeted_marginal",
    )

    assert score_first.selected_memory_ids == ["mem_long"]
    assert budgeted.selected_memory_ids == ["mem_short_a", "mem_short_b"]
    assert "Short high-value fact A." in budgeted.memory_block
    assert "Short high-value fact B." in budgeted.memory_block
    assert "long operational narrative" not in budgeted.memory_block


def test_temporal_memory_includes_validity_window_in_rendered_block() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_bounded",
                final_score=0.9,
                canonical_text="User painted a lake sunrise.",
                valid_from="2041-05-15T00:00:00+00:00",
                valid_to="2041-05-31T23:59:59+00:00",
                temporal_type="bounded",
            ),
            _candidate(
                "mem_open",
                final_score=0.8,
                canonical_text="User signed up for pottery class.",
                valid_from="2023-07-02T00:00:00+00:00",
                temporal_type="ephemeral",
            ),
            _candidate(
                "mem_no_time",
                final_score=0.7,
                canonical_text="User prefers direct answers.",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[],
    )

    # Bounded memory shows both dates
    assert (
        "valid_window: 2041-05-15T00:00:00+00:00 to 2041-05-31T23:59:59+00:00"
        in context.memory_block
    )
    # Open-ended (only valid_from) shows from-date and ?
    assert "valid_window: 2023-07-02T00:00:00+00:00 to ?" in context.memory_block
    # Non-temporal memory has no valid_window segment
    # (verified by making sure the line for mem_no_time does not include the marker)
    lines = context.memory_block.splitlines()
    no_time_line = next(
        line for line in lines if "User prefers direct answers." in line
    )
    no_time_header = lines[lines.index(no_time_line) - 1]
    assert "valid_window:" not in no_time_header


def test_event_triggered_memory_renders_event_time_before_source_window() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_event",
                final_score=0.9,
                canonical_text="Melanie celebrated her daughter's birthday last night with a concert.",
                valid_from="2023-08-13T00:00:00+00:00",
                valid_to="2023-08-14T14:24:00+00:00",
                temporal_type="event_triggered",
                payload_json={
                    "source_message_window_start_occurred_at": "2023-08-14T14:24:00+00:00",
                    "source_message_window_end_occurred_at": "2023-08-14T14:24:00+00:00",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(500),
        conversation_messages=[],
    )

    header = context.memory_block.splitlines()[1]
    assert (
        "event_time: 2023-08-13T00:00:00+00:00 to 2023-08-14T14:24:00+00:00" in header
    )
    assert (
        "source_window: 2023-08-14T14:24:00+00:00 to 2023-08-14T14:24:00+00:00"
        in header
    )
    assert header.index("event_time:") < header.index("source_window:")


def test_resolved_date_is_rendered_in_memory_metadata() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_resolved",
                final_score=0.9,
                canonical_text="Caroline attended the conference on a Saturday.",
                resolved_date="2024-06-15",
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert "resolved_date: 2024-06-15" in context.memory_block


def test_exact_recall_memory_includes_source_quote_from_source_message() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_badge_expiry",
                final_score=0.9,
                canonical_text="Nia's observatory access badge expired.",
                payload_json={
                    "source_message_ids": ["msg_1"],
                    "source_message_window_start_occurred_at": "2024-02-12T08:40:00+00:00",
                    "source_message_window_end_occurred_at": "2024-02-12T08:40:00+00:00",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "seq": 2,
                "text": (
                    "Nia: My observatory access badge expired yesterday, so I "
                    "requested a replacement."
                ),
                "occurred_at": "2024-02-12T08:40:00+00:00",
            }
        ],
        query_type="temporal",
        exact_recall_mode=True,
    )

    assert (
        "source_window: 2024-02-12T08:40:00+00:00 to 2024-02-12T08:40:00+00:00"
        in context.memory_block
    )
    assert (
        "source_quote: user @ 2024-02-12T08:40:00+00:00 seq 2: "
        "Nia: My observatory access badge expired yesterday"
        in context.memory_block
    )


def test_memory_entry_prefers_evidence_packet_quotes_when_hydrated() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_kiln_setting",
                final_score=0.9,
                canonical_text="Mira's preferred kiln setting is cone six.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "speaker_relation_to_subject": "self_report",
                        "confidence": 0.91,
                        "rationale": "Mira answers Theo's kiln-setting question.",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "Cone six gives the glaze finish I want.",
                                "seq": 2,
                                "occurred_at": "2023-01-20T16:04:00+00:00",
                                "metadata_json": {"message_role": "user"},
                            },
                            {
                                "span_role": "trigger",
                                "quote_text": "Which kiln setting do you prefer?",
                                "seq": 1,
                                "occurred_at": "2023-01-20T16:03:00+00:00",
                                "metadata_json": {"message_role": "assistant"},
                            },
                        ],
                    }
                ],
                payload_json={"source_message_ids": ["msg_1"]},
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(700),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "seq": 2,
                "text": "This fallback source quote should not render.",
                "occurred_at": "2023-01-20T16:04:00+00:00",
            }
        ],
        query_type="slot_fill",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=False,
    )

    assert "evidence_packet: support: contextual_direct" in context.memory_block
    assert (
        "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 2: "
        "Cone six gives the glaze finish I want."
    ) in context.memory_block
    assert (
        "trigger_quote: assistant @ 2023-01-20T16:03:00+00:00 seq 1: "
        "Which kiln setting do you prefer?"
    ) in context.memory_block
    assert "fallback source quote" not in context.memory_block


def test_source_quote_options_scale_with_large_context_budget() -> None:
    default_options = ContextComposer._source_quote_options(
        query_type="temporal",
        exact_recall_mode=True,
    )
    expanded_options = ContextComposer._source_quote_options(
        query_type="temporal",
        exact_recall_mode=True,
        context_budget_tokens=32_000,
    )

    assert expanded_options.max_entries == default_options.max_entries + 2
    assert expanded_options.max_messages == 4
    assert expanded_options.max_chars > default_options.max_chars
    assert expanded_options.max_message_chars > default_options.max_message_chars


def test_exact_recall_keeps_compact_source_quote_when_full_quote_exceeds_budget() -> (
    None
):
    composer = _composer()
    candidate = _candidate(
        "mem_job_loss",
        final_score=0.9,
        canonical_text="Jon left his banker job to start a business.",
        payload_json={"source_message_ids": ["msg_1", "msg_2", "msg_3"]},
    )
    source_messages = [
        {
            "id": f"msg_{index}",
            "role": "user",
            "seq": index,
            "occurred_at": "2023-01-20T16:04:00+00:00",
            "text": (
                "Jon: Lost my job as a banker yesterday, so I'm gonna start my own "
                "business. This extra wording is deliberately long enough to make "
                "the full three-message source quote too expensive for this test."
            ),
        }
        for index in range(1, 4)
    ]
    source_messages_by_id = {str(message["id"]): message for message in source_messages}
    contract_block = ContextComposer.render_contract_block(_contract(), _resolved_policy(600))
    compact_options = composer._compact_source_quote_options(
        composer._source_quote_options(query_type="temporal", exact_recall_mode=True)
    )
    compact_block = composer._format_memory_entry(
        1,
        candidate,
        source_messages_by_id=source_messages_by_id,
        source_quote_options=compact_options,
    )
    full_block = composer._format_memory_entry(
        1,
        candidate,
        source_messages_by_id=source_messages_by_id,
        source_quote_options=composer._source_quote_options(
            query_type="temporal", exact_recall_mode=True
        ),
    )
    budget = (
        composer.estimate_tokens(contract_block)
        + composer.estimate_tokens("[Retrieved Memories]\n")
        + composer.estimate_tokens(compact_block)
    )
    assert composer.estimate_tokens(full_block) > composer.estimate_tokens(
        compact_block
    )

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(budget),
        conversation_messages=source_messages,
        query_type="temporal",
        exact_recall_mode=True,
    )

    assert (
        "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 1: Jon: Lost my job as a banker yesterday"
        in context.memory_block
    )
    assert "seq 2:" not in context.memory_block


def test_exact_recall_source_quote_snips_around_query_terms() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_public_name",
                final_score=0.9,
                canonical_text="The user's public name is Núria Pau.",
                payload_json={"source_message_ids": ["msg_1"]},
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(700),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "seq": 3,
                "text": (
                    "This opening filler is intentionally long enough that a "
                    "plain prefix quote would hide the important part. "
                    "I was checking whether voice transcription arrived, then "
                    "I said my name is Núria Pau, P-A-U, como paz en catalán."
                ),
                "occurred_at": "2026-01-09T22:09:25+00:00",
            }
        ],
        query_text="¿Dije explícitamente que Pau significa paz en catalán?",
        query_type="slot_fill",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=False,
    )

    assert (
        "source_quote: user @ 2026-01-09T22:09:25+00:00 seq 3:"
        in context.memory_block
    )
    assert "paz en catalán" in context.memory_block


def test_exact_recall_prefers_direct_user_source_over_assistant_echo() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_assistant_echo",
                final_score=0.95,
                canonical_text="The user's public name is Núria Pau.",
                payload_json={"source_message_ids": ["msg_assistant"]},
            ),
            _candidate(
                "mem_user_direct",
                final_score=0.94,
                canonical_text="The user said Núria Pau means peace in Catalan.",
                payload_json={"source_message_ids": ["msg_user"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(900),
        conversation_messages=[
            {
                "id": "msg_assistant",
                "role": "assistant",
                "seq": 4,
                "text": "Núria Pau, nice entrepreneurial name.",
                "occurred_at": "2026-01-09T22:09:26+00:00",
            },
            {
                "id": "msg_user",
                "role": "user",
                "seq": 3,
                "text": "Me llamo Núria Pau, P-A-U, como paz en catalán.",
                "occurred_at": "2026-01-09T22:09:25+00:00",
            },
        ],
        query_text="¿Qué significa Pau?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids[:2] == ["mem_user_direct", "mem_assistant_echo"]
    assert context.memory_block.index("peace in Catalan") < context.memory_block.index(
        "public name is Núria Pau"
    )


def test_temporal_source_quotes_are_limited_to_top_ranked_entries() -> None:
    composer = _composer()
    policy = _resolved_policy(1600).model_copy(
        update={
            "retrieval_params": _resolved_policy(1600).retrieval_params.model_copy(
                update={"final_context_items": 5}
            )
        }
    )
    candidates = [
        _candidate(
            f"mem_event_{index}",
            final_score=1.0 - (index * 0.01),
            canonical_text=f"Caroline attended event {index} yesterday.",
            payload_json={"source_message_ids": [f"msg_{index}"]},
            valid_from="2023-05-07T00:00:00+00:00",
            temporal_type="event_triggered",
        )
        for index in range(1, 6)
    ]
    messages = [
        {
            "id": f"msg_{index}",
            "role": "user",
            "seq": index,
            "occurred_at": "2023-05-08T13:56:00+00:00",
            "text": f"Caroline: I attended event {index} yesterday.",
        }
        for index in range(1, 6)
    ]

    context = composer.compose(
        scored_candidates=candidates,
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=messages,
        query_text="When did Caroline attend the event?",
        query_type="temporal",
        exact_recall_mode=True,
    )

    assert context.memory_block.count("source_quote:") == 4
    assert "mem_event_5" in context.selected_memory_ids


def test_default_source_quote_renders_for_non_exact_queries() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_job_loss",
                final_score=0.9,
                canonical_text="Jon is no longer in a secure banker job.",
                payload_json={"source_message_ids": ["msg_1"]},
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "seq": 4,
                "occurred_at": "2023-01-20T16:04:00+00:00",
                "text": "Jon: Lost my job as a banker yesterday.",
            }
        ],
    )

    assert "mem_job_loss" in context.selected_memory_ids
    assert (
        "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 4: Jon: Lost my job as a banker yesterday."
        in context.memory_block
    )


def test_source_quote_with_zero_query_token_overlap_is_not_vetoed() -> None:
    composer = _composer()
    source_text = "Liora packed vellum logbooks before sunrise."
    query_text = "Which tool preference should be remembered?"

    assert ContextComposer._quote_query_relevance(source_text, query_text) == 0.0

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_source_backed",
                final_score=0.9,
                canonical_text="The retained item is backed by a source message.",
                payload_json={"source_message_ids": ["msg_zero_overlap"]},
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "confidence": 0.91,
                        "rationale": "The support edge points at the retained item.",
                    }
                ],
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(700),
        conversation_messages=[
            {
                "id": "msg_zero_overlap",
                "role": "user",
                "seq": 7,
                "occurred_at": "2026-02-14T09:30:00+00:00",
                "text": source_text,
            }
        ],
        query_text=query_text,
        query_type="default",
    )

    assert "evidence_packet: support: contextual_direct" in context.memory_block
    assert (
        "source_quote: user @ 2026-02-14T09:30:00+00:00 seq 7: "
        "Liora packed vellum logbooks before sunrise."
    ) in context.memory_block


def test_default_source_quote_drops_under_budget_without_dropping_memory() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_budget_source",
        final_score=0.9,
        canonical_text="The user keeps a compact travel planning note.",
        payload_json={"source_message_ids": ["msg_budget"]},
    )
    source_messages = [
        {
            "id": "msg_budget",
            "role": "user",
            "seq": 3,
            "occurred_at": "2026-03-01T08:00:00+00:00",
            "text": (
                "I keep a compact travel planning note with flight windows, "
                "hotel preferences, packing reminders, and the one thing I "
                "must check before leaving for the airport."
            ),
        }
    ]
    source_messages_by_id = {str(message["id"]): message for message in source_messages}
    bare_block = composer._format_memory_entry(1, candidate)
    quoted_block = composer._format_memory_entry(
        1,
        candidate,
        source_messages_by_id=source_messages_by_id,
        source_quote_options=composer._source_quote_options(
            query_type="default",
            exact_recall_mode=False,
        ),
        query_text="What should I remember about travel planning?",
    )
    assert composer.estimate_tokens(quoted_block) > composer.estimate_tokens(bare_block)

    policy = _resolved_policy(800)
    contract_tokens = composer.estimate_tokens(
        composer.render_contract_block({}, policy)
    )
    memory_header_tokens = composer.estimate_tokens("[Retrieved Memories]\n")
    bare_tokens = composer.estimate_tokens(bare_block)
    budget = contract_tokens + memory_header_tokens + bare_tokens

    context = composer.compose(
        scored_candidates=[candidate],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(budget),
        conversation_messages=source_messages,
        query_text="What should I remember about travel planning?",
        query_type="default",
    )

    assert context.selected_memory_ids == ["mem_budget_source"]
    assert "The user keeps a compact travel planning note." in context.memory_block
    assert "source_quote:" not in context.memory_block


def test_source_quote_never_evicts_a_later_memory_slot() -> None:
    """An earlier entry's source quote must never displace a later entry's slot.

    Invariant: enabling source quotes must not change the SET of selected
    memory entries relative to the same composition with quotes disabled.
    Quotes are funded strictly from budget left over after every bare
    admission. Here both candidates fit bare within the budget, but candidate
    #1's quote alone would consume the whole memory region; the quote must drop
    (or compact) rather than evict candidate #2.
    """
    composer = _composer()
    first = _candidate(
        "mem_a",
        final_score=0.95,
        canonical_text="Caroline planned a trip to Lisbon for the holidays.",
        payload_json={"source_message_ids": ["msg_a"]},
    )
    second = _candidate(
        "mem_b",
        final_score=0.90,
        canonical_text="Caroline also booked a museum tour.",
        payload_json={"source_message_ids": ["msg_b"]},
    )
    source_messages = [
        {
            "id": "msg_a",
            "role": "user",
            "seq": 1,
            "occurred_at": "2026-03-01T08:00:00+00:00",
            "text": (
                "I planned a long trip to Lisbon for the winter holidays with "
                "my whole family and a packed museum itinerary."
            ),
        },
        {
            "id": "msg_b",
            "role": "user",
            "seq": 2,
            "occurred_at": "2026-03-01T08:05:00+00:00",
            "text": "I also booked a museum tour.",
        },
    ]
    source_messages_by_id = {
        str(message["id"]): message for message in source_messages
    }
    query_text = "Where did Caroline plan to travel and what did she book?"

    bare_a = composer._format_memory_entry(1, first)
    bare_b = composer._format_memory_entry(2, second)
    quoted_a = composer._format_memory_entry(
        1,
        first,
        source_messages_by_id=source_messages_by_id,
        source_quote_options=composer._ranked_source_quote_options(
            composer._source_quote_options(
                query_type="default", exact_recall_mode=False
            ),
            rank=1,
        ),
        query_text=query_text,
    )
    bare_a_tokens = composer.estimate_tokens(bare_a)
    bare_b_tokens = composer.estimate_tokens(bare_b)
    quoted_a_tokens = composer.estimate_tokens(quoted_a)
    # Both bare entries fit together; candidate #1's quote alone fills the room.
    assert bare_a_tokens + bare_b_tokens < quoted_a_tokens

    policy = _resolved_policy(800)
    contract_tokens = composer.estimate_tokens(
        composer.render_contract_block({}, policy)
    )
    memory_header_tokens = composer.estimate_tokens("[Retrieved Memories]\n")
    # Memory region exactly holds candidate #1's quoted form. With quotes
    # disabled, both bare entries (bare_a + bare_b) fit in the same region.
    budget = contract_tokens + memory_header_tokens + quoted_a_tokens

    context = composer.compose(
        scored_candidates=[first, second],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(budget),
        conversation_messages=source_messages,
        query_text=query_text,
        query_type="default",
    )

    # Both memory slots survive; the quote did not evict the later entry.
    assert context.selected_memory_ids == ["mem_a", "mem_b"]
    assert "Caroline planned a trip to Lisbon for the holidays." in context.memory_block
    assert "Caroline also booked a museum tour." in context.memory_block
    assert context.items_included == 2
    assert context.items_dropped == 0
    assert context.total_tokens_estimate <= context.budget_tokens


def test_source_quote_respects_message_raw_inclusion_policy() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_sensitive_source",
                final_score=0.9,
                canonical_text="The retained memory is safe to show.",
                payload_json={"source_message_ids": ["msg_1"]},
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[
            {
                "id": "msg_1",
                "role": "user",
                "text": "Large skipped source text should not be mirrored.",
                "include_raw": 0,
            }
        ],
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert "The retained memory is safe to show." in context.memory_block
    assert "source_quote:" not in context.memory_block


def test_conversation_chunk_summary_includes_source_window_and_excerpt() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_chunk",
                final_score=0.9,
                canonical_text="Oren registered for a paper marbling workshop.",
                object_type="summary_view",
                scope="conversation",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_excerpt_messages": [
                        {
                            "role": "assistant",
                            "occurred_at": "2023-07-03T13:36:00",
                            "text": (
                                "Oren: I registered for a paper marbling workshop yesterday."
                            ),
                        }
                    ],
                    "source_message_window_start_occurred_at": "2023-07-03T13:36:00",
                    "source_message_window_end_occurred_at": "2023-07-03T13:36:00",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(500),
        conversation_messages=[],
    )

    assert (
        "source_window: 2023-07-03T13:36:00 to 2023-07-03T13:36:00"
        in context.memory_block
    )
    assert (
        "source_excerpt: assistant @ 2023-07-03T13:36:00: "
        "Oren: I registered for a paper marbling workshop yesterday."
        in context.memory_block
    )


def test_verbatim_evidence_search_candidate_includes_source_window() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "raw_cnv_1_1_2",
                final_score=0.9,
                canonical_text=(
                    "user: My calibration targets are cobalt and quartz\n"
                    "assistant: Noted."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_window_start_occurred_at": "2026-04-04T11:00:00+00:00",
                    "source_message_window_end_occurred_at": "2026-04-04T11:01:00+00:00",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(500),
        conversation_messages=[],
    )

    assert (
        "source_window: 2026-04-04T11:00:00+00:00 to 2026-04-04T11:01:00+00:00"
        in context.memory_block
    )


def test_evidence_obligation_reserves_literal_support_for_source_summary() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_episode",
                final_score=0.98,
                canonical_text="Ilya compared acoustic panel suppliers.",
                object_type="summary_view",
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["vew_conv_10_12"],
                },
            ),
            _candidate(
                "mem_distractor",
                final_score=0.96,
                canonical_text="Ilya discussed unrelated freight schedules.",
            ),
            _candidate(
                "vew_conv_10_12",
                final_score=0.52,
                canonical_text=(
                    "[user] Ilya: Comparing acoustic panel suppliers has been "
                    "on my agenda."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_10", "msg_11", "msg_12"],
                    "source_message_window_start_occurred_at": "2023-07-01T10:00:00",
                    "source_message_window_end_occurred_at": "2023-07-01T10:02:00",
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(800, 2),
        conversation_messages=[],
        query_text="Which suppliers did Ilya compare?",
        query_type="slot_fill",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids[0] == "vew_conv_10_12"
    assert "Comparing acoustic panel suppliers" in context.memory_block


def test_evidence_obligation_keeps_near_tie_literal_windows() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_conv_10_12",
                final_score=0.90,
                canonical_text="[user] The appointment was moved to Tuesday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_10", "msg_11", "msg_12"],
                    "source_message_window_start_occurred_at": "2023-07-01T10:00:00",
                    "source_message_window_end_occurred_at": "2023-07-01T10:02:00",
                },
            ),
            _candidate(
                "mem_distractor",
                final_score=0.89,
                canonical_text="A compact but unrelated scheduling preference.",
            ),
            _candidate(
                "vew_conv_10_12_dup",
                final_score=0.88,
                canonical_text="[user] The appointment was moved to Tuesday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_10", "msg_11", "msg_12"],
                    "source_message_window_start_occurred_at": "2023-07-01T10:00:00",
                    "source_message_window_end_occurred_at": "2023-07-01T10:02:00",
                },
            ),
            _candidate(
                "vew_conv_20_22",
                final_score=0.84,
                canonical_text="[assistant] Later they confirmed Tuesday morning.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_20", "msg_21", "msg_22"],
                    "source_message_window_start_occurred_at": "2023-07-02T09:00:00",
                    "source_message_window_end_occurred_at": "2023-07-02T09:02:00",
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(900, 2),
        conversation_messages=[],
        query_text="When was the appointment moved?",
        query_type="temporal",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["vew_conv_10_12", "vew_conv_20_22"]
    assert "unrelated scheduling preference" not in context.memory_block
    assert "vew_conv_10_12_dup" not in context.selected_memory_ids


def test_budgeted_marginal_honors_evidence_obligation_windows() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_conv_10_12",
                final_score=0.80,
                canonical_text=(
                    "[user] The appointment was moved to Tuesday after the "
                    "first plan fell through."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_10", "msg_11", "msg_12"],
                },
            ),
            _candidate(
                "mem_short",
                final_score=0.95,
                canonical_text="Short unrelated fact.",
            ),
            _candidate(
                "vew_conv_20_22",
                final_score=0.75,
                canonical_text="[assistant] They confirmed Tuesday morning later.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_20", "msg_21", "msg_22"],
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(900, 2),
        conversation_messages=[],
        query_text="When was the appointment moved?",
        query_type="temporal",
        composer_strategy="budgeted_marginal",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["vew_conv_10_12", "vew_conv_20_22"]
    assert "Short unrelated fact." not in context.memory_block


def test_source_required_list_reserves_unique_source_backed_item_over_summaries() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_paris_high",
                final_score=0.96,
                canonical_text="Caroline mentioned visiting Paris.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_paris"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            ),
            _candidate(
                "sum_paris_dup",
                final_score=0.94,
                canonical_text="Another summary says Caroline talked about Paris.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_paris"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            ),
            _candidate(
                "mem_rome_source",
                final_score=0.44,
                canonical_text="Caroline said Rome was also one of the cities.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_rome"],
                    "value_norm_key": "rome",
                    "value_text": "Rome",
                },
                llm_applicability=0.7,
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(900, 2),
        conversation_messages=[],
        query_text="Which cities did Caroline mention?",
        query_type="broad_list",
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert "mem_rome_source" in context.selected_memory_ids
    assert "sum_paris_dup" not in context.selected_memory_ids
    assert context.coverage_state == "complete"
    assert context.allowed_values == [
        {
            "display_text": "Rome",
            "normalized_key": "value|rome",
            "evidence_ids": [
                "memory:mem_rome_source",
                "message:msg_rome",
            ],
            "memory_ids": ["mem_rome_source"],
        }
    ]


def test_source_required_list_reserves_distinct_values_from_same_source() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_northglass",
                final_score=0.95,
                canonical_text="Rhea mentioned Northglass Station.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "northglass",
                    "value_text": "Northglass",
                },
            ),
            _candidate(
                "mem_unrelated",
                final_score=0.94,
                canonical_text="Rhea also discussed unrelated archive logistics.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_other"],
                    "value_norm_key": "logistics",
                    "value_text": "Logistics",
                },
            ),
            _candidate(
                "mem_ember_shoal",
                final_score=0.41,
                canonical_text="Rhea mentioned Ember Shoal in the same sentence.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "ember_shoal",
                    "value_text": "Ember Shoal",
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(900, 2),
        conversation_messages=[],
        query_text="Which field stations did Rhea mention?",
        query_type="broad_list",
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["mem_northglass", "mem_ember_shoal"]
    assert [item["display_text"] for item in context.allowed_values] == [
        "Northglass",
        "Ember Shoal",
    ]
    assert "Rhea also discussed unrelated archive logistics." not in context.memory_block


def test_source_coverage_reserve_seeds_groups_from_window_reservations() -> None:
    window = _candidate(
        "vew_combo",
        final_score=0.99,
        canonical_text=(
            "Rhea inspected Northglass, Ember Shoal, and Cloudbreak Stations."
        ),
        payload_json={
            "source_kind_variant": "conversation_window",
            "source_message_ids": ["msg_combo"],
        },
    )
    same_source_duplicate = _candidate(
        "mem_combo_duplicate",
        final_score=0.98,
        canonical_text="Rhea discussed the same inspection record.",
        payload_json={"source_message_ids": ["msg_combo"]},
    )
    northglass = _candidate(
        "mem_northglass",
        final_score=0.97,
        canonical_text="Rhea inspected Northglass Station.",
        payload_json={
            "source_message_ids": ["msg_northglass"],
            "value_norm_key": "northglass",
            "value_text": "Northglass",
        },
    )
    ember_shoal = _candidate(
        "mem_ember_shoal",
        final_score=0.96,
        canonical_text="Rhea inspected Ember Shoal Station.",
        payload_json={
            "source_message_ids": ["msg_ember_shoal"],
            "value_norm_key": "ember_shoal",
            "value_text": "Ember Shoal",
        },
    )
    cloudbreak = _candidate(
        "mem_cloudbreak",
        final_score=0.95,
        canonical_text="Rhea inspected Cloudbreak Station.",
        payload_json={
            "source_message_ids": ["msg_cloudbreak"],
            "value_norm_key": "cloudbreak",
            "value_text": "Cloudbreak",
        },
    )

    reserved = ContextComposer._evidence_obligation_candidates(
        [window, same_source_duplicate, northglass, ember_shoal, cloudbreak],
        max_items=4,
        query_type="slot_fill",
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
        exact_recall_mode=True,
        source_messages_by_id={},
    )

    assert [candidate.memory_id for candidate in reserved] == [
        "vew_combo",
        "mem_northglass",
        "mem_ember_shoal",
        "mem_cloudbreak",
    ]


def test_fact_facets_and_base_preserve_values_with_shared_quote_once() -> None:
    composer = _composer()
    shared_quote = "Caroline said she lived in Paris, then Rome."
    shared_packet = {
        "support_kind": "direct",
        "evidence_polarity": "supports",
        "spans": [
            {
                "span_role": "source",
                "message_id": "msg_combo",
                "quote_text": shared_quote,
            }
        ],
    }

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_city_base",
                final_score=0.96,
                canonical_text="Caroline said she lived in Paris and Rome.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_combo"],
                    "coverage_members": [
                        {"member_key": "paris", "display_text": "Paris"},
                        {"member_key": "rome", "display_text": "Rome"},
                    ],
                },
                evidence_packets=[shared_packet],
            ),
            _candidate(
                "mff_city_paris",
                final_score=0.95,
                canonical_text="Caroline / city: Paris",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "fact_facet",
                    "source_memory_ids": ["mem_city_base"],
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                    "fact_facet": {
                        "fact_id": "mff_city_paris",
                        "surface_class": "structured",
                    },
                },
                evidence_packets=[shared_packet],
            ),
            _candidate(
                "mff_city_rome",
                final_score=0.94,
                canonical_text="Caroline / city: Rome",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "fact_facet",
                    "source_memory_ids": ["mem_city_base"],
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "rome",
                    "value_text": "Rome",
                    "fact_facet": {
                        "fact_id": "mff_city_rome",
                        "surface_class": "structured",
                    },
                },
                evidence_packets=[shared_packet],
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(1000, 3),
        conversation_messages=[],
        query_text="Which cities did Caroline live in?",
        query_type="broad_list",
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        exact_recall_mode=True,
        enable_evidence_obligation_coverage=True,
        fact_facet_span_coadmission_enabled=True,
    )

    assert context.coverage_state == "complete"
    assert set(context.selected_memory_ids) == {
        "mem_city_base",
        "mff_city_paris",
        "mff_city_rome",
    }
    assert {"Paris", "Rome"}.issubset(
        {item["display_text"] for item in context.allowed_values}
    )
    assert context.memory_block.count(shared_quote) == 1


def test_raw_context_shape_reaches_source_coverage_reserve() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_high",
                final_score=0.95,
                canonical_text="A summary mentions there is raw context elsewhere.",
                object_type="summary_view",
                payload_json={"source_message_ids": ["msg_raw"]},
            ),
            _candidate(
                "mem_raw_source",
                final_score=0.42,
                canonical_text="Verbatim raw context that should be exposed.",
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_raw"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(700, 1),
        conversation_messages=[],
        query_text="Show the raw context.",
        query_type="default",
        answer_shape="raw_context",
        coverage_mode="top_support",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["mem_raw_source"]
    assert "Verbatim raw context that should be exposed." in context.memory_block


def test_source_required_summary_only_support_is_insufficient() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_origin_port",
                final_score=0.95,
                canonical_text="Episode summary says Vela transferred from Arbor Bay.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_origin_port"],
                    "value_norm_key": "arbor_bay",
                    "value_text": "Arbor Bay",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(700, 1),
        conversation_messages=[],
        query_text="Which port did Vela transfer from?",
        query_type="slot_fill",
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["sum_origin_port"]
    assert context.coverage_state == "insufficient"
    assert context.allowed_values == []
    assert context.support_map == {}


def test_evidence_reserve_does_not_affect_open_domain_questions() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_paris_high",
                final_score=0.96,
                canonical_text="Caroline mentioned visiting Paris.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_paris"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            ),
            _candidate(
                "mem_rome_source",
                final_score=0.44,
                canonical_text="Caroline said Rome was also one of the cities.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_rome"],
                    "value_norm_key": "rome",
                    "value_text": "Rome",
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(700, 1),
        conversation_messages=[],
        query_text="What do you remember about Caroline's travel?",
        query_type="default",
        answer_shape="open_domain",
        coverage_mode="top_support",
        source_precision="preferred",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["sum_paris_high"]
    assert context.coverage_state == "unknown"


def test_empty_canonical_text_candidates_are_filtered_out() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate("mem_blank", final_score=0.99, canonical_text="   "),
            _candidate(
                "mem_good", final_score=0.50, canonical_text="Useful retained memory."
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(300),
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_good"]
    assert "Useful retained memory." in context.memory_block
    assert context.items_included + context.items_dropped == 1


def test_state_block_reserves_budget_before_memory_selection() -> None:
    composer = _composer()
    candidate = _candidate(
        "mem_1", final_score=0.9, canonical_text="Short memory that otherwise fits."
    )
    no_state = composer.compose(
        scored_candidates=[candidate],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )
    tight_budget = no_state.total_tokens_estimate

    with_state = composer.compose(
        scored_candidates=[candidate],
        current_contract=_contract(),
        user_state={"urgency": "high", "active_topics": ["websocket", "fastapi"]},
        resolved_policy=_resolved_policy(tight_budget),
        conversation_messages=[],
    )

    assert with_state.state_block.startswith("[Current User State]")
    assert with_state.selected_memory_ids == []
    assert with_state.total_tokens_estimate <= with_state.budget_tokens


def test_token_estimation_is_reasonable_and_counts_balance() -> None:
    composer = _composer()
    assert composer.estimate_tokens("abcd") == 1
    assert composer.estimate_tokens("abcde") == 2

    context = composer.compose(
        scored_candidates=[
            _candidate("mem_1", final_score=0.9, canonical_text="One."),
            _candidate("mem_2", final_score=0.8, canonical_text="Two."),
            _candidate("mem_3", final_score=0.7, canonical_text="Three."),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(200),
        conversation_messages=[],
    )

    assert context.total_tokens_estimate >= composer.estimate_tokens(
        context.contract_block
    )
    assert context.items_included + context.items_dropped == 3


def test_final_context_items_caps_selected_memories_even_when_budget_allows_more() -> (
    None
):
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 2}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate("mem_1", final_score=0.9, canonical_text="One."),
            _candidate("mem_2", final_score=0.8, canonical_text="Two."),
            _candidate("mem_3", final_score=0.7, canonical_text="Three."),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_1", "mem_2"]
    assert "Three." not in context.memory_block
    assert context.items_included == 2
    assert context.items_dropped == 1


def test_workspace_rollup_appears_as_dedicated_block_before_memories() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate("mem_1", final_score=0.9, canonical_text="Useful memory.")
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
        workspace_rollup={"summary_text": "This workspace prefers incremental fixes."},
    )

    assert context.workspace_block.startswith("[Workspace Context]")
    assert "This workspace prefers incremental fixes." in context.workspace_block
    assert context.memory_block.startswith("[Retrieved Memories]")


def test_workspace_rollup_respects_eight_percent_budget_with_truncation() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(100),
        conversation_messages=[],
        workspace_rollup={"summary_text": "x" * 200},
    )

    assert context.workspace_block.startswith("[Workspace Context]")
    assert composer.estimate_tokens(context.workspace_block) <= 8
    assert context.workspace_block.endswith("...")


def test_workspace_rollup_none_keeps_workspace_block_empty() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(200),
        conversation_messages=[],
        workspace_rollup=None,
    )

    assert context.workspace_block == ""


def test_hierarchical_summary_reserves_budget_for_supporting_l0_memory() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_theme",
                final_score=0.99,
                canonical_text="Thematic profile summary.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "thematic_profile",
                    "hierarchy_level": 2,
                    "source_object_ids": ["mem_support"],
                    "source_claim_signatures": [],
                },
                updated_at="2026-03-30T12:00:00+00:00",
            ),
            _candidate(
                "mem_support",
                final_score=0.4,
                canonical_text="Supporting atomic belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "patch_first",
                },
                updated_at="2026-03-30T11:00:00+00:00",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_support", "sum_mem_theme"]
    assert "Supporting atomic belief." in context.memory_block
    assert "Thematic profile summary." in context.memory_block


def test_budgeted_marginal_hierarchical_summary_uses_incremental_support_cost() -> None:
    composer = _composer()
    support = _candidate(
        "mem_support",
        final_score=0.82,
        canonical_text="Compact supporting fact.",
        object_type="belief",
        scope="global_user",
        payload_json={
            "claim_key": "workflow.debugging.style",
            "claim_value": "patch_first",
        },
    )
    summary = _candidate(
        "sum_mem_theme",
        final_score=0.78,
        canonical_text="Thematic profile summary grounded by support.",
        object_type="summary_view",
        scope="global_user",
        payload_json={
            "summary_kind": "thematic_profile",
            "hierarchy_level": 2,
            "source_object_ids": ["mem_support"],
            "source_claim_signatures": [],
        },
    )

    context = composer.compose(
        scored_candidates=[summary, support],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(500),
        conversation_messages=[],
        composer_strategy="budgeted_marginal",
    )

    assert context.selected_memory_ids == ["mem_support", "sum_mem_theme"]
    assert context.memory_block.count("Compact supporting fact.") == 1
    assert "Thematic profile summary grounded by support." in context.memory_block


def test_conflicting_fresher_l0_demotes_hierarchical_summary() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_theme",
                final_score=0.95,
                canonical_text="Outdated thematic profile summary.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "thematic_profile",
                    "hierarchy_level": 2,
                    "source_object_ids": ["mem_old"],
                    "source_claim_signatures": [
                        {
                            "claim_key": "workflow.debugging.style",
                            "claim_value": "patch_first",
                        }
                    ],
                },
                updated_at="2026-03-30T10:00:00+00:00",
            ),
            _candidate(
                "mem_old",
                final_score=0.5,
                canonical_text="Older supporting belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "patch_first",
                },
                updated_at="2026-03-30T09:00:00+00:00",
            ),
            _candidate(
                "mem_fresh",
                final_score=0.45,
                canonical_text="Fresher contradictory belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "investigate_breadth_first",
                },
                updated_at="2026-03-30T12:00:00+00:00",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert "Outdated thematic profile summary." not in context.memory_block
    assert "Fresher contradictory belief." in context.memory_block


def test_conflicting_fresher_fact_facet_uses_span_coadmission_flag() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_theme",
                final_score=0.95,
                canonical_text="Outdated thematic profile summary.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "thematic_profile",
                    "hierarchy_level": 2,
                    "source_object_ids": ["mem_old"],
                    "source_claim_signatures": [
                        {
                            "claim_key": "workflow.debugging.style",
                            "claim_value": "patch_first",
                        }
                    ],
                },
                updated_at="2026-03-30T10:00:00+00:00",
            ),
            _candidate(
                "mem_old",
                final_score=0.5,
                canonical_text="Older supporting belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "patch_first",
                },
                updated_at="2026-03-30T09:00:00+00:00",
            ),
            _candidate(
                "mff_fresh_debug_style",
                final_score=0.05,
                canonical_text="workflow.debugging.style: investigate_breadth_first",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "investigate_breadth_first",
                    "source_kind_variant": "fact_facet",
                    "fact_facet": {
                        "fact_id": "mff_fresh_debug_style",
                        "surface_class": "structured",
                    },
                },
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": (
                                    "I now debug by investigating breadth first."
                                ),
                            }
                        ],
                    }
                ],
                updated_at="2026-03-30T12:00:00+00:00",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(600),
        conversation_messages=[],
        fact_facet_span_coadmission_enabled=True,
    )

    assert "source_span: I now debug by investigating breadth first." in context.memory_block
    assert "fact_facet_span_coadmitted: true" in context.memory_block
    assert (
        "fact_facet_pointer: workflow.debugging.style: investigate_breadth_first"
        in context.memory_block
    )


def test_budgeted_marginal_blocks_stale_summary_after_conflicting_l0_selected() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_theme",
                final_score=0.95,
                canonical_text="Outdated thematic profile summary.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "thematic_profile",
                    "hierarchy_level": 2,
                    "source_object_ids": ["mem_old"],
                    "source_claim_signatures": [
                        {
                            "claim_key": "workflow.debugging.style",
                            "claim_value": "patch_first",
                        }
                    ],
                },
                updated_at="2026-03-30T10:00:00+00:00",
            ),
            _candidate(
                "mem_old",
                final_score=0.3,
                canonical_text="Older supporting belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "patch_first",
                },
                updated_at="2026-03-30T09:00:00+00:00",
            ),
            _candidate(
                "mem_fresh",
                final_score=0.86,
                canonical_text="Fresher contradictory belief.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "investigate_breadth_first",
                },
                updated_at="2026-03-30T12:00:00+00:00",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(500),
        conversation_messages=[],
        composer_strategy="budgeted_marginal",
    )

    assert "mem_fresh" in context.selected_memory_ids
    assert "sum_mem_theme" not in context.selected_memory_ids
    assert "Outdated thematic profile summary." not in context.memory_block


def test_thematic_profile_can_ground_through_episode_to_nested_l0_support() -> None:
    composer = _composer()
    policy = _resolved_policy(400).model_copy(
        update={
            "retrieval_params": _resolved_policy(400).retrieval_params.model_copy(
                update={"final_context_items": 2}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_mem_theme",
                final_score=0.95,
                canonical_text="Theme derived from an episode only.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "thematic_profile",
                    "hierarchy_level": 2,
                    "source_object_ids": ["sum_mem_episode"],
                    "source_claim_signatures": [],
                },
                updated_at="2026-03-30T12:00:00+00:00",
            ),
            _candidate(
                "sum_mem_episode",
                final_score=0.7,
                canonical_text="Episode carrying the real support.",
                object_type="summary_view",
                scope="global_user",
                payload_json={
                    "summary_kind": "episode",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_support"],
                    "source_claim_signatures": [],
                },
                updated_at="2026-03-30T11:00:00+00:00",
            ),
            _candidate(
                "mem_support",
                final_score=0.4,
                canonical_text="Nested atomic support.",
                object_type="belief",
                scope="global_user",
                payload_json={
                    "claim_key": "workflow.debugging.style",
                    "claim_value": "patch_first",
                },
                updated_at="2026-03-30T10:00:00+00:00",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
    )

    assert context.selected_memory_ids == ["mem_support", "sum_mem_theme"]
    assert "Nested atomic support." in context.memory_block
    assert "Theme derived from an episode only." in context.memory_block


def test_broad_query_admission_protects_top_k_rank() -> None:
    # CS-2.2: the diversity reranker may no longer displace a top-K
    # (pre-diversity rank) candidate with a lower-ranked one. The redundant but
    # higher-ranked "busy with apprentices" summary keeps its seat; the lower-ranked
    # "outdoors" summary is dropped. Production broad-list COVERAGE comes from the
    # evidence-obligation path (disabled in this composer-only unit test), not
    # from letting a lower-ranked item outrank a higher-ranked one.
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_clockwork",
                final_score=0.91,
                canonical_text=(
                    "Orin's apprentices crowded around the clockwork exhibit "
                    "at the gallery."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_busy",
                final_score=0.90,
                canonical_text="Orin is busy with the apprentices.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_glasswork",
                final_score=0.87,
                canonical_text=(
                    "Orin took the apprentices to a glassworking studio where they "
                    "enjoyed shaping bright glass."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_nature",
                final_score=0.86,
                canonical_text=(
                    "Orin's workshop group enjoys stargazing, sketching, and working "
                    "outdoors together."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_canoe_practice",
                final_score=0.88,
                canonical_text="Orin practiced canoeing with the apprentices.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="What do Orin's apprentices like?",
        query_type="broad_list",
    )

    assert set(context.selected_memory_ids) == {
        "mem_clockwork",
        "mem_busy",
        "mem_canoe_practice",
        "mem_glasswork",
    }
    assert "busy with the apprentices" in context.memory_block
    assert "mem_nature" not in context.selected_memory_ids


def test_broad_query_admission_keeps_higher_ranked_shared_source_carrier() -> None:
    # CS-2.2: the composer no longer drops a higher-ranked carrier that shares a
    # source message with another top-K carrier -- both top-2 carriers are
    # admitted and the lower-ranked distinct one is dropped by the item cap. In
    # production the CS-2.3 fusion dedupe collapses same-source carriers BEFORE
    # the composer, so this backstop no longer needs to reorder by source.
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 2}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_first_glasswork",
                final_score=0.94,
                canonical_text="Orin's apprentices loved glassworking at the studio.",
                payload_json={"source_message_ids": ["msg_glasswork"]},
            ),
            _candidate(
                "mem_duplicate_glasswork",
                final_score=0.93,
                canonical_text="The glassworking studio delighted Orin's apprentices.",
                payload_json={"source_message_ids": ["msg_glasswork"]},
            ),
            _candidate(
                "mem_canoe_practice",
                final_score=0.82,
                canonical_text="Orin's apprentices also enjoyed canoe practice.",
                payload_json={"source_message_ids": ["msg_canoe_practice"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which activities did Orin's apprentices enjoy?",
        query_type="broad_list",
    )

    assert context.selected_memory_ids == [
        "mem_first_glasswork",
        "mem_duplicate_glasswork",
    ]
    assert "mem_canoe_practice" not in context.selected_memory_ids


def test_broad_query_admission_fills_top_k_by_rank() -> None:
    # CS-2.2: the top-K seats go to the highest pre-diversity ranks in order; a
    # lower-ranked item cannot be promoted ahead of a higher-ranked one for
    # coverage. mem_generic_survey (rank 3) keeps its seat over mem_cavern (rank 4).
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 3}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_plateau",
                final_score=0.94,
                canonical_text="Orin catalogued fossils on a desert plateau.",
                payload_json={"source_message_ids": ["msg_plateau"]},
            ),
            _candidate(
                "mem_tidal_flats",
                final_score=0.92,
                canonical_text="Orin catalogued fossils beside the tidal flats.",
                payload_json={"source_message_ids": ["msg_tidal_flats"]},
            ),
            _candidate(
                "mem_generic_survey",
                final_score=0.88,
                canonical_text="Orin catalogued fossils during another survey.",
                payload_json={"source_message_ids": ["msg_plateau"]},
            ),
            _candidate(
                "mem_cavern",
                final_score=0.85,
                canonical_text="Orin catalogued fossils inside a limestone cavern.",
                payload_json={"source_message_ids": ["msg_cavern"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Where has Orin catalogued fossils?",
        query_type="broad_list",
    )

    assert set(context.selected_memory_ids) == {
        "mem_plateau",
        "mem_tidal_flats",
        "mem_generic_survey",
    }
    assert "mem_cavern" not in context.selected_memory_ids


def test_broad_query_selection_handles_unicode_tokens_mechanically() -> None:
    composer = _composer()
    policy = _resolved_policy(700).model_copy(
        update={
            "retrieval_params": _resolved_policy(700).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_automatas",
                final_score=0.91,
                canonical_text=(
                    "Élodie's apprentices crowded around the automaton exhibit "
                    "at the gallery."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_encargos",
                final_score=0.90,
                canonical_text="Élodie balances apprentice training with commissions.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_vidrio",
                final_score=0.87,
                canonical_text=(
                    "Élodie took the apprentices to a glassworking studio to shape "
                    "colored glass."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_cartografia",
                final_score=0.86,
                canonical_text=(
                    "The workshop group practices sky surveys, botanical sketches, "
                    "and outdoor mapping."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_canoa",
                final_score=0.88,
                canonical_text="Élodie joined a canoe practice with the apprentices.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="¿Qué disfrutan los aprendices de Élodie?",
        query_type="broad_list",
    )
    default_context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_automatas",
                final_score=0.91,
                canonical_text=(
                    "Élodie's apprentices crowded around the automaton exhibit "
                    "at the gallery."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_encargos",
                final_score=0.90,
                canonical_text="Élodie balances apprentice training with commissions.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_vidrio",
                final_score=0.87,
                canonical_text=(
                    "Élodie took the apprentices to a glassworking studio to shape "
                    "colored glass."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_cartografia",
                final_score=0.86,
                canonical_text=(
                    "The workshop group practices sky surveys, botanical sketches, "
                    "and outdoor mapping."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_canoa",
                final_score=0.88,
                canonical_text="Élodie joined a canoe practice with the apprentices.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="¿Qué disfrutan los aprendices de Élodie?",
        query_type="default",
    )

    # CS-2.2: broad-list top-K now follows pre-diversity rank (unicode tokens are
    # still folded mechanically for scoring), so it matches the default arm --
    # the higher-ranked "encargos" summary keeps its seat over lower-ranked
    # "cartografia". Diversity no longer displaces a higher-ranked top-K item.
    assert set(context.selected_memory_ids) == {
        "mem_automatas",
        "mem_encargos",
        "mem_vidrio",
        "mem_canoa",
    }
    assert "mem_cartografia" not in context.selected_memory_ids
    assert set(default_context.selected_memory_ids) == {
        "mem_automatas",
        "mem_encargos",
        "mem_vidrio",
        "mem_canoa",
    }


def _window_candidate(
    memory_id: str,
    *,
    final_score: float,
    canonical_text: str | None = None,
) -> ScoredCandidate:
    """A token-heavy verbatim conversation window (CS-2.2 verbatim class)."""
    return _candidate(
        memory_id,
        final_score=final_score,
        canonical_text=(
            canonical_text
            if canonical_text is not None
            else "User: " + ("blah " * 40) + " Assistant: " + ("ctx " * 40)
        ),
        object_type="evidence",
        payload_json={
            "source_kind_variant": "conversation_window",
            "source_message_ids": [memory_id + "_m"],
        },
    )


def _compose_with_pre_cs22_selection_order(composer: ContextComposer, **kwargs):
    """Run compose() under the pre-CS-2.2 admission order (counterfactual).

    Reverts `_rank_respecting_selection_order` to the plain diversity reranker
    over ALL candidates -- the exact pre-CS-2.2 behavior. Regression fixtures
    use this to prove they are load-bearing: the same scenario must LOSE the
    gold under the old order and keep it under the rank-respecting one.
    """
    with mock.patch.object(
        ContextComposer,
        "_rank_respecting_selection_order",
        staticmethod(
            lambda candidates, **order_kwargs: ContextComposer._selection_order(
                candidates,
                **order_kwargs,
            )
        ),
    ):
        return composer.compose(**kwargs)


def test_rank_respecting_order_protects_top_k_and_reranks_remainder() -> None:
    # CS-2.2: the pre-diversity top-K keep their exact rank order; only the
    # rank>K remainder is handed to the diversity reranker.
    cands = [
        _candidate("m1", final_score=0.95, canonical_text="Alpha fact one.",
                   object_type="summary_view"),
        _candidate("m2", final_score=0.90, canonical_text="Beta fact two.",
                   object_type="summary_view"),
        _candidate("m3", final_score=0.85,
                   canonical_text="Gamma fact about mineral catalogues and telescope logs.",
                   object_type="summary_view"),
        _candidate("m4", final_score=0.80,
                   canonical_text="Delta fact about loom maintenance schedules.",
                   object_type="summary_view"),
    ]
    order = ContextComposer._rank_respecting_selection_order(
        list(cands),
        max_items=2,
        query_text="which workshop records matter",
        query_type="broad_list",
        exact_recall_mode=False,
        source_messages_by_id={},
    )
    assert [candidate.memory_id for candidate in order[:2]] == ["m1", "m2"]
    assert {candidate.memory_id for candidate in order[2:]} == {"m3", "m4"}


def test_selection_order_still_penalizes_redundant_candidates() -> None:
    # The diversity reranker still demotes a higher-scored redundant candidate
    # below a lower-scored diverse one -- it just now governs the remainder only.
    cands = [
        _candidate("mem_clockwork", final_score=0.91,
                   canonical_text="Orin's apprentices crowded around the clockwork exhibit at the gallery.",
                   object_type="summary_view"),
        _candidate("mem_busy", final_score=0.90,
                   canonical_text="Orin is busy with the apprentices.",
                   object_type="summary_view"),
        _candidate("mem_glasswork", final_score=0.87,
                   canonical_text="Orin took the apprentices to a glassworking studio where they enjoyed shaping bright glass.",
                   object_type="summary_view"),
        _candidate("mem_nature", final_score=0.86,
                   canonical_text="Sky surveys, botanical sketches, and outdoor mapping fill the workshop weekends.",
                   object_type="summary_view"),
        _candidate("mem_canoe_practice", final_score=0.88,
                   canonical_text="Orin practiced canoeing with the apprentices.",
                   object_type="summary_view"),
    ]
    order = ContextComposer._selection_order(
        list(cands),
        max_items=4,
        query_text="What do Orin's apprentices like?",
        query_type="broad_list",
        exact_recall_mode=False,
        source_messages_by_id={},
    )
    ids = [candidate.memory_id for candidate in order]
    assert ids.index("mem_nature") < ids.index("mem_busy")


def test_summary_class_cap_protects_lower_ranked_direct_evidence() -> None:
    # CS-2.2 per-class cap: three bulky, higher-ranked summaries cannot consume
    # the whole budget; the compact, lower-ranked direct-evidence fact still gets
    # a seat, and the over-cap summary is labelled class_cap_reached.
    bulky_summary = (
        "Vela discussed at great length the harbor archive, restoration methods, "
        "navigation records and many catalog topics over numerous long conversations "
        "spanning years and many collections and tangents. "
    ) * 3
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )
    context = composer.compose(
        scored_candidates=[
            _candidate("mem_sum1", final_score=0.95, canonical_text=bulky_summary,
                       object_type="summary_view"),
            _candidate("mem_sum2", final_score=0.93, canonical_text=bulky_summary,
                       object_type="summary_view"),
            _candidate("mem_sum3", final_score=0.91, canonical_text=bulky_summary,
                       object_type="summary_view"),
            _candidate("mem_direct", final_score=0.80,
                       canonical_text="Vela transferred from Arbor Bay three cycles ago.",
                       object_type="evidence"),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which port did Vela transfer from?",
        query_type="slot_fill",
    )
    assert "mem_direct" in context.selected_memory_ids
    assert context.composer_eviction_reasons.get("mem_sum3") == "class_cap_reached"


def test_class_capped_hierarchical_pair_co_skips_without_orphan_l0() -> None:
    # CS-2.2 review M3: when the summary class share is already spent, the L1
    # summary's class-cap check runs BEFORE its supporting L0 is admitted, so the
    # pair co-skips. No orphan L0 may hold a seat for a summary the cap rejected;
    # the freed seats go to direct evidence.
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_a",
                final_score=0.95,
                canonical_text=(
                    "Vela says the dock crew, archivists, and conservators formed a "
                    "strong restoration team through a difficult salvage season."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "sum_b",
                final_score=0.93,
                canonical_text=(
                    "Vela describes a brass sextant from the old harbor office and "
                    "discusses navigation marks and preservation history at length."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "sum_l1",
                final_score=0.90,
                canonical_text=(
                    "Vela summarized the port transfer across sessions and its "
                    "effects on the archive."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_l0"],
                },
            ),
            _candidate(
                "mem_direct1",
                final_score=0.85,
                canonical_text="Vela transferred from Arbor Bay three cycles ago.",
                object_type="evidence",
            ),
            _candidate(
                "mem_direct2",
                final_score=0.80,
                canonical_text="Vela catalogs navigation logs at Harbor Nine.",
                object_type="evidence",
            ),
            _candidate(
                "mem_l0",
                final_score=0.20,
                canonical_text=(
                    "[user] Vela: I transferred here three cycles ago from Arbor Bay."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_l0"],
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which port did Vela transfer from?",
        query_type="slot_fill",
    )
    assert context.composer_eviction_reasons.get("sum_l1") == "class_cap_reached"
    assert "mem_l0" not in context.selected_memory_ids
    assert {"mem_direct1", "mem_direct2"} <= set(context.selected_memory_ids)


def test_pair_promoted_l0_does_not_read_as_diversity_demotion() -> None:
    # CS-2.2 review follow-up: an L0 promoted out of feed order by the
    # hierarchical pairing branch is a policy-funded selection. A remainder
    # candidate that lost its seat to that promotion reads item_cap_reached,
    # never diversity_demoted.
    bulky_summary = (
        "Vela recounted an extended narrative about harbor archives, restoration "
        "methods, navigation records and catalog topics across many long "
        "conversations with tangents and themes. "
    ) * 3
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 3}
            )
        }
    )
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_bulky",
                final_score=0.95,
                canonical_text=bulky_summary,
                object_type="summary_view",
            ),
            _candidate(
                "sum_l1",
                final_score=0.90,
                canonical_text=(
                    "Vela summarized the port transfer across sessions."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 1,
                    "source_object_ids": ["mem_l0"],
                },
            ),
            _candidate(
                "mem_direct1",
                final_score=0.85,
                canonical_text="Vela transferred from Arbor Bay three cycles ago.",
                object_type="evidence",
            ),
            _candidate(
                "mem_direct2",
                final_score=0.80,
                canonical_text="Vela catalogs navigation logs at Harbor Nine.",
                object_type="evidence",
            ),
            _candidate(
                "mem_l0",
                final_score=0.20,
                canonical_text=(
                    "[user] Vela: I transferred here three cycles ago from Arbor Bay."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_l0"],
                },
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which port did Vela transfer from?",
        query_type="slot_fill",
    )
    # The pairing branch promoted mem_l0 (with sum_l1); the over-cap bulky
    # summary reads class_cap_reached, and the direct fact that lost the last
    # seat to the promotion reads item_cap_reached -- NOT diversity_demoted.
    assert "mem_l0" in context.selected_memory_ids
    assert context.composer_eviction_reasons.get("sum_bulky") == "class_cap_reached"
    assert context.composer_eviction_reasons.get("mem_direct2") == "item_cap_reached"


def _compact_gold_vs_bulky_windows_candidates() -> list[ScoredCandidate]:
    # Shape under test: top-ranked compact facts (the gold redundant with the
    # rank-1 decoy so the diversity reranker penalizes it) + many mid-rank
    # bulky verbatim windows with rich, mutually distinct text (so the reranker
    # promotes them).
    return [
        _candidate(
            "mem_decoy",
            final_score=0.95,
            canonical_text="Soren discussed his clockwork finch at the market.",
            object_type="evidence",
        ),
        _candidate(
            "mem_gold",
            final_score=0.92,
            canonical_text="Soren assembled a clockwork finch named Pip.",
            object_type="evidence",
        ),
        _window_candidate(
            "mem_w1",
            final_score=0.90,
            canonical_text=(
                "User: we calibrated the river gauge sensors, sampling intervals, "
                "telemetry packets and the upstream reference marks across several "
                "measurement stations."
            ),
        ),
        _window_candidate(
            "mem_w2",
            final_score=0.89,
            canonical_text=(
                "User: the glass annealing trial covered cooling curves, furnace "
                "zones, stress checks and optical inspection steps for each panel."
            ),
        ),
        _window_candidate(
            "mem_w3",
            final_score=0.88,
            canonical_text=(
                "User: the archive humidity review included sensor placement, seal "
                "inspection, airflow readings and storage cabinet adjustments."
            ),
        ),
        _window_candidate(
            "mem_w4",
            final_score=0.87,
            canonical_text=(
                "User: the acoustic baffle test explored resonance bands, mounting "
                "angles, vibration isolation and measurements from the sound chamber."
            ),
        ),
        _window_candidate(
            "mem_w5",
            final_score=0.86,
            canonical_text=(
                "User: the botanical specimen audit covered label formats, drying "
                "times, cabinet indexes and provenance notes for the herbarium."
            ),
        ),
    ]


def test_top_ranked_compact_gold_survives_bulky_windows() -> None:
    # Regression: a compact top-ranked gold fact must not be displaced by
    # many mid-rank bulky verbatim windows that the diversity reranker would
    # otherwise promote ahead of it.
    composer = _composer()
    policy = _resolved_policy(700).model_copy(
        update={
            "retrieval_params": _resolved_policy(700).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )
    kwargs = dict(
        scored_candidates=_compact_gold_vs_bulky_windows_candidates(),
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which automaton did Soren assemble?",
        query_type="broad_list",
    )

    # Counterfactual guard: under the pre-CS-2.2 admission order the diversity
    # reranker promotes the bulky windows past BOTH top-ranked compact facts and
    # the gold dies -- proving this fixture exercises the fix.
    pre_fix = _compose_with_pre_cs22_selection_order(composer, **kwargs)
    assert "mem_gold" not in pre_fix.selected_memory_ids

    context = composer.compose(**kwargs)
    assert "mem_gold" in context.selected_memory_ids
    assert "mem_decoy" in context.selected_memory_ids
    # The bulky windows beyond capacity are evicted with a precise composer cause,
    # never the compact gold.
    assert any(
        context.composer_eviction_reasons.get(window)
        in {"class_cap_reached", "budget_exhausted", "item_cap_reached"}
        for window in ("mem_w3", "mem_w4", "mem_w5")
    )
    assert "mem_gold" not in context.composer_eviction_reasons


def _multi_hop_chain_candidates() -> list[ScoredCandidate]:
    # Three compact hops across sessions that share chain entities (the club,
    # the destination), so the diversity reranker sees them as mutually
    # redundant, plus bulky diverse distractor windows it promotes instead.
    return [
        _candidate(
            "mem_hopA",
            final_score=0.95,
            canonical_text="Vela joined the coastal signal corps in March.",
            object_type="evidence",
            payload_json={"source_message_ids": ["sA"]},
        ),
        _candidate(
            "mem_hopB",
            final_score=0.93,
            canonical_text="The signal corps planned a survey at Cinder Reach in April.",
            object_type="evidence",
            payload_json={"source_message_ids": ["sB"]},
        ),
        _candidate(
            "mem_hopC",
            final_score=0.91,
            canonical_text="Vela noted that Cinder Reach belongs to the Aster Union.",
            object_type="evidence",
            payload_json={"source_message_ids": ["sC"]},
        ),
        _window_candidate(
            "mem_d1",
            final_score=0.90,
            canonical_text=(
                "User: we compared ceramic filter meshes, pressure limits, pump "
                "curves and maintenance intervals for the workshop coolant loop."
            ),
        ),
        _window_candidate(
            "mem_d2",
            final_score=0.89,
            canonical_text=(
                "User: the map restoration project covered paper fibers, archival "
                "adhesives, flattening boards and pigment stability under low light."
            ),
        ),
        _window_candidate(
            "mem_d3",
            final_score=0.88,
            canonical_text=(
                "User: the telescope housing review included gasket tolerances, "
                "fastener torque, thermal expansion and alignment checks."
            ),
        ),
    ]


def test_multi_hop_three_hops_across_sessions_all_admitted() -> None:
    # Multi-hop regression: three compact top-ranked hops must ALL be admitted;
    # pre-CS-2.2 the diversity reranker displaced the answer-bearing hop with
    # bulky diverse distractors.
    composer = _composer()
    policy = _resolved_policy(700).model_copy(
        update={
            "retrieval_params": _resolved_policy(700).retrieval_params.model_copy(
                update={"final_context_items": 4}
            )
        }
    )
    kwargs = dict(
        scored_candidates=_multi_hop_chain_candidates(),
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which federation contains the signal corps destination?",
        query_type="slot_fill",
    )

    # Counterfactual guard: under the pre-CS-2.2 admission order the chain is
    # broken -- the answer-bearing hop (Cinder Reach -> Aster Union) loses its seat to a
    # bulky distractor window.
    pre_fix = _compose_with_pre_cs22_selection_order(composer, **kwargs)
    assert "mem_hopC" not in pre_fix.selected_memory_ids

    context = composer.compose(**kwargs)
    assert {"mem_hopA", "mem_hopB", "mem_hopC"} <= set(context.selected_memory_ids)
    # The displaced distractor carries a precise composer cause; no hop does.
    assert context.composer_eviction_reasons.get("mem_d2") == "item_cap_reached"
    assert not any(
        hop in context.composer_eviction_reasons
        for hop in ("mem_hopA", "mem_hopB", "mem_hopC")
    )


def test_slot_fill_query_selection_keeps_complementary_origin_fact() -> None:
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 3}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_transfer",
                final_score=0.91,
                canonical_text=(
                    "Vela says the harbor crew supported the archive for three cycles "
                    "after the transfer from Arbor Bay."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_generic",
                final_score=0.90,
                canonical_text=(
                    "Vela says the dock crew, archivists, and conservators formed a "
                    "strong restoration team through a difficult salvage season."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_arbor_bay",
                final_score=0.84,
                canonical_text=(
                    "Vela describes a brass sextant from the Arbor Bay harbor office and "
                    "discusses its navigation marks."
                ),
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which port did Vela work at before transferring to Harbor Nine?",
        query_type="slot_fill",
    )

    assert context.selected_memory_ids == [
        "mem_transfer",
        "mem_generic",
        "mem_arbor_bay",
    ]
    assert "strong restoration team" in context.memory_block


def test_slot_fill_query_selection_handles_unicode_tokens_mechanically() -> None:
    composer = _composer()
    policy = _resolved_policy(500).model_copy(
        update={
            "retrieval_params": _resolved_policy(500).retrieval_params.model_copy(
                update={"final_context_items": 3}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_move",
                final_score=0.91,
                canonical_text=(
                    "The clockmaker moved to Delft in 2025. "
                    "Before that, she lived in Arbordale."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_generic",
                final_score=0.90,
                canonical_text=(
                    "The clockmaker repairs mechanical clocks and now works "
                    "beside a canal in Delft."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_arbordale",
                final_score=0.84,
                canonical_text=(
                    "Before moving to Delft, the clockmaker lived "
                    "in Arbordale."
                ),
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="¿De qué país se mudó Caroline hace cuatro años?",
        query_type="slot_fill",
    )

    assert context.selected_memory_ids == ["mem_move", "mem_generic", "mem_arbordale"]
    assert "Arbordale" in context.memory_block


def test_content_tokens_preserve_unicode_words() -> None:
    tokens = ContextComposer._content_tokens("¿Qué les gusta a los hijos de Mélanie?")
    assert {"qué", "les", "gusta", "hijos", "mélanie"}.issubset(tokens)

    move_tokens = ContextComposer._content_tokens(
        "¿De qué país se mudó Caroline hace 4 años?"
    )
    assert {"qué", "país", "mudó", "caroline", "años"}.issubset(move_tokens)


def test_exact_recall_mode_promotes_l0_evidence_above_summary() -> None:
    """Wave 1 batch 2 (1-D): concrete evidence must beat higher-level summaries."""
    composer = _composer()
    summary_candidate = _candidate(
        "mem_summary",
        final_score=0.95,
        canonical_text="Abstract episode summary covering family background",
        object_type="summary_view",
        payload_json={"hierarchy_level": 1, "summary_kind": "episode"},
    )
    evidence_candidate = _candidate(
        "mem_evidence",
        final_score=0.60,
        canonical_text="User said: my birthday is 14 march 1988",
        object_type="evidence",
    )

    context = composer.compose(
        scored_candidates=[summary_candidate, evidence_candidate],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(),
        conversation_messages=[],
        query_text="What is my birthday?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids[0] == "mem_evidence"

    # Without exact recall the default flow runs. The summary starts
    # with the higher final_score so it is considered first; the
    # hierarchical-summary path may replace it with its supporting L0
    # evidence, but either way the concrete evidence ends up selected.
    context_default = composer.compose(
        scored_candidates=[summary_candidate, evidence_candidate],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(),
        conversation_messages=[],
        query_text="What is my birthday?",
        query_type="slot_fill",
        exact_recall_mode=False,
    )
    assert "mem_evidence" in context_default.selected_memory_ids


def test_exact_recall_slot_fill_keeps_top_scored_evidence_before_diversity() -> None:
    composer = _composer()
    policy = _resolved_policy(700).model_copy(
        update={
            "retrieval_params": _resolved_policy(700).retrieval_params.model_copy(
                update={"final_context_items": 2}
            )
        }
    )

    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_delft_2041",
                final_score=0.42,
                canonical_text="I attended the Delft optics fair in 2041.",
                object_type="evidence",
            ),
            _candidate(
                "mem_delft_2047",
                final_score=0.36,
                canonical_text="I attended the Delft optics fair in 2047.",
                object_type="evidence",
            ),
            _candidate(
                "mem_rich_distractor",
                final_score=0.35,
                canonical_text=(
                    "The Rotterdam lens workshop used a blue alignment laser. "
                    "It was crowded."
                ),
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_unrelated"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Which years did I attend the Delft optics fair?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids == ["mem_delft_2041", "mem_delft_2047"]


def test_budgeted_marginal_exact_recall_keeps_l0_evidence_ahead_of_summary() -> None:
    composer = _composer()
    summary_candidate = _candidate(
        "mem_summary",
        final_score=0.95,
        canonical_text="Abstract episode summary covering family background",
        object_type="summary_view",
        payload_json={
            "hierarchy_level": 1,
            "summary_kind": "episode",
            "source_object_ids": ["mem_evidence"],
        },
    )
    evidence_candidate = _candidate(
        "mem_evidence",
        final_score=0.60,
        canonical_text="User said: my birthday is 14 march 1988",
        object_type="evidence",
    )

    context = composer.compose(
        scored_candidates=[summary_candidate, evidence_candidate],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(),
        conversation_messages=[],
        query_text="What is my birthday?",
        query_type="slot_fill",
        exact_recall_mode=True,
        composer_strategy="budgeted_marginal",
    )

    assert context.selected_memory_ids[0] == "mem_evidence"


def test_fact_facet_span_coadmission_renders_source_span_as_primary_context() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mff_rate_limit",
                final_score=0.91,
                canonical_text="usr_1 / rate_limit: 100 requests per minute",
                payload_json={
                    "source_kind_variant": "fact_facet",
                    "fact_facet": {
                        "fact_id": "mff_rate_limit",
                        "surface_class": "structured",
                    },
                },
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "evidence_polarity": "supports",
                        "speaker_relation_to_subject": "unknown",
                        "confidence": 0.91,
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": (
                                    "We need Redis-backed FastAPI rate limiting "
                                    "at The prism crate is in archive bay four.."
                                ),
                            }
                        ],
                    }
                ],
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(700),
        conversation_messages=[],
        query_text="What rate limit did I mention for the FastAPI service?",
        query_type="slot_fill",
        exact_recall_mode=True,
        fact_facet_span_coadmission_enabled=True,
    )

    assert "fact_facet_span_coadmitted: true" in context.memory_block
    assert (
        "source_span: We need Redis-backed FastAPI rate limiting at The prism crate is in archive bay four.."
        in context.memory_block
    )
    assert (
        "fact_facet_pointer: usr_1 / rate_limit: 100 requests per minute"
        in context.memory_block
    )


def _oversized_window(memory_id: str, source_msg: str, *, final_score: float):
    # Verbatim conversation window: many times the token cost of a tiny direct
    # fact, but low score. Sized so a single window admitted ahead of the facts
    # consumes enough of the budget to evict a gold fact (the compact-gold
    # eviction symptom).
    filler = "unrelated logistics scheduling travel weather chitchat " * 8
    return _candidate(
        memory_id,
        final_score=final_score,
        canonical_text=filler,
        object_type="evidence",
        payload_json={
            "source_kind_variant": "conversation_window",
            "source_message_ids": [source_msg],
        },
    )


def test_evidence_obligation_window_gate_preserves_tiny_direct_facts() -> None:
    # Shape under test: two tiny high-score source-message-backed direct facts must
    # survive even though several oversized low-score verbatim windows are present.
    # Without the absolute-score gate the windows are reserved first and exhaust
    # the budget, evicting the gold facts.
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_manager",
                final_score=0.98,
                canonical_text="Caroline's manager is Diane.",
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_manager"]},
            ),
            _candidate(
                "mem_role",
                final_score=0.98,
                canonical_text="Caroline works as a data analyst.",
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_role"]},
            ),
            _oversized_window("vew_window_a", "msg_w_a", final_score=0.29),
            _oversized_window("vew_window_b", "msg_w_b", final_score=0.28),
            _oversized_window("vew_window_c", "msg_w_c", final_score=0.27),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(200, 8),
        conversation_messages=[],
        query_text="Who is Caroline's manager?",
        query_type="slot_fill",
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
        fact_facet_span_coadmission_enabled=True,
    )

    assert "mem_manager" in context.selected_memory_ids
    assert "mem_role" in context.selected_memory_ids
    assert "vew_window_a" not in context.selected_memory_ids
    assert "vew_window_b" not in context.selected_memory_ids
    assert "vew_window_c" not in context.selected_memory_ids


def test_evidence_obligation_window_gate_inert_without_direct_evidence() -> None:
    # Temporal query (an EVIDENCE_OBLIGATION_QUERY_TYPE) with no source-grounded
    # direct evidence in the pool: the absolute-score gate must degrade gracefully
    # and the near-tie windows still qualify exactly as before.
    reserved = ContextComposer._evidence_obligation_candidates(
        [
            _candidate(
                "vew_conv_10_12",
                final_score=0.90,
                canonical_text="[user] The appointment was moved to Tuesday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_10", "msg_11", "msg_12"],
                },
            ),
            # A weak distractor with no source ids -> grounding 0.0 -> not
            # "direct evidence", so the gate stays inert.
            _candidate(
                "mem_distractor",
                final_score=0.30,
                canonical_text="A compact but unrelated scheduling preference.",
            ),
            _candidate(
                "vew_conv_20_22",
                final_score=0.84,
                canonical_text="[assistant] Later they confirmed Tuesday morning.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_20", "msg_21", "msg_22"],
                },
            ),
        ],
        max_items=8,
        query_type="temporal",
        answer_shape="temporal",
        coverage_mode="current_state",
        source_precision="preferred",
        exact_recall_mode=False,
        source_messages_by_id={},
    )

    reserved_ids = [candidate.memory_id for candidate in reserved]
    # Window-vs-window near-tie: 0.84 >= 0.90 * 0.92 = 0.828, so both windows qualify.
    assert "vew_conv_10_12" in reserved_ids
    assert "vew_conv_20_22" in reserved_ids


def test_source_coverage_ranking_prefers_structured_value_then_score() -> None:
    # Guards the higher sort keys (A2 only swapped the 4th/5th): a structured-value
    # fact must outrank a non-structured source-backed candidate even when the
    # latter has a higher final_score and higher grounding.
    structured = _candidate(
        "mem_structured",
        final_score=0.50,
        canonical_text="Caroline lived in Rome.",
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_rome"],
            "value_norm_key": "rome",
            "value_text": "Rome",
        },
    )
    # Higher score and higher grounding (conversation_window -> 1.0), but no
    # structured value -> must rank after the structured-value fact.
    ungrounded_window = _candidate(
        "vew_window",
        final_score=0.99,
        canonical_text="Caroline talked at length about many cities once.",
        object_type="evidence",
        payload_json={
            "source_kind_variant": "conversation_window",
            "source_message_ids": ["msg_window"],
        },
    )

    ranked = ContextComposer._rank_source_coverage_candidates(
        [ungrounded_window, structured],
        source_messages_by_id={},
        query_type="broad_list",
        answer_shape="list",
    )

    assert ranked[0].memory_id == "mem_structured"


def test_source_coverage_ranking_score_beats_grounding_tiebreak() -> None:
    # A2: with structured-value and source-backed tied, -final_score now precedes
    # -grounding, so the higher-score lower-grounding fact ranks first.
    high_score_low_grounding = _candidate(
        "mem_high_score",
        final_score=0.98,
        canonical_text="Caroline's manager is Diane.",
        object_type="evidence",
        payload_json={"source_message_ids": ["msg_manager"]},  # grounding 0.85
    )
    low_score_high_grounding = _candidate(
        "vew_window",
        final_score=0.29,
        canonical_text="A long unrelated conversation window.",
        object_type="evidence",
        payload_json={
            "source_kind_variant": "conversation_window",  # grounding 1.0
            "source_message_ids": ["msg_window"],
        },
    )

    ranked = ContextComposer._rank_source_coverage_candidates(
        [low_score_high_grounding, high_score_low_grounding],
        source_messages_by_id={},
        query_type="slot_fill",
        answer_shape="single_fact",
    )

    assert ranked[0].memory_id == "mem_high_score"


# ---------------------------------------------------------------------------
# Phase B: exhaustive_known_set coverage-member selection and metadata.
# ---------------------------------------------------------------------------


def _member_candidate(
    memory_id: str,
    *,
    final_score: float,
    canonical_text: str,
    members: list[tuple[str, str]],
    object_type: str = "evidence",
    source_message_ids: list[str] | None = None,
    extra_payload: dict | None = None,
    llm_applicability: float = 0.7,
) -> ScoredCandidate:
    payload: dict = {
        "coverage_members": [
            {"member_key": member_key, "display_text": display_text}
            for member_key, display_text in members
        ]
    }
    if source_message_ids is not None:
        payload["source_message_ids"] = source_message_ids
    if extra_payload:
        payload.update(extra_payload)
    return _candidate(
        memory_id,
        final_score=final_score,
        canonical_text=canonical_text,
        object_type=object_type,
        payload_json=payload,
        llm_applicability=llm_applicability,
    )


def _verbatim_window_candidate(
    memory_id: str,
    *,
    final_score: float,
    members: list[tuple[str, str]] | None = None,
) -> ScoredCandidate:
    payload: dict = {
        "source_kind_variant": "conversation_window",
        "source_message_ids": [f"msg_{memory_id}"],
    }
    if members is not None:
        payload["coverage_members"] = [
            {"member_key": member_key, "display_text": display_text}
            for member_key, display_text in members
        ]
    return _candidate(
        memory_id,
        final_score=final_score,
        canonical_text=f"{memory_id} conversation window evidence.",
        object_type="evidence",
        payload_json=payload,
    )


def _source_grounded_candidate(
    memory_id: str,
    *,
    final_score: float,
) -> ScoredCandidate:
    return _candidate(
        memory_id,
        final_score=final_score,
        canonical_text=f"{memory_id} direct supporting evidence.",
        object_type="evidence",
        payload_json={
            "source_message_ids": [f"msg_{memory_id}"],
            "coverage_members": [],
        },
    )


def _exhaustive_compose(
    composer: ContextComposer,
    candidates: list[ScoredCandidate],
    *,
    final_context_items: int,
    context_budget_tokens: int = 5000,
    enable_final_answer_evidence_pack: bool = False,
):
    return composer.compose(
        scored_candidates=candidates,
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(
            context_budget_tokens, final_context_items
        ),
        conversation_messages=[],
        query_text="List all of the entries.",
        query_type="broad_list",
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
        enable_final_answer_evidence_pack=enable_final_answer_evidence_pack,
    )


def test_exhaustive_selects_all_members_despite_duplicate_carriers() -> None:
    composer = _composer()
    context = _exhaustive_compose(
        composer,
        [
            _member_candidate(
                "mem_alpha_1",
                final_score=0.97,
                canonical_text="Subject A is part of the set.",
                members=[("alpha", "Subject A")],
                source_message_ids=["msg_a1"],
            ),
            _member_candidate(
                "mem_alpha_2",
                final_score=0.96,
                canonical_text="Subject A is again part of the set.",
                members=[("alpha", "Subject A")],
                source_message_ids=["msg_a2"],
            ),
            _member_candidate(
                "mem_alpha_3",
                final_score=0.95,
                canonical_text="Subject A is mentioned a third time.",
                members=[("alpha", "Subject A")],
                source_message_ids=["msg_a3"],
            ),
            _member_candidate(
                "mem_beta",
                final_score=0.50,
                canonical_text="Subject B is part of the set.",
                members=[("beta", "Subject B")],
                source_message_ids=["msg_b"],
            ),
            _member_candidate(
                "mem_gamma",
                final_score=0.45,
                canonical_text="Subject C is part of the set.",
                members=[("gamma", "Subject C")],
                source_message_ids=["msg_c"],
            ),
        ],
        final_context_items=3,
    )

    assert context.coverage_state == "complete"
    display = {item["display_text"] for item in context.allowed_values}
    assert display == {"Subject A", "Subject B", "Subject C"}
    selected = set(context.selected_memory_ids)
    assert "mem_beta" in selected
    assert "mem_gamma" in selected
    # Exactly one carrier of the duplicated member is needed.
    alpha_carriers = {"mem_alpha_1", "mem_alpha_2", "mem_alpha_3"}
    assert len(selected & alpha_carriers) == 1


def test_exhaustive_selects_more_members_than_default_with_expanded_budget() -> None:
    composer = _composer()
    members = [(f"m{i}", f"Member {i}") for i in range(7)]
    candidates = [
        _member_candidate(
            f"mem_{member_key}",
            final_score=0.9 - 0.05 * index,
            canonical_text=f"{display} is part of the set.",
            members=[(member_key, display)],
            source_message_ids=[f"msg_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]
    # The pipeline raises final_context_items to the distinct member count before
    # composing (see _expand_exhaustive_coverage_budget). With that expansion in
    # effect, every member must survive the reservation cap and be selected. The
    # pipeline-level expansion itself is asserted in the retrieval_pipeline tests.
    context = _exhaustive_compose(
        composer, candidates, final_context_items=len(members)
    )

    assert context.coverage_state == "complete"
    assert {item["display_text"] for item in context.allowed_values} == {
        display for _, display in members
    }
    assert len(context.selected_memory_ids) == len(members)


def test_exhaustive_keeps_reserved_windows_and_all_member_carriers() -> None:
    composer = _composer()
    members = [(f"m{i}", f"Member {i}") for i in range(5)]
    windows = [
        _verbatim_window_candidate(f"vew_{index}", final_score=0.90)
        for index in range(3)
    ]
    member_carriers = [
        _member_candidate(
            f"mem_{member_key}",
            final_score=0.82 - 0.01 * index,
            canonical_text=f"{display} is part of the set.",
            members=[(member_key, display)],
            source_message_ids=[f"msg_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]
    candidates = [
        *windows,
        *member_carriers,
        _source_grounded_candidate("mem_direct", final_score=0.95),
    ]

    context = _exhaustive_compose(
        composer,
        candidates,
        final_context_items=len(windows) + len(members),
    )

    selected = set(context.selected_memory_ids)
    assert {window.memory_id for window in windows} <= selected
    assert {carrier.memory_id for carrier in member_carriers} <= selected
    assert {item["display_text"] for item in context.allowed_values} == {
        display for _, display in members
    }


def test_exhaustive_answer_pack_keeps_reserved_windows_and_members() -> None:
    composer = _composer()
    members = [(f"m{i}", f"Member {i}") for i in range(5)]
    windows = [
        _verbatim_window_candidate(f"vew_pack_{index}", final_score=0.90)
        for index in range(3)
    ]
    member_carriers = [
        _member_candidate(
            f"mem_pack_{member_key}",
            final_score=0.82 - 0.01 * index,
            canonical_text=f"{display} is part of the set.",
            members=[(member_key, display)],
            source_message_ids=[f"msg_pack_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]

    context = _exhaustive_compose(
        composer,
        [
            *windows,
            *member_carriers,
            _source_grounded_candidate("mem_pack_direct", final_score=0.95),
        ],
        final_context_items=len(windows) + len(members),
        enable_final_answer_evidence_pack=True,
    )

    selected = set(context.selected_memory_ids)
    assert {window.memory_id for window in windows} <= selected
    assert {carrier.memory_id for carrier in member_carriers} <= selected


def test_exhaustive_window_carried_member_does_not_need_redundant_carrier() -> None:
    composer = _composer()
    window = _verbatim_window_candidate(
        "vew_alpha",
        final_score=0.90,
        members=[("alpha", "Subject Alpha")],
    )
    alpha_carrier = _member_candidate(
        "mem_alpha",
        final_score=0.82,
        canonical_text="Subject Alpha is part of the set.",
        members=[("alpha", "Subject Alpha")],
        source_message_ids=["msg_alpha"],
    )
    beta_carrier = _member_candidate(
        "mem_beta",
        final_score=0.80,
        canonical_text="Subject Beta is part of the set.",
        members=[("beta", "Subject Beta")],
        source_message_ids=["msg_beta"],
    )
    candidates = [
        window,
        alpha_carrier,
        beta_carrier,
        _source_grounded_candidate("mem_direct", final_score=0.95),
    ]

    assert (
        ContextComposer.exhaustive_coverage_floor(
            candidates,
            active_presence_id=None,
            allow_intimacy_context=True,
        )
        == 2
    )
    context = _exhaustive_compose(composer, candidates, final_context_items=2)

    assert context.selected_memory_ids == ["vew_alpha", "mem_beta"]
    assert "mem_alpha" not in context.selected_memory_ids
    assert {item["display_text"] for item in context.allowed_values} == {
        "Subject Alpha",
        "Subject Beta",
    }


def test_exhaustive_floor_matches_composer_coercion_for_window_gate() -> None:
    composer = _composer()
    blocked_direct = _source_grounded_candidate("mem_blocked_direct", final_score=1.0)
    blocked_direct.memory_object["intimacy_boundary"] = "safety_blocked"
    members = [(f"m{i}", f"Member {i}") for i in range(5)]
    windows = [
        _verbatim_window_candidate(f"vew_coerced_{index}", final_score=0.60)
        for index in range(3)
    ]
    member_carriers = [
        _member_candidate(
            f"mem_coerced_{member_key}",
            final_score=0.82 - 0.01 * index,
            canonical_text=f"{display} is part of the set.",
            members=[(member_key, display)],
            source_message_ids=[f"msg_coerced_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]
    candidates = [
        blocked_direct,
        *windows,
        *member_carriers,
        _source_grounded_candidate("mem_direct", final_score=0.70),
    ]

    assert (
        ContextComposer.exhaustive_coverage_floor(
            candidates,
            active_presence_id=None,
            allow_intimacy_context=True,
        )
        == len(windows) + len(members)
    )
    context = _exhaustive_compose(
        composer,
        candidates,
        final_context_items=len(windows) + len(members),
    )

    selected = set(context.selected_memory_ids)
    assert {window.memory_id for window in windows} <= selected
    assert {carrier.memory_id for carrier in member_carriers} <= selected
    assert "mem_blocked_direct" not in selected


def test_exhaustive_single_carrier_covers_two_members() -> None:
    composer = _composer()
    context = _exhaustive_compose(
        composer,
        [
            _member_candidate(
                "mem_combo",
                final_score=0.97,
                canonical_text="Subject A and Subject B are both in the set.",
                members=[("alpha", "Subject A"), ("beta", "Subject B")],
                source_message_ids=["msg_combo"],
            ),
            _member_candidate(
                "mem_gamma",
                final_score=0.40,
                canonical_text="Subject C is in the set.",
                members=[("gamma", "Subject C")],
                source_message_ids=["msg_c"],
            ),
        ],
        final_context_items=5,
    )

    assert context.coverage_state == "complete"
    assert {item["display_text"] for item in context.allowed_values} == {
        "Subject A",
        "Subject B",
        "Subject C",
    }
    # The multi-member carrier is reserved/selected exactly once.
    assert context.selected_memory_ids.count("mem_combo") == 1
    # And it covers both of its members.
    combo_keys = {
        item["normalized_key"]
        for item in context.allowed_values
        if "mem_combo" in item["memory_ids"]
    }
    assert combo_keys == {"value|alpha", "value|beta"}


def test_exhaustive_member_bearing_belief_is_admitted_and_counted() -> None:
    composer = _composer()
    # A belief carries members but is NOT source-backed (no packets, no source
    # message ids, object_type belief). It must still be admitted + counted via
    # the member-keys admission path.
    belief = _member_candidate(
        "blf_member",
        final_score=0.95,
        canonical_text="The assistant believes Subject A belongs to the set.",
        members=[("alpha", "Subject A")],
        object_type="belief",
    )
    assert not ContextComposer._is_source_backed_coverage_candidate(belief)
    assert ContextComposer._exhaustive_index_admits(belief)

    context = _exhaustive_compose(
        composer,
        [
            belief,
            _member_candidate(
                "mem_beta",
                final_score=0.40,
                canonical_text="Subject B belongs to the set.",
                members=[("beta", "Subject B")],
                source_message_ids=["msg_b"],
            ),
        ],
        final_context_items=5,
    )

    assert "blf_member" in context.selected_memory_ids
    assert {item["display_text"] for item in context.allowed_values} == {
        "Subject A",
        "Subject B",
    }
    assert context.coverage_state == "complete"


def test_exhaustive_unkeyed_source_backed_evidence_caps_at_partial() -> None:
    composer = _composer()
    context = _exhaustive_compose(
        composer,
        [
            _member_candidate(
                "mem_alpha",
                final_score=0.96,
                canonical_text="Subject A belongs to the set.",
                members=[("alpha", "Subject A")],
                source_message_ids=["msg_a"],
            ),
            # Source-backed evidence with NO coverage_members key and no value_*
            # key: unkeyed residue that blocks a "complete" claim.
            _candidate(
                "mem_unkeyed",
                final_score=0.94,
                canonical_text="Some supporting context that was never processed.",
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_unkeyed"]},
            ),
        ],
        final_context_items=5,
    )

    assert context.coverage_state == "partial"
    assert any(
        slot["reason"] == "unkeyed_supported_evidence_present"
        for slot in context.missing_slots
    )
    assert "Subject A" in {item["display_text"] for item in context.allowed_values}


def test_exhaustive_summary_with_member_key_does_not_mint_phantom_member() -> None:
    composer = _composer()
    # A summary-view carries a value_* key (rule #2) but is barred from
    # selection. It must NOT be admitted to the index, so it cannot mint a
    # phantom member that would flip complete -> partial.
    context = _exhaustive_compose(
        composer,
        [
            _member_candidate(
                "mem_alpha",
                final_score=0.50,
                canonical_text="Subject A belongs to the set.",
                members=[("alpha", "Subject A")],
                source_message_ids=["msg_a"],
            ),
            _candidate(
                "sum_phantom",
                final_score=0.97,
                canonical_text="A summary mentions Subject Z exists somewhere.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_z"],
                    "value_norm_key": "zeta",
                    "value_text": "Subject Z",
                },
            ),
        ],
        final_context_items=5,
    )

    assert context.coverage_state == "complete"
    display = {item["display_text"] for item in context.allowed_values}
    assert display == {"Subject A"}
    assert "Subject Z" not in display
    assert context.missing_slots == []


def test_exhaustive_member_with_clean_and_withheld_carriers_is_covered() -> None:
    composer = _composer()
    withheld_carrier = _member_candidate(
        "mem_alpha_secret",
        final_score=0.99,
        canonical_text="Subject A's secret is fixture-secret-Q7X9.",
        members=[("alpha", "Subject A")],
        source_message_ids=["msg_secret"],
        extra_payload={
            "value_norm_key": "fixture-secret-Q7X9",
            "value_text": "fixture-secret-Q7X9",
        },
    )
    withheld_carrier.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )
    clean_carrier = _member_candidate(
        "mem_alpha_clean",
        final_score=0.98,
        canonical_text="Subject A belongs to the set.",
        members=[("alpha", "Subject A")],
        source_message_ids=["msg_clean"],
    )

    context = _exhaustive_compose(
        composer,
        [withheld_carrier, clean_carrier],
        final_context_items=5,
    )

    # A clean carrier of the member is selected -> the member is covered, NOT a
    # redaction gap.
    assert context.coverage_state == "complete"
    assert "Subject A" in {item["display_text"] for item in context.allowed_values}
    assert context.missing_slots == []


def test_exhaustive_covered_member_never_uses_withheld_value_display() -> None:
    composer = _composer()
    secret_label = "fixture-secret-Q7X9"
    # Two value_* carriers of the SAME member (normalized value "rome"): the
    # higher-ranked one is withheld and its display text is a secret-tagged
    # literal; the other is clean. With at least one clean selected carrier the
    # member is covered, and its answer-facing display must come from the clean
    # carrier, never the withheld carrier's secret display.
    withheld_first = _candidate(
        "mem_rome_secret",
        final_score=0.99,
        canonical_text=f"The rome credential is {secret_label}.",
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_secret"],
            "value_norm_key": "rome",
            "value_text": secret_label,
        },
    )
    withheld_first.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )
    clean_carrier = _candidate(
        "mem_rome_clean",
        final_score=0.98,
        canonical_text="Caroline lived in Rome.",
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_clean"],
            "value_norm_key": "rome",
            "value_text": "Rome",
        },
    )

    context = _exhaustive_compose(
        composer,
        [withheld_first, clean_carrier],
        final_context_items=5,
    )

    assert context.coverage_state == "complete"
    assert "Rome" in {item["display_text"] for item in context.allowed_values}
    leaked = json.dumps(
        {
            "allowed_values": context.allowed_values,
            "missing_slots": context.missing_slots,
            "answer_support": answer_support_prompt_payload(context),
        }
    )
    assert secret_label not in leaked


def test_exhaustive_missing_member_never_uses_withheld_value_display() -> None:
    composer = _composer()
    secret_label = "fixture-secret-Q7X9"
    # A genuine gap: member "rome" has a withheld value_* carrier (display = a
    # secret literal) AND a clean value_* carrier, but neither fits the tiny
    # budget. The named missing slot must use the clean carrier's display, never
    # the withheld secret literal.
    alpha_clean = _member_candidate(
        "mem_alpha",
        final_score=0.99,
        canonical_text="Subject Alpha belongs to the set.",
        members=[("alpha", "Subject Alpha")],
        source_message_ids=["msg_a"],
    )
    withheld_rome = _candidate(
        "mem_rome_secret",
        final_score=0.50,
        canonical_text=f"The rome credential is {secret_label}.",
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_b1"],
            "value_norm_key": "rome",
            "value_text": secret_label,
        },
    )
    withheld_rome.memory_object.update(
        {
            "privacy_level": 3,
            "memory_category": "pin_or_password",
            "preserve_verbatim": True,
        }
    )
    clean_rome = _candidate(
        "mem_rome_clean",
        final_score=0.40,
        canonical_text="Caroline lived in Rome. " * 60,
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_b2"],
            "value_norm_key": "rome",
            "value_text": "Rome",
        },
    )

    context = _exhaustive_compose(
        composer,
        [alpha_clean, withheld_rome, clean_rome],
        final_context_items=6,
        context_budget_tokens=360,
    )

    assert context.coverage_state == "partial"
    missing_display = {slot["display_text"] for slot in context.missing_slots}
    assert "Rome" in missing_display
    leaked = json.dumps(
        {
            "allowed_values": context.allowed_values,
            "missing_slots": context.missing_slots,
            "answer_support": answer_support_prompt_payload(context),
        }
    )
    assert secret_label not in leaked


def test_exhaustive_member_null_display_falls_back_to_canonical_text() -> None:
    composer = _composer()
    # A processed member whose display_text is null must fall back to the
    # candidate canonical text, not surface an empty label.
    candidate = _candidate(
        "mem_alpha",
        final_score=0.95,
        canonical_text="Subject Alpha is part of the set.",
        object_type="evidence",
        payload_json={
            "source_message_ids": ["msg_a"],
            "coverage_members": [{"member_key": "alpha", "display_text": None}],
        },
    )

    context = _exhaustive_compose(composer, [candidate], final_context_items=5)

    assert context.coverage_state == "complete"
    displays = [item["display_text"] for item in context.allowed_values]
    assert displays == ["Subject Alpha is part of the set."]
    assert "" not in displays


def test_exhaustive_budget_too_small_reports_missing_members() -> None:
    composer = _composer()
    members = [(f"m{i}", f"Member {i}") for i in range(6)]
    candidates = [
        _member_candidate(
            f"mem_{member_key}",
            final_score=0.9 - 0.05 * index,
            canonical_text=f"{display} is part of the set. " * 12,
            members=[(member_key, display)],
            source_message_ids=[f"msg_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]
    # Tiny token budget: only some members fit. The rest become missing_slots,
    # never a RuntimeError.
    context = _exhaustive_compose(
        composer,
        candidates,
        final_context_items=6,
        context_budget_tokens=320,
    )

    assert context.coverage_state == "partial"
    assert context.allowed_values  # at least one member fit
    assert context.missing_slots  # at least one member did not
    covered = {item["display_text"] for item in context.allowed_values}
    missing = {slot["display_text"] for slot in context.missing_slots}
    # Every declared member is either covered or named as missing.
    assert covered | missing == {display for _, display in members}
    assert covered.isdisjoint(missing)


def test_exhaustive_all_members_dropped_is_insufficient() -> None:
    composer = _composer()
    members = [(f"m{i}", f"Member {i}") for i in range(4)]
    candidates = [
        _member_candidate(
            f"mem_{member_key}",
            final_score=0.9 - 0.05 * index,
            canonical_text=f"{display} is part of the set. " * 40,
            members=[(member_key, display)],
            source_message_ids=[f"msg_{member_key}"],
        )
        for index, (member_key, display) in enumerate(members)
    ]
    # Token budget below even one member's footprint after the fixed blocks:
    # nothing fits -> insufficient, and never a RuntimeError.
    context = _exhaustive_compose(
        composer,
        candidates,
        final_context_items=4,
        context_budget_tokens=120,
    )

    assert context.coverage_state == "insufficient"
    assert context.allowed_values == []


def test_exhaustive_coverage_independent_of_ranking_and_budget_property() -> None:
    composer = _composer()
    # Property/invariant check: N members, D duplicate carriers, adversarial
    # ranking (duplicates ranked above unique members), varying budget. When all
    # members fit, all are covered and the state is complete; otherwise the
    # state is an honest partial/insufficient and every member is accounted for.
    member_count = 5
    members = [(f"m{i}", f"Member {i}") for i in range(member_count)]
    candidates: list[ScoredCandidate] = []
    # Three near-duplicate carriers of the first member, ranked highest.
    for dup in range(3):
        candidates.append(
            _member_candidate(
                f"mem_m0_dup{dup}",
                final_score=0.99 - 0.001 * dup,
                canonical_text="Member 0 is part of the set.",
                members=[("m0", "Member 0")],
                source_message_ids=[f"msg_m0_{dup}"],
            )
        )
    # Remaining members ranked below the duplicates.
    for index, (member_key, display) in enumerate(members[1:], start=1):
        candidates.append(
            _member_candidate(
                f"mem_{member_key}",
                final_score=0.5 - 0.01 * index,
                canonical_text=f"{display} is part of the set.",
                members=[(member_key, display)],
                source_message_ids=[f"msg_{member_key}"],
            )
        )

    all_member_displays = {display for _, display in members}

    for final_context_items in (1, 2, 3, member_count, member_count + 5):
        for context_budget_tokens in (200, 600, 5000):
            context = _exhaustive_compose(
                composer,
                candidates,
                final_context_items=final_context_items,
                context_budget_tokens=context_budget_tokens,
            )
            covered = {item["display_text"] for item in context.allowed_values}
            missing = {
                slot["display_text"]
                for slot in context.missing_slots
                if slot["reason"] != "unkeyed_supported_evidence_present"
            }
            # Coverage and missing partitions never overlap.
            assert covered.isdisjoint(missing)
            if context.coverage_state == "complete":
                assert covered == all_member_displays
                assert missing == set()
            elif context.coverage_state == "partial":
                assert covered
                assert covered | missing == all_member_displays
            else:
                assert context.coverage_state == "insufficient"
                assert covered == set()
