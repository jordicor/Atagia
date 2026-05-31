"""Tests for final context composition."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from atagia.core.clock import FrozenClock
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import ScoredCandidate
from atagia.services.answer_postcondition import _verification_prompt
from atagia.services.chat_support import answer_support_prompt_payload

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


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
                "mem_gina",
                final_score=0.88,
                canonical_text="Gina's favorite dance style is contemporary.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "Contemporary dance really speaks to me.",
                                "occurred_at": "2023-01-20T16:04:00+00:00",
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
        query_text="What is Gina's favorite style of dance?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.memory_block.startswith("[Final Answer Evidence Pack]")
    assert context.answer_evidence_memory_ids == ["mem_gina"]
    assert context.answer_evidence_items[0]["supporting_quote"].startswith(
        "user @ 2023-01-20T16:04:00+00:00 seq 1: Contemporary dance"
    )
    assert "Evidence 1" in context.answer_evidence_block
    assert "support_kind: contextual_direct" in context.answer_evidence_block
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_items[0]["selected_for_answer_pack"] is True
    assert context.answer_evidence_items[0]["normalization"]["speaker_role"] == "user"
    assert (
        context.answer_evidence_items[0]["normalization"]["evidence_occurred_at"]
        == "2023-01-20T16:04:00+00:00"
    )


def test_answer_evidence_diagnostic_is_populated_without_rendering_pack() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_gina",
                final_score=0.88,
                canonical_text="Gina's favorite dance style is contemporary.",
                valid_from="2023-01-20T16:04:00+00:00",
                temporal_type="event_triggered",
                resolved_date="2023-01-20T16:04:00+00:00",
                payload_json={
                    "source_message_ids": ["msg_2"],
                    "source_message_window_start_occurred_at": (
                        "2023-01-20T16:03:00+00:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2023-01-20T16:04:00+00:00"
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
                                "quote_text": "Contemporary dance really speaks to me.",
                                "occurred_at": "2023-01-20T16:04:00+00:00",
                                "seq": 2,
                                "metadata_json": {"message_role": "user"},
                            },
                            {
                                "span_role": "trigger",
                                "message_id": "msg_1",
                                "quote_text": "What's your favorite style?",
                                "occurred_at": "2023-01-20T16:03:00+00:00",
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
        query_text="What is Gina's favorite style of dance?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=False,
    )

    assert context.answer_evidence_block == ""
    assert context.answer_evidence_memory_ids == []
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_sufficiency["rendered"] is False
    assert context.answer_evidence_items[0]["selected_for_answer_pack"] is False
    normalization = context.answer_evidence_items[0]["normalization"]
    assert normalization["resolved_date"] == "2023-01-20T16:04:00+00:00"
    assert normalization["source_message_ids"] == ["msg_2"]
    assert normalization["evidence_packet_message_ids"] == ["msg_2", "msg_1"]
    assert (
        normalization["source_window_start"] == "2023-01-20T16:03:00+00:00"
    )
    assert "What's your favorite style?" in normalization["trigger_quote"]


def test_answer_evidence_pack_does_not_promote_low_score_quote() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_high_no_quote",
                final_score=0.95,
                canonical_text="Gina discussed dance classes.",
            ),
            _candidate(
                "mem_low_literal",
                final_score=0.31,
                canonical_text="Gina's favorite dance style is contemporary.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "Contemporary dance really speaks to me.",
                                "occurred_at": "2023-01-20T16:04:00+00:00",
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
        query_text="What is Gina's favorite style of dance?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_block == ""
    assert context.answer_evidence_memory_ids == []
    assert context.answer_evidence_items[0]["memory_id"] == "mem_low_literal"
    assert context.answer_evidence_sufficiency["state"] == "weak_low_applicability"
    assert "Final Answer Evidence Pack" not in context.memory_block


def test_answer_evidence_prefers_query_relevant_source_quote_over_first_packet_span() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_studio_launch",
                final_score=0.89,
                canonical_text=(
                    "Gina described Jon's studio launch and encouraged him."
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
                                "quote_text": "Jon took a short trip last week.",
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
                "text": "Jon took a short trip last week.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_2",
                "role": "user",
                "seq": 2,
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="How does Gina describe the studio Jon opened?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["quote_source"] == "source_message"
    assert "studio looks amazing" in context.answer_evidence_items[0]["supporting_quote"]
    assert "short trip" not in context.answer_evidence_items[0]["supporting_quote"]


def test_answer_evidence_keeps_named_speaker_prefix_for_quote_relevance() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_studio_opening",
                final_score=0.99,
                canonical_text=(
                    "[user] Gina: When are you opening the studio?\n"
                    "[assistant] Jon: The official opening night is tomorrow.\n"
                    "[user] Gina: Congrats, Jon! The studio looks amazing.\n"
                    "[assistant] Jon: Thanks, Gina! I'm excited!"
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
                "text": "Gina: When are you opening the studio?",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": "Jon: The official opening night is tomorrow.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Jon: Thanks, Gina! I'm excited!",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="How does Gina describe the studio that Jon has opened?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_memory_ids == ["vew_studio_opening"]
    assert "seq 280" in context.answer_evidence_items[0]["supporting_quote"]
    assert "studio looks amazing" in context.answer_evidence_items[0]["supporting_quote"]


def test_answer_evidence_ranks_query_relevant_quote_before_higher_score_distractor() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_flooring",
                final_score=0.95,
                canonical_text="Gina liked the studio flooring.",
                evidence_packets=[
                    {
                        "support_kind": "direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "The Marley flooring has the right grip.",
                                "occurred_at": "2023-01-29T14:32:00+00:00",
                                "seq": 1,
                                "metadata_json": {"message_role": "user"},
                            }
                        ],
                    }
                ],
            ),
            _candidate(
                "sum_studio_launch",
                final_score=0.72,
                canonical_text="Gina described Jon's studio launch.",
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
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            }
        ],
        query_text="How does Gina describe the studio Jon opened?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["memory_id"] == "sum_studio_launch"
    assert "studio looks amazing" in context.answer_evidence_items[0]["supporting_quote"]


def test_broad_list_answer_evidence_prefers_material_applicability_over_ir_noise() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_paris",
                final_score=0.86,
                canonical_text="[assistant] Jon: I visited Paris yesterday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_paris"],
                },
                llm_applicability=1.0,
                retrieval_score=0.75,
            ),
            _candidate(
                "vew_ir_noise",
                final_score=0.35,
                canonical_text="[assistant] Jon: The studio floor needs work.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_noise"],
                },
                llm_applicability=0.0,
                retrieval_score=0.98,
            ),
            _candidate(
                "sum_rome",
                final_score=0.20,
                canonical_text=(
                    "Jon mentioned taking a short trip to Rome to clear his mind."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_rome"],
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
                "id": "msg_paris",
                "role": "assistant",
                "seq": 32,
                "text": "Jon: I visited Paris yesterday.",
                "occurred_at": "2023-01-28T14:32:00",
            },
            {
                "id": "msg_noise",
                "role": "assistant",
                "seq": 34,
                "text": "Jon: The studio floor needs work.",
                "occurred_at": "2023-01-29T14:32:00",
            },
            {
                "id": "msg_rome",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Jon: Took a short trip last week to Rome to clear my mind "
                    "a little."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
        ],
        query_text="Which cities has Jon visited?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=True,
    )

    assert [item["memory_id"] for item in context.answer_evidence_items[:2]] == [
        "vew_paris",
        "sum_rome",
    ]
    assert "Rome" in context.answer_evidence_block
    assert "vew_ir_noise" not in context.answer_evidence_memory_ids


def test_broad_list_evidence_obligation_reserves_applicable_source_linked_summary() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_paris",
                final_score=0.86,
                canonical_text="[assistant] Jon: I visited Paris yesterday.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_paris"],
                },
                llm_applicability=1.0,
            ),
            _candidate(
                "mem_distractor",
                final_score=0.82,
                canonical_text="Jon discussed dance studio flooring.",
                llm_applicability=0.1,
            ),
            _candidate(
                "sum_unrelated",
                final_score=0.81,
                canonical_text="Jon discussed unrelated dance studio logistics.",
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
                canonical_text="[assistant] Jon: Marley flooring seems practical.",
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_unrelated"],
                },
                llm_applicability=0.1,
            ),
            _candidate(
                "sum_rome",
                final_score=0.20,
                canonical_text=(
                    "Jon mentioned taking a short trip to Rome to clear his mind."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_rome"],
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
                "id": "msg_paris",
                "role": "assistant",
                "seq": 32,
                "text": "Jon: I visited Paris yesterday.",
                "occurred_at": "2023-01-28T14:32:00",
            },
            {
                "id": "msg_rome",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Jon: Took a short trip last week to Rome to clear my mind "
                    "a little."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
            {
                "id": "msg_unrelated",
                "role": "assistant",
                "seq": 36,
                "text": "Jon: Marley flooring seems practical.",
                "occurred_at": "2023-01-29T14:32:00",
            },
        ],
        query_text="Which cities has Jon visited?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["vew_paris", "sum_rome"]
    assert "Rome" in context.memory_block
    assert "dance studio flooring" not in context.memory_block


def test_broad_list_answer_evidence_renders_material_direct_quote_below_score_floor() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_high_ir_noise",
                final_score=0.92,
                canonical_text="Jon discussed studio logistics.",
                llm_applicability=0.0,
                retrieval_score=0.95,
            ),
            _candidate(
                "sum_rome",
                final_score=0.20,
                canonical_text=(
                    "Jon mentioned taking a short trip to Rome to clear his mind."
                ),
                object_type="summary_view",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_message_ids": ["msg_rome"],
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
                "id": "msg_rome",
                "role": "assistant",
                "seq": 275,
                "text": (
                    "Jon: Took a short trip last week to Rome to clear my mind "
                    "a little."
                ),
                "occurred_at": "2023-06-19T10:04:00",
            },
        ],
        query_text="Which cities has Jon visited?",
        query_type="broad_list",
        exact_recall_mode=True,
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"
    assert context.answer_evidence_memory_ids == ["sum_rome"]
    assert "Rome" in context.answer_evidence_block


def test_memory_entry_adds_query_relevant_source_quote_when_packet_span_is_weak() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_studio_launch",
                final_score=0.89,
                canonical_text="Gina described Jon's studio launch and encouraged him.",
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
                                "quote_text": "Jon took a short trip last week.",
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
                "text": "Jon took a short trip last week.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_2",
                "role": "user",
                "seq": 2,
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="How does Gina describe the studio Jon opened?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=False,
    )

    assert "evidence_packet: support: inferred" in context.memory_block
    assert "source_quote: user @ 2023-06-19T10:04:00+00:00 seq 2:" in context.memory_block
    assert "studio looks amazing" in context.memory_block


def test_summary_memory_entry_renders_short_source_chain_from_first_query_match() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "sum_grand_opening",
                final_score=0.88,
                canonical_text=(
                    "Jon's dance studio was nearly ready, with Gina excited "
                    "for the grand opening."
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
                                    "Still working on opening a dance studio."
                                ),
                                "occurred_at": "2023-06-19T10:04:00+00:00",
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
                "text": "Jon: Still working on opening a dance studio.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_278",
                "role": "user",
                "seq": 278,
                "text": "Gina: When are you opening the studio?",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": (
                    "Jon: The official opening night is tomorrow. I'm working "
                    "hard to make everything just right."
                ),
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Jon: Thanks, Gina! I'm excited!",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_282",
                "role": "user",
                "seq": 282,
                "text": "Gina: Take some time to savor it.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_283",
                "role": "assistant",
                "seq": 283,
                "text": "Jon: I want to savor all the good vibes.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="What does Jon plan to do at the grand opening?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert "source_chain:" in context.memory_block
    assert "seq 279: Jon: The official opening night is tomorrow" in context.memory_block
    assert "seq 283: Jon: I want to savor all the good vibes." in context.memory_block
    assert any(
        "savor all the good vibes" in line
        for line in context.answer_evidence_items[0]["source_chain"]
    )


def test_answer_evidence_uses_verbatim_window_text_when_source_messages_are_absent() -> (
    None
):
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_conv_281_283",
                final_score=0.91,
                canonical_text=(
                    "[user] Take some time to savor it.\n"
                    "[assistant] I want to savor all the good vibes."
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_281", "msg_282", "msg_283"],
                },
            )
        ],
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(1200),
        conversation_messages=[],
        query_text="What does Jon plan to do at the grand opening?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    assert context.answer_evidence_items[0]["quote_source"] == "verbatim_evidence_window"
    assert "savor all the good vibes" in context.answer_evidence_items[0]["supporting_quote"]
    assert context.answer_evidence_sufficiency["state"] == "sufficient_direct_quote"


def test_summary_source_window_answer_evidence_renders_source_chain() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "vew_grand_opening_277_278",
                final_score=0.98,
                canonical_text=(
                    "[assistant] Still working on opening a dance studio.\n"
                    "[user] When are you opening the studio?"
                ),
                object_type="evidence",
                payload_json={
                    "source_kind_variant": "conversation_window",
                    "source_message_ids": ["msg_277", "msg_278"],
                },
            ),
            _candidate(
                "ssw_sum_grand_opening_277_283",
                final_score=0.93,
                canonical_text=(
                    "assistant @ 2023-06-19T10:04:00 seq 277: Jon: Still "
                    "working on opening a dance studio.\n"
                    "user @ 2023-06-19T10:04:00 seq 278: Gina: When are you "
                    "opening the studio?\n"
                    "assistant @ 2023-06-19T10:04:00 seq 279: Jon: The official "
                    "opening night is tomorrow.\n"
                    "user @ 2023-06-19T10:04:00 seq 280: Gina: Congrats, Jon! "
                    "The studio looks amazing.\n"
                    "assistant @ 2023-06-19T10:04:00 seq 281: Jon: Thanks, Gina! "
                    "I'm excited!\n"
                    "user @ 2023-06-19T10:04:00 seq 282: Gina: Take some time "
                    "to savor it.\n"
                    "assistant @ 2023-06-19T10:04:00 seq 283: Jon: I want to "
                    "savor all the good vibes."
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
                        "2023-06-19T10:04:00+00:00"
                    ),
                    "source_message_window_end_occurred_at": (
                        "2023-06-19T10:04:00+00:00"
                    ),
                },
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
                "text": "Jon: Still working on opening a dance studio.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_278",
                "role": "user",
                "seq": 278,
                "text": "Gina: When are you opening the studio?",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_279",
                "role": "assistant",
                "seq": 279,
                "text": "Jon: The official opening night is tomorrow.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_280",
                "role": "user",
                "seq": 280,
                "text": "Gina: Congrats, Jon! The studio looks amazing.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_281",
                "role": "assistant",
                "seq": 281,
                "text": "Jon: Thanks, Gina! I'm excited!",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_282",
                "role": "user",
                "seq": 282,
                "text": "Gina: Take some time to savor it.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
            {
                "id": "msg_283",
                "role": "assistant",
                "seq": 283,
                "text": "Jon: I want to savor all the good vibes.",
                "occurred_at": "2023-06-19T10:04:00+00:00",
            },
        ],
        query_text="What does Jon plan to do at the grand opening?",
        query_type="slot_fill",
        enable_final_answer_evidence_pack=True,
    )

    source_chain = context.answer_evidence_items[0]["source_chain"]
    assert context.answer_evidence_memory_ids == ["ssw_sum_grand_opening_277_283"]
    assert context.memory_block.startswith("[Final Answer Evidence Pack]")
    assert any("seq 283: Jon: I want to savor all the good vibes." in line for line in source_chain)
    assert "seq 283: Jon: I want to savor all the good vibes." in context.answer_evidence_block


def test_literal_evidence_is_selected_before_higher_scoring_summary() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_summary",
                final_score=0.99,
                canonical_text="Gina discussed several store design ideas.",
                object_type="summary_view",
                payload_json={"hierarchy_level": 1, "source_message_ids": ["msg_1"]},
            ),
            _candidate(
                "mem_literal",
                final_score=0.62,
                canonical_text="Gina designed the space, furniture, and decor.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "I designed the space, furniture, and decor.",
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
                "text": "Gina talked about her store designs.",
            }
        ],
        query_text="What did Gina design for her store?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids == ["mem_literal"]
    assert "I designed the space, furniture, and decor." in context.memory_block


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
    secret_literal = "K8sN0d3Jump!2024"
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
        canonical_text="The production jump host password is K8sN0d3Jump!2024.",
        payload_json={
            "source_message_ids": ["msg_jump_host"],
            "value_norm_key": "K8sN0d3Jump!2024",
            "value_text": "K8sN0d3Jump!2024",
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
    assert "K8sN0d3Jump!2024" not in context.memory_block
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
    assert "K8sN0d3Jump!2024" not in serialized_support
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
    secret_message_text = "The production jump host password is K8sN0d3Jump!2024."
    candidate = _candidate(
        "mem_secret_quote",
        final_score=0.93,
        canonical_text="The production jump host password is K8sN0d3Jump!2024.",
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
                valid_from="2022-05-15T00:00:00+00:00",
                valid_to="2022-05-31T23:59:59+00:00",
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
        "valid_window: 2022-05-15T00:00:00+00:00 to 2022-05-31T23:59:59+00:00"
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
                "mem_job_loss",
                final_score=0.9,
                canonical_text="Jon is no longer in a secure banker job.",
                payload_json={
                    "source_message_ids": ["msg_1"],
                    "source_message_window_start_occurred_at": "2023-01-20T16:04:00+00:00",
                    "source_message_window_end_occurred_at": "2023-01-20T16:04:00+00:00",
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
                "text": "Jon: Lost my job as a banker yesterday, so I'm gonna start my own business.",
                "occurred_at": "2023-01-20T16:04:00+00:00",
            }
        ],
        query_type="temporal",
        exact_recall_mode=True,
    )

    assert (
        "source_window: 2023-01-20T16:04:00+00:00 to 2023-01-20T16:04:00+00:00"
        in context.memory_block
    )
    assert (
        "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 2: Jon: Lost my job as a banker yesterday"
        in context.memory_block
    )


def test_memory_entry_prefers_evidence_packet_quotes_when_hydrated() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_gina",
                final_score=0.9,
                canonical_text="Gina's favorite dance style is contemporary.",
                evidence_packets=[
                    {
                        "support_kind": "contextual_direct",
                        "evidence_polarity": "supports",
                        "speaker_relation_to_subject": "self_report",
                        "confidence": 0.91,
                        "rationale": "Gina answers Jon's favorite-dance question.",
                        "spans": [
                            {
                                "span_role": "source",
                                "quote_text": "Contemporary dance really speaks to me.",
                                "seq": 2,
                                "occurred_at": "2023-01-20T16:04:00+00:00",
                                "metadata_json": {"message_role": "user"},
                            },
                            {
                                "span_role": "trigger",
                                "quote_text": "What's your fave?",
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
    assert "source_quote: user @ 2023-01-20T16:04:00+00:00 seq 2: Contemporary dance really speaks to me." in context.memory_block
    assert "trigger_quote: assistant @ 2023-01-20T16:03:00+00:00 seq 1: What's your fave?" in context.memory_block
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
    source_text = "Liora packed saffron notebooks before sunrise."
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
        "Liora packed saffron notebooks before sunrise."
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
                canonical_text="Melanie signed up for a pottery class.",
                object_type="summary_view",
                scope="conversation",
                payload_json={
                    "summary_kind": "conversation_chunk",
                    "hierarchy_level": 0,
                    "source_excerpt_messages": [
                        {
                            "role": "assistant",
                            "occurred_at": "2023-07-03T13:36:00",
                            "text": "Melanie: I just signed up for a pottery class yesterday.",
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
        "source_excerpt: assistant @ 2023-07-03T13:36:00: Melanie: I just signed up for a pottery class yesterday."
        in context.memory_block
    )


def test_verbatim_evidence_search_candidate_includes_source_window() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate(
                "raw_cnv_1_1_2",
                final_score=0.9,
                canonical_text="user: My allergy is peanut and shellfish\nassistant: Noted.",
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
                canonical_text="Caroline looked into adoption options.",
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
                canonical_text="Caroline talked about unrelated travel logistics.",
            ),
            _candidate(
                "vew_conv_10_12",
                final_score=0.52,
                canonical_text=(
                    "[user] Caroline: Researching adoption agencies has been "
                    "on my mind."
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
        query_text="What did Caroline research?",
        query_type="slot_fill",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids[0] == "vew_conv_10_12"
    assert "Researching adoption agencies" in context.memory_block


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
                "mem_paris",
                final_score=0.95,
                canonical_text="Caroline mentioned Paris.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "paris",
                    "value_text": "Paris",
                },
            ),
            _candidate(
                "mem_unrelated",
                final_score=0.94,
                canonical_text="Caroline also discussed unrelated travel logistics.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_other"],
                    "value_norm_key": "logistics",
                    "value_text": "Logistics",
                },
            ),
            _candidate(
                "mem_rome",
                final_score=0.41,
                canonical_text="Caroline mentioned Rome in the same sentence.",
                object_type="evidence",
                payload_json={
                    "source_message_ids": ["msg_combo"],
                    "value_norm_key": "rome",
                    "value_text": "Rome",
                },
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

    assert context.selected_memory_ids == ["mem_paris", "mem_rome"]
    assert [item["display_text"] for item in context.allowed_values] == [
        "Paris",
        "Rome",
    ]
    assert "Caroline also discussed unrelated travel logistics." not in context.memory_block


def test_source_coverage_reserve_seeds_groups_from_window_reservations() -> None:
    window = _candidate(
        "vew_combo",
        final_score=0.99,
        canonical_text="Caroline said she lived in Paris, Rome, and Lisbon.",
        payload_json={
            "source_kind_variant": "conversation_window",
            "source_message_ids": ["msg_combo"],
        },
    )
    same_source_duplicate = _candidate(
        "mem_combo_duplicate",
        final_score=0.98,
        canonical_text="Caroline discussed the same source message.",
        payload_json={"source_message_ids": ["msg_combo"]},
    )
    paris = _candidate(
        "mem_paris",
        final_score=0.97,
        canonical_text="Caroline lived in Paris.",
        payload_json={
            "source_message_ids": ["msg_paris"],
            "value_norm_key": "paris",
            "value_text": "Paris",
        },
    )
    rome = _candidate(
        "mem_rome",
        final_score=0.96,
        canonical_text="Caroline lived in Rome.",
        payload_json={
            "source_message_ids": ["msg_rome"],
            "value_norm_key": "rome",
            "value_text": "Rome",
        },
    )
    lisbon = _candidate(
        "mem_lisbon",
        final_score=0.95,
        canonical_text="Caroline lived in Lisbon.",
        payload_json={
            "source_message_ids": ["msg_lisbon"],
            "value_norm_key": "lisbon",
            "value_text": "Lisbon",
        },
    )

    reserved = ContextComposer._evidence_obligation_candidates(
        [window, same_source_duplicate, paris, rome, lisbon],
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
        "mem_paris",
        "mem_rome",
        "mem_lisbon",
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
                payload_json={"source_message_ids": ["msg_combo"]},
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
                "sum_country",
                final_score=0.95,
                canonical_text="Episode summary says Caroline moved from Sweden.",
                object_type="summary_view",
                payload_json={
                    "source_message_ids": ["msg_country"],
                    "value_norm_key": "sweden",
                    "value_text": "Sweden",
                },
            )
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_policy_with_final_context_items(700, 1),
        conversation_messages=[],
        query_text="Where did Caroline move from?",
        query_type="slot_fill",
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
        enable_evidence_obligation_coverage=True,
    )

    assert context.selected_memory_ids == ["sum_country"]
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


def test_broad_query_selection_prefers_diverse_specific_facets() -> None:
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
                "mem_dinosaurs",
                final_score=0.91,
                canonical_text="Melanie's kids were excited about the dinosaur exhibit at the museum.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_busy",
                final_score=0.90,
                canonical_text="Melanie mentions being busy with the kids and work.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_pottery",
                final_score=0.87,
                canonical_text="Melanie took her kids to a pottery workshop where they loved getting creative with clay.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_nature",
                final_score=0.86,
                canonical_text="Melanie's family enjoys camping, hiking, and spending time in nature together.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_swimming",
                final_score=0.88,
                canonical_text="Melanie went swimming with her kids after the conversation.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="What do Melanie's kids like?",
        query_type="broad_list",
    )

    assert set(context.selected_memory_ids) == {
        "mem_dinosaurs",
        "mem_pottery",
        "mem_nature",
        "mem_swimming",
    }
    assert "busy with the kids and work" not in context.memory_block


def test_broad_query_selection_penalizes_duplicate_source_messages() -> None:
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
                "mem_first_pottery",
                final_score=0.94,
                canonical_text="Melanie's kids loved pottery at the workshop.",
                payload_json={"source_message_ids": ["msg_pottery"]},
            ),
            _candidate(
                "mem_duplicate_pottery",
                final_score=0.93,
                canonical_text="The pottery workshop made Melanie's kids happy.",
                payload_json={"source_message_ids": ["msg_pottery"]},
            ),
            _candidate(
                "mem_swimming",
                final_score=0.82,
                canonical_text="Melanie's kids also enjoyed swimming.",
                payload_json={"source_message_ids": ["msg_swimming"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="What activities did Melanie's kids enjoy?",
        query_type="broad_list",
    )

    assert context.selected_memory_ids == ["mem_first_pottery", "mem_swimming"]


def test_broad_query_selection_preserves_lower_scored_new_list_coverage() -> None:
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
                "mem_mountains",
                final_score=0.94,
                canonical_text="Melanie camped in the mountains with her family.",
                payload_json={"source_message_ids": ["msg_mountains"]},
            ),
            _candidate(
                "mem_beach",
                final_score=0.92,
                canonical_text="Melanie camped at the beach with her family.",
                payload_json={"source_message_ids": ["msg_beach"]},
            ),
            _candidate(
                "mem_generic_trip",
                final_score=0.88,
                canonical_text="Melanie camped with her family during another trip.",
                payload_json={"source_message_ids": ["msg_mountains"]},
            ),
            _candidate(
                "mem_forest",
                final_score=0.85,
                canonical_text="Melanie went camping in the forest with her kids.",
                payload_json={"source_message_ids": ["msg_forest"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Where has Melanie camped?",
        query_type="broad_list",
    )

    assert set(context.selected_memory_ids) == {
        "mem_mountains",
        "mem_beach",
        "mem_forest",
    }
    assert "another trip" not in context.memory_block


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
                "mem_dinosaurios",
                final_score=0.91,
                canonical_text="Mélanie's kids were excited about the dinosaur exhibit at the museum.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_ocupada",
                final_score=0.90,
                canonical_text="Mélanie mentions being busy with the kids and work.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_ceramica",
                final_score=0.87,
                canonical_text="Mélanie took her kids to a pottery workshop where they loved getting creative with clay.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_naturaleza",
                final_score=0.86,
                canonical_text="Mélanie's family enjoys camping, hiking, and spending time in nature together.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_natacion",
                final_score=0.88,
                canonical_text="Mélanie went swimming with her kids after the conversation.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="¿Qué les gusta a los hijos de Mélanie?",
        query_type="broad_list",
    )
    default_context = composer.compose(
        scored_candidates=[
            _candidate(
                "mem_dinosaurios",
                final_score=0.91,
                canonical_text="Mélanie's kids were excited about the dinosaur exhibit at the museum.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_ocupada",
                final_score=0.90,
                canonical_text="Mélanie mentions being busy with the kids and work.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_ceramica",
                final_score=0.87,
                canonical_text="Mélanie took her kids to a pottery workshop where they loved getting creative with clay.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_naturaleza",
                final_score=0.86,
                canonical_text="Mélanie's family enjoys camping, hiking, and spending time in nature together.",
                object_type="summary_view",
            ),
            _candidate(
                "mem_natacion",
                final_score=0.88,
                canonical_text="Mélanie went swimming with her kids after the conversation.",
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="¿Qué les gusta a los hijos de Mélanie?",
        query_type="default",
    )

    assert set(context.selected_memory_ids) == {
        "mem_dinosaurios",
        "mem_ceramica",
        "mem_naturaleza",
        "mem_natacion",
    }
    assert "busy with the kids and work" not in context.memory_block
    assert set(default_context.selected_memory_ids) == {
        "mem_dinosaurios",
        "mem_ocupada",
        "mem_ceramica",
        "mem_natacion",
    }


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
                "mem_move",
                final_score=0.91,
                canonical_text=(
                    "Caroline says her friends supported her for four years since moving "
                    "from her home country."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_generic",
                final_score=0.90,
                canonical_text=(
                    "Caroline says her friends, family, and mentors were a strong support "
                    "system for her through a tough breakup."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_sweden",
                final_score=0.84,
                canonical_text=(
                    "Caroline describes a necklace from her Swedish grandmother and talks "
                    "about her roots."
                ),
                object_type="summary_view",
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="Where did Caroline move from 4 years ago?",
        query_type="slot_fill",
    )

    assert context.selected_memory_ids == ["mem_move", "mem_generic", "mem_sweden"]
    assert "strong support system" in context.memory_block


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
                    "Caroline says her friends supported her for four years since moving "
                    "from her home country."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_generic",
                final_score=0.90,
                canonical_text=(
                    "Caroline says her friends, family, and mentors were a strong support "
                    "system for her through a tough breakup."
                ),
                object_type="summary_view",
            ),
            _candidate(
                "mem_sweden",
                final_score=0.84,
                canonical_text=(
                    "Caroline describes a necklace from her Swedish grandmother and talks "
                    "about her roots."
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

    assert context.selected_memory_ids == ["mem_move", "mem_generic", "mem_sweden"]
    assert "strong support system" in context.memory_block


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
                "mem_paris_2022",
                final_score=0.42,
                canonical_text="Jolene bought the pendant in Paris one year ago.",
                object_type="evidence",
            ),
            _candidate(
                "mem_paris_2010",
                final_score=0.36,
                canonical_text="Jolene's mother gave her the pendant in Paris in 2010.",
                object_type="evidence",
            ),
            _candidate(
                "mem_rich_distractor",
                final_score=0.35,
                canonical_text=(
                    "Jolene has many unrelated hobbies, deadlines, activities, "
                    "and reflective life updates."
                ),
                object_type="evidence",
                payload_json={"source_message_ids": ["msg_unrelated"]},
            ),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=policy,
        conversation_messages=[],
        query_text="How many times has Jolene been to France?",
        query_type="slot_fill",
        exact_recall_mode=True,
    )

    assert context.selected_memory_ids == ["mem_paris_2022", "mem_paris_2010"]


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
                                    "at 100 requests per minute per API key."
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
        "source_span: We need Redis-backed FastAPI rate limiting at 100 requests per minute per API key."
        in context.memory_block
    )
    assert (
        "fact_facet_pointer: usr_1 / rate_limit: 100 requests per minute"
        in context.memory_block
    )
