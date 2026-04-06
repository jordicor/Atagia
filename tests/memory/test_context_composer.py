"""Tests for final context composition."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from atagia.core.clock import FrozenClock
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import ScoredCandidate

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _resolved_policy(context_budget_tokens: int = 5300):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()["coding_debug"]
    resolved = PolicyResolver().resolve(manifest, None, None)
    return resolved.model_copy(update={"context_budget_tokens": context_budget_tokens})


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
) -> ScoredCandidate:
    return ScoredCandidate(
        memory_id=memory_id,
        memory_object={
            "id": memory_id,
            "object_type": object_type,
            "confidence": confidence,
            "scope": scope,
            "canonical_text": canonical_text,
            "payload_json": payload_json or {},
            "updated_at": updated_at,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
        llm_applicability=0.7,
        retrieval_score=0.6,
        vitality_boost=0.2,
        confirmation_boost=0.0,
        need_boost=0.0,
        penalty=0.0,
        final_score=final_score,
    )


def _contract() -> dict[str, dict]:
    return {
        "depth": {"label": "detailed explanations preferred", "score": 0.72},
        "directness": {"label": "high", "score": 0.85},
    }


def _composer() -> ContextComposer:
    return ContextComposer(FrozenClock(datetime(2026, 3, 30, 22, 0, tzinfo=timezone.utc)))


def test_normal_composition_includes_contract_and_memories_within_budget() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate("mem_2", final_score=0.74, canonical_text="FastAPI and SQLite are the current stack."),
            _candidate("mem_1", final_score=0.91, canonical_text="User prefers patch-style debugging help."),
        ],
        current_contract=_contract(),
        user_state=None,
        resolved_policy=_resolved_policy(400),
        conversation_messages=[],
    )

    assert context.contract_block.startswith("[Interaction Contract]")
    assert "depth: detailed explanations preferred (confidence: 0.72)" in context.contract_block
    assert context.memory_block.startswith("[Retrieved Memories]")
    assert "User prefers patch-style debugging help." in context.memory_block
    assert "FastAPI and SQLite are the current stack." in context.memory_block
    assert context.selected_memory_ids == ["mem_1", "mem_2"]
    assert context.items_included == 2
    assert context.items_dropped == 0
    assert context.total_tokens_estimate <= context.budget_tokens
    assert context.state_block == ""


def test_budget_exhaustion_keeps_only_top_candidates_that_fit() -> None:
    composer = _composer()
    first = _candidate("mem_1", final_score=0.95, canonical_text="Short top-priority memory.")
    contract_block = composer._format_contract_block(_contract(), _resolved_policy(400))
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
    later_fit = _candidate("mem_fit", final_score=0.6, canonical_text="Short memory that fits.")
    contract_block = composer._format_contract_block(_contract(), _resolved_policy(400))
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
    first = _candidate("mem_high", final_score=0.95, canonical_text="High-priority memory.")
    contract_block = composer._format_contract_block(_contract(), _resolved_policy(300))
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
            ),
            _candidate(
                "mem_open",
                final_score=0.8,
                canonical_text="User signed up for pottery class.",
                valid_from="2023-07-02T00:00:00+00:00",
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
    assert "valid: 2022-05-15T00:00:00+00:00 to 2022-05-31T23:59:59+00:00" in context.memory_block
    # Open-ended (only valid_from) shows from-date and ?
    assert "valid: 2023-07-02T00:00:00+00:00 to ?" in context.memory_block
    # Non-temporal memory has no valid: segment
    # (verified by making sure the line for mem_no_time does not include the marker)
    lines = context.memory_block.splitlines()
    no_time_line = next(line for line in lines if "User prefers direct answers." in line)
    no_time_header = lines[lines.index(no_time_line) - 1]
    assert "valid:" not in no_time_header


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

    assert "source_window: 2023-07-03T13:36:00 to 2023-07-03T13:36:00" in context.memory_block
    assert "source_excerpt: assistant @ 2023-07-03T13:36:00: Melanie: I just signed up for a pottery class yesterday." in context.memory_block


def test_empty_canonical_text_candidates_are_filtered_out() -> None:
    composer = _composer()
    context = composer.compose(
        scored_candidates=[
            _candidate("mem_blank", final_score=0.99, canonical_text="   "),
            _candidate("mem_good", final_score=0.50, canonical_text="Useful retained memory."),
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
    candidate = _candidate("mem_1", final_score=0.9, canonical_text="Short memory that otherwise fits.")
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

    assert context.total_tokens_estimate >= composer.estimate_tokens(context.contract_block)
    assert context.items_included + context.items_dropped == 3


def test_final_context_items_caps_selected_memories_even_when_budget_allows_more() -> None:
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
        scored_candidates=[_candidate("mem_1", final_score=0.9, canonical_text="Useful memory.")],
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
                final_score=0.95,
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
                payload_json={"claim_key": "workflow.debugging.style", "claim_value": "patch_first"},
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
                        {"claim_key": "workflow.debugging.style", "claim_value": "patch_first"}
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
                payload_json={"claim_key": "workflow.debugging.style", "claim_value": "patch_first"},
                updated_at="2026-03-30T09:00:00+00:00",
            ),
            _candidate(
                "mem_fresh",
                final_score=0.45,
                canonical_text="Fresher contradictory belief.",
                object_type="belief",
                scope="global_user",
                payload_json={"claim_key": "workflow.debugging.style", "claim_value": "investigate_breadth_first"},
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
                payload_json={"claim_key": "workflow.debugging.style", "claim_value": "patch_first"},
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
    )

    assert set(context.selected_memory_ids) == {
        "mem_dinosaurs",
        "mem_pottery",
        "mem_nature",
        "mem_swimming",
    }
    assert "busy with the kids and work" not in context.memory_block


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
    )

    assert context.selected_memory_ids[:2] == ["mem_move", "mem_sweden"]
    assert "strong support system" in context.memory_block
