"""Tests for retrieval comparison logic."""

from __future__ import annotations

import pytest

from atagia.models.schemas_memory import ComposedContext, RetrievalPlan, ScoredCandidate
from atagia.models.schemas_replay import PipelineResult
from atagia.memory.retrieval_comparator import RetrievalComparator


def _pipeline_result(*, selected_ids: list[str], scores: dict[str, float], tokens: int = 100) -> PipelineResult:
    resolved_fts_queries = ["retry"]
    return PipelineResult(
        detected_needs=[],
        retrieval_plan=RetrievalPlan(
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            fts_queries=resolved_fts_queries,
            sub_query_plans=[
                {
                    "text": resolved_fts_queries[0],
                    "fts_queries": resolved_fts_queries,
                }
            ],
            query_type="default",
            scope_filter=[],
            status_filter=[],
            vector_limit=0,
            max_candidates=10,
            max_context_items=8,
            privacy_ceiling=1,
            retrieval_levels=[0],
            require_evidence_regrounding=False,
            need_driven_boosts={},
            skip_retrieval=False,
        ),
        raw_candidates=[],
        scored_candidates=[
            ScoredCandidate(
                memory_id=memory_id,
                memory_object={"id": memory_id, "canonical_text": memory_id},
                llm_applicability=score,
                retrieval_score=score,
                vitality_boost=0.0,
                confirmation_boost=0.0,
                need_boost=0.0,
                penalty=0.0,
                final_score=score,
            )
            for memory_id, score in scores.items()
        ],
        composed_context=ComposedContext(
            contract_block="[Interaction Contract]",
            workspace_block="",
            memory_block="memories",
            state_block="",
            selected_memory_ids=selected_ids,
            total_tokens_estimate=tokens,
            budget_tokens=500,
            items_included=len(selected_ids),
            items_dropped=0,
        ),
        current_contract={},
        user_state={},
        stage_timings={},
    )


def test_compare_identical_results_has_full_overlap_and_zero_deltas() -> None:
    comparator = RetrievalComparator()
    original_event = {
        "selected_memory_ids_json": ["mem_1", "mem_2"],
        "context_view_json": {
            "contract_block": "[Interaction Contract]",
            "workspace_block": "",
            "memory_block": "memories",
            "state_block": "",
            "total_tokens_estimate": 100,
        },
        "outcome_json": {
            "scored_candidates": [
                {"memory_id": "mem_1", "final_score": 0.8},
                {"memory_id": "mem_2", "final_score": 0.6},
            ]
        },
    }

    comparison = comparator.compare(
        original_event,
        _pipeline_result(selected_ids=["mem_1", "mem_2"], scores={"mem_1": 0.8, "mem_2": 0.6}),
    )

    assert comparison.memories_in_both == ["mem_1", "mem_2"]
    assert comparison.memories_only_original == []
    assert comparison.memories_only_replay == []
    assert all(delta.delta == 0.0 for delta in comparison.score_deltas)
    assert comparison.overlap_ratio == 1.0


def test_compare_different_results_reports_set_diffs_and_score_changes() -> None:
    comparator = RetrievalComparator()
    original_event = {
        "selected_memory_ids_json": ["mem_1", "mem_2"],
        "context_view_json": {
            "contract_block": "A",
            "workspace_block": "",
            "memory_block": "X",
            "state_block": "",
            "total_tokens_estimate": 80,
        },
        "outcome_json": {
            "scored_candidates": [
                {"memory_id": "mem_1", "final_score": 0.4},
                {"memory_id": "mem_2", "final_score": 0.7},
            ]
        },
    }

    comparison = comparator.compare(
        original_event,
        _pipeline_result(selected_ids=["mem_2", "mem_3"], scores={"mem_2": 0.9, "mem_3": 0.5}, tokens=120),
    )

    assert comparison.memories_in_both == ["mem_2"]
    assert comparison.memories_only_original == ["mem_1"]
    assert comparison.memories_only_replay == ["mem_3"]
    assert comparison.score_deltas[0].memory_id == "mem_2"
    assert comparison.score_deltas[0].delta == pytest.approx(0.2)
    assert comparison.memory_block_changed is True
    assert comparison.overlap_ratio == 1 / 3


def test_compare_handles_empty_results() -> None:
    comparator = RetrievalComparator()
    original_event = {
        "selected_memory_ids_json": [],
        "context_view_json": {
            "contract_block": "",
            "workspace_block": "",
            "memory_block": "",
            "state_block": "",
            "total_tokens_estimate": 0,
        },
        "outcome_json": {},
    }

    comparison = comparator.compare(original_event, _pipeline_result(selected_ids=[], scores={}, tokens=0))

    assert comparison.memories_in_both == []
    assert comparison.overlap_ratio == 0.0
    assert comparison.score_deltas == []
