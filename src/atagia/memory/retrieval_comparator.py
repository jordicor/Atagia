"""Pure comparison logic for original vs replayed retrieval results."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_replay import PipelineResult, RetrievalComparison, ScoreDelta


class RetrievalComparator:
    """Compare original retrieval traces against replay results."""

    def compare(
        self,
        original_event: dict[str, Any],
        replay_result: PipelineResult,
    ) -> RetrievalComparison:
        original_context = original_event.get("context_view_json") or {}
        original_selected = self._selected_memory_ids(original_event)
        replay_selected = list(replay_result.composed_context.selected_memory_ids)

        original_set = set(original_selected)
        replay_set = set(replay_selected)
        in_both = sorted(original_set & replay_set)
        only_original = sorted(original_set - replay_set)
        only_replay = sorted(replay_set - original_set)
        union = original_set | replay_set

        original_scores = self._original_score_map(original_event)
        replay_scores = {
            candidate.memory_id: candidate.final_score
            for candidate in replay_result.scored_candidates
        }
        score_deltas = [
            ScoreDelta(
                memory_id=memory_id,
                original_score=float(original_scores[memory_id]),
                replay_score=float(replay_scores[memory_id]),
                delta=float(replay_scores[memory_id]) - float(original_scores[memory_id]),
            )
            for memory_id in in_both
            if memory_id in original_scores and memory_id in replay_scores
        ]

        return RetrievalComparison(
            memories_in_both=in_both,
            memories_only_original=only_original,
            memories_only_replay=only_replay,
            score_deltas=score_deltas,
            contract_block_changed=str(original_context.get("contract_block", "")) != replay_result.composed_context.contract_block,
            workspace_block_changed=str(original_context.get("workspace_block", "")) != replay_result.composed_context.workspace_block,
            memory_block_changed=str(original_context.get("memory_block", "")) != replay_result.composed_context.memory_block,
            state_block_changed=str(original_context.get("state_block", "")) != replay_result.composed_context.state_block,
            original_items_count=len(original_selected),
            replay_items_count=len(replay_selected),
            overlap_ratio=(len(in_both) / len(union)) if union else 0.0,
            original_total_tokens=int(original_context.get("total_tokens_estimate", 0) or 0),
            replay_total_tokens=replay_result.composed_context.total_tokens_estimate,
        )

    @staticmethod
    def _selected_memory_ids(original_event: dict[str, Any]) -> list[str]:
        selected = original_event.get("selected_memory_ids_json")
        if isinstance(selected, list):
            return [str(memory_id) for memory_id in selected]
        context_view = original_event.get("context_view_json") or {}
        context_selected = context_view.get("selected_memory_ids")
        if isinstance(context_selected, list):
            return [str(memory_id) for memory_id in context_selected]
        return []

    @staticmethod
    def _original_score_map(original_event: dict[str, Any]) -> dict[str, float]:
        outcome = original_event.get("outcome_json") or {}
        scored_candidates = outcome.get("scored_candidates")
        if not isinstance(scored_candidates, list):
            return {}
        score_map: dict[str, float] = {}
        for candidate in scored_candidates:
            if not isinstance(candidate, dict):
                continue
            memory_id = candidate.get("memory_id")
            final_score = candidate.get("final_score")
            if memory_id is None or final_score is None:
                continue
            score_map[str(memory_id)] = float(final_score)
        return score_map
