"""Final context composition within token budgets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Literal

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.memory.intimacy_boundary_policy import candidate_allows_intimacy_boundary
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_memory import ComposedContext, QueryType, ScoredCandidate


ComposerStrategy = Literal["score_first", "budgeted_marginal"]


@dataclass(frozen=True, slots=True)
class _SelectionProfile:
    diversity_weight: float
    richness_weight: float
    source_grounding_weight: float
    coverage_gain_weight: float
    pool_extra: int


@dataclass(frozen=True, slots=True)
class _SelectionAction:
    items: tuple[ScoredCandidate, ...]
    blocked_ids: frozenset[str]
    token_cost: int
    utility: float
    exact_priority: int
    first_order: int


@dataclass(slots=True)
class _MemorySelection:
    selected: list[ScoredCandidate]
    memory_lines: list[str]


@dataclass(frozen=True, slots=True)
class _SourceQuoteOptions:
    enabled: bool
    max_messages: int = 3
    max_chars: int = 520
    max_message_chars: int = 180
    max_entries: int | None = None


class ContextComposer:
    """Formats scored memories into a prompt-ready context view."""

    CONTRACT_BLOCK_RATIO = 0.35
    WORKSPACE_BLOCK_RATIO = 0.08

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def compose(
        self,
        scored_candidates: list[ScoredCandidate | dict[str, Any]],
        current_contract: dict[str, dict[str, Any]],
        user_state: dict[str, Any] | None,
        resolved_policy: ResolvedPolicy,
        conversation_messages: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None = None,
        query_text: str | None = None,
        query_type: QueryType = "default",
        exact_recall_mode: bool = False,
        composer_strategy: ComposerStrategy | None = None,
    ) -> ComposedContext:
        coerced_candidates = [
            candidate
            for candidate in (
                self._coerce_scored_candidate(candidate)
                for candidate in scored_candidates
            )
            if str(candidate.memory_object.get("canonical_text", "")).strip()
            and candidate_allows_intimacy_boundary(
                candidate.memory_object,
                allow_intimacy_context=resolved_policy.allow_intimacy_context,
            )
        ]
        # Wave 1 batch 2 (1-D): when exact recall is flagged, push L0
        # evidence ahead of L1/L2 summaries before any diversity pass.
        candidates = sorted(
            coerced_candidates,
            key=lambda candidate: (
                self._exact_recall_level_priority(candidate, exact_recall_mode),
                -candidate.final_score,
                candidate.memory_id,
            ),
        )
        candidates = self._selection_order(
            candidates,
            max_items=resolved_policy.retrieval_params.final_context_items,
            query_text=query_text,
            query_type=query_type,
        )
        candidate_by_id = {candidate.memory_id: candidate for candidate in candidates}
        budget_tokens = resolved_policy.context_budget_tokens
        remaining_budget = budget_tokens
        contract_block, remaining_budget = self._budget_block(
            self._format_contract_block(current_contract, resolved_policy),
            total_budget=budget_tokens,
            remaining_budget=remaining_budget,
            ratio=self.CONTRACT_BLOCK_RATIO,
            min_tokens=1,
            header="[Interaction Contract]",
        )
        workspace_block, remaining_budget = self._budget_block(
            self._format_workspace_block(workspace_rollup),
            total_budget=budget_tokens,
            remaining_budget=remaining_budget,
            ratio=self.WORKSPACE_BLOCK_RATIO,
            min_tokens=0,
            header="[Workspace Context]",
        )
        state_block, remaining_budget = self._budget_block(
            self._format_state_block(user_state),
            total_budget=remaining_budget,
            remaining_budget=remaining_budget,
            ratio=1.0,
            min_tokens=0,
            header="[Current User State]",
        )
        memory_header = "[Retrieved Memories]\n"
        if candidates:
            header_tokens = self.estimate_tokens(memory_header)
            if remaining_budget >= header_tokens:
                remaining_budget -= header_tokens
            else:
                remaining_budget = 0

        source_messages_by_id = self._source_messages_by_id(conversation_messages)
        source_quote_options = self._source_quote_options(
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
        )
        max_items = resolved_policy.retrieval_params.final_context_items
        strategy = composer_strategy or "score_first"
        if strategy == "score_first":
            selection = self._select_score_first(
                candidates,
                candidate_by_id=candidate_by_id,
                remaining_budget=remaining_budget,
                max_items=max_items,
                source_messages_by_id=source_messages_by_id,
                source_quote_options=source_quote_options,
            )
        elif strategy == "budgeted_marginal":
            selection = self._select_budgeted_marginal(
                candidates,
                candidate_by_id=candidate_by_id,
                remaining_budget=remaining_budget,
                max_items=max_items,
                query_text=query_text,
                query_type=query_type,
                exact_recall_mode=exact_recall_mode,
                source_messages_by_id=source_messages_by_id,
                source_quote_options=source_quote_options,
            )
        else:
            raise ValueError(f"Unsupported composer strategy: {strategy}")

        memory_block = (
            memory_header + "\n".join(selection.memory_lines)
            if selection.memory_lines
            else ""
        )
        total_tokens_estimate = (
            self.estimate_tokens(contract_block)
            + self.estimate_tokens(workspace_block)
            + self.estimate_tokens(memory_block)
            + self.estimate_tokens(state_block)
        )
        if total_tokens_estimate > budget_tokens:
            raise RuntimeError("Context composition exceeded the resolved token budget")
        selected = selection.selected
        items_included = len(selected)
        items_dropped = len(candidates) - items_included
        return ComposedContext(
            contract_block=contract_block,
            workspace_block=workspace_block,
            memory_block=memory_block,
            state_block=state_block,
            selected_memory_ids=[candidate.memory_id for candidate in selected],
            total_tokens_estimate=total_tokens_estimate,
            budget_tokens=budget_tokens,
            items_included=items_included,
            items_dropped=items_dropped,
        )

    @classmethod
    def _select_score_first(
        cls,
        candidates: list[ScoredCandidate],
        *,
        candidate_by_id: dict[str, ScoredCandidate],
        remaining_budget: int,
        max_items: int,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
    ) -> _MemorySelection:
        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()
        memory_lines: list[str] = []
        for candidate in candidates:
            if len(selected) >= max_items or remaining_budget <= 0:
                break
            if candidate.memory_id in selected_ids:
                continue

            if cls._is_hierarchical_summary_candidate(candidate):
                conflicting_l0 = cls._find_conflicting_fresher_l0(
                    candidate,
                    candidates,
                    selected_ids,
                )
                if conflicting_l0 is not None:
                    remaining_budget = cls._append_candidate_if_possible(
                        conflicting_l0,
                        selected=selected,
                        selected_ids=selected_ids,
                        memory_lines=memory_lines,
                        remaining_budget=remaining_budget,
                        max_items=max_items,
                        source_messages_by_id=source_messages_by_id,
                        source_quote_options=source_quote_options,
                    )
                    continue

                supporting_l0 = cls._supporting_l0_candidate(
                    candidate,
                    candidate_by_id,
                    selected_ids,
                )
                if supporting_l0 is None:
                    continue
                if supporting_l0.memory_id not in selected_ids:
                    required_items = 2
                    required_tokens = cls.estimate_tokens(
                        cls._format_memory_entry(len(selected) + 1, supporting_l0)
                    ) + cls.estimate_tokens(
                        cls._format_memory_entry(len(selected) + 2, candidate)
                    )
                    if (
                        len(selected) + required_items > max_items
                        or required_tokens > remaining_budget
                    ):
                        continue
                    remaining_budget = cls._append_candidate_if_possible(
                        supporting_l0,
                        selected=selected,
                        selected_ids=selected_ids,
                        memory_lines=memory_lines,
                        remaining_budget=remaining_budget,
                        max_items=max_items,
                        source_messages_by_id=source_messages_by_id,
                        source_quote_options=source_quote_options,
                    )
                remaining_budget = cls._append_candidate_if_possible(
                    candidate,
                    selected=selected,
                    selected_ids=selected_ids,
                    memory_lines=memory_lines,
                    remaining_budget=remaining_budget,
                    max_items=max_items,
                    source_messages_by_id=source_messages_by_id,
                    source_quote_options=source_quote_options,
                )
                continue

            remaining_budget = cls._append_candidate_if_possible(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                memory_lines=memory_lines,
                remaining_budget=remaining_budget,
                max_items=max_items,
                source_messages_by_id=source_messages_by_id,
                source_quote_options=source_quote_options,
            )
        return _MemorySelection(
            selected=selected,
            memory_lines=memory_lines,
        )

    @classmethod
    def _select_budgeted_marginal(
        cls,
        candidates: list[ScoredCandidate],
        *,
        candidate_by_id: dict[str, ScoredCandidate],
        remaining_budget: int,
        max_items: int,
        query_text: str | None,
        query_type: QueryType,
        exact_recall_mode: bool,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
    ) -> _MemorySelection:
        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()
        blocked_ids: set[str] = set()
        memory_lines: list[str] = []
        order_by_id = {
            candidate.memory_id: index for index, candidate in enumerate(candidates)
        }
        query_tokens = cls._content_tokens(query_text or "")
        repeated_pool_tokens = cls._repeated_pool_tokens(candidates)
        profile = cls._selection_profile(query_type)

        while len(selected) < max_items and remaining_budget > 0:
            best_action: _SelectionAction | None = None
            best_key: tuple[float, float, float, float, float, str] | None = None
            for candidate in candidates:
                if (
                    candidate.memory_id in selected_ids
                    or candidate.memory_id in blocked_ids
                ):
                    continue
                action = cls._budgeted_action_for_candidate(
                    candidate,
                    candidates=candidates,
                    candidate_by_id=candidate_by_id,
                    selected=selected,
                    selected_ids=selected_ids,
                    blocked_ids=blocked_ids,
                    remaining_budget=remaining_budget,
                    max_items=max_items,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                    profile=profile,
                    exact_recall_mode=exact_recall_mode,
                    order_by_id=order_by_id,
                )
                if action is None:
                    continue
                density = action.utility / max(1, action.token_cost)
                action_key = (
                    float(-action.exact_priority),
                    density,
                    action.utility,
                    float(-action.token_cost),
                    float(-action.first_order),
                    "|".join(item.memory_id for item in action.items),
                )
                if best_key is None or action_key > best_key:
                    best_key = action_key
                    best_action = action

            if best_action is None:
                break

            for item in best_action.items:
                remaining_budget = cls._append_candidate_if_possible(
                    item,
                    selected=selected,
                    selected_ids=selected_ids,
                    memory_lines=memory_lines,
                    remaining_budget=remaining_budget,
                    max_items=max_items,
                    source_messages_by_id=source_messages_by_id,
                    source_quote_options=source_quote_options,
                )
            blocked_ids.update(best_action.blocked_ids)

        return _MemorySelection(
            selected=selected,
            memory_lines=memory_lines,
        )

    @classmethod
    def _selection_order(
        cls,
        candidates: list[ScoredCandidate],
        *,
        max_items: int,
        query_text: str | None,
        query_type: QueryType,
    ) -> list[ScoredCandidate]:
        if len(candidates) <= 1 or not query_text or max_items <= 0:
            return candidates

        profile = cls._selection_profile(query_type)
        if (
            profile.diversity_weight <= 0.0
            and profile.richness_weight <= 0.0
            and profile.source_grounding_weight <= 0.0
            and profile.coverage_gain_weight <= 0.0
        ):
            return candidates

        pool_size = min(
            len(candidates),
            max(max_items * 3, max_items + profile.pool_extra),
        )
        pool = list(candidates[:pool_size])
        remainder = candidates[pool_size:]
        query_tokens = cls._content_tokens(query_text)
        repeated_pool_tokens = cls._repeated_pool_tokens(pool)
        selected: list[ScoredCandidate] = []
        covered_tokens: set[str] = set()

        while pool:
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(pool):
                redundancy = (
                    max(
                        cls._candidate_similarity(
                            candidate,
                            chosen,
                            ignored_tokens=repeated_pool_tokens,
                        )
                        for chosen in selected
                    )
                    if selected
                    else 0.0
                )
                richness = cls._candidate_richness(
                    candidate,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                )
                source_grounding = cls._candidate_source_grounding(candidate)
                coverage_gain = (
                    cls._candidate_coverage_gain(
                        candidate,
                        covered_tokens=covered_tokens,
                        query_tokens=query_tokens,
                        repeated_pool_tokens=repeated_pool_tokens,
                    )
                    if selected
                    else 0.0
                )
                selection_score = (
                    float(candidate.final_score)
                    + (profile.richness_weight * richness)
                    + (profile.source_grounding_weight * source_grounding)
                    + (profile.coverage_gain_weight * coverage_gain)
                    - (profile.diversity_weight * redundancy)
                )
                if selection_score > best_score:
                    best_index = index
                    best_score = selection_score
                    continue
                if math.isclose(selection_score, best_score):
                    incumbent = pool[best_index]
                    if (
                        float(candidate.final_score),
                        candidate.memory_id,
                    ) > (
                        float(incumbent.final_score),
                        incumbent.memory_id,
                    ):
                        best_index = index
                        best_score = selection_score
            selected_candidate = pool.pop(best_index)
            selected.append(selected_candidate)
            covered_tokens.update(
                cls._candidate_coverage_tokens(
                    selected_candidate,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                )
            )

        return [*selected, *remainder]

    @staticmethod
    def _selection_profile(query_type: QueryType) -> _SelectionProfile:
        if query_type == "broad_list":
            return _SelectionProfile(
                diversity_weight=0.32,
                richness_weight=0.12,
                source_grounding_weight=0.04,
                coverage_gain_weight=0.10,
                pool_extra=24,
            )
        if query_type == "temporal":
            return _SelectionProfile(
                diversity_weight=0.03,
                richness_weight=0.01,
                source_grounding_weight=0.04,
                coverage_gain_weight=0.0,
                pool_extra=10,
            )
        if query_type == "slot_fill":
            return _SelectionProfile(
                diversity_weight=0.18,
                richness_weight=0.07,
                source_grounding_weight=0.04,
                coverage_gain_weight=0.0,
                pool_extra=16,
            )
        return _SelectionProfile(
            diversity_weight=0.08,
            richness_weight=0.03,
            source_grounding_weight=0.0,
            coverage_gain_weight=0.0,
            pool_extra=12,
        )

    @classmethod
    def _budgeted_action_for_candidate(
        cls,
        candidate: ScoredCandidate,
        *,
        candidates: list[ScoredCandidate],
        candidate_by_id: dict[str, ScoredCandidate],
        selected: list[ScoredCandidate],
        selected_ids: set[str],
        blocked_ids: set[str],
        remaining_budget: int,
        max_items: int,
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
        profile: _SelectionProfile,
        exact_recall_mode: bool,
        order_by_id: dict[str, int],
    ) -> _SelectionAction | None:
        action_items: tuple[ScoredCandidate, ...]
        action_blocked_ids: frozenset[str] = frozenset()
        if cls._is_hierarchical_summary_candidate(candidate):
            conflicting_l0 = cls._find_conflicting_fresher_l0(
                candidate,
                candidates,
                set(),
            )
            if conflicting_l0 is not None:
                blocked_ids.add(candidate.memory_id)
                if conflicting_l0.memory_id in selected_ids:
                    return None
                action_items = (conflicting_l0,)
                action_blocked_ids = frozenset({candidate.memory_id})
            else:
                supporting_l0 = cls._supporting_l0_candidate(
                    candidate,
                    candidate_by_id,
                    selected_ids,
                )
                if supporting_l0 is None:
                    blocked_ids.add(candidate.memory_id)
                    return None
                if (
                    supporting_l0.memory_id not in selected_ids
                    and supporting_l0.memory_id in blocked_ids
                ):
                    return None
                action_items = (
                    (candidate,)
                    if supporting_l0.memory_id in selected_ids
                    else (supporting_l0, candidate)
                )
        else:
            action_items = (candidate,)

        new_items = tuple(
            item
            for item in action_items
            if item.memory_id not in selected_ids and item.memory_id not in blocked_ids
        )
        if not new_items:
            return None
        if len(selected) + len(new_items) > max_items:
            return None

        token_cost = cls._action_token_cost(new_items, selected_count=len(selected))
        if token_cost > remaining_budget:
            return None

        utility = sum(
            cls._budgeted_marginal_utility(
                item,
                selected=selected,
                query_tokens=query_tokens,
                repeated_pool_tokens=repeated_pool_tokens,
                profile=profile,
            )
            for item in new_items
        )
        exact_priority = max(
            cls._exact_recall_level_priority(item, exact_recall_mode)
            for item in new_items
        )
        first_order = min(
            order_by_id.get(item.memory_id, len(order_by_id)) for item in new_items
        )
        return _SelectionAction(
            items=new_items,
            blocked_ids=action_blocked_ids,
            token_cost=token_cost,
            utility=utility,
            exact_priority=exact_priority,
            first_order=first_order,
        )

    @classmethod
    def _action_token_cost(
        cls,
        items: tuple[ScoredCandidate, ...],
        *,
        selected_count: int,
    ) -> int:
        return sum(
            cls.estimate_tokens(cls._format_memory_entry(selected_count + offset, item))
            for offset, item in enumerate(items, start=1)
        )

    @classmethod
    def _budgeted_marginal_utility(
        cls,
        candidate: ScoredCandidate,
        *,
        selected: list[ScoredCandidate],
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
        profile: _SelectionProfile,
    ) -> float:
        redundancy = (
            max(
                cls._candidate_similarity(
                    candidate,
                    chosen,
                    ignored_tokens=repeated_pool_tokens,
                )
                for chosen in selected
            )
            if selected
            else 0.0
        )
        richness = cls._candidate_richness(
            candidate,
            query_tokens=query_tokens,
            repeated_pool_tokens=repeated_pool_tokens,
        )
        source_grounding = cls._candidate_source_grounding(candidate)
        coverage_gain = (
            cls._candidate_coverage_gain(
                candidate,
                covered_tokens=cls._selected_coverage_tokens(
                    selected,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                ),
                query_tokens=query_tokens,
                repeated_pool_tokens=repeated_pool_tokens,
            )
            if selected
            else 0.0
        )
        return max(
            0.0,
            float(candidate.final_score)
            + (profile.richness_weight * richness)
            + (profile.source_grounding_weight * source_grounding)
            + (profile.coverage_gain_weight * coverage_gain)
            - (profile.diversity_weight * redundancy),
        )

    @classmethod
    def _candidate_similarity(
        cls,
        candidate: ScoredCandidate,
        other: ScoredCandidate,
        *,
        ignored_tokens: set[str],
    ) -> float:
        candidate_tokens = cls._content_tokens(
            str(candidate.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        other_tokens = cls._content_tokens(
            str(other.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        source_similarity = cls._candidate_source_similarity(candidate, other)
        if not candidate_tokens or not other_tokens:
            return source_similarity
        overlap = candidate_tokens & other_tokens
        union = candidate_tokens | other_tokens
        similarity = len(overlap) / len(union)
        similarity = max(similarity, source_similarity)
        candidate_window = cls._candidate_source_window(candidate)
        if (
            candidate_window is not None
            and candidate_window == cls._candidate_source_window(other)
        ):
            similarity = max(similarity, 0.55)
        return min(1.0, similarity)

    @classmethod
    def _candidate_source_similarity(
        cls,
        candidate: ScoredCandidate,
        other: ScoredCandidate,
    ) -> float:
        candidate_ids = set(cls._candidate_source_message_ids(candidate))
        other_ids = set(cls._candidate_source_message_ids(other))
        if not candidate_ids or not other_ids:
            return 0.0
        overlap = candidate_ids & other_ids
        if not overlap:
            return 0.0
        return len(overlap) / max(1, min(len(candidate_ids), len(other_ids)))

    @classmethod
    def _candidate_source_grounding(cls, candidate: ScoredCandidate) -> float:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return 0.0
        if payload_json.get("source_kind_variant") == "conversation_window":
            return 1.0
        if cls._candidate_source_message_ids(candidate):
            return 0.85
        if payload_json.get("source_excerpt_messages"):
            return 0.75
        if cls._candidate_source_window(candidate) is not None:
            return 0.45
        return 0.0

    @classmethod
    def _candidate_richness(
        cls,
        candidate: ScoredCandidate,
        *,
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
    ) -> float:
        ignored_tokens = {*query_tokens, *repeated_pool_tokens}
        candidate_tokens = cls._content_tokens(
            str(candidate.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        if not candidate_tokens:
            return 0.0
        richness = min(1.0, sum(len(token) for token in candidate_tokens) / 40.0)
        if str(candidate.memory_object.get("object_type")) in {
            "belief",
            "interaction_contract",
            "state_snapshot",
        }:
            return richness * 0.7
        return richness

    @classmethod
    def _candidate_coverage_gain(
        cls,
        candidate: ScoredCandidate,
        *,
        covered_tokens: set[str],
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
    ) -> float:
        if cls._candidate_source_grounding(candidate) <= 0.0:
            return 0.0
        candidate_tokens = cls._candidate_coverage_tokens(
            candidate,
            query_tokens=query_tokens,
            repeated_pool_tokens=repeated_pool_tokens,
        )
        if not candidate_tokens:
            return 0.0
        new_tokens = candidate_tokens - covered_tokens
        if not new_tokens:
            return 0.0
        return min(1.0, sum(len(token) for token in new_tokens) / 40.0)

    @classmethod
    def _selected_coverage_tokens(
        cls,
        selected: list[ScoredCandidate],
        *,
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
    ) -> set[str]:
        covered_tokens: set[str] = set()
        for candidate in selected:
            covered_tokens.update(
                cls._candidate_coverage_tokens(
                    candidate,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                )
            )
        return covered_tokens

    @classmethod
    def _candidate_coverage_tokens(
        cls,
        candidate: ScoredCandidate,
        *,
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
    ) -> set[str]:
        return cls._content_tokens(
            str(candidate.memory_object.get("canonical_text", "")),
            ignored_tokens={*query_tokens, *repeated_pool_tokens},
        )

    @staticmethod
    def _candidate_source_window(candidate: ScoredCandidate) -> tuple[str, str] | None:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        start = str(
            payload_json.get("source_message_window_start_occurred_at")
            or payload_json.get("source_window_start_occurred_at")
            or ""
        ).strip()
        end = str(
            payload_json.get("source_message_window_end_occurred_at")
            or payload_json.get("source_window_end_occurred_at")
            or ""
        ).strip()
        if not start and not end:
            return None
        return (start, end)

    @classmethod
    def _content_tokens(
        cls,
        text: str,
        *,
        ignored_tokens: set[str] | None = None,
    ) -> set[str]:
        ignored = ignored_tokens or set()
        return {
            normalized
            for token in cls._tokenize(text)
            if (normalized := cls._normalize_token(token))
            and normalized not in ignored
            and len(normalized) > 1
        }

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.strip("_")
        if not normalized:
            return ""
        if not any(character.isalnum() for character in normalized):
            return ""
        return normalized

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for character in text.lower():
            if character.isalnum() or character == "_":
                current.append(character)
                continue
            if current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens

    @classmethod
    def _repeated_pool_tokens(cls, candidates: list[ScoredCandidate]) -> set[str]:
        token_counts: dict[str, int] = {}
        for candidate in candidates:
            for token in cls._content_tokens(
                str(candidate.memory_object.get("canonical_text", ""))
            ):
                token_counts[token] = token_counts.get(token, 0) + 1
        return {token for token, count in token_counts.items() if count >= 2}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def _coerce_scored_candidate(
        candidate: ScoredCandidate | dict[str, Any],
    ) -> ScoredCandidate:
        if isinstance(candidate, ScoredCandidate):
            return candidate
        return ScoredCandidate.model_validate(candidate)

    @staticmethod
    def _exact_recall_level_priority(
        candidate: ScoredCandidate,
        exact_recall_mode: bool,
    ) -> int:
        """Return the exact-recall hierarchy bucket for a candidate.

        Lower is better. Concrete L0 evidence (including verbatim evidence search
        windows) sits in bucket 0, while episode/thematic summaries
        drop to bucket 1. Outside exact-recall mode all candidates are
        treated equally so ordering stays score-driven.
        """
        if not exact_recall_mode:
            return 0
        memory_object = candidate.memory_object
        object_type = str(memory_object.get("object_type"))
        if object_type != "summary_view":
            return 0
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return 1
        try:
            hierarchy_level = int(payload_json.get("hierarchy_level", 0))
        except (TypeError, ValueError):
            return 1
        return 0 if hierarchy_level == 0 else 1

    def _format_contract_block(
        self,
        current_contract: dict[str, dict[str, Any]],
        resolved_policy: ResolvedPolicy,
    ) -> str:
        ordered_dimensions = list(resolved_policy.contract_dimensions_priority)
        ordered_dimensions.extend(
            sorted(
                dimension
                for dimension in current_contract
                if dimension not in resolved_policy.contract_dimensions_priority
            )
        )
        lines = ["[Interaction Contract]"]
        for dimension in ordered_dimensions:
            value = current_contract.get(dimension)
            if value is None:
                continue
            lines.append(f"- {dimension}: {self._format_contract_value(value)}")
        return "\n".join(lines)

    @staticmethod
    def _format_contract_value(value: dict[str, Any]) -> str:
        if not value:
            return "unspecified"
        label = value.get("label")
        if label is None:
            extras = {
                key: item
                for key, item in value.items()
                if key not in {"score", "confidence"}
            }
            label = json_utils.dumps(extras or value, sort_keys=True)
        rendered = str(label)
        confidence = value.get("confidence", value.get("score"))
        if isinstance(confidence, (int, float)):
            rendered += f" (confidence: {float(confidence):.2f})"
        return rendered

    @classmethod
    def _format_workspace_block(
        cls,
        workspace_rollup: dict[str, Any] | None,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        if not workspace_rollup:
            return ""
        summary_text = str(workspace_rollup.get("summary_text", "")).strip()
        if not summary_text:
            return ""
        block = f"[Workspace Context]\n{summary_text}"
        if budget_tokens is None:
            return block
        max_tokens = max(1, math.ceil(budget_tokens * cls.WORKSPACE_BLOCK_RATIO))
        return cls._truncate_block(
            block,
            max_tokens=max_tokens,
            header="[Workspace Context]",
        )

    @staticmethod
    def _format_memory_entry(
        index: int,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]] | None = None,
        source_quote_options: _SourceQuoteOptions | None = None,
    ) -> str:
        memory_object = candidate.memory_object
        confidence = float(memory_object.get("confidence", 0.0))
        payload_json = memory_object.get("payload_json") or {}
        is_conversation_chunk = (
            memory_object.get("object_type") == "summary_view"
            and isinstance(payload_json, dict)
            and payload_json.get("summary_kind") == "conversation_chunk"
        )
        metadata_parts = [
            f"{memory_object.get('object_type')}",
            f"confidence: {confidence:.2f}",
            f"scope: {memory_object.get('scope')}",
        ]
        privacy_level = memory_object.get("privacy_level")
        if privacy_level is not None:
            metadata_parts.append(f"privacy_level: {privacy_level}")
        memory_category = str(memory_object.get("memory_category") or "").strip()
        if memory_category:
            metadata_parts.append(f"memory_category: {memory_category}")
        intimacy_boundary = str(memory_object.get("intimacy_boundary") or "").strip()
        if intimacy_boundary and intimacy_boundary != "ordinary":
            metadata_parts.append(f"intimacy_boundary: {intimacy_boundary}")
        if bool(memory_object.get("preserve_verbatim")):
            metadata_parts.append("preserve_verbatim: true")
        temporal_type = str(memory_object.get("temporal_type") or "unknown")
        valid_from = memory_object.get("valid_from")
        valid_to = memory_object.get("valid_to")
        if valid_from or valid_to:
            if temporal_type == "event_triggered":
                metadata_parts.append(
                    f"event_time: {valid_from or '?'} to {valid_to or '?'}"
                )
            else:
                metadata_parts.append(
                    f"valid_window: {valid_from or '?'} to {valid_to or '?'}"
                )
        source_window = ContextComposer._candidate_source_window(candidate)
        if source_window is not None:
            source_window_start, source_window_end = source_window
            metadata_parts.append(
                f"source_window: {source_window_start or '?'} to {source_window_end or '?'}"
            )
        if candidate.resolved_date:
            metadata_parts.append(f"resolved_date: {candidate.resolved_date}")
        lines = [
            f"{index}. ({', '.join(metadata_parts)})\n"
            f"   {memory_object.get('canonical_text', '')}"
        ]
        source_quote = ContextComposer._source_quote_for_candidate(
            candidate,
            source_messages_by_id=source_messages_by_id or {},
            options=source_quote_options or _SourceQuoteOptions(enabled=False),
        )
        if source_quote:
            lines.append(f"   source_quote: {source_quote}")
        elif is_conversation_chunk:
            excerpt = ContextComposer._conversation_chunk_excerpt(payload_json)
            if excerpt:
                lines.append(f"   source_excerpt: {excerpt}")
        return "\n".join(lines)

    @staticmethod
    def _format_state_block(user_state: dict[str, Any] | None) -> str:
        if not user_state:
            return ""
        lines = ["[Current User State]"]
        for key, value in sorted(user_state.items()):
            if isinstance(value, (dict, list)):
                rendered = json_utils.dumps(value, sort_keys=True)
            else:
                rendered = str(value)
            lines.append(f"- {key}: {rendered}")
        return "\n".join(lines)

    @classmethod
    def _budget_block(
        cls,
        text: str,
        *,
        total_budget: int,
        remaining_budget: int,
        ratio: float,
        min_tokens: int,
        header: str,
    ) -> tuple[str, int]:
        if not text or remaining_budget <= 0:
            return "", remaining_budget
        allocated_tokens = min(
            remaining_budget,
            max(min_tokens, math.ceil(total_budget * ratio)),
        )
        block = cls._truncate_block(
            text,
            max_tokens=allocated_tokens,
            header=header,
        )
        return block, max(0, remaining_budget - cls.estimate_tokens(block))

    @classmethod
    def _truncate_block(cls, text: str, *, max_tokens: int, header: str = "") -> str:
        if max_tokens <= 0 or not text:
            return ""
        if cls.estimate_tokens(text) <= max_tokens:
            return text

        max_chars = max_tokens * 4
        if not header:
            if max_chars <= 3:
                return text[:max_chars]
            return f"{text[: max_chars - 3].rstrip()}..."

        if max_chars <= len(header):
            return header[:max_chars].rstrip()

        suffix = "..."
        content = text[len(header) :].strip()
        available_chars = max(0, max_chars - len(header) - len(suffix))
        if available_chars == 0:
            return header.rstrip()
        truncated = content[:available_chars].rstrip()
        return f"{header}{truncated}{suffix}"

    @classmethod
    def _append_candidate_if_possible(
        cls,
        candidate: ScoredCandidate,
        *,
        selected: list[ScoredCandidate],
        selected_ids: set[str],
        memory_lines: list[str],
        remaining_budget: int,
        max_items: int,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
    ) -> int:
        if (
            candidate.memory_id in selected_ids
            or remaining_budget <= 0
            or len(selected) >= max_items
        ):
            return remaining_budget
        candidate_block = cls._format_memory_entry(len(selected) + 1, candidate)
        candidate_tokens = cls.estimate_tokens(candidate_block)
        if candidate_tokens > remaining_budget:
            return remaining_budget
        ranked_source_quote_options = cls._ranked_source_quote_options(
            source_quote_options,
            rank=len(selected) + 1,
        )
        candidate_block_with_source = cls._format_memory_entry(
            len(selected) + 1,
            candidate,
            source_messages_by_id=source_messages_by_id,
            source_quote_options=ranked_source_quote_options,
        )
        candidate_tokens_with_source = cls.estimate_tokens(candidate_block_with_source)
        if candidate_tokens_with_source <= remaining_budget:
            candidate_block = candidate_block_with_source
            candidate_tokens = candidate_tokens_with_source
        else:
            compact_source_options = cls._compact_source_quote_options(
                ranked_source_quote_options
            )
            if compact_source_options != ranked_source_quote_options:
                compact_candidate_block = cls._format_memory_entry(
                    len(selected) + 1,
                    candidate,
                    source_messages_by_id=source_messages_by_id,
                    source_quote_options=compact_source_options,
                )
                compact_candidate_tokens = cls.estimate_tokens(compact_candidate_block)
                if compact_candidate_tokens <= remaining_budget:
                    candidate_block = compact_candidate_block
                    candidate_tokens = compact_candidate_tokens
        memory_lines.append(candidate_block)
        selected.append(candidate)
        selected_ids.add(candidate.memory_id)
        return remaining_budget - candidate_tokens

    @staticmethod
    def _is_hierarchical_summary_candidate(candidate: ScoredCandidate) -> bool:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if candidate.memory_object.get(
            "object_type"
        ) != "summary_view" or not isinstance(payload_json, dict):
            return False
        return int(payload_json.get("hierarchy_level", -1)) in {1, 2}

    @classmethod
    def _supporting_l0_candidate(
        cls,
        candidate: ScoredCandidate,
        candidate_by_id: dict[str, ScoredCandidate],
        selected_ids: set[str],
    ) -> ScoredCandidate | None:
        source_ids = cls._candidate_source_ids(candidate)
        if not source_ids:
            return None
        supporting_candidate = cls._best_supporting_l0_candidate(
            source_ids=source_ids,
            candidate_by_id=candidate_by_id,
            selected_ids=selected_ids,
            visited={candidate.memory_id},
            allow_selected=False,
        )
        if supporting_candidate is not None:
            return supporting_candidate
        return cls._best_supporting_l0_candidate(
            source_ids=source_ids,
            candidate_by_id=candidate_by_id,
            selected_ids=selected_ids,
            visited={candidate.memory_id},
            allow_selected=True,
        )

    @staticmethod
    def _candidate_source_ids(candidate: ScoredCandidate) -> list[str]:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return []
        return [
            str(item).strip()
            for item in payload_json.get("source_object_ids", [])
            if str(item).strip()
        ]

    @classmethod
    def _best_supporting_l0_candidate(
        cls,
        *,
        source_ids: list[str],
        candidate_by_id: dict[str, ScoredCandidate],
        selected_ids: set[str],
        visited: set[str],
        allow_selected: bool,
    ) -> ScoredCandidate | None:
        direct_support: list[ScoredCandidate] = []
        recursive_support: list[ScoredCandidate] = []
        for source_id in source_ids:
            supporting_candidate = candidate_by_id.get(source_id)
            if supporting_candidate is None:
                continue
            if supporting_candidate.memory_object.get("object_type") != "summary_view":
                if allow_selected or supporting_candidate.memory_id not in selected_ids:
                    direct_support.append(supporting_candidate)
                continue
            if supporting_candidate.memory_id in visited:
                continue
            nested_source_ids = cls._candidate_source_ids(supporting_candidate)
            if not nested_source_ids:
                continue
            recursive_support_candidate = cls._best_supporting_l0_candidate(
                source_ids=nested_source_ids,
                candidate_by_id=candidate_by_id,
                selected_ids=selected_ids,
                visited={*visited, supporting_candidate.memory_id},
                allow_selected=allow_selected,
            )
            if recursive_support_candidate is not None:
                recursive_support.append(recursive_support_candidate)
        all_support = [*direct_support, *recursive_support]
        if not all_support:
            return None
        return sorted(
            all_support,
            key=lambda item: (-item.final_score, item.memory_id),
        )[0]

    @classmethod
    def _find_conflicting_fresher_l0(
        cls,
        candidate: ScoredCandidate,
        all_candidates: list[ScoredCandidate],
        selected_ids: set[str],
    ) -> ScoredCandidate | None:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        claim_signatures = payload_json.get("source_claim_signatures", [])
        if not isinstance(claim_signatures, list) or not claim_signatures:
            return None
        summary_updated_at = cls._parse_candidate_datetime(
            candidate.memory_object.get("updated_at")
        )
        source_object_ids = {
            str(item).strip()
            for item in payload_json.get("source_object_ids", [])
            if str(item).strip()
        }
        expected_values_by_key: dict[str, set[str]] = {}
        for signature in claim_signatures:
            if not isinstance(signature, dict):
                continue
            claim_key = str(signature.get("claim_key") or "").strip()
            if not claim_key:
                continue
            expected_values_by_key.setdefault(claim_key, set()).add(
                json_utils.dumps(signature.get("claim_value"), sort_keys=True)
            )
        if not expected_values_by_key:
            return None

        conflicting: list[ScoredCandidate] = []
        for other in all_candidates:
            if other.memory_id in selected_ids or other.memory_id in source_object_ids:
                continue
            if other.memory_object.get("object_type") != "belief":
                continue
            other_payload = other.memory_object.get("payload_json") or {}
            if not isinstance(other_payload, dict):
                continue
            claim_key = str(other_payload.get("claim_key") or "").strip()
            if claim_key not in expected_values_by_key:
                continue
            claim_value = json_utils.dumps(
                other_payload.get("claim_value"), sort_keys=True
            )
            if claim_value in expected_values_by_key[claim_key]:
                continue
            other_updated_at = cls._parse_candidate_datetime(
                other.memory_object.get("updated_at")
            )
            if (
                summary_updated_at is not None
                and other_updated_at is not None
                and other_updated_at <= summary_updated_at
            ):
                continue
            conflicting.append(other)
        if not conflicting:
            return None
        return sorted(
            conflicting, key=lambda item: (-item.final_score, item.memory_id)
        )[0]

    @staticmethod
    def _parse_candidate_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @staticmethod
    def _conversation_chunk_excerpt(payload_json: dict[str, Any]) -> str:
        excerpt_messages = payload_json.get("source_excerpt_messages")
        if not isinstance(excerpt_messages, list):
            return ""
        rendered: list[str] = []
        for item in excerpt_messages:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            role = str(item.get("role", "")).strip() or "unknown"
            occurred_at = str(item.get("occurred_at", "")).strip()
            prefix = role
            if occurred_at:
                prefix += f" @ {occurred_at}"
            rendered.append(f"{prefix}: {text}")
        return " | ".join(rendered)

    @staticmethod
    def _source_quote_options(
        *,
        query_type: QueryType,
        exact_recall_mode: bool,
    ) -> _SourceQuoteOptions:
        enabled = exact_recall_mode or query_type in {
            "temporal",
            "slot_fill",
            "broad_list",
        }
        if not enabled:
            return _SourceQuoteOptions(enabled=False)
        if query_type == "temporal":
            max_entries = 4
        elif query_type in {"slot_fill", "broad_list"}:
            max_entries = 6
        else:
            max_entries = 5
        return _SourceQuoteOptions(
            enabled=True,
            max_entries=max_entries,
        )

    @staticmethod
    def _ranked_source_quote_options(
        options: _SourceQuoteOptions,
        *,
        rank: int,
    ) -> _SourceQuoteOptions:
        if (
            not options.enabled
            or options.max_entries is None
            or rank <= options.max_entries
        ):
            return options
        return _SourceQuoteOptions(enabled=False)

    @staticmethod
    def _compact_source_quote_options(
        options: _SourceQuoteOptions,
    ) -> _SourceQuoteOptions:
        if not options.enabled:
            return options
        return _SourceQuoteOptions(
            enabled=True,
            max_messages=min(options.max_messages, 1),
            max_chars=min(options.max_chars, 220),
            max_message_chars=min(options.max_message_chars, 180),
            max_entries=options.max_entries,
        )

    @staticmethod
    def _source_messages_by_id(
        conversation_messages: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        messages_by_id: dict[str, dict[str, Any]] = {}
        for message in conversation_messages:
            message_id = str(message.get("id") or "").strip()
            if not message_id or message_id in messages_by_id:
                continue
            messages_by_id[message_id] = message
        return messages_by_id

    @classmethod
    def _source_quote_for_candidate(
        cls,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        options: _SourceQuoteOptions,
    ) -> str:
        if not options.enabled:
            return ""
        memory_object = candidate.memory_object
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return ""
        if payload_json.get("source_kind_variant") == "conversation_window":
            return ""
        if not source_messages_by_id:
            return ""
        source_message_ids = cls._candidate_source_message_ids(candidate)
        if not source_message_ids:
            return ""

        rendered: list[str] = []
        total_chars = 0
        for source_message_id in source_message_ids:
            if len(rendered) >= options.max_messages:
                break
            message = source_messages_by_id.get(source_message_id)
            if message is None or not cls._message_allows_source_quote(message):
                continue
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            text = cls._truncate_inline(text, options.max_message_chars)
            prefix_parts = [str(message.get("role") or "unknown").strip() or "unknown"]
            occurred_at = str(message.get("occurred_at") or "").strip()
            if occurred_at:
                prefix_parts.append(f"@ {occurred_at}")
            seq = message.get("seq")
            if seq is not None:
                prefix_parts.append(f"seq {seq}")
            segment = f"{' '.join(prefix_parts)}: {text}"
            segment = cls._truncate_inline(
                segment, max(1, options.max_chars - total_chars)
            )
            if not segment:
                continue
            rendered.append(segment)
            total_chars += len(segment)
            if total_chars >= options.max_chars:
                break
        return " | ".join(rendered)

    @staticmethod
    def _candidate_source_message_ids(candidate: ScoredCandidate) -> list[str]:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return []
        source_ids = payload_json.get("source_message_ids") or []
        if not isinstance(source_ids, list):
            return []
        seen: set[str] = set()
        normalized: list[str] = []
        for source_id in source_ids:
            value = str(source_id).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    @staticmethod
    def _message_allows_source_quote(message: dict[str, Any]) -> bool:
        include_raw = message.get("include_raw")
        if include_raw is None:
            return True
        if isinstance(include_raw, str):
            return include_raw.strip().lower() not in {"0", "false", "no"}
        if int(include_raw) == 0:
            return False
        return True

    @staticmethod
    def _truncate_inline(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 3:
            return normalized[:max_chars]
        return f"{normalized[: max_chars - 3].rstrip()}..."
