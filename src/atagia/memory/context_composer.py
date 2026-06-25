"""Final context composition within token budgets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Literal

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.text_utils import truncate_inline
from atagia.memory.high_risk_policy import HighRiskDisclosureAction, disclosure_action
from atagia.memory.intimacy_boundary_policy import candidate_allows_intimacy_boundary
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.models.schemas_memory import (
    AnswerShape,
    ComposedContext,
    CoverageMode,
    EvidenceCoverageState,
    MemoryCategory,
    QueryType,
    ScoredCandidate,
    SourcePrecision,
)


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
    remaining_budget: int


@dataclass(slots=True)
class _AnswerEvidencePack:
    block: str
    candidates: list[ScoredCandidate]
    items: list[dict[str, Any]]
    sufficiency: dict[str, Any]


@dataclass(slots=True)
class _CoverageMetadata:
    state: EvidenceCoverageState
    support_map: dict[str, list[str]]
    allowed_values: list[dict[str, Any]]
    missing_slots: list[dict[str, Any]]


_COVERAGE_VALUE_PAYLOAD_KEYS = (
    "value_norm_key",
    "value_key",
    "normalized_key",
    "value_text",
    "value",
    "display_text",
    "surface",
    "subject_surface",
)
_COVERAGE_DISPLAY_PAYLOAD_KEYS = ("value_text", "value", "display_text", "surface")
_COVERAGE_MISSING_SLOT_MODES = frozenset({"exhaustive_known_set", "chronology"})


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
    EXACT_DIRECT_USER_SOURCE_BOOST = 0.03
    EVIDENCE_OBLIGATION_QUERY_TYPES = frozenset({"temporal", "slot_fill", "broad_list"})
    EVIDENCE_NEAR_TIE_SCORE_RATIO = 0.92
    # Absolute-score floor for reserving a verbatim evidence window when
    # source-grounded direct-evidence candidates exist in the pool. A window only
    # qualifies if its final_score >= best_direct_evidence_score * this ratio.
    # Distinct from EVIDENCE_NEAR_TIE_SCORE_RATIO (which only compares windows to
    # each other): this gate prevents low-score, token-heavy windows from being
    # reserved ahead of tiny high-score direct facts and exhausting the budget.
    EVIDENCE_WINDOW_VS_DIRECT_SCORE_RATIO = 0.72
    EVIDENCE_OBLIGATION_MAX_WINDOWS = 3
    EVIDENCE_OBLIGATION_MAX_ITEMS = 4
    EVIDENCE_DUPLICATE_SOURCE_SIMILARITY = 0.67
    FINAL_ANSWER_EVIDENCE_QUERY_TYPES = frozenset(
        {"temporal", "slot_fill", "broad_list"}
    )
    FINAL_ANSWER_EVIDENCE_MAX_ITEMS = 6
    FINAL_ANSWER_EVIDENCE_BUDGET_RATIO = 0.28
    FINAL_ANSWER_EVIDENCE_MAX_TOKENS = 900
    FINAL_ANSWER_EVIDENCE_MIN_TOKENS = 160
    FINAL_ANSWER_EVIDENCE_MIN_SCORE = 0.45
    FINAL_ANSWER_EVIDENCE_RELATIVE_SCORE_RATIO = 0.72
    BROAD_LIST_MATERIAL_LLM_APPLICABILITY = 0.15
    SOURCE_CHAIN_MAX_MESSAGES = 7
    SOURCE_CHAIN_MAX_CHARS = 960
    RENDERED_SOURCE_QUOTE_DEDUPE_MIN_CHARS = 24

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def compose(
        self,
        scored_candidates: list[ScoredCandidate | dict[str, Any]],
        current_contract: dict[str, dict[str, Any]],
        user_state: dict[str, Any] | None,
        resolved_policy: ResolvedRetrievalPolicy,
        conversation_messages: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None = None,
        query_text: str | None = None,
        query_type: QueryType = "default",
        answer_shape: AnswerShape = "open_domain",
        coverage_mode: CoverageMode = "top_support",
        source_precision: SourcePrecision = "preferred",
        exact_recall_mode: bool = False,
        composer_strategy: ComposerStrategy | None = None,
        enable_evidence_obligation_coverage: bool = False,
        enable_final_answer_evidence_pack: bool = False,
        fact_facet_span_coadmission_enabled: bool = False,
        active_presence_id: str | None = None,
        active_realm_id: str | None = None,
        redact_high_risk_secret_literals: bool = True,
    ) -> ComposedContext:
        coerced_candidates = [
            candidate
            for candidate in (
                self._coerce_scored_candidate(candidate)
                for candidate in scored_candidates
            )
            if str(candidate.memory_object.get("canonical_text", "")).strip()
            and self._presence_boundary_allowed(candidate, active_presence_id)
            and candidate_allows_intimacy_boundary(
                candidate.memory_object,
                allow_intimacy_context=resolved_policy.allow_intimacy_context,
            )
        ]
        source_messages_by_id = self._source_messages_by_id(conversation_messages)
        prefer_literal_evidence = self._prefer_literal_evidence(
            query_type=query_type,
            answer_shape=answer_shape,
            exact_recall_mode=exact_recall_mode,
        )
        candidates = sorted(
            coerced_candidates,
            key=lambda candidate: (
                self._exact_recall_level_priority(candidate, exact_recall_mode),
                -self._source_weighted_score(
                    candidate,
                    source_messages_by_id=source_messages_by_id,
                    exact_recall_mode=exact_recall_mode,
                    query_text=query_text,
                ),
                self._literal_evidence_priority(
                    candidate,
                    prefer_literal_evidence=prefer_literal_evidence,
                ),
                candidate.memory_id,
            ),
        )
        candidates = self._selection_order(
            candidates,
            max_items=resolved_policy.retrieval_params.final_context_items,
            query_text=query_text,
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
            source_messages_by_id=source_messages_by_id,
        )
        evidence_obligation_candidates = (
            self._evidence_obligation_candidates(
                candidates,
                max_items=resolved_policy.retrieval_params.final_context_items,
                query_type=query_type,
                answer_shape=answer_shape,
                coverage_mode=coverage_mode,
                source_precision=source_precision,
                exact_recall_mode=exact_recall_mode,
                source_messages_by_id=source_messages_by_id,
            )
            if enable_evidence_obligation_coverage
            else []
        )
        evidence_obligation_ids = {
            candidate.memory_id for candidate in evidence_obligation_candidates
        }
        if evidence_obligation_candidates:
            candidates = [
                *evidence_obligation_candidates,
                *[
                    candidate
                    for candidate in candidates
                    if candidate.memory_id not in evidence_obligation_ids
                ],
            ]
        candidate_by_id = {candidate.memory_id: candidate for candidate in candidates}
        budget_tokens = resolved_policy.context_budget_tokens
        remaining_budget = budget_tokens
        contract_block, remaining_budget = self._budget_block(
            ContextComposer.render_contract_block(current_contract, resolved_policy),
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

        source_quote_options = self._source_quote_options(
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
            context_budget_tokens=budget_tokens,
        )
        max_items = resolved_policy.retrieval_params.final_context_items
        answer_evidence_pack = self._build_answer_evidence_pack(
            candidates,
            query_text=query_text,
            query_type=query_type,
            answer_shape=answer_shape,
            exact_recall_mode=exact_recall_mode,
            source_messages_by_id=source_messages_by_id,
            source_quote_options=source_quote_options,
            max_items=max_items,
            remaining_budget=remaining_budget,
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            render_block=enable_final_answer_evidence_pack,
        )
        answer_evidence_ids = {
            candidate.memory_id for candidate in answer_evidence_pack.candidates
        }
        if answer_evidence_pack.block:
            remaining_budget = max(
                0,
                remaining_budget - self.estimate_tokens(answer_evidence_pack.block),
            )
            candidates = [
                *answer_evidence_pack.candidates,
                *[
                    candidate
                    for candidate in candidates
                    if candidate.memory_id not in answer_evidence_ids
                ],
            ]
            candidate_by_id = {candidate.memory_id: candidate for candidate in candidates}
        regular_max_items = max(0, max_items - len(answer_evidence_ids))
        regular_source_quote_options = (
            _SourceQuoteOptions(enabled=False)
            if answer_evidence_pack.block
            else source_quote_options
        )
        selection_candidates = (
            [
                candidate
                for candidate in candidates
                if candidate.memory_id not in answer_evidence_ids
            ]
            if answer_evidence_pack.block
            else candidates
        )
        strategy = composer_strategy or "score_first"
        if strategy == "score_first":
            selection = self._select_score_first(
                selection_candidates,
                candidate_by_id=candidate_by_id,
                remaining_budget=remaining_budget,
                max_items=regular_max_items,
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                fact_facet_span_coadmission_enabled=(
                    fact_facet_span_coadmission_enabled
                ),
            )
        elif strategy == "budgeted_marginal":
            selection = self._select_budgeted_marginal(
                selection_candidates,
                candidate_by_id=candidate_by_id,
                remaining_budget=remaining_budget,
                max_items=regular_max_items,
                query_text=query_text,
                query_type=query_type,
                exact_recall_mode=exact_recall_mode,
                evidence_obligation_ids=frozenset(evidence_obligation_ids),
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                fact_facet_span_coadmission_enabled=(
                    fact_facet_span_coadmission_enabled
                ),
            )
        else:
            raise ValueError(f"Unsupported composer strategy: {strategy}")

        # Phase 2: upgrade selected entries to their quote-bearing form, funded
        # strictly from budget left over after every bare admission. This keeps
        # the set of selected memory entries identical to a quote-disabled run.
        self._upgrade_entries_with_source_quotes(
            selection,
            query_text=query_text,
            source_messages_by_id=source_messages_by_id,
            source_quote_options=regular_source_quote_options,
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            fact_facet_span_coadmission_enabled=fact_facet_span_coadmission_enabled,
        )

        selected = [
            *answer_evidence_pack.candidates,
            *[
                candidate
                for candidate in selection.selected
                if candidate.memory_id not in answer_evidence_ids
            ],
        ]
        memory_sections = []
        if answer_evidence_pack.block:
            memory_sections.append(answer_evidence_pack.block)
        if selection.memory_lines:
            memory_sections.append(memory_header + "\n".join(selection.memory_lines))
        memory_block = "\n\n".join(memory_sections)
        total_tokens_estimate = (
            self.estimate_tokens(contract_block)
            + self.estimate_tokens(workspace_block)
            + self.estimate_tokens(memory_block)
            + self.estimate_tokens(state_block)
        )
        if total_tokens_estimate > budget_tokens:
            raise RuntimeError("Context composition exceeded the resolved token budget")
        items_included = len(selected)
        items_dropped = len(candidates) - items_included
        coverage_metadata = self._coverage_metadata(
            candidates=candidates,
            selected=selected,
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
            source_precision=source_precision,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
        )
        return ComposedContext(
            contract_block=contract_block,
            workspace_block=workspace_block,
            answer_evidence_block=answer_evidence_pack.block,
            answer_evidence_memory_ids=[
                candidate.memory_id for candidate in answer_evidence_pack.candidates
            ],
            answer_evidence_items=answer_evidence_pack.items,
            answer_evidence_sufficiency=answer_evidence_pack.sufficiency,
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
            source_precision=source_precision,
            coverage_state=coverage_metadata.state,
            support_map=coverage_metadata.support_map,
            allowed_values=coverage_metadata.allowed_values,
            missing_slots=coverage_metadata.missing_slots,
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
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        fact_facet_span_coadmission_enabled: bool = False,
    ) -> _MemorySelection:
        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()
        memory_lines: list[str] = []
        rendered_source_quote_keys: set[str] = set()
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
                        active_realm_id=active_realm_id,
                        redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                        fact_facet_span_coadmission_enabled=(
                            fact_facet_span_coadmission_enabled
                        ),
                        rendered_source_quote_keys=rendered_source_quote_keys,
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
                    required_tokens = cls._action_token_cost(
                        (supporting_l0, candidate),
                        selected_count=len(selected),
                        active_realm_id=active_realm_id,
                        redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                        fact_facet_span_coadmission_enabled=(
                            fact_facet_span_coadmission_enabled
                        ),
                        rendered_source_quote_keys=rendered_source_quote_keys,
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
                        active_realm_id=active_realm_id,
                        redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                        fact_facet_span_coadmission_enabled=(
                            fact_facet_span_coadmission_enabled
                        ),
                        rendered_source_quote_keys=rendered_source_quote_keys,
                    )
                remaining_budget = cls._append_candidate_if_possible(
                    candidate,
                    selected=selected,
                    selected_ids=selected_ids,
                    memory_lines=memory_lines,
                    remaining_budget=remaining_budget,
                    max_items=max_items,
                    active_realm_id=active_realm_id,
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    fact_facet_span_coadmission_enabled=(
                        fact_facet_span_coadmission_enabled
                    ),
                    rendered_source_quote_keys=rendered_source_quote_keys,
                )
                continue

            remaining_budget = cls._append_candidate_if_possible(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                memory_lines=memory_lines,
                remaining_budget=remaining_budget,
                max_items=max_items,
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                fact_facet_span_coadmission_enabled=(
                    fact_facet_span_coadmission_enabled
                ),
                rendered_source_quote_keys=rendered_source_quote_keys,
            )
        return _MemorySelection(
            selected=selected,
            memory_lines=memory_lines,
            remaining_budget=remaining_budget,
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
        evidence_obligation_ids: frozenset[str],
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        fact_facet_span_coadmission_enabled: bool = False,
    ) -> _MemorySelection:
        selected: list[ScoredCandidate] = []
        selected_ids: set[str] = set()
        blocked_ids: set[str] = set()
        memory_lines: list[str] = []
        rendered_source_quote_keys: set[str] = set()
        order_by_id = {
            candidate.memory_id: index for index, candidate in enumerate(candidates)
        }
        query_tokens = cls._content_tokens(query_text or "")
        repeated_pool_tokens = cls._repeated_pool_tokens(candidates)
        profile = cls._selection_profile(query_type)

        for candidate in candidates:
            if candidate.memory_id not in evidence_obligation_ids:
                continue
            if len(selected) >= max_items or remaining_budget <= 0:
                break
            remaining_budget = cls._append_candidate_if_possible(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                memory_lines=memory_lines,
                remaining_budget=remaining_budget,
                max_items=max_items,
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                fact_facet_span_coadmission_enabled=(
                    fact_facet_span_coadmission_enabled
                ),
                rendered_source_quote_keys=rendered_source_quote_keys,
            )

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
                    active_realm_id=active_realm_id,
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    fact_facet_span_coadmission_enabled=(
                        fact_facet_span_coadmission_enabled
                    ),
                    rendered_source_quote_keys=rendered_source_quote_keys,
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
                    active_realm_id=active_realm_id,
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    fact_facet_span_coadmission_enabled=(
                        fact_facet_span_coadmission_enabled
                    ),
                    rendered_source_quote_keys=rendered_source_quote_keys,
                )
            blocked_ids.update(best_action.blocked_ids)

        return _MemorySelection(
            selected=selected,
            memory_lines=memory_lines,
            remaining_budget=remaining_budget,
        )

    @classmethod
    def _upgrade_entries_with_source_quotes(
        cls,
        selection: _MemorySelection,
        *,
        query_text: str | None,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        fact_facet_span_coadmission_enabled: bool,
    ) -> None:
        """Attach source quotes to already-selected entries, in place.

        Walks the selected entries in rank order and, for ranks within the
        per-composition ``max_entries`` cap, upgrades each entry's rendered
        text to a quote-bearing form (full quote -> compact quote -> keep
        bare) funded only by the budget left over after every bare admission.
        Selection membership and order never change; only the rendered text of
        an entry and the running leftover budget are mutated. The cross-entry
        source-quote dedup chain is replayed here so a quote rendered on an
        earlier entry still suppresses its duplicate on a later one.
        """
        if not source_quote_options.enabled or not selection.selected:
            return
        leftover_budget = selection.remaining_budget
        rendered_source_quote_keys: set[str] = set()
        for index, candidate in enumerate(selection.selected):
            rank = index + 1
            current_block = selection.memory_lines[index]
            current_tokens = cls.estimate_tokens(current_block)
            ranked_source_quote_options = cls._ranked_source_quote_options(
                source_quote_options,
                rank=rank,
            )
            if ranked_source_quote_options.enabled:
                upgraded_block = cls._best_quote_bearing_block_within_budget(
                    candidate,
                    rank=rank,
                    bare_tokens=current_tokens,
                    leftover_budget=leftover_budget,
                    query_text=query_text,
                    source_messages_by_id=source_messages_by_id,
                    ranked_source_quote_options=ranked_source_quote_options,
                    active_realm_id=active_realm_id,
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    fact_facet_span_coadmission_enabled=(
                        fact_facet_span_coadmission_enabled
                    ),
                    suppressed_source_quote_keys=frozenset(
                        rendered_source_quote_keys
                    ),
                )
                if upgraded_block is not None:
                    upgraded_tokens = cls.estimate_tokens(upgraded_block)
                    leftover_budget -= upgraded_tokens - current_tokens
                    selection.memory_lines[index] = upgraded_block
                    current_block = upgraded_block
            rendered_source_quote_keys.update(
                cls._source_quote_keys_from_rendered_memory_entry(current_block)
            )

    @classmethod
    def _best_quote_bearing_block_within_budget(
        cls,
        candidate: ScoredCandidate,
        *,
        rank: int,
        bare_tokens: int,
        leftover_budget: int,
        query_text: str | None,
        source_messages_by_id: dict[str, dict[str, Any]],
        ranked_source_quote_options: _SourceQuoteOptions,
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        fact_facet_span_coadmission_enabled: bool,
        suppressed_source_quote_keys: frozenset[str],
    ) -> str | None:
        """Return the richest quote-bearing block that fits leftover budget.

        Tries the full quote first, then the compact quote, returning the
        first whose token delta over the bare form fits the remaining
        leftover budget. Returns ``None`` when no quote-bearing form fits or
        when neither adds anything (the caller then keeps the bare entry).
        """
        full_block = cls._format_memory_entry(
            rank,
            candidate,
            source_messages_by_id=source_messages_by_id,
            source_quote_options=ranked_source_quote_options,
            query_text=query_text,
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            fact_facet_span_coadmission_enabled=fact_facet_span_coadmission_enabled,
            suppressed_source_quote_keys=suppressed_source_quote_keys,
        )
        full_tokens = cls.estimate_tokens(full_block)
        if full_tokens <= bare_tokens:
            # No quote was added (or it shrank the entry); nothing to upgrade.
            return None
        if full_tokens - bare_tokens <= leftover_budget:
            return full_block
        compact_source_options = cls._compact_source_quote_options(
            ranked_source_quote_options
        )
        if compact_source_options == ranked_source_quote_options:
            return None
        compact_block = cls._format_memory_entry(
            rank,
            candidate,
            source_messages_by_id=source_messages_by_id,
            source_quote_options=compact_source_options,
            query_text=query_text,
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            fact_facet_span_coadmission_enabled=fact_facet_span_coadmission_enabled,
            suppressed_source_quote_keys=suppressed_source_quote_keys,
        )
        compact_tokens = cls.estimate_tokens(compact_block)
        if compact_tokens <= bare_tokens:
            return None
        if compact_tokens - bare_tokens <= leftover_budget:
            return compact_block
        return None

    @classmethod
    def _selection_order(
        cls,
        candidates: list[ScoredCandidate],
        *,
        max_items: int,
        query_text: str | None,
        query_type: QueryType,
        exact_recall_mode: bool,
        source_messages_by_id: dict[str, dict[str, Any]],
    ) -> list[ScoredCandidate]:
        if len(candidates) <= 1 or not query_text or max_items <= 0:
            return candidates
        if exact_recall_mode and query_type != "broad_list":
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
                    cls._source_weighted_score(
                        candidate,
                        source_messages_by_id=source_messages_by_id,
                        exact_recall_mode=exact_recall_mode,
                        query_text=query_text,
                    )
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
                        cls._source_weighted_score(
                            candidate,
                            source_messages_by_id=source_messages_by_id,
                            exact_recall_mode=exact_recall_mode,
                            query_text=query_text,
                        ),
                        candidate.memory_id,
                    ) > (
                        cls._source_weighted_score(
                            incumbent,
                            source_messages_by_id=source_messages_by_id,
                            exact_recall_mode=exact_recall_mode,
                            query_text=query_text,
                        ),
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

    @classmethod
    def _evidence_obligation_candidates(
        cls,
        candidates: list[ScoredCandidate],
        *,
        max_items: int,
        query_type: QueryType,
        answer_shape: AnswerShape,
        coverage_mode: CoverageMode,
        source_precision: SourcePrecision,
        exact_recall_mode: bool,
        source_messages_by_id: dict[str, dict[str, Any]],
    ) -> list[ScoredCandidate]:
        source_coverage_required = cls._source_coverage_reserve_applies(
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
            source_precision=source_precision,
            exact_recall_mode=exact_recall_mode,
        )
        exhaustive_coverage = coverage_mode == "exhaustive_known_set"
        if max_items <= 0 or not candidates:
            return []
        if (
            not exact_recall_mode
            and query_type not in cls.EVIDENCE_OBLIGATION_QUERY_TYPES
            and not source_coverage_required
        ):
            return []

        candidate_by_id = {candidate.memory_id: candidate for candidate in candidates}
        reserved: list[ScoredCandidate] = []
        reserved_ids: set[str] = set()

        def reserve(
            candidate: ScoredCandidate | None,
            *,
            allow_source_linked_summary: bool = False,
            dedupe_evidence_source: bool = True,
        ) -> bool:
            if candidate is None or candidate.memory_id in reserved_ids:
                return False
            if cls._is_summary_like_candidate(
                candidate
            ) and not allow_source_linked_summary:
                return False
            if allow_source_linked_summary and not cls._candidate_has_quoteable_source(
                candidate
            ):
                return False
            # Exhaustive lists must be able to reserve one carrier per member, so
            # the EVIDENCE_OBLIGATION_MAX_ITEMS cap is lifted (bounded only by
            # max_items, the lifted selection budget). All other modes keep the
            # 4-item reservation cap byte-identically.
            reserve_cap = (
                max_items
                if exhaustive_coverage
                else min(max_items, cls.EVIDENCE_OBLIGATION_MAX_ITEMS)
            )
            if len(reserved) >= reserve_cap:
                return False
            if dedupe_evidence_source and any(
                cls._is_duplicate_evidence_source(candidate, existing)
                for existing in reserved
            ):
                return False
            reserved.append(candidate)
            reserved_ids.add(candidate.memory_id)
            return True

        for candidate in cls._near_tie_verbatim_evidence_windows(candidates):
            reserve(candidate)

        if exhaustive_coverage and source_coverage_required:
            # Coverage-first ranked greedy: reserve one carrier per still-missing
            # member. A single carrier can cover several members; a reserved
            # carrier marks ALL its members covered so duplicate carriers of the
            # same member never consume distinct slots.
            covered: set[str] = set()
            for candidate in reserved:
                if cls._exhaustive_index_admits(candidate):
                    covered |= cls._coverage_member_keys(candidate)
            for candidate in cls._rank_source_coverage_candidates(
                candidates,
                source_messages_by_id=source_messages_by_id,
                query_type=query_type,
                answer_shape=answer_shape,
            ):
                if not cls._exhaustive_index_admits(candidate):
                    continue
                member_keys = cls._coverage_member_keys(candidate)
                if not member_keys or member_keys <= covered:
                    continue
                if reserve(candidate, dedupe_evidence_source=False):
                    covered |= member_keys
        elif source_coverage_required:
            reserved_groups = {
                cls._coverage_group_key(candidate)
                for candidate in reserved
                if cls._is_source_backed_coverage_candidate(candidate)
            }
            for candidate in cls._rank_source_coverage_candidates(
                candidates,
                source_messages_by_id=source_messages_by_id,
                query_type=query_type,
                answer_shape=answer_shape,
            ):
                if not cls._is_source_backed_coverage_candidate(candidate):
                    continue
                group_key = cls._coverage_group_key(candidate)
                if group_key in reserved_groups:
                    continue
                if reserve(candidate, dedupe_evidence_source=False):
                    reserved_groups.add(group_key)

        if query_type == "broad_list":
            for candidate in cls._rank_answer_evidence_candidates(
                candidates,
                source_messages_by_id=source_messages_by_id,
                query_text=None,
                query_type=query_type,
            ):
                if (
                    cls._broad_list_material_answer_evidence_priority(
                        candidate,
                        query_type=query_type,
                    )
                    <= 0
                ):
                    continue
                reserve(
                    candidate,
                    allow_source_linked_summary=cls._is_summary_like_candidate(
                        candidate
                    ),
                )

        for candidate in candidates:
            if not cls._is_summary_like_candidate(candidate):
                continue
            support = cls._supporting_l0_candidate(
                candidate,
                candidate_by_id,
                reserved_ids,
            )
            if support is not None and cls._is_source_grounded_evidence_candidate(
                support
            ):
                reserve(support)

        return reserved

    @staticmethod
    def _source_coverage_reserve_applies(
        *,
        answer_shape: AnswerShape,
        coverage_mode: CoverageMode,
        source_precision: SourcePrecision,
        exact_recall_mode: bool,
    ) -> bool:
        if source_precision != "required":
            return False
        if answer_shape in {"list", "temporal", "raw_context"}:
            return True
        if answer_shape == "single_fact" and (
            exact_recall_mode or coverage_mode == "current_state"
        ):
            return True
        return False

    @classmethod
    def _rank_source_coverage_candidates(
        cls,
        candidates: list[ScoredCandidate],
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        query_type: QueryType,
        answer_shape: AnswerShape,
    ) -> list[ScoredCandidate]:
        ranked = cls._rank_answer_evidence_candidates(
            candidates,
            source_messages_by_id=source_messages_by_id,
            query_text=None,
            query_type=query_type,
            dedupe_evidence_source=False,
        )
        return sorted(
            ranked,
            key=lambda candidate: (
                -int(cls._is_source_backed_coverage_candidate(candidate)),
                -int(cls._coverage_group_has_structured_value(candidate)),
                -(
                    float(candidate.llm_applicability)
                    if answer_shape == "list"
                    else 0.0
                ),
                -float(candidate.final_score),
                -cls._candidate_source_grounding(candidate),
                candidate.memory_id,
            ),
        )

    @classmethod
    def _coverage_metadata(
        cls,
        *,
        candidates: list[ScoredCandidate],
        selected: list[ScoredCandidate],
        answer_shape: AnswerShape,
        coverage_mode: CoverageMode,
        source_precision: SourcePrecision,
        redact_high_risk_secret_literals: bool,
    ) -> _CoverageMetadata:
        if answer_shape == "open_domain":
            return _CoverageMetadata(
                state="unknown",
                support_map={},
                allowed_values=[],
                missing_slots=[],
            )
        if coverage_mode == "exhaustive_known_set":
            return cls._exhaustive_coverage_metadata(
                candidates=candidates,
                selected=selected,
                answer_shape=answer_shape,
                coverage_mode=coverage_mode,
                source_precision=source_precision,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            )
        selected_ids = {candidate.memory_id for candidate in selected}
        source_groups = cls._source_backed_coverage_groups(candidates)
        raw_selected_groups = {
            group_key: group_candidates
            for group_key, group_candidates in source_groups.items()
            if any(candidate.memory_id in selected_ids for candidate in group_candidates)
        }
        support_map: dict[str, list[str]] = {}
        allowed_values: list[dict[str, Any]] = []
        selected_group_count = 0
        redacted_gap_count = 0
        for group_key, group_candidates in raw_selected_groups.items():
            if redact_high_risk_secret_literals and cls._coverage_group_withheld(
                group_candidates
            ):
                redacted_gap_count += 1
                continue
            selected_group_candidates = [
                candidate
                for candidate in group_candidates
                if candidate.memory_id in selected_ids
            ]
            if not selected_group_candidates:
                continue
            selected_group_count += 1
            evidence_ids = cls._coverage_support_ids(selected_group_candidates)
            for candidate in selected_group_candidates:
                support_map[candidate.memory_id] = evidence_ids
            allowed_values.append(
                {
                    "display_text": cls._coverage_display_text(
                        selected_group_candidates[0],
                    ),
                    "normalized_key": cls._coverage_serialized_key(group_key),
                    "evidence_ids": evidence_ids,
                    "memory_ids": [
                        candidate.memory_id for candidate in selected_group_candidates
                    ],
                }
            )

        missing_slots: list[dict[str, Any]] = []
        if coverage_mode in _COVERAGE_MISSING_SLOT_MODES:
            for group_key, group_candidates in source_groups.items():
                if group_key in raw_selected_groups:
                    continue
                if redact_high_risk_secret_literals and cls._coverage_group_withheld(
                    group_candidates
                ):
                    redacted_gap_count += 1
                    continue
                missing_slots.append(
                    {
                        "normalized_key": cls._coverage_serialized_key(group_key),
                        "display_text": cls._coverage_display_text(
                            group_candidates[0],
                        ),
                        "evidence_ids": cls._coverage_support_ids(group_candidates),
                        "reason": "source_backed_group_not_selected",
                    }
                )

        state = cls._coverage_state_for_metadata(
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
            source_precision=source_precision,
            source_group_count=len(source_groups),
            selected_group_count=selected_group_count,
            missing_slot_count=len(missing_slots) + redacted_gap_count,
        )
        return _CoverageMetadata(
            state=state,
            support_map=support_map,
            allowed_values=allowed_values,
            missing_slots=missing_slots,
        )

    @classmethod
    def _exhaustive_coverage_metadata(
        cls,
        *,
        candidates: list[ScoredCandidate],
        selected: list[ScoredCandidate],
        answer_shape: AnswerShape,
        coverage_mode: CoverageMode,
        source_precision: SourcePrecision,
        redact_high_risk_secret_literals: bool,
    ) -> _CoverageMetadata:
        """Member->candidates inverted index for exhaustive known-set coverage.

        Each admitted candidate appears under EVERY member it carries, so a
        member is covered iff >= 1 of its carriers was selected, and a single
        multi-member selection covers all of its members. Redaction is decided
        per member (a member is withheld only if ALL its carriers are withheld).
        An admitted source-backed carrier that is UNKEYED (no member identity)
        is unresolved residue that caps the state at ``partial``.
        """
        selected_ids = {candidate.memory_id for candidate in selected}

        # Build the inverted index over admitted candidates, and detect unkeyed
        # source-backed residue (rule #3) that blocks a "complete" claim.
        member_to_candidates: dict[str, list[ScoredCandidate]] = {}
        member_display: dict[str, str] = {}
        clean_member_display: dict[str, str] = {}
        unkeyed_residue_count = 0
        for candidate in candidates:
            if not cls._exhaustive_index_admits(candidate):
                continue
            member_keys = cls._coverage_member_keys(candidate)
            if not member_keys:
                # Admitted only via the source-backed gate but carrying no member
                # identity: cannot be proven to be part of the known set.
                unkeyed_residue_count += 1
                continue
            display_map = cls._coverage_member_display_map(candidate)
            # A withheld carrier's display can be the raw secret literal (legacy
            # value_* carriers reuse the candidate display text). Such a label
            # must never reach allowed_values/missing_slots, so the answer-facing
            # display for a member is sourced ONLY from a clean (non-withheld)
            # carrier. A member with all-withheld carriers becomes a redacted_gap
            # whose display is never emitted, so its fallback label is harmless.
            candidate_withheld = (
                redact_high_risk_secret_literals
                and cls._coverage_candidate_withheld(candidate)
            )
            for member_key in member_keys:
                member_to_candidates.setdefault(member_key, []).append(candidate)
                candidate_display = display_map.get(
                    member_key,
                    cls._coverage_display_text(candidate),
                )
                if candidate_withheld:
                    # Only fill from a withheld carrier when nothing is set yet,
                    # so any later clean carrier overrides it.
                    member_display.setdefault(member_key, candidate_display)
                elif member_key not in clean_member_display:
                    clean_member_display[member_key] = candidate_display
                    member_display[member_key] = candidate_display

        support_map: dict[str, list[str]] = {}
        allowed_values: list[dict[str, Any]] = []
        missing_slots: list[dict[str, Any]] = []
        covered_member_count = 0
        redacted_gap_count = 0
        for member_key, carriers in member_to_candidates.items():
            selected_carriers = [
                candidate
                for candidate in carriers
                if candidate.memory_id in selected_ids
            ]
            clean_selected_carriers = [
                candidate
                for candidate in selected_carriers
                if not (
                    redact_high_risk_secret_literals
                    and cls._coverage_candidate_withheld(candidate)
                )
            ]
            normalized_key = cls._coverage_serialized_key(("value", member_key))
            display_text = member_display[member_key]
            if clean_selected_carriers:
                covered_member_count += 1
                evidence_ids = cls._coverage_support_ids(clean_selected_carriers)
                for candidate in clean_selected_carriers:
                    support_map[candidate.memory_id] = evidence_ids
                allowed_values.append(
                    {
                        "display_text": display_text,
                        "normalized_key": normalized_key,
                        "evidence_ids": evidence_ids,
                        "memory_ids": [
                            candidate.memory_id
                            for candidate in clean_selected_carriers
                        ],
                    }
                )
                continue
            # No clean selected carrier: the member is a gap. If every carrier
            # is withheld it is a redaction gap (no display); otherwise it is a
            # genuine missing slot the answer must be told about.
            if redact_high_risk_secret_literals and all(
                cls._coverage_candidate_withheld(candidate) for candidate in carriers
            ):
                redacted_gap_count += 1
                continue
            missing_slots.append(
                {
                    "normalized_key": normalized_key,
                    "display_text": display_text,
                    "evidence_ids": cls._coverage_support_ids(carriers),
                    "reason": "coverage_member_not_selected",
                }
            )

        if unkeyed_residue_count > 0:
            missing_slots.append(
                {
                    "normalized_key": cls._coverage_serialized_key(
                        ("unkeyed", "ungrouped_supported_evidence")
                    ),
                    "display_text": "ungrouped supported evidence present",
                    "evidence_ids": [],
                    "reason": "unkeyed_supported_evidence_present",
                }
            )

        state = cls._coverage_state_for_metadata(
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
            source_precision=source_precision,
            source_group_count=len(member_to_candidates),
            selected_group_count=covered_member_count,
            missing_slot_count=len(missing_slots) + redacted_gap_count,
        )
        return _CoverageMetadata(
            state=state,
            support_map=support_map,
            allowed_values=allowed_values,
            missing_slots=missing_slots,
        )

    @classmethod
    def _coverage_state_for_metadata(
        cls,
        *,
        answer_shape: AnswerShape,
        coverage_mode: CoverageMode,
        source_precision: SourcePrecision,
        source_group_count: int,
        selected_group_count: int,
        missing_slot_count: int,
    ) -> EvidenceCoverageState:
        if answer_shape == "open_domain":
            return "unknown"
        if source_precision == "required" and source_group_count <= 0:
            return "insufficient"
        if selected_group_count <= 0:
            return "insufficient" if source_precision == "required" else "unknown"
        if (
            missing_slot_count > 0
            and coverage_mode in _COVERAGE_MISSING_SLOT_MODES
        ):
            return "partial"
        return "complete"

    @classmethod
    def _source_backed_coverage_groups(
        cls,
        candidates: list[ScoredCandidate],
    ) -> dict[tuple[str, ...], list[ScoredCandidate]]:
        groups: dict[tuple[str, ...], list[ScoredCandidate]] = {}
        for candidate in candidates:
            if not cls._is_source_backed_coverage_candidate(candidate):
                continue
            groups.setdefault(cls._coverage_group_key(candidate), []).append(candidate)
        return groups

    @classmethod
    def _is_source_backed_coverage_candidate(
        cls,
        candidate: ScoredCandidate,
    ) -> bool:
        """Strict composer-stage classifier for answer coverage qualifications."""
        if cls._is_summary_like_candidate(candidate):
            return False
        memory_object = candidate.memory_object
        if cls._candidate_has_source_packet_quote(candidate):
            return True
        if cls._is_verbatim_evidence_window_candidate(candidate):
            return True
        if cls._candidate_has_quoteable_source(candidate):
            return True
        source_kind = str(memory_object.get("source_kind") or "").strip()
        if source_kind in {"verbatim", "extracted"} and str(
            memory_object.get("object_type") or ""
        ) in {"evidence", "interaction_contract", "state_snapshot"}:
            return True
        return False

    @classmethod
    def _exhaustive_index_admits(cls, candidate: ScoredCandidate) -> bool:
        """Admission rule for the exhaustive member index / reservation.

        Admit iff the candidate carries non-empty member keys AND is not
        summary-like, OR it already passes the strict source-backed gate.

        The member-keys path is required for member-bearing beliefs, which fail
        the source-backed gate yet must be enumerable. The ``not summary-like``
        clause blocks summary-views that can carry a ``value_*`` key but are
        barred from selection: admitting one would mint a member that can never
        be selected, producing a phantom missing_slot.
        """
        if cls._coverage_member_keys(candidate) and not cls._is_summary_like_candidate(
            candidate
        ):
            return True
        return cls._is_source_backed_coverage_candidate(candidate)

    @staticmethod
    def _coverage_group_has_structured_value(candidate: ScoredCandidate) -> bool:
        if ContextComposer._coverage_member_keys(candidate):
            return True
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        return any(
            ContextComposer._optional_text(payload_json.get(key)) is not None
            for key in _COVERAGE_VALUE_PAYLOAD_KEYS
        )

    @staticmethod
    def _coverage_members_payload(candidate: ScoredCandidate) -> list[Any] | None:
        """Return the raw ``coverage_members`` list when the key is present.

        Returns ``None`` when the key is absent (legacy/unprocessed row). A
        present-but-empty list means "processed, no enumerable members" and is
        returned verbatim so callers can distinguish it from absence.
        """
        payload_json = candidate.memory_object.get("payload_json")
        if not isinstance(payload_json, dict):
            return None
        if "coverage_members" not in payload_json:
            return None
        members = payload_json["coverage_members"]
        return members if isinstance(members, list) else []

    @classmethod
    def _coverage_member_keys(cls, candidate: ScoredCandidate) -> frozenset[str]:
        """Mechanical resolution ladder for a candidate's member identities.

        Shared by reservation and metadata so they can never disagree:
          1. ``coverage_members`` key present -> the set of normalized
             ``member_key`` values (possibly empty).
          2. else a legacy ``_COVERAGE_VALUE_PAYLOAD_KEYS`` value present ->
             single-element set with the normalized value (preserving the
             existing ``("value", normalized)`` grouping shape).
          3. else -> empty set (treated as UNKEYED by callers).
        """
        members = cls._coverage_members_payload(candidate)
        if members is not None:
            keys: set[str] = set()
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_key = cls._optional_text(member.get("member_key"))
                if member_key is not None:
                    keys.add(cls._normalize_coverage_key(member_key))
            return frozenset(keys)

        payload_json = candidate.memory_object.get("payload_json") or {}
        if isinstance(payload_json, dict):
            for key in _COVERAGE_VALUE_PAYLOAD_KEYS:
                value = cls._optional_text(payload_json.get(key))
                if value is not None:
                    return frozenset({cls._normalize_coverage_key(value)})
        return frozenset()

    @classmethod
    def _coverage_member_display_map(
        cls,
        candidate: ScoredCandidate,
    ) -> dict[str, str]:
        """Map each member_key of a candidate to its answer-facing display text.

        For the ``coverage_members`` payload, the display text is the member's
        own ``display_text``. For the legacy ``value_*`` fallback (rule #2), the
        single member reuses the candidate-level coverage display text so legacy
        ``allowed_values`` rendering is byte-identical.
        """
        members = cls._coverage_members_payload(candidate)
        if members is not None:
            mapping: dict[str, str] = {}
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_key = cls._optional_text(member.get("member_key"))
                if member_key is None:
                    continue
                display = cls._optional_text(member.get("display_text"))
                if display is None:
                    # No usable label on this member: leave the key absent so the
                    # caller's canonical-text fallback fires instead of an empty
                    # label.
                    continue
                normalized = cls._normalize_coverage_key(member_key)
                mapping[normalized] = cls._truncate_inline(display, 160)
            return mapping

        payload_json = candidate.memory_object.get("payload_json") or {}
        if isinstance(payload_json, dict):
            for key in _COVERAGE_VALUE_PAYLOAD_KEYS:
                value = cls._optional_text(payload_json.get(key))
                if value is not None:
                    return {
                        cls._normalize_coverage_key(value): cls._coverage_display_text(
                            candidate
                        )
                    }
        return {}

    @classmethod
    def _coverage_group_key(cls, candidate: ScoredCandidate) -> tuple[str, ...]:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if isinstance(payload_json, dict):
            for key in _COVERAGE_VALUE_PAYLOAD_KEYS:
                value = cls._optional_text(payload_json.get(key))
                if value is not None:
                    return ("value", cls._normalize_coverage_key(value))

        source_ids = cls._candidate_source_message_ids(candidate)
        packet_ids = cls._evidence_packet_message_ids(candidate.memory_object)
        object_ids = cls._candidate_source_ids(candidate)
        if source_ids or packet_ids or object_ids:
            return (
                "source",
                *sorted({*source_ids, *packet_ids, *object_ids}),
            )

        support_kind = cls._best_evidence_packet_field(
            candidate.memory_object,
            "support_kind",
        )
        if support_kind:
            return ("support", cls._normalize_coverage_key(support_kind))
        return ("candidate", candidate.memory_id)

    @staticmethod
    def _normalize_coverage_key(value: str) -> str:
        return " ".join(str(value).casefold().split())

    @classmethod
    def _coverage_support_ids(
        cls,
        candidates: list[ScoredCandidate],
    ) -> list[str]:
        seen: set[str] = set()
        support_ids: list[str] = []

        def append(value: str) -> None:
            normalized = value.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            support_ids.append(normalized)

        for candidate in candidates:
            append(f"memory:{candidate.memory_id}")
            for message_id in cls._candidate_source_message_ids(candidate):
                append(f"message:{message_id}")
            for message_id in cls._evidence_packet_message_ids(candidate.memory_object):
                append(f"evidence_message:{message_id}")
            for source_id in cls._candidate_source_ids(candidate):
                append(f"source_memory:{source_id}")
            source_window = cls._candidate_source_window(candidate)
            if source_window is not None:
                append(f"source_window:{source_window[0]}:{source_window[1]}")
        return support_ids

    @classmethod
    def _coverage_display_text(
        cls,
        candidate: ScoredCandidate,
    ) -> str:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if isinstance(payload_json, dict):
            for key in _COVERAGE_DISPLAY_PAYLOAD_KEYS:
                value = cls._optional_text(payload_json.get(key))
                if value is not None:
                    return cls._truncate_inline(value, 160)
        return cls._truncate_inline(
            str(candidate.memory_object.get("canonical_text") or ""),
            160,
        )

    @classmethod
    def _coverage_group_withheld(
        cls,
        group_candidates: list[ScoredCandidate],
    ) -> bool:
        return any(cls._coverage_candidate_withheld(candidate) for candidate in group_candidates)

    @classmethod
    def _coverage_candidate_withheld(cls, candidate: ScoredCandidate) -> bool:
        return (
            cls._disclosure_action(candidate.memory_object)
            is HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
        )

    @classmethod
    def _coverage_serialized_key(cls, group_key: tuple[str, ...]) -> str:
        return "|".join(group_key)

    @classmethod
    def _build_answer_evidence_pack(
        cls,
        candidates: list[ScoredCandidate],
        *,
        query_text: str | None,
        query_type: QueryType,
        answer_shape: AnswerShape,
        exact_recall_mode: bool,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
        max_items: int,
        remaining_budget: int,
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        render_block: bool,
    ) -> _AnswerEvidencePack:
        best_score = cls._best_answer_evidence_score(
            candidates,
            source_messages_by_id=source_messages_by_id,
            query_text=query_text,
        )
        if (
            not candidates
            or max_items <= 0
            or remaining_budget < cls.FINAL_ANSWER_EVIDENCE_MIN_TOKENS
            or not cls._prefer_literal_evidence(
                query_type=query_type,
                answer_shape=answer_shape,
                exact_recall_mode=exact_recall_mode,
            )
        ):
            return _AnswerEvidencePack(
                block="",
                candidates=[],
                items=[],
                sufficiency=cls._answer_evidence_sufficiency(
                    items=[],
                    best_score=best_score,
                    score_floor=cls._answer_evidence_score_floor(best_score),
                    query_type=query_type,
                    applies=bool(
                        candidates
                        and max_items > 0
                        and remaining_budget >= cls.FINAL_ANSWER_EVIDENCE_MIN_TOKENS
                        and cls._prefer_literal_evidence(
                            query_type=query_type,
                            answer_shape=answer_shape,
                            exact_recall_mode=exact_recall_mode,
                        )
                    ),
                    render_block_requested=render_block,
                    rendered_memory_ids=[],
                ),
            )

        allocated_tokens = min(
            remaining_budget,
            cls.FINAL_ANSWER_EVIDENCE_MAX_TOKENS,
            max(
                cls.FINAL_ANSWER_EVIDENCE_MIN_TOKENS,
                math.ceil(remaining_budget * cls.FINAL_ANSWER_EVIDENCE_BUDGET_RATIO),
            ),
        )
        score_floor = cls._answer_evidence_score_floor(best_score)
        ranked = cls._rank_answer_evidence_candidates(
            candidates,
            source_messages_by_id=source_messages_by_id,
            query_text=query_text,
            query_type=query_type,
        )
        max_pack_items = min(
            max_items,
            cls.FINAL_ANSWER_EVIDENCE_MAX_ITEMS,
            source_quote_options.max_entries or cls.FINAL_ANSWER_EVIDENCE_MAX_ITEMS,
        )
        analysis_pairs: list[tuple[ScoredCandidate, dict[str, Any]]] = []
        for candidate in ranked:
            item = cls._answer_evidence_item(
                index=len(analysis_pairs) + 1,
                candidate=candidate,
                source_messages_by_id=source_messages_by_id,
                source_quote_options=source_quote_options,
                query_text=query_text,
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            )
            if item is None:
                continue
            item["selected_for_answer_pack"] = False
            analysis_pairs.append((candidate, item))
            if len(analysis_pairs) >= max_pack_items:
                break

        analysis_items = [item for _, item in analysis_pairs]
        sufficiency = cls._answer_evidence_sufficiency(
            items=analysis_items,
            best_score=best_score,
            score_floor=score_floor,
            query_type=query_type,
            applies=True,
            render_block_requested=render_block,
            rendered_memory_ids=[],
        )
        if not render_block or sufficiency.get("state") != "sufficient_direct_quote":
            return _AnswerEvidencePack(
                block="",
                candidates=[],
                items=analysis_items,
                sufficiency=sufficiency,
            )

        qualified_pairs = [
            (candidate, item)
            for candidate, item in analysis_pairs
            if cls._answer_evidence_item_is_renderable(
                item,
                score_floor=score_floor,
                query_type=query_type,
            )
        ]
        for item_count in range(max_pack_items, 0, -1):
            selected_pairs = qualified_pairs[:item_count]
            if not selected_pairs:
                continue
            selected: list[ScoredCandidate] = []
            rendered_items: list[dict[str, Any]] = []
            for candidate, item in selected_pairs:
                selected.append(candidate)
                rendered_items.append(item)
            block = cls._format_answer_evidence_pack(rendered_items)
            if block and cls.estimate_tokens(block) <= allocated_tokens:
                rendered_ids = [candidate.memory_id for candidate in selected]
                for item in analysis_items:
                    item["selected_for_answer_pack"] = item["memory_id"] in rendered_ids
                return _AnswerEvidencePack(
                    block=block,
                    candidates=selected,
                    items=analysis_items,
                    sufficiency={
                        **sufficiency,
                        "rendered": True,
                        "rendered_memory_ids": rendered_ids,
                        "rendered_item_count": len(rendered_items),
                    },
                )
        return _AnswerEvidencePack(
            block="",
            candidates=[],
            items=analysis_items,
            sufficiency={
                **sufficiency,
                "rendered": False,
                "rendered_memory_ids": [],
                "rendered_item_count": 0,
                "rationale_codes": [
                    *list(sufficiency.get("rationale_codes") or []),
                    "render_block_exceeds_budget",
                ],
            },
        )

    @classmethod
    def _rank_answer_evidence_candidates(
        cls,
        candidates: list[ScoredCandidate],
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        query_text: str | None,
        query_type: QueryType,
        dedupe_evidence_source: bool = True,
    ) -> list[ScoredCandidate]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -(
                    float(candidate.llm_applicability)
                    if query_type == "broad_list"
                    else 0.0
                ),
                -cls._broad_list_material_answer_evidence_priority(
                    candidate,
                    query_type=query_type,
                ),
                -cls._candidate_quote_query_relevance(
                    candidate,
                    source_messages_by_id=source_messages_by_id,
                    query_text=query_text,
                ),
                -cls._answer_evidence_source_chain_breadth(candidate),
                -cls._source_weighted_score(
                    candidate,
                    source_messages_by_id=source_messages_by_id,
                    exact_recall_mode=True,
                    query_text=query_text,
                ),
                cls._literal_evidence_priority(
                    candidate,
                    prefer_literal_evidence=True,
                ),
                -cls._candidate_source_grounding(candidate),
                candidate.memory_id,
            ),
        )
        ranked: list[ScoredCandidate] = []
        for candidate in ordered:
            if cls._literal_evidence_priority(
                candidate,
                prefer_literal_evidence=True,
            ) >= 5:
                continue
            if dedupe_evidence_source and any(
                cls._is_duplicate_evidence_source(candidate, item) for item in ranked
            ):
                continue
            ranked.append(candidate)
        return ranked

    @classmethod
    def _answer_evidence_source_chain_breadth(cls, candidate: ScoredCandidate) -> float:
        if not cls._has_sequential_source_chain(candidate):
            return 0.0
        source_message_count = len(cls._candidate_source_message_ids(candidate))
        if source_message_count <= 0:
            return 0.0
        return min(source_message_count, cls.SOURCE_CHAIN_MAX_MESSAGES) / max(
            1,
            cls.SOURCE_CHAIN_MAX_MESSAGES,
        )

    @classmethod
    def _candidate_quote_query_relevance(
        cls,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        query_text: str | None,
    ) -> float:
        if not query_text:
            return 0.0
        memory_object = candidate.memory_object
        scores: list[float] = []
        for quote, _ in cls._evidence_packet_quote_candidates(memory_object):
            scores.append(cls._quote_query_relevance(quote, query_text))
        for source_message_id in cls._candidate_source_message_ids(candidate):
            message = source_messages_by_id.get(source_message_id)
            if message is None or not cls._message_allows_source_quote(message):
                continue
            text = str(message.get("text") or "").strip()
            if text:
                scores.append(cls._quote_query_relevance(text, query_text))
        if cls._is_verbatim_evidence_window_candidate(candidate):
            scores.append(
                cls._quote_query_relevance(
                    str(memory_object.get("canonical_text") or ""),
                    query_text,
                )
            )
        return max(scores, default=0.0)

    @classmethod
    def _answer_evidence_item(
        cls,
        *,
        index: int,
        candidate: ScoredCandidate,
        source_messages_by_id: dict[str, dict[str, Any]],
        source_quote_options: _SourceQuoteOptions,
        query_text: str | None,
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
    ) -> dict[str, Any] | None:
        memory_object = candidate.memory_object
        if (
            redact_high_risk_secret_literals
            and cls._disclosure_action(memory_object)
            is HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
        ):
            return None
        packet_quote, packet_quote_role = (
            cls._best_evidence_packet_quote_with_role_for_query(
                memory_object,
                query_text=query_text,
            )
        )
        quote_source = (
            f"evidence_packet_{packet_quote_role}" if packet_quote_role else ""
        )
        source_quote = packet_quote
        source_message_quote = cls._source_quote_for_candidate(
            candidate,
            source_messages_by_id=source_messages_by_id,
            options=cls._compact_source_quote_options(source_quote_options),
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            query_text=query_text,
        )
        if source_message_quote and (
            not source_quote
            or cls._quote_query_relevance(source_message_quote, query_text)
            > cls._quote_query_relevance(source_quote, query_text)
        ):
            source_quote = source_message_quote
            quote_source = "source_message"
        if not source_quote and cls._is_verbatim_evidence_window_candidate(candidate):
            source_quote = str(memory_object.get("canonical_text") or "").strip()
            if source_quote:
                quote_source = "verbatim_evidence_window"
        if not source_quote:
            payload_json = memory_object.get("payload_json") or {}
            if isinstance(payload_json, dict):
                source_quote = cls._conversation_chunk_excerpt(payload_json)
                if source_quote:
                    quote_source = "conversation_chunk_excerpt"
        if not source_quote:
            return None
        claim = str(memory_object.get("canonical_text") or "")
        support_kind = cls._best_evidence_packet_field(memory_object, "support_kind")
        date = cls._answer_evidence_date(candidate, memory_object)
        speaker = cls._answer_evidence_speaker(memory_object, source_messages_by_id)
        source = cls._answer_evidence_source(candidate, memory_object)
        why_selected = cls._answer_evidence_reason(candidate)
        realm_note = cls._realm_note(memory_object, active_realm_id=active_realm_id)
        if realm_note:
            source = f"{source}; {realm_note}"
        source_chain = cls._source_chain_for_candidate(
            candidate,
            source_messages_by_id=source_messages_by_id,
            options=source_quote_options,
            query_text=query_text,
        )
        normalization = cls._answer_evidence_normalization(
            candidate,
            memory_object=memory_object,
            source_messages_by_id=source_messages_by_id,
            quote_source=quote_source,
        )
        return {
            "index": index,
            "memory_id": candidate.memory_id,
            "claim": cls._truncate_inline(claim, 260),
            "supporting_quote": cls._truncate_inline(source_quote, 300),
            "quote_source": quote_source,
            "date": date,
            "speaker": speaker,
            "source": source,
            "support_kind": support_kind,
            "why_selected": why_selected,
            "object_type": str(memory_object.get("object_type") or ""),
            "source_kind": str(memory_object.get("source_kind") or ""),
            "channels": list(memory_object.get("retrieval_sources") or []),
            "final_score": round(float(candidate.final_score), 6),
            "llm_applicability": round(float(candidate.llm_applicability), 6),
            "retrieval_score": round(float(candidate.retrieval_score), 6),
            "normalization": normalization,
            "source_chain": source_chain,
        }

    @staticmethod
    def _format_answer_evidence_pack(items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["[Final Answer Evidence Pack]"]
        for item in items:
            lines.append(f"Evidence {item['index']}")
            lines.append(f"- claim: {item['claim']}")
            quote = str(item.get("supporting_quote") or "").strip()
            lines.append(f"- supporting_quote: {quote or 'not available'}")
            lines.append(f"- date: {item.get('date') or 'unknown'}")
            lines.append(f"- speaker: {item.get('speaker') or 'unknown'}")
            lines.append(f"- source: {item.get('source') or item.get('memory_id')}")
            support_kind = str(item.get("support_kind") or "").strip()
            if support_kind:
                lines.append(f"- support_kind: {support_kind}")
            source_chain = [
                str(line).strip()
                for line in item.get("source_chain") or []
                if str(line).strip()
            ]
            if source_chain:
                lines.append("- source_chain:")
                for chain_line in source_chain:
                    lines.append(f"  - {chain_line}")
            lines.append(f"- why_selected: {item.get('why_selected') or 'selected evidence'}")
        return "\n".join(lines)

    @classmethod
    def _best_evidence_packet_quote(cls, memory_object: dict[str, Any]) -> str:
        quote, _ = cls._best_evidence_packet_quote_with_role(memory_object)
        return quote

    @classmethod
    def _best_evidence_packet_quote_with_role_for_query(
        cls,
        memory_object: dict[str, Any],
        *,
        query_text: str | None,
    ) -> tuple[str, str]:
        candidates = cls._evidence_packet_quote_candidates(memory_object)
        if not candidates:
            return ("", "")
        if not query_text:
            return candidates[0]
        return max(
            candidates,
            key=lambda candidate: (
                cls._quote_query_relevance(candidate[0], query_text),
                1 if candidate[1] == "source" else 0,
            ),
        )

    @classmethod
    def _best_evidence_packet_quote_with_role(
        cls,
        memory_object: dict[str, Any],
    ) -> tuple[str, str]:
        candidates = cls._evidence_packet_quote_candidates(memory_object)
        return candidates[0] if candidates else ("", "")

    @classmethod
    def _evidence_packet_quote_candidates(
        cls,
        memory_object: dict[str, Any],
    ) -> list[tuple[str, str]]:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return []
        candidates: list[tuple[str, str]] = []
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            spans = packet.get("spans")
            if not isinstance(spans, list):
                continue
            for span_role in ("source", "trigger"):
                for span in spans:
                    if not isinstance(span, dict) or span.get("span_role") != span_role:
                        continue
                    quote = str(span.get("quote_text") or "").strip()
                    if not quote:
                        continue
                    prefix_parts: list[str] = []
                    metadata = span.get("metadata_json") or {}
                    if isinstance(metadata, dict):
                        role = str(metadata.get("message_role") or "").strip()
                        if role:
                            prefix_parts.append(role)
                    occurred_at = str(span.get("occurred_at") or "").strip()
                    if occurred_at:
                        prefix_parts.append(f"@ {occurred_at}")
                    seq = span.get("seq")
                    if seq is not None:
                        prefix_parts.append(f"seq {seq}")
                    prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""
                    candidates.append(
                        (cls._truncate_inline(f"{prefix}{quote}", 300), span_role)
                    )
        return candidates

    @classmethod
    def _best_evidence_packet_quote_for_role(
        cls,
        memory_object: dict[str, Any],
        span_role: str,
    ) -> str:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return ""
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            spans = packet.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict) or span.get("span_role") != span_role:
                    continue
                quote = str(span.get("quote_text") or "").strip()
                if not quote:
                    continue
                prefix_parts: list[str] = []
                metadata = span.get("metadata_json") or {}
                if isinstance(metadata, dict):
                    role = str(metadata.get("message_role") or "").strip()
                    if role:
                        prefix_parts.append(role)
                occurred_at = str(span.get("occurred_at") or "").strip()
                if occurred_at:
                    prefix_parts.append(f"@ {occurred_at}")
                seq = span.get("seq")
                if seq is not None:
                    prefix_parts.append(f"seq {seq}")
                prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""
                return cls._truncate_inline(f"{prefix}{quote}", 300)
        return ""

    @staticmethod
    def _best_evidence_packet_field(
        memory_object: dict[str, Any],
        field_name: str,
    ) -> str:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return ""
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            value = str(packet.get(field_name) or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def _answer_evidence_date(
        cls,
        candidate: ScoredCandidate,
        memory_object: dict[str, Any],
    ) -> str:
        if candidate.resolved_date:
            return candidate.resolved_date
        for key in ("valid_from", "valid_to", "occurred_at", "created_at"):
            value = str(memory_object.get(key) or "").strip()
            if value:
                return value
        packets = memory_object.get("evidence_packets")
        if isinstance(packets, list):
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                spans = packet.get("spans")
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    occurred_at = str(span.get("occurred_at") or "").strip()
                    if occurred_at:
                        return occurred_at
        source_window = cls._candidate_source_window(candidate)
        if source_window is not None:
            start, end = source_window
            return start if start == end or not end else f"{start} to {end}"
        return ""

    @classmethod
    def _answer_evidence_speaker(
        cls,
        memory_object: dict[str, Any],
        source_messages_by_id: dict[str, dict[str, Any]],
    ) -> str:
        packets = memory_object.get("evidence_packets")
        if isinstance(packets, list):
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                spans = packet.get("spans")
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    metadata = span.get("metadata_json") or {}
                    if isinstance(metadata, dict):
                        role = str(metadata.get("message_role") or "").strip()
                        if role:
                            return role
        payload_json = memory_object.get("payload_json") or {}
        if isinstance(payload_json, dict):
            for source_message_id in payload_json.get("source_message_ids") or []:
                message = source_messages_by_id.get(str(source_message_id))
                if message is None:
                    continue
                role = str(message.get("role") or "").strip()
                if role:
                    return role
        return ""

    @classmethod
    def _answer_evidence_source(
        cls,
        candidate: ScoredCandidate,
        memory_object: dict[str, Any],
    ) -> str:
        parts = [f"memory_id={candidate.memory_id}"]
        packets = memory_object.get("evidence_packets")
        if isinstance(packets, list):
            message_ids: list[str] = []
            seqs: list[str] = []
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                spans = packet.get("spans")
                if not isinstance(spans, list):
                    continue
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    message_id = str(span.get("message_id") or "").strip()
                    if message_id and message_id not in message_ids:
                        message_ids.append(message_id)
                    seq = span.get("seq")
                    if seq is not None:
                        text_seq = str(seq)
                        if text_seq not in seqs:
                            seqs.append(text_seq)
            if message_ids:
                parts.append("message_id=" + ",".join(message_ids[:3]))
            if seqs:
                parts.append("seq=" + ",".join(seqs[:3]))
        source_window = cls._candidate_source_window(candidate)
        if source_window is not None:
            start, end = source_window
            parts.append(f"source_window={start or '?'} to {end or '?'}")
        return "; ".join(parts)

    @classmethod
    def _answer_evidence_reason(cls, candidate: ScoredCandidate) -> str:
        memory_object = candidate.memory_object
        if cls._candidate_has_source_packet_quote(candidate):
            return "direct evidence packet quote for the asked fact"
        if cls._is_verbatim_evidence_window_candidate(candidate):
            return "verbatim source window for the asked fact"
        if cls._candidate_source_message_ids(candidate):
            return "source-linked memory with quoteable message support"
        if str(memory_object.get("object_type") or "") == "summary_view":
            return "summary selected only with source support"
        return "scored evidence selected for the question"

    @classmethod
    def _best_answer_evidence_score(
        cls,
        candidates: list[ScoredCandidate],
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        query_text: str | None,
    ) -> float:
        if not candidates:
            return 0.0
        return max(
            cls._source_weighted_score(
                candidate,
                source_messages_by_id=source_messages_by_id,
                exact_recall_mode=True,
                query_text=query_text,
            )
            for candidate in candidates
        )

    @classmethod
    def _answer_evidence_score_floor(cls, best_score: float) -> float:
        if best_score <= 0:
            return cls.FINAL_ANSWER_EVIDENCE_MIN_SCORE
        return max(
            cls.FINAL_ANSWER_EVIDENCE_MIN_SCORE,
            best_score * cls.FINAL_ANSWER_EVIDENCE_RELATIVE_SCORE_RATIO,
        )

    @classmethod
    def _answer_evidence_item_is_renderable(
        cls,
        item: dict[str, Any],
        *,
        score_floor: float,
        query_type: QueryType,
    ) -> bool:
        if (
            query_type == "broad_list"
            and cls._answer_evidence_item_is_direct(item)
            and float(item.get("llm_applicability") or 0.0)
            >= cls.BROAD_LIST_MATERIAL_LLM_APPLICABILITY
        ):
            return True
        if float(item.get("final_score") or 0.0) < score_floor:
            return False
        if str(item.get("support_kind") or "") in {"direct", "contextual_direct"}:
            return True
        return str(item.get("quote_source") or "") in {
            "evidence_packet_source",
            "source_message",
            "verbatim_evidence_window",
        }

    @classmethod
    def _answer_evidence_sufficiency(
        cls,
        *,
        items: list[dict[str, Any]],
        best_score: float,
        score_floor: float,
        query_type: QueryType,
        applies: bool,
        render_block_requested: bool,
        rendered_memory_ids: list[str],
    ) -> dict[str, Any]:
        rationale_codes: list[str] = []
        direct_items = [
            item for item in items if cls._answer_evidence_item_is_direct(item)
        ]
        broad_list_material_direct_items = [
            item
            for item in direct_items
            if query_type == "broad_list"
            and float(item.get("llm_applicability") or 0.0)
            >= cls.BROAD_LIST_MATERIAL_LLM_APPLICABILITY
        ]
        score_floor_direct_items = [
            item
            for item in direct_items
            if float(item.get("final_score") or 0.0) >= score_floor
        ]
        top_item = items[0] if items else None
        if not applies:
            state = "not_applicable_query_type"
            confidence = 0.0
            rationale_codes.append("query_type_not_evidence_obligated")
        elif not items:
            state = "insufficient_no_quoteable_evidence"
            confidence = 0.0
            rationale_codes.append("no_quoteable_evidence_item")
        elif not direct_items:
            state = "weak_no_direct_quote"
            confidence = max(0.0, min(1.0, float(top_item.get("final_score") or 0.0)))
            rationale_codes.append("no_direct_quote_item")
        elif not score_floor_direct_items and not broad_list_material_direct_items:
            state = "weak_low_applicability"
            confidence = max(0.0, min(1.0, float(top_item.get("final_score") or 0.0)))
            rationale_codes.append("direct_evidence_below_score_floor")
        else:
            confidence_item = (
                broad_list_material_direct_items[0]
                if broad_list_material_direct_items
                else score_floor_direct_items[0]
            )
            state = "sufficient_direct_quote"
            confidence = max(0.0, min(1.0, float(confidence_item.get("final_score") or 0.0)))
            rationale_codes.append("direct_quote_item_available")

        return {
            "state": state,
            "confidence": round(confidence, 6),
            "rationale_codes": rationale_codes,
            "best_candidate_score": round(float(best_score), 6),
            "score_floor": round(float(score_floor), 6),
            "quote_item_count": len(items),
            "direct_quote_item_count": len(direct_items),
            "top_memory_id": str(top_item.get("memory_id") or "") if top_item else "",
            "top_support_kind": str(top_item.get("support_kind") or "") if top_item else "",
            "top_quote_source": str(top_item.get("quote_source") or "") if top_item else "",
            "render_block_requested": render_block_requested,
            "rendered": bool(rendered_memory_ids),
            "rendered_memory_ids": rendered_memory_ids,
            "rendered_item_count": len(rendered_memory_ids),
        }

    @staticmethod
    def _answer_evidence_item_is_direct(item: dict[str, Any]) -> bool:
        if str(item.get("support_kind") or "") in {"direct", "contextual_direct"}:
            return True
        return str(item.get("quote_source") or "") in {
            "evidence_packet_source",
            "source_message",
            "verbatim_evidence_window",
        }

    @classmethod
    def _broad_list_material_answer_evidence_priority(
        cls,
        candidate: ScoredCandidate,
        *,
        query_type: QueryType,
    ) -> int:
        if query_type != "broad_list":
            return 0
        if (
            float(candidate.llm_applicability)
            < cls.BROAD_LIST_MATERIAL_LLM_APPLICABILITY
        ):
            return 0
        if cls._literal_evidence_priority(candidate, prefer_literal_evidence=True) >= 5:
            return 0
        if cls._candidate_has_source_packet_quote(candidate):
            return 3
        if cls._candidate_has_quoteable_source(candidate):
            return 2
        if cls._is_verbatim_evidence_window_candidate(candidate):
            return 1
        return 0

    @classmethod
    def _answer_evidence_normalization(
        cls,
        candidate: ScoredCandidate,
        *,
        memory_object: dict[str, Any],
        source_messages_by_id: dict[str, dict[str, Any]],
        quote_source: str,
    ) -> dict[str, Any]:
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            payload_json = {}
        normalized: dict[str, Any] = {
            "quote_source": quote_source,
        }
        source_message_ids = cls._candidate_source_message_ids(candidate)
        if source_message_ids:
            normalized["source_message_ids"] = source_message_ids
        packet_message_ids = cls._evidence_packet_message_ids(memory_object)
        if packet_message_ids:
            normalized["evidence_packet_message_ids"] = packet_message_ids
        source_window = cls._candidate_source_window(candidate)
        if source_window is not None:
            start, end = source_window
            normalized["source_window_start"] = start
            normalized["source_window_end"] = end
        for key in ("valid_from", "valid_to", "temporal_type", "created_at", "updated_at"):
            value = str(memory_object.get(key) or "").strip()
            if value:
                normalized[key] = value
        if candidate.resolved_date:
            normalized["resolved_date"] = candidate.resolved_date
        occurred_at = cls._first_evidence_packet_occurred_at(memory_object)
        if occurred_at:
            normalized["evidence_occurred_at"] = occurred_at
        speaker = cls._answer_evidence_speaker(memory_object, source_messages_by_id)
        if speaker:
            normalized["speaker_role"] = speaker
        trigger_quote = cls._best_evidence_packet_quote_for_role(
            memory_object,
            "trigger",
        )
        if trigger_quote:
            normalized["trigger_quote"] = trigger_quote
        return normalized

    @staticmethod
    def _evidence_packet_message_ids(memory_object: dict[str, Any]) -> list[str]:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return []
        seen: set[str] = set()
        message_ids: list[str] = []
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            spans = packet.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                message_id = str(span.get("message_id") or "").strip()
                if message_id and message_id not in seen:
                    seen.add(message_id)
                    message_ids.append(message_id)
        return message_ids

    @staticmethod
    def _first_evidence_packet_occurred_at(memory_object: dict[str, Any]) -> str:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return ""
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            spans = packet.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                occurred_at = str(span.get("occurred_at") or "").strip()
                if occurred_at:
                    return occurred_at
        return ""

    @staticmethod
    def _prefer_literal_evidence(
        *,
        query_type: QueryType,
        answer_shape: AnswerShape,
        exact_recall_mode: bool,
    ) -> bool:
        return (
            exact_recall_mode
            or query_type in ContextComposer.FINAL_ANSWER_EVIDENCE_QUERY_TYPES
            or answer_shape
            in {"single_fact", "list", "temporal", "raw_context"}
        )

    @classmethod
    def _literal_evidence_priority(
        cls,
        candidate: ScoredCandidate,
        *,
        prefer_literal_evidence: bool,
    ) -> int:
        if not prefer_literal_evidence:
            return 0
        is_summary = cls._is_summary_like_candidate(candidate)
        if cls._candidate_has_source_packet_quote(candidate):
            return 0 if not is_summary else 3
        if cls._is_verbatim_evidence_window_candidate(candidate):
            return 1
        if not is_summary:
            if str(candidate.memory_object.get("object_type") or "") == "evidence":
                return 2
            if cls._candidate_source_grounding(candidate) > 0.0:
                return 3
            return 4
        if (
            cls._candidate_source_grounding(candidate) > 0.0
            or cls._summary_has_source_objects(candidate)
        ):
            return 3
        return 5

    @staticmethod
    def _candidate_has_source_packet_quote(candidate: ScoredCandidate) -> bool:
        packets = candidate.memory_object.get("evidence_packets")
        if not isinstance(packets, list):
            return False
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            spans = packet.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if (
                    isinstance(span, dict)
                    and span.get("span_role") == "source"
                    and str(span.get("quote_text") or "").strip()
                ):
                    return True
        return False

    @staticmethod
    def _summary_has_source_objects(candidate: ScoredCandidate) -> bool:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        source_object_ids = payload_json.get("source_object_ids") or []
        return isinstance(source_object_ids, list) and bool(source_object_ids)

    @classmethod
    def _near_tie_verbatim_evidence_windows(
        cls,
        candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        windows = [
            candidate
            for candidate in candidates
            if cls._is_verbatim_evidence_window_candidate(candidate)
        ]
        if not windows:
            return []
        ordered = sorted(
            windows,
            key=lambda candidate: (-candidate.final_score, candidate.memory_id),
        )
        best_score = float(ordered[0].final_score)
        minimum_score = (
            best_score * cls.EVIDENCE_NEAR_TIE_SCORE_RATIO
            if best_score > 0.0
            else best_score
        )
        # Absolute-score gate: when source-grounded direct-evidence candidates
        # exist in the pool, a window must clear a fraction of the best direct
        # evidence score to be reserved. This stops low-score, token-heavy windows
        # from preempting tiny high-score direct facts under a tight budget. If no
        # direct evidence exists the gate is inert (window-only cases unaffected).
        direct_evidence_scores = [
            float(candidate.final_score)
            for candidate in candidates
            if cls._is_source_grounded_evidence_candidate(candidate)
            and not cls._is_verbatim_evidence_window_candidate(candidate)
            and not cls._is_summary_like_candidate(candidate)
        ]
        if direct_evidence_scores:
            direct_floor = (
                max(direct_evidence_scores) * cls.EVIDENCE_WINDOW_VS_DIRECT_SCORE_RATIO
            )
            minimum_score = max(minimum_score, direct_floor)
        selected: list[ScoredCandidate] = []
        for candidate in ordered:
            if len(selected) >= cls.EVIDENCE_OBLIGATION_MAX_WINDOWS:
                break
            if float(candidate.final_score) < minimum_score:
                continue
            if any(
                cls._is_duplicate_evidence_source(candidate, existing)
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return selected

    @classmethod
    def _is_source_grounded_evidence_candidate(
        cls,
        candidate: ScoredCandidate,
    ) -> bool:
        return (
            not cls._is_summary_like_candidate(candidate)
            and cls._candidate_source_grounding(candidate) >= 0.85
        )

    @staticmethod
    def _candidate_has_quoteable_source(candidate: ScoredCandidate) -> bool:
        if ContextComposer._candidate_source_message_ids(candidate):
            return True
        payload_json = candidate.memory_object.get("payload_json") or {}
        return isinstance(payload_json, dict) and bool(
            ContextComposer._conversation_chunk_excerpt(payload_json)
        )

    @staticmethod
    def _is_summary_like_candidate(candidate: ScoredCandidate) -> bool:
        memory_object = candidate.memory_object
        if memory_object.get("object_type") == "summary_view":
            return True
        return memory_object.get("source_kind") == "summarized"

    @staticmethod
    def _is_verbatim_evidence_window_candidate(candidate: ScoredCandidate) -> bool:
        memory_object = candidate.memory_object
        payload_json = memory_object.get("payload_json") or {}
        return bool(memory_object.get("is_verbatim_evidence_window")) or (
            isinstance(payload_json, dict)
            and payload_json.get("source_kind_variant") == "conversation_window"
        )

    @staticmethod
    def _has_sequential_source_chain(candidate: ScoredCandidate) -> bool:
        memory_object = candidate.memory_object
        payload_json = memory_object.get("payload_json") or {}
        return bool(memory_object.get("is_verbatim_evidence_window")) or (
            isinstance(payload_json, dict)
            and payload_json.get("source_kind_variant")
            in {"conversation_window", "summary_source_window"}
        )

    @classmethod
    def _is_duplicate_evidence_source(
        cls,
        candidate: ScoredCandidate,
        other: ScoredCandidate,
    ) -> bool:
        candidate_window = cls._candidate_source_window(candidate)
        other_window = cls._candidate_source_window(other)
        if (
            candidate_window is not None
            and other_window is not None
            and candidate_window == other_window
        ):
            return True
        return (
            cls._candidate_source_similarity(candidate, other)
            >= cls.EVIDENCE_DUPLICATE_SOURCE_SIMILARITY
        )

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
        active_realm_id: str | None,
        redact_high_risk_secret_literals: bool,
        fact_facet_span_coadmission_enabled: bool = False,
        rendered_source_quote_keys: set[str] | None = None,
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

        token_cost = cls._action_token_cost(
            new_items,
            selected_count=len(selected),
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            fact_facet_span_coadmission_enabled=fact_facet_span_coadmission_enabled,
            rendered_source_quote_keys=rendered_source_quote_keys,
        )
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
        active_realm_id: str | None = None,
        redact_high_risk_secret_literals: bool = True,
        fact_facet_span_coadmission_enabled: bool = False,
        rendered_source_quote_keys: set[str] | None = None,
    ) -> int:
        quote_keys = set(rendered_source_quote_keys or ())
        total = 0
        for offset, item in enumerate(items, start=1):
            entry = cls._format_memory_entry(
                selected_count + offset,
                item,
                active_realm_id=active_realm_id,
                redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                fact_facet_span_coadmission_enabled=(
                    fact_facet_span_coadmission_enabled
                ),
                suppressed_source_quote_keys=frozenset(quote_keys),
            )
            total += cls.estimate_tokens(entry)
            quote_keys.update(cls._source_quote_keys_from_rendered_memory_entry(entry))
        return total

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

    @classmethod
    def _source_weighted_score(
        cls,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        exact_recall_mode: bool,
        query_text: str | None,
    ) -> float:
        score = float(candidate.final_score)
        if exact_recall_mode and cls._has_direct_user_source(
            candidate,
            source_messages_by_id,
            query_text=query_text,
        ):
            score += cls.EXACT_DIRECT_USER_SOURCE_BOOST
        return score

    @classmethod
    def _has_direct_user_source(
        cls,
        candidate: ScoredCandidate,
        source_messages_by_id: dict[str, dict[str, Any]],
        *,
        query_text: str | None,
    ) -> bool:
        query_tokens = cls._content_tokens(query_text or "")
        for source_message_id in cls._candidate_source_message_ids(candidate):
            message = source_messages_by_id.get(source_message_id)
            if message is None:
                continue
            if str(message.get("role") or "").strip().lower() == "user":
                if not query_tokens:
                    return True
                candidate_tokens = cls._content_tokens(
                    " ".join(
                        (
                            str(candidate.memory_object.get("canonical_text") or ""),
                            str(message.get("text") or ""),
                        )
                    )
                )
                if query_tokens & candidate_tokens:
                    return True
        return False

    @staticmethod
    def render_contract_block(
        current_contract: dict[str, dict[str, Any]],
        resolved_policy: ResolvedRetrievalPolicy,
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
            lines.append(
                f"- {dimension}: {ContextComposer._format_contract_value(value)}"
            )
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
        realm_note = ContextComposer._contract_realm_note(value)
        if realm_note:
            rendered += f" [{realm_note}]"
        return rendered

    @staticmethod
    def _contract_realm_note(value: dict[str, Any]) -> str | None:
        realm_payload = value.get("realm")
        if not isinstance(realm_payload, dict):
            return None
        realm_id = ContextComposer._optional_text(
            realm_payload.get("realm_id") or realm_payload.get("active_realm_id")
        )
        if realm_id is None:
            return None
        display_name = ContextComposer._optional_text(
            realm_payload.get("display_name")
            or realm_payload.get("active_realm_display_name")
        )
        active_id = ContextComposer._optional_text(
            realm_payload.get("active_request_realm_id")
        )
        mode = ContextComposer._optional_text(
            realm_payload.get("cross_realm_mode") or realm_payload.get("bridge_mode")
        )
        label = display_name or realm_id
        if active_id is not None and mode in {"attributed", "applicable"}:
            return f"realm: in Realm {label} [cross_realm: {mode}; active={active_id}]"
        return f"realm: in Realm {label}"

    @classmethod
    def _format_workspace_block(
        cls,
        workspace_rollup: dict[str, Any] | None,
    ) -> str:
        if not workspace_rollup:
            return ""
        summary_text = str(workspace_rollup.get("summary_text", "")).strip()
        if not summary_text:
            return ""
        return f"[Workspace Context]\n{summary_text}"

    @staticmethod
    def _presence_boundary_allowed(
        candidate: ScoredCandidate,
        active_presence_id: str | None,
    ) -> bool:
        if active_presence_id is None:
            return True
        memory_presence_id = ContextComposer._optional_text(
            candidate.memory_object.get("active_presence_id")
        )
        if memory_presence_id is None or memory_presence_id == active_presence_id:
            return True
        kind = ContextComposer._presence_payload_kind(candidate.memory_object, "active")
        return kind is not None and kind != "unknown"

    @staticmethod
    def _presence_payload_kind(memory_object: dict[str, Any], role: str) -> str | None:
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        attribution = payload_json.get("presence_attribution") or {}
        if not isinstance(attribution, dict):
            return None
        presence = attribution.get(role) or {}
        if not isinstance(presence, dict):
            return None
        return ContextComposer._optional_text(presence.get("kind"))

    @staticmethod
    def _presence_attribution_note(memory_object: dict[str, Any]) -> str | None:
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        attribution = payload_json.get("presence_attribution") or {}
        if not isinstance(attribution, dict):
            return None
        active = attribution.get("active") or {}
        source = attribution.get("source") or {}
        if not isinstance(active, dict):
            active = {}
        if not isinstance(source, dict):
            source = {}
        active_id = ContextComposer._optional_text(
            memory_object.get("active_presence_id") or active.get("presence_id")
        )
        source_id = ContextComposer._optional_text(
            memory_object.get("source_presence_id") or source.get("presence_id")
        )
        parts: list[str] = []
        if active_id is not None:
            active_label = ContextComposer._presence_label(active, active_id)
            parts.append(f"active={active_label}")
        if source_id is not None and source_id != active_id:
            source_label = ContextComposer._presence_label(source, source_id)
            parts.append(f"source={source_label}")
        return "; ".join(parts) if parts else None

    @staticmethod
    def _presence_label(presence: dict[str, Any], fallback_id: str) -> str:
        display_name = ContextComposer._optional_text(presence.get("display_name"))
        kind = ContextComposer._optional_text(presence.get("kind"))
        label = display_name or fallback_id
        if kind is not None:
            return f"{label} [{kind}]"
        return label

    @staticmethod
    def _space_boundary_note(memory_object: dict[str, Any]) -> str | None:
        payload_json = memory_object.get("payload_json") or {}
        space_payload = (
            payload_json.get("space_boundary")
            if isinstance(payload_json, dict)
            else None
        )
        space_id = ContextComposer._optional_text(memory_object.get("space_id"))
        boundary_mode = ContextComposer._optional_text(
            memory_object.get("space_boundary_mode")
        )
        display_name: str | None = None
        if isinstance(space_payload, dict):
            space_id = space_id or ContextComposer._optional_text(
                space_payload.get("active_space_id")
            )
            boundary_mode = boundary_mode or ContextComposer._optional_text(
                space_payload.get("boundary_mode")
            )
            display_name = ContextComposer._optional_text(
                space_payload.get("display_name")
            )
        if space_id is None:
            return None
        label = display_name or space_id
        if boundary_mode is not None:
            return f"{label} [{boundary_mode}]"
        return label

    @staticmethod
    def _mind_perspective_note(memory_object: dict[str, Any]) -> str | None:
        payload_json = memory_object.get("payload_json") or {}
        mind_payload = (
            payload_json.get("mind_perspective")
            if isinstance(payload_json, dict)
            else None
        )
        owner_id = ContextComposer._optional_text(memory_object.get("memory_owner_id"))
        source_id = ContextComposer._optional_text(memory_object.get("source_mind_id"))
        relation = ContextComposer._optional_text(memory_object.get("mind_relation"))
        grant_kind = ContextComposer._optional_text(
            memory_object.get("mind_grant_kind")
        )
        grant_target_kind = ContextComposer._optional_text(
            memory_object.get("mind_grant_target_kind")
        )
        grant_target_id = ContextComposer._optional_text(
            memory_object.get("mind_grant_target_id")
        )
        if isinstance(mind_payload, dict):
            owner_id = owner_id or ContextComposer._optional_text(
                mind_payload.get("memory_owner_id")
            )
            source_id = source_id or ContextComposer._optional_text(
                mind_payload.get("source_mind_id")
            )
            relation = relation or ContextComposer._optional_text(
                mind_payload.get("mind_relation")
            )
            grant_kind = grant_kind or ContextComposer._optional_text(
                mind_payload.get("grant_kind")
            )
            grant_target_kind = grant_target_kind or ContextComposer._optional_text(
                mind_payload.get("grant_target_kind")
            )
            grant_target_id = grant_target_id or ContextComposer._optional_text(
                mind_payload.get("grant_target_id")
            )
        if owner_id is None:
            return None
        parts = [f"owned by {owner_id}"]
        if relation == "granted" and grant_kind is not None:
            grant = f"granted: {grant_kind}"
            if grant_target_kind is not None and grant_target_id is not None:
                grant = f"{grant}; target={grant_target_kind}:{grant_target_id}"
            parts.append(f"[{grant}]")
        elif relation == "same":
            parts.append("[same]")
        if source_id is not None and source_id != owner_id:
            parts.append(f"source={source_id}")
        return " ".join(parts)

    @staticmethod
    def _realm_note(
        memory_object: dict[str, Any],
        *,
        active_realm_id: str | None,
    ) -> str | None:
        payload_json = memory_object.get("payload_json") or {}
        realm_payload = (
            payload_json.get("realm") if isinstance(payload_json, dict) else None
        )
        realm_id = ContextComposer._optional_text(memory_object.get("realm_id"))
        display_name: str | None = None
        bridge_mode = ContextComposer._optional_text(
            memory_object.get("realm_bridge_mode")
            or memory_object.get("cross_realm_mode")
        )
        if isinstance(realm_payload, dict):
            realm_id = realm_id or ContextComposer._optional_text(
                realm_payload.get("realm_id") or realm_payload.get("active_realm_id")
            )
            display_name = ContextComposer._optional_text(
                realm_payload.get("display_name")
                or realm_payload.get("active_realm_display_name")
            )
            bridge_mode = bridge_mode or ContextComposer._optional_text(
                realm_payload.get("cross_realm_mode")
                or realm_payload.get("bridge_mode")
            )
        if realm_id is None:
            return None
        active_id = ContextComposer._optional_text(active_realm_id)
        label = display_name or realm_id
        if active_id is None:
            return f"in Realm {label}"
        if realm_id == active_id:
            return f"in Realm {label} [same]"
        if bridge_mode in {"attributed", "applicable"}:
            return f"in Realm {label} [cross_realm: {bridge_mode}; active={active_id}]"
        return f"in Realm {label} [cross_realm: unknown; active={active_id}]"

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _format_memory_entry(
        index: int,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]] | None = None,
        source_quote_options: _SourceQuoteOptions | None = None,
        query_text: str | None = None,
        active_realm_id: str | None = None,
        redact_high_risk_secret_literals: bool = True,
        fact_facet_span_coadmission_enabled: bool = False,
        suppressed_source_quote_keys: frozenset[str] = frozenset(),
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
            if redact_high_risk_secret_literals:
                metadata_parts.append(f"privacy_level: {privacy_level}")
            else:
                metadata_parts.append(
                    f"privacy_classification_non_blocking: level_{privacy_level}"
                )
        memory_category = str(memory_object.get("memory_category") or "").strip()
        if memory_category:
            if redact_high_risk_secret_literals:
                metadata_parts.append(f"memory_category: {memory_category}")
            else:
                metadata_parts.append(
                    f"memory_category_non_blocking: {memory_category}"
                )
        intimacy_boundary = str(memory_object.get("intimacy_boundary") or "").strip()
        if intimacy_boundary and intimacy_boundary != "ordinary":
            metadata_parts.append(f"intimacy_boundary: {intimacy_boundary}")
        if bool(memory_object.get("preserve_verbatim")):
            metadata_parts.append("preserve_verbatim: true")
        presence_note = ContextComposer._presence_attribution_note(memory_object)
        if presence_note:
            metadata_parts.append(f"presence: {presence_note}")
        space_note = ContextComposer._space_boundary_note(memory_object)
        if space_note:
            metadata_parts.append(f"space: {space_note}")
        mind_note = ContextComposer._mind_perspective_note(memory_object)
        if mind_note:
            metadata_parts.append(f"mind: {mind_note}")
        realm_note = ContextComposer._realm_note(
            memory_object,
            active_realm_id=active_realm_id,
        )
        if realm_note:
            metadata_parts.append(f"realm: {realm_note}")
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
        disclosure = ContextComposer._disclosure_action(memory_object)
        withhold_literal = (
            redact_high_risk_secret_literals
            and disclosure is HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
        )
        if withhold_literal:
            metadata_parts.append(f"disclosure_action: {disclosure.value}")
        elif disclosure is HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL:
            metadata_parts.append(
                "privacy_restrictions_inactive: high_risk_secret_literal_unredacted"
            )
        memory_text = (
            "Protected high-risk memory present; raw value withheld. "
            "Use a host-managed secure reveal or verification flow."
            if withhold_literal
            else str(memory_object.get("canonical_text", ""))
        )
        coadmitted_source_span = ""
        if (
            fact_facet_span_coadmission_enabled
            and not withhold_literal
            and ContextComposer._is_fact_facet_memory_object(memory_object)
        ):
            candidate_source_span = (
                ContextComposer._best_evidence_packet_quote_with_role_for_query(
                    memory_object,
                    query_text=query_text,
                )[0]
            )
            if (
                candidate_source_span
                and not ContextComposer._source_quote_is_suppressed(
                    candidate_source_span,
                    suppressed_source_quote_keys,
                )
            ):
                coadmitted_source_span = candidate_source_span
                metadata_parts.append("fact_facet_span_coadmitted: true")
                memory_text = f"source_span: {coadmitted_source_span}"
        lines = [f"{index}. ({', '.join(metadata_parts)})\n   {memory_text}"]
        if not withhold_literal:
            evidence_packet_lines = ContextComposer._evidence_packet_lines(
                memory_object
            )
            if suppressed_source_quote_keys:
                evidence_packet_lines = [
                    line
                    for line in evidence_packet_lines
                    if not ContextComposer._source_quote_line_is_suppressed(
                        line,
                        suppressed_source_quote_keys,
                    )
                ]
            if coadmitted_source_span:
                fact_pointer = str(memory_object.get("canonical_text") or "").strip()
                if fact_pointer:
                    lines.append(f"   fact_facet_pointer: {fact_pointer}")
                evidence_packet_lines = [
                    line
                    for line in evidence_packet_lines
                    if ContextComposer._normalize_quote_for_compare(
                        coadmitted_source_span
                    )
                    not in ContextComposer._normalize_quote_for_compare(line)
                ]
            if evidence_packet_lines:
                lines.extend(evidence_packet_lines)
                source_quote = ContextComposer._source_quote_for_candidate(
                    candidate,
                    source_messages_by_id=source_messages_by_id or {},
                    options=source_quote_options or _SourceQuoteOptions(enabled=False),
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    query_text=query_text,
                )
                if (
                    source_quote
                    and not ContextComposer._source_quote_is_suppressed(
                        source_quote,
                        suppressed_source_quote_keys,
                    )
                    and query_text
                    and ContextComposer._normalize_quote_for_compare(source_quote)
                    not in ContextComposer._normalize_quote_for_compare(
                        "\n".join(evidence_packet_lines)
                    )
                ):
                    lines.append(f"   source_quote: {source_quote}")
            else:
                source_quote = ContextComposer._source_quote_for_candidate(
                    candidate,
                    source_messages_by_id=source_messages_by_id or {},
                    options=source_quote_options or _SourceQuoteOptions(enabled=False),
                    redact_high_risk_secret_literals=redact_high_risk_secret_literals,
                    query_text=query_text,
                )
                if source_quote and not ContextComposer._source_quote_is_suppressed(
                    source_quote,
                    suppressed_source_quote_keys,
                ):
                    lines.append(f"   source_quote: {source_quote}")
                elif is_conversation_chunk:
                    excerpt = ContextComposer._conversation_chunk_excerpt(payload_json)
                    if excerpt:
                        lines.append(f"   source_excerpt: {excerpt}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_quote_for_compare(value: str) -> str:
        return " ".join(str(value).casefold().split())

    @classmethod
    def _source_quote_is_suppressed(
        cls,
        value: str,
        suppressed_source_quote_keys: frozenset[str],
    ) -> bool:
        normalized = cls._normalize_quote_for_compare(value)
        if len(normalized) < cls.RENDERED_SOURCE_QUOTE_DEDUPE_MIN_CHARS:
            return False
        return any(normalized in key for key in suppressed_source_quote_keys)

    @classmethod
    def _source_quote_line_is_suppressed(
        cls,
        line: str,
        suppressed_source_quote_keys: frozenset[str],
    ) -> bool:
        stripped = line.strip()
        if not (
            stripped.startswith("source_span:")
            or stripped.startswith("source_quote:")
        ):
            return False
        return cls._source_quote_is_suppressed(
            stripped.split(":", 1)[1].strip(),
            suppressed_source_quote_keys,
        )

    @classmethod
    def _source_quote_keys_from_rendered_memory_entry(cls, entry: str) -> set[str]:
        keys: set[str] = set()
        for line in entry.splitlines():
            stripped = line.strip()
            if not (
                stripped.startswith("source_span:")
                or stripped.startswith("source_quote:")
            ):
                continue
            normalized = cls._normalize_quote_for_compare(
                stripped.split(":", 1)[1].strip()
            )
            if len(normalized) >= cls.RENDERED_SOURCE_QUOTE_DEDUPE_MIN_CHARS:
                keys.add(normalized)
        return keys

    @staticmethod
    def _is_fact_facet_memory_object(memory_object: dict[str, Any]) -> bool:
        if bool(memory_object.get("is_fact_facet_candidate")):
            return True
        payload_json = memory_object.get("payload_json") or {}
        return (
            isinstance(payload_json, dict)
            and payload_json.get("source_kind_variant") == "fact_facet"
            and isinstance(payload_json.get("fact_facet"), dict)
        )

    @staticmethod
    def _evidence_packet_lines(memory_object: dict[str, Any]) -> list[str]:
        packets = memory_object.get("evidence_packets")
        if not isinstance(packets, list) or not packets:
            return []
        lines: list[str] = []
        for packet in packets[:2]:
            if not isinstance(packet, dict):
                continue
            support_kind = str(packet.get("support_kind") or "").strip()
            polarity = str(packet.get("evidence_polarity") or "").strip()
            speaker_relation = str(
                packet.get("speaker_relation_to_subject") or ""
            ).strip()
            confidence = packet.get("confidence")
            metadata_parts = []
            if support_kind:
                metadata_parts.append(f"support: {support_kind}")
            if polarity:
                metadata_parts.append(f"polarity: {polarity}")
            if speaker_relation:
                metadata_parts.append(f"speaker_relation: {speaker_relation}")
            if isinstance(confidence, (float, int)):
                metadata_parts.append(f"confidence: {float(confidence):.2f}")
            if metadata_parts:
                lines.append(f"   evidence_packet: {', '.join(metadata_parts)}")
            spans = packet.get("spans")
            if isinstance(spans, list):
                lines.extend(
                    ContextComposer._evidence_span_lines(
                        spans,
                        span_role="source",
                        label="source_quote",
                        max_spans=2,
                    )
                )
                lines.extend(
                    ContextComposer._evidence_span_lines(
                        spans,
                        span_role="trigger",
                        label="trigger_quote",
                        max_spans=2,
                    )
                )
            rationale = str(packet.get("rationale") or "").strip()
            if rationale:
                lines.append(
                    "   rationale: "
                    + ContextComposer._truncate_inline(rationale, 180)
                )
        return lines

    @staticmethod
    def _evidence_span_lines(
        spans: list[Any],
        *,
        span_role: str,
        label: str,
        max_spans: int,
    ) -> list[str]:
        lines: list[str] = []
        for span in spans:
            if len(lines) >= max_spans:
                break
            if not isinstance(span, dict) or span.get("span_role") != span_role:
                continue
            quote = str(span.get("quote_text") or "").strip()
            if not quote:
                continue
            prefix_parts: list[str] = []
            metadata = span.get("metadata_json") or {}
            if isinstance(metadata, dict):
                role = str(metadata.get("message_role") or "").strip()
                if role:
                    prefix_parts.append(role)
            occurred_at = str(span.get("occurred_at") or "").strip()
            if occurred_at:
                prefix_parts.append(f"@ {occurred_at}")
            seq = span.get("seq")
            if seq is not None:
                prefix_parts.append(f"seq {seq}")
            prefix = f"{' '.join(prefix_parts)}: " if prefix_parts else ""
            lines.append(
                f"   {label}: "
                + ContextComposer._truncate_inline(f"{prefix}{quote}", 260)
            )
        return lines

    @staticmethod
    def _disclosure_action(memory_object: dict[str, Any]) -> HighRiskDisclosureAction:
        raw_category = str(
            memory_object.get("memory_category") or MemoryCategory.UNKNOWN.value
        )
        try:
            memory_category = MemoryCategory(raw_category)
        except ValueError:
            memory_category = MemoryCategory.UNKNOWN
        try:
            privacy_level = int(memory_object.get("privacy_level") or 0)
        except (TypeError, ValueError):
            privacy_level = 0
        return disclosure_action(
            memory_category=memory_category,
            privacy_level=privacy_level,
            preserve_verbatim=bool(memory_object.get("preserve_verbatim")),
        )

    @staticmethod
    def _format_state_block(user_state: dict[str, Any] | None) -> str:
        if not user_state:
            return ""
        lines = ["[Current User State]"]
        for key, value in sorted(user_state.items()):
            if isinstance(value, dict) and "value" in value and "realm" in value:
                rendered = str(value.get("value"))
                realm_note = ContextComposer._contract_realm_note(value)
                if realm_note:
                    rendered += f" [{realm_note}]"
            elif isinstance(value, (dict, list)):
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
        active_realm_id: str | None = None,
        redact_high_risk_secret_literals: bool = True,
        fact_facet_span_coadmission_enabled: bool = False,
        rendered_source_quote_keys: set[str] | None = None,
    ) -> int:
        """Admit a candidate priced on its bare (quote-free) form only.

        Source quotes are never attached here. They are funded in a separate
        post-selection upgrade pass from budget left over after all bare
        admissions, so enabling quotes can never change the set of selected
        memory entries (see ``_upgrade_entries_with_source_quotes``).
        """
        if (
            candidate.memory_id in selected_ids
            or remaining_budget <= 0
            or len(selected) >= max_items
        ):
            return remaining_budget
        candidate_block = cls._format_memory_entry(
            len(selected) + 1,
            candidate,
            active_realm_id=active_realm_id,
            redact_high_risk_secret_literals=redact_high_risk_secret_literals,
            fact_facet_span_coadmission_enabled=fact_facet_span_coadmission_enabled,
            suppressed_source_quote_keys=frozenset(rendered_source_quote_keys or ()),
        )
        candidate_tokens = cls.estimate_tokens(candidate_block)
        if candidate_tokens > remaining_budget:
            return remaining_budget
        memory_lines.append(candidate_block)
        if rendered_source_quote_keys is not None:
            rendered_source_quote_keys.update(
                cls._source_quote_keys_from_rendered_memory_entry(candidate_block)
            )
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
        context_budget_tokens: int | None = None,
    ) -> _SourceQuoteOptions:
        enabled = exact_recall_mode or query_type in {
            "temporal",
            "slot_fill",
            "broad_list",
            "default",
        }
        if not enabled:
            return _SourceQuoteOptions(enabled=False)
        if query_type == "temporal":
            max_entries = 4
        elif query_type in {"slot_fill", "broad_list"}:
            max_entries = 6
        elif query_type == "default":
            return _SourceQuoteOptions(
                enabled=True,
                max_messages=2,
                max_chars=520,
                max_message_chars=180,
                max_entries=2,
            )
        else:
            max_entries = 5
        max_messages = 3
        max_chars = 520
        max_message_chars = 180
        if context_budget_tokens is not None and context_budget_tokens >= 8_000:
            scale = min(3.0, max(1.0, context_budget_tokens / 10_000))
            max_messages = 4
            max_chars = int(520 * scale)
            max_message_chars = int(180 * min(2.5, scale))
            max_entries += 2
        return _SourceQuoteOptions(
            enabled=True,
            max_messages=max_messages,
            max_chars=max_chars,
            max_message_chars=max_message_chars,
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
    def _source_chain_for_candidate(
        cls,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        options: _SourceQuoteOptions,
        query_text: str | None,
    ) -> list[str]:
        if not options.enabled or not query_text or not source_messages_by_id:
            return []
        memory_object = candidate.memory_object
        if str(memory_object.get("object_type") or "") != "summary_view" and not (
            cls._has_sequential_source_chain(candidate)
        ):
            return []
        source_message_ids = cls._candidate_source_message_ids(candidate)
        if len(source_message_ids) <= options.max_messages:
            return []

        entries: list[tuple[int, dict[str, Any], str]] = []
        for source_index, source_message_id in enumerate(source_message_ids):
            message = source_messages_by_id.get(source_message_id)
            if message is None or not cls._message_allows_source_quote(message):
                continue
            text = str(message.get("text") or "").strip()
            if text:
                entries.append((source_index, message, text))
        if len(entries) <= options.max_messages and not cls._has_sequential_source_chain(
            candidate
        ):
            return []

        anchor_offset = cls._source_chain_anchor_offset(entries, query_text)
        if anchor_offset is None:
            return []
        max_messages = min(
            cls.SOURCE_CHAIN_MAX_MESSAGES,
            max(options.max_messages, cls.SOURCE_CHAIN_MAX_MESSAGES),
        )
        start = anchor_offset
        if start + max_messages > len(entries):
            start = max(0, len(entries) - max_messages)
        selected_entries = entries[start : start + max_messages]

        rendered: list[str] = []
        total_chars = 0
        max_chars = max(options.max_chars, cls.SOURCE_CHAIN_MAX_CHARS)
        for _, message, text in selected_entries:
            prefix_parts = [str(message.get("role") or "unknown").strip() or "unknown"]
            occurred_at = str(message.get("occurred_at") or "").strip()
            if occurred_at:
                prefix_parts.append(f"@ {occurred_at}")
            seq = message.get("seq")
            if seq is not None:
                prefix_parts.append(f"seq {seq}")
            text = cls._truncate_inline(text, max(options.max_message_chars, 220))
            segment = f"{' '.join(prefix_parts)}: {text}"
            segment = cls._truncate_inline(segment, max(1, max_chars - total_chars))
            if not segment:
                continue
            rendered.append(segment)
            total_chars += len(segment)
            if total_chars >= max_chars:
                break
        return rendered

    @classmethod
    def _source_chain_anchor_offset(
        cls,
        entries: list[tuple[int, dict[str, Any], str]],
        query_text: str,
    ) -> int | None:
        best_offset: int | None = None
        best_score = 0.0
        for offset, (_, _, text) in enumerate(entries):
            score = cls._quote_query_relevance(text, query_text)
            if score > best_score or (
                score == best_score
                and best_offset is not None
                and cls._question_quote_priority(text)
                < cls._question_quote_priority(entries[best_offset][2])
            ):
                best_score = score
                best_offset = offset
        return best_offset

    @classmethod
    def _source_quote_for_candidate(
        cls,
        candidate: ScoredCandidate,
        *,
        source_messages_by_id: dict[str, dict[str, Any]],
        options: _SourceQuoteOptions,
        redact_high_risk_secret_literals: bool,
        query_text: str | None = None,
    ) -> str:
        if not options.enabled:
            return ""
        memory_object = candidate.memory_object
        if (
            redact_high_risk_secret_literals
            and cls._disclosure_action(memory_object)
            is HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
        ):
            return ""
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return ""
        if not source_messages_by_id:
            return ""
        source_message_ids = cls._candidate_source_message_ids(candidate)
        if not source_message_ids:
            return ""

        message_candidates: list[tuple[int, float, dict[str, Any], str]] = []
        for index, source_message_id in enumerate(source_message_ids):
            message = source_messages_by_id.get(source_message_id)
            if message is None or not cls._message_allows_source_quote(message):
                continue
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            message_candidates.append(
                (
                    index,
                    cls._quote_query_relevance(text, query_text),
                    message,
                    text,
                )
            )
        if not message_candidates:
            return ""
        if query_text:
            message_candidates = sorted(
                message_candidates,
                key=lambda item: (
                    -item[1],
                    cls._question_quote_priority(item[3]),
                    -item[0],
                ),
            )

        rendered: list[str] = []
        total_chars = 0
        for _, _, message, text in message_candidates:
            if len(rendered) >= options.max_messages:
                break
            text = cls._source_quote_snippet(
                text,
                max_chars=options.max_message_chars,
                query_text=query_text,
            )
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

    @classmethod
    def _quote_query_relevance(cls, text: str, query_text: str | None) -> float:
        query_tokens = cls._quote_relevance_tokens(query_text or "")
        if not query_tokens:
            return 0.0
        text_tokens = cls._quote_relevance_tokens(
            cls._strip_speaker_prefix_for_relevance(text)
        )
        if not text_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(1, len(query_tokens))

    @classmethod
    def _quote_relevance_tokens(cls, text: str) -> set[str]:
        return {token for token in cls._content_tokens(text) if len(token) >= 4}

    @staticmethod
    def _strip_speaker_prefix_for_relevance(text: str) -> str:
        prefix, separator, rest = str(text).partition(":")
        if (
            separator
            and prefix
            and prefix.strip().lower() in {"assistant", "system", "tool", "user"}
        ):
            return rest
        return text

    @staticmethod
    def _question_quote_priority(text: str) -> int:
        return 1 if str(text).strip().endswith("?") else 0

    @classmethod
    def _source_quote_snippet(
        cls,
        text: str,
        *,
        max_chars: int,
        query_text: str | None,
    ) -> str:
        if len(text) <= max_chars:
            return text
        anchor = cls._source_quote_anchor(text, query_text)
        if anchor is None:
            return cls._truncate_inline(text, max_chars)
        context_before = max(20, max_chars // 3)
        start = max(0, anchor - context_before)
        end = min(len(text), start + max_chars)
        if end - start < max_chars:
            start = max(0, end - max_chars)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = f"... {snippet.lstrip()}"
        if end < len(text):
            snippet = f"{snippet.rstrip()}..."
        return snippet

    @classmethod
    def _source_quote_anchor(cls, text: str, query_text: str | None) -> int | None:
        if not query_text:
            return None
        lowered = text.lower()
        best_index: int | None = None
        for token in cls._tokenize(query_text):
            if len(token) < 4:
                continue
            index = lowered.find(token)
            if index < 0:
                continue
            if best_index is None or index < best_index:
                best_index = index
        return best_index

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
        return truncate_inline(text, max_chars)
