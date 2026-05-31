"""Mechanical repair of lean need-detection plan linkage.

The primary need-detection call returns a lean ``QueryPlanCore`` whose schema
omits the cross-field validators that the rich ``QueryIntelligenceResult``
enforces (hint<->sub_query linkage, anchor referencing, anchor dedup). Medium
models drift on those constraints, and the old rich schema turned that drift
into hard structured-output failures.

This module repairs that drift deterministically instead of raising:

* sparse hints whose ``sub_query_text`` does not match any sub-query are
  re-linked to the closest existing sub-query (exact match, then unique
  index/position fallback) or dropped when no safe target exists;
* duplicate hint targets keep only the first hint;
* runtime anchors are re-linked the same way and deduplicated by
  ``(sub_query_text, anchor_type, original_surface)``.

All operations are mechanical (string equality and list position only). No
semantic interpretation, regex, or keyword matching is performed here. Dropped
items are reported through ``RepairOutcome.trace_events`` so the loss is
visible in the retrieval trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atagia.models.schemas_memory import (
    QueryType,
    RuntimeAnchor,
    SparseQueryHint,
    TemporaryScaffoldingTrace,
    _normalize_sparse_hint_precision_fields,
)


@dataclass(slots=True)
class RepairOutcome:
    """Result of repairing lean plan linkage before building the rich object."""

    sub_queries: list[str]
    sparse_query_hints: list[SparseQueryHint]
    anchors: list[RuntimeAnchor]
    trace_events: list[TemporaryScaffoldingTrace] = field(default_factory=list)


def _repair_trace_event(*, mechanism: str, trace_flag: str) -> TemporaryScaffoldingTrace:
    return TemporaryScaffoldingTrace(
        component="need_detection",
        mechanism=mechanism,
        trace_flag=trace_flag,
        intended_metric="lean_query_plan_linkage_recovery",
        replacement_architecture=(
            "schema-stable route classifier with materialized retrieval surfaces"
        ),
        retirement_condition=(
            "retire when the primary plan model returns linkage-consistent "
            "hints and anchors on retained replay without mechanical repair"
        ),
    )


def _resolve_linked_sub_query(
    referenced: str,
    sub_queries: list[str],
    *,
    fallback_index: int,
) -> str | None:
    """Return the sub-query a hint/anchor should attach to, or None to drop.

    Mirrors the exact-match rule the rich validator enforced. When the
    referenced text matches no sub-query, fall back to position only when it is
    unambiguous: a single sub-query absorbs everything, otherwise the
    item at ``fallback_index`` is used when that index exists. Anything else is
    dropped rather than guessed.
    """
    if referenced in sub_queries:
        return referenced
    if not sub_queries:
        return None
    if len(sub_queries) == 1:
        return sub_queries[0]
    if 0 <= fallback_index < len(sub_queries):
        return sub_queries[fallback_index]
    return None


def repair_query_plan_linkage(
    *,
    sub_queries: list[str],
    sparse_query_hints: list[SparseQueryHint],
    anchors: list[RuntimeAnchor],
    query_type: QueryType,
    callback_bias: bool,
) -> RepairOutcome:
    """Repair hint/anchor linkage so the rich object can be built without raising."""
    trace_events: list[TemporaryScaffoldingTrace] = []

    repaired_hints: list[SparseQueryHint] = []
    seen_hint_targets: set[str] = set()
    dropped_hint = False
    relinked_hint = False
    for index, hint in enumerate(sparse_query_hints):
        target = _resolve_linked_sub_query(
            hint.sub_query_text, sub_queries, fallback_index=index
        )
        if target is None:
            dropped_hint = True
            continue
        if target != hint.sub_query_text:
            relinked_hint = True
            hint = hint.model_copy(update={"sub_query_text": target})
        if target in seen_hint_targets:
            dropped_hint = True
            continue
        seen_hint_targets.add(target)
        repaired_hints.append(
            _normalize_sparse_hint_precision_fields(
                hint,
                query_type=query_type,
                callback_bias=callback_bias,
            )
        )
    if relinked_hint:
        trace_events.append(
            _repair_trace_event(
                mechanism="lean_plan_sparse_hint_relinked",
                trace_flag="sparse_query_hint:relinked_to_sub_query",
            )
        )
    if dropped_hint:
        trace_events.append(
            _repair_trace_event(
                mechanism="lean_plan_sparse_hint_dropped",
                trace_flag="sparse_query_hint:dropped_unlinkable",
            )
        )

    repaired_anchors: list[RuntimeAnchor] = []
    seen_anchor_signatures: set[tuple[str, str, str]] = set()
    dropped_anchor = False
    relinked_anchor = False
    for index, anchor in enumerate(anchors):
        target = _resolve_linked_sub_query(
            anchor.sub_query_text, sub_queries, fallback_index=index
        )
        if target is None:
            dropped_anchor = True
            continue
        if target != anchor.sub_query_text:
            relinked_anchor = True
            anchor = anchor.model_copy(update={"sub_query_text": target})
        signature = (anchor.sub_query_text, anchor.anchor_type, anchor.original_surface)
        if signature in seen_anchor_signatures:
            dropped_anchor = True
            continue
        seen_anchor_signatures.add(signature)
        repaired_anchors.append(anchor)
    if relinked_anchor:
        trace_events.append(
            _repair_trace_event(
                mechanism="lean_plan_anchor_relinked",
                trace_flag="runtime_anchor:relinked_to_sub_query",
            )
        )
    if dropped_anchor:
        trace_events.append(
            _repair_trace_event(
                mechanism="lean_plan_anchor_dropped",
                trace_flag="runtime_anchor:dropped_unlinkable_or_duplicate",
            )
        )

    return RepairOutcome(
        sub_queries=list(sub_queries),
        sparse_query_hints=repaired_hints,
        anchors=repaired_anchors,
        trace_events=trace_events,
    )
