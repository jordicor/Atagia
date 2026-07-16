"""Mechanical duplicate-carrier collapse at the fusion-to-scoring seam.

Near-duplicate carriers of one fact each consume shortlist slots, scoring
card calls, and composer budget. This module
collapses them AFTER fusion and policy filtering but BEFORE scoring, using
provenance joins only:

- equal facet value identity (payload ``value_norm_key``) on overlapping
  source spans,
- equal exhaustive-coverage member-key sets resolved by the SAME shared
  ladder the composer uses (``coverage_keys.resolve_member_keys`` — modern
  ``coverage_members`` first, legacy ``value_*`` payload keys as fallback;
  equal sets only, so no list member can be lost; applied ONLY under
  exhaustive known-set coverage — see ``collapse_duplicate_carriers``),
- identical content token sequence on overlapping source spans
  (chunk-duplicate extraction of the same utterance).

NO LLM calls, NO text-similarity heuristics beyond exact token identity,
NO semantic guessing: carriers whose provenance differs stay separate.
Carriers only join within the same mechanical carrier class (object type,
source kind, channel-specific kind flags), so a summary can never absorb a
source-backed evidence row.

Within a group the kept representative is the best fused carrier
(highest normalized ``rrf_score``, earliest fusion position on ties), and
it takes the group's earliest position so the group's collective standing
in the fused ordering is preserved. Collapsed carriers stay recoverable:
their ids are recorded on the representative (``deduped_carrier_ids``) and
in the result mapping for custody labeling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any, Final

from atagia.memory.coverage_keys import normalize_coverage_key, resolve_member_keys

# Mechanical tokenization only (case/whitespace/punctuation normalization for
# exact token-identity comparison). This performs no semantic interpretation.
_CONTENT_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class CarrierDedupeResult:
    """Outcome of one collapse pass over a fused candidate pool."""

    candidates: list[dict[str, Any]]
    collapsed_into: dict[str, str]
    collapsed_ids_by_representative: dict[str, list[str]]

    @property
    def collapsed_count(self) -> int:
        return len(self.collapsed_into)


@dataclass(slots=True)
class _CarrierFeatures:
    candidate_id: str
    position: int
    rrf_score: float
    class_key: tuple[str, str, bool, bool, bool, bool]
    span_ids: frozenset[str]
    value_key: str | None
    member_keys: frozenset[str]
    content_tokens: tuple[str, ...]


def collapse_duplicate_carriers(
    candidates: list[dict[str, Any]],
    *,
    member_key_collapse: bool = False,
) -> CarrierDedupeResult:
    """Collapse duplicate carriers of one fact into their best fused carrier.

    ``member_key_collapse`` enables the member-key join and must only be set
    for exhaustive known-set retrieval (``coverage_mode ==
    "exhaustive_known_set"``), where the member key IS the unit of answer and
    the composer's exhaustive reservation already treats same-member carriers
    as one slot. Outside that mode, carriers sharing a member key can carry
    DISTINCT secondary facts (a date, an event) around the same enumerable
    value, so collapsing them would guess — the span-overlap joins remain the
    only provenance strong enough there.
    """
    if len(candidates) < 2:
        return CarrierDedupeResult(
            candidates=list(candidates),
            collapsed_into={},
            collapsed_ids_by_representative={},
        )

    features = [
        _carrier_features(candidate, position)
        for position, candidate in enumerate(candidates)
    ]
    parent = list(range(len(features)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    _join_by_value_key(features, union)
    if member_key_collapse:
        _join_by_member_keys(features, union)
    _join_by_content_tokens(features, union)

    groups: dict[int, list[int]] = {}
    for index in range(len(features)):
        groups.setdefault(find(index), []).append(index)

    representative_by_index: dict[int, int] = {}
    for group_indexes in groups.values():
        if len(group_indexes) < 2:
            continue
        representative = min(
            group_indexes,
            key=lambda index: (-features[index].rrf_score, features[index].position),
        )
        for index in group_indexes:
            if index == representative:
                continue
            # B3 exhaustive-coverage guard: a collapsed carrier may not take
            # any member key out of the pool. Release carriers whose member
            # keys are not fully carried by the representative (possible only
            # via cross-rule bridging; equal-set member joins pass trivially).
            if not features[index].member_keys <= features[representative].member_keys:
                continue
            # Facet-identity guard: never collapse a carrier whose structured
            # value identity differs from the representative's.
            if (
                features[index].value_key is not None
                and features[index].value_key != features[representative].value_key
            ):
                continue
            representative_by_index[index] = representative

    if not representative_by_index:
        return CarrierDedupeResult(
            candidates=list(candidates),
            collapsed_into={},
            collapsed_ids_by_representative={},
        )

    collapsed_indexes_by_representative: dict[int, list[int]] = {}
    for index, representative in representative_by_index.items():
        collapsed_indexes_by_representative.setdefault(representative, []).append(index)

    collapsed_into: dict[str, str] = {}
    collapsed_ids_by_representative: dict[str, list[str]] = {}
    emitted_representatives: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        collapsed_to = representative_by_index.get(index)
        if collapsed_to is None and index not in collapsed_indexes_by_representative:
            deduped.append(candidate)
            continue
        representative = collapsed_to if collapsed_to is not None else index
        if representative in emitted_representatives:
            continue
        emitted_representatives.add(representative)
        collapsed_indexes = sorted(
            collapsed_indexes_by_representative[representative],
            key=lambda item: features[item].position,
        )
        collapsed_ids = [features[item].candidate_id for item in collapsed_indexes]
        representative_id = features[representative].candidate_id
        representative_candidate = dict(candidates[representative])
        representative_candidate["deduped_carrier_ids"] = list(collapsed_ids)
        _merge_collapsed_spans_into_representative(
            representative_candidate,
            [candidates[item] for item in collapsed_indexes],
        )
        deduped.append(representative_candidate)
        collapsed_ids_by_representative[representative_id] = collapsed_ids
        for collapsed_id in collapsed_ids:
            collapsed_into[collapsed_id] = representative_id

    return CarrierDedupeResult(
        candidates=deduped,
        collapsed_into=collapsed_into,
        collapsed_ids_by_representative=collapsed_ids_by_representative,
    )


def _merge_collapsed_spans_into_representative(
    representative_candidate: dict[str, Any],
    collapsed_candidates: list[dict[str, Any]],
) -> None:
    """Union the collapsed carriers' source spans onto the representative.

    A duplicate carrier of the same fact often points at a DIFFERENT
    utterance of that fact; those utterances stay quote-fundable through the
    representative or the collapse silently loses evidence detail (e.g. the
    one restatement that carried a date). This mirrors what ingest-time
    dedupe already does when repeated extractions merge their
    ``source_message_ids`` into one row. The runtime candidate dict is
    updated; stored memory rows are never modified.
    """
    payload_json = representative_candidate.get("payload_json")
    payload = dict(payload_json) if isinstance(payload_json, dict) else {}
    for key in ("source_message_ids", "source_object_ids"):
        merged: list[str] = []
        seen: set[str] = set()

        def append_ids(raw_ids: Any) -> None:
            if not isinstance(raw_ids, list):
                return
            for raw_id in raw_ids:
                value = str(raw_id).strip()
                if value and value not in seen:
                    seen.add(value)
                    merged.append(value)

        append_ids(payload.get(key))
        for collapsed in collapsed_candidates:
            collapsed_payload = collapsed.get("payload_json")
            if isinstance(collapsed_payload, dict):
                append_ids(collapsed_payload.get(key))
        if merged:
            payload[key] = merged
    representative_candidate["payload_json"] = payload


def _carrier_features(candidate: dict[str, Any], position: int) -> _CarrierFeatures:
    payload_json = candidate.get("payload_json")
    payload = payload_json if isinstance(payload_json, dict) else {}
    return _CarrierFeatures(
        candidate_id=str(candidate["id"]),
        position=position,
        rrf_score=_normalized_rrf_score(candidate.get("rrf_score")),
        class_key=(
            str(candidate.get("object_type") or ""),
            str(candidate.get("source_kind") or ""),
            bool(candidate.get("is_verbatim_pin")),
            bool(candidate.get("is_artifact_chunk")),
            bool(candidate.get("is_fact_facet_candidate")),
            bool(candidate.get("is_verbatim_evidence_window")),
        ),
        span_ids=_span_ids(payload),
        value_key=_normalized_key_or_none(payload.get("value_norm_key")),
        member_keys=resolve_member_keys(payload),
        content_tokens=tuple(
            _CONTENT_TOKEN_PATTERN.findall(
                str(candidate.get("canonical_text") or "").casefold()
            )
        ),
    )


def _normalized_rrf_score(value: Any) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _span_ids(payload: dict[str, Any]) -> frozenset[str]:
    span_ids: set[str] = set()
    for key in ("source_message_ids", "source_object_ids"):
        raw_ids = payload.get(key)
        if not isinstance(raw_ids, list):
            continue
        for raw_id in raw_ids:
            value = str(raw_id).strip()
            if value:
                span_ids.add(value)
    return frozenset(span_ids)


def _normalized_key_or_none(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_coverage_key(str(value))
    return normalized or None


def _join_by_value_key(
    features: list[_CarrierFeatures],
    union: Callable[[int, int], None],
) -> None:
    """Join carriers with the same facet value identity on overlapping spans."""
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, feature in enumerate(features):
        if feature.value_key is None or not feature.span_ids:
            continue
        buckets.setdefault((feature.class_key, feature.value_key), []).append(index)
    _join_span_overlaps(features, buckets, union)


def _join_by_member_keys(
    features: list[_CarrierFeatures],
    union: Callable[[int, int], None],
) -> None:
    """Join carriers whose coverage member-key sets are equal.

    Equal sets only: a carrier enumerating a different member set (even a
    subset) stays separate so no exhaustive-list member can lose its carrier.
    Span overlap is NOT required — the member key is the engine's own
    enumerable identity, and the composer's exhaustive reservation already
    treats same-member carriers as one slot.
    """
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, feature in enumerate(features):
        if not feature.member_keys:
            continue
        buckets.setdefault(
            (feature.class_key, feature.member_keys), []
        ).append(index)
    for bucket in buckets.values():
        for index in bucket[1:]:
            union(bucket[0], index)


def _join_by_content_tokens(
    features: list[_CarrierFeatures],
    union: Callable[[int, int], None],
) -> None:
    """Join carriers with identical content tokens on overlapping spans."""
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, feature in enumerate(features):
        if not feature.content_tokens or not feature.span_ids:
            continue
        buckets.setdefault(
            (feature.class_key, feature.content_tokens), []
        ).append(index)
    _join_span_overlaps(features, buckets, union)


def _join_span_overlaps(
    features: list[_CarrierFeatures],
    buckets: dict[tuple[Any, ...], list[int]],
    union: Callable[[int, int], None],
) -> None:
    """Union bucket members that share at least one source span id."""
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        first_holder_by_span: dict[str, int] = {}
        for index in bucket:
            for span_id in features[index].span_ids:
                holder = first_holder_by_span.get(span_id)
                if holder is None:
                    first_holder_by_span[span_id] = index
                else:
                    union(holder, index)
