"""Shared coverage member-key resolution (single source of truth).

The composer's evidence-coverage machinery and the fusion-stage carrier
dedupe MUST resolve a carrier's member identities identically:
if the dedupe sees fewer member keys than the composer (e.g. a legacy row
whose members come from the ``value_*`` payload ladder instead of
``coverage_members``), a collapse can silently shrink the member universe
the composer later enumerates — a latent exhaustive-coverage (B3) bypass.
Both components import THIS resolver so they can never disagree.
"""

from __future__ import annotations

from typing import Any, Final

# Legacy payload keys that identify a carrier's structured value when the
# modern ``coverage_members`` list is absent. Order is the resolution order.
COVERAGE_VALUE_PAYLOAD_KEYS: Final[tuple[str, ...]] = (
    "value_norm_key",
    "value_key",
    "normalized_key",
    "value_text",
    "value",
    "display_text",
    "surface",
    "subject_surface",
)


def normalize_coverage_key(value: str) -> str:
    """Mechanical key normalization: casefold + whitespace collapse."""
    return " ".join(str(value).casefold().split())


def resolve_member_keys(payload_json: Any) -> frozenset[str]:
    """Mechanical resolution ladder for a carrier's member identities.

    1. ``coverage_members`` key present -> the set of normalized
       ``member_key`` values (possibly empty; a non-list value means
       "processed, no enumerable members").
    2. else a legacy ``COVERAGE_VALUE_PAYLOAD_KEYS`` value present ->
       single-element set with the normalized value.
    3. else -> empty set (treated as UNKEYED by callers).
    """
    payload = payload_json if isinstance(payload_json, dict) else {}
    if "coverage_members" in payload:
        members = payload["coverage_members"]
        keys: set[str] = set()
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_key = _optional_text(member.get("member_key"))
                if member_key is not None:
                    keys.add(normalize_coverage_key(member_key))
        return frozenset(keys)
    for key in COVERAGE_VALUE_PAYLOAD_KEYS:
        value = _optional_text(payload.get(key))
        if value is not None:
            return frozenset({normalize_coverage_key(value)})
    return frozenset()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
