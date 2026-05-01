"""Deterministic policy helpers for intimacy-bound memory."""

from __future__ import annotations

import re
from typing import Any

from atagia.models.schemas_memory import IntimacyBoundary, MemoryScope

_SQL_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RESTRICTED_INTIMACY_BOUNDARIES: frozenset[IntimacyBoundary] = frozenset(
    {
        IntimacyBoundary.ROMANTIC_PRIVATE,
        IntimacyBoundary.INTIMACY_PRIVATE,
        IntimacyBoundary.INTIMACY_PREFERENCE_PRIVATE,
        IntimacyBoundary.INTIMACY_BOUNDARY,
        IntimacyBoundary.AMBIGUOUS_INTIMATE,
    }
)
BLOCKED_INTIMACY_BOUNDARIES: frozenset[IntimacyBoundary] = frozenset(
    {IntimacyBoundary.SAFETY_BLOCKED}
)
NON_ORDINARY_INTIMACY_BOUNDARIES = (
    RESTRICTED_INTIMACY_BOUNDARIES | BLOCKED_INTIMACY_BOUNDARIES
)
INTIMACY_FILTER_REASON = "policy_filtered_intimacy_boundary"


def normalize_intimacy_boundary(value: Any) -> IntimacyBoundary:
    """Normalize untrusted row or model output values to a conservative boundary."""
    if isinstance(value, IntimacyBoundary):
        return value
    if value is None:
        return IntimacyBoundary.ORDINARY
    try:
        return IntimacyBoundary(str(value).strip())
    except ValueError:
        return IntimacyBoundary.AMBIGUOUS_INTIMATE


def is_restricted_intimacy_boundary(value: Any) -> bool:
    return normalize_intimacy_boundary(value) in RESTRICTED_INTIMACY_BOUNDARIES


def is_blocked_intimacy_boundary(value: Any) -> bool:
    return normalize_intimacy_boundary(value) in BLOCKED_INTIMACY_BOUNDARIES


def allows_intimacy_boundary(value: Any, *, allow_intimacy_context: bool) -> bool:
    boundary = normalize_intimacy_boundary(value)
    if boundary is IntimacyBoundary.ORDINARY:
        return True
    if boundary in BLOCKED_INTIMACY_BOUNDARIES:
        return False
    return allow_intimacy_context and boundary in RESTRICTED_INTIMACY_BOUNDARIES


def candidate_intimacy_boundary(candidate: dict[str, Any]) -> IntimacyBoundary:
    value = candidate.get("intimacy_boundary")
    if value is not None:
        return normalize_intimacy_boundary(value)
    payload = candidate.get("payload_json") or {}
    if isinstance(payload, dict):
        return normalize_intimacy_boundary(payload.get("intimacy_boundary"))
    return IntimacyBoundary.ORDINARY


def candidate_allows_intimacy_boundary(
    candidate: dict[str, Any],
    *,
    allow_intimacy_context: bool,
) -> bool:
    return allows_intimacy_boundary(
        candidate_intimacy_boundary(candidate),
        allow_intimacy_context=allow_intimacy_context,
    )


def _validated_sql_alias(alias: str) -> str:
    normalized = alias.strip()
    if not _SQL_ALIAS_RE.fullmatch(normalized):
        raise ValueError(f"Invalid SQL alias for intimacy boundary clause: {alias!r}")
    return normalized


def memory_object_intimacy_sql_clause(alias: str, *, allow_intimacy_context: bool) -> str:
    alias = _validated_sql_alias(alias)
    expression = f"COALESCE({alias}.intimacy_boundary, '{IntimacyBoundary.ORDINARY.value}')"
    if allow_intimacy_context:
        return f"{expression} != '{IntimacyBoundary.SAFETY_BLOCKED.value}'"
    return f"{expression} = '{IntimacyBoundary.ORDINARY.value}'"


def coalesced_intimacy_sql_clause(
    primary_alias: str,
    fallback_alias: str,
    *,
    allow_intimacy_context: bool,
) -> str:
    primary_alias = _validated_sql_alias(primary_alias)
    fallback_alias = _validated_sql_alias(fallback_alias)
    expression = (
        f"COALESCE({primary_alias}.intimacy_boundary, "
        f"{fallback_alias}.intimacy_boundary, "
        f"'{IntimacyBoundary.ORDINARY.value}')"
    )
    if allow_intimacy_context:
        return f"{expression} != '{IntimacyBoundary.SAFETY_BLOCKED.value}'"
    return f"{expression} = '{IntimacyBoundary.ORDINARY.value}'"


def minimum_privacy_for_intimacy_boundary(
    value: Any,
    *,
    privacy_level: int,
) -> int:
    if normalize_intimacy_boundary(value) in NON_ORDINARY_INTIMACY_BOUNDARIES:
        return max(int(privacy_level), 2)
    return int(privacy_level)


def constrained_scope_for_intimacy_boundary(
    value: Any,
    *,
    scope: MemoryScope,
) -> MemoryScope:
    if normalize_intimacy_boundary(value) in NON_ORDINARY_INTIMACY_BOUNDARIES and scope in {
        MemoryScope.GLOBAL_USER,
        MemoryScope.ASSISTANT_MODE,
    }:
        return MemoryScope.CONVERSATION
    return scope


def strongest_intimacy_boundary(rows: list[dict[str, Any]]) -> IntimacyBoundary:
    """Return the strongest boundary represented in source rows."""
    if not rows:
        return IntimacyBoundary.ORDINARY
    priority = {
        IntimacyBoundary.SAFETY_BLOCKED: 6,
        IntimacyBoundary.AMBIGUOUS_INTIMATE: 5,
        IntimacyBoundary.INTIMACY_BOUNDARY: 4,
        IntimacyBoundary.INTIMACY_PREFERENCE_PRIVATE: 3,
        IntimacyBoundary.INTIMACY_PRIVATE: 2,
        IntimacyBoundary.ROMANTIC_PRIVATE: 1,
        IntimacyBoundary.ORDINARY: 0,
    }
    boundaries = [candidate_intimacy_boundary(row) for row in rows]
    return max(boundaries, key=lambda boundary: priority[boundary])
