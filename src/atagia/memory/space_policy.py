"""Space boundary eligibility helpers."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import RetrievalPlan, SpaceBoundaryMode


def space_visibility_sql_clause(
    plan: RetrievalPlan,
    *,
    alias: str,
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Space visibility before candidate ranking."""

    return space_visibility_sql_clause_for_context(
        active_space_id=plan.active_space_id,
        active_space_boundary_mode=plan.active_space_boundary_mode,
        alias=alias,
    )


def space_visibility_sql_clause_for_context(
    *,
    active_space_id: str | None,
    active_space_boundary_mode: SpaceBoundaryMode | str | None,
    alias: str,
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Space visibility for a retrieval context."""

    prefix = f"{alias}." if alias else ""
    if active_space_id is None:
        return (
            f"({prefix}space_id IS NULL "
            f"OR COALESCE({prefix}space_boundary_mode, 'focus') IN ('focus', 'tagged'))",
            [],
        )

    mode = _boundary_mode(active_space_boundary_mode)
    if mode is SpaceBoundaryMode.SEVERANCE:
        return f"({prefix}space_id = ?)", [active_space_id]

    if mode is SpaceBoundaryMode.TAGGED:
        return (
            f"({prefix}space_id IS NULL "
            f"OR {prefix}space_id = ? "
            f"OR COALESCE({prefix}space_boundary_mode, 'focus') IN ('focus', 'tagged'))",
            [active_space_id],
        )

    return (
        f"({prefix}space_id IS NULL "
        f"OR {prefix}space_id = ? "
        f"OR COALESCE({prefix}space_boundary_mode, 'focus') = 'tagged')",
        [active_space_id],
    )


def candidate_allows_space_boundary(
    candidate: dict[str, Any],
    plan: RetrievalPlan,
) -> bool:
    """Return whether a decoded candidate is visible in the active Space."""

    candidate_space_id = _optional_text(candidate.get("space_id"))
    candidate_mode = _boundary_mode(candidate.get("space_boundary_mode"))
    if plan.active_space_id is None:
        return candidate_space_id is None or candidate_mode in {
            SpaceBoundaryMode.FOCUS,
            SpaceBoundaryMode.TAGGED,
        }

    if candidate_space_id == plan.active_space_id:
        return True

    active_mode = _boundary_mode(plan.active_space_boundary_mode)
    if active_mode is SpaceBoundaryMode.SEVERANCE:
        return False

    if candidate_space_id is None:
        return True

    if active_mode is SpaceBoundaryMode.TAGGED:
        return candidate_mode in {SpaceBoundaryMode.FOCUS, SpaceBoundaryMode.TAGGED}

    return candidate_mode is SpaceBoundaryMode.TAGGED


def _boundary_mode(value: Any) -> SpaceBoundaryMode:
    if isinstance(value, SpaceBoundaryMode):
        return value
    try:
        return SpaceBoundaryMode(str(value or SpaceBoundaryMode.FOCUS.value))
    except ValueError:
        return SpaceBoundaryMode.FOCUS


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "candidate_allows_space_boundary",
    "space_visibility_sql_clause",
    "space_visibility_sql_clause_for_context",
]
