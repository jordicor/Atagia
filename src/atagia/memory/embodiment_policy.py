"""Embodiment body/device eligibility helpers."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import RetrievalPlan


def embodiment_visibility_sql_clause(
    plan: RetrievalPlan,
    *,
    alias: str,
    embodiment_column: str = "embodiment_id",
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Embodiment visibility before ranking."""

    return embodiment_visibility_sql_clause_for_context(
        active_embodiment_id=plan.active_embodiment_id,
        alias=alias,
        embodiment_column=embodiment_column,
    )


def embodiment_visibility_sql_clause_for_context(
    *,
    active_embodiment_id: str | None,
    alias: str,
    embodiment_column: str = "embodiment_id",
) -> tuple[str, list[Any]]:
    """Return SQL that enforces direct-if-same-body visibility."""

    prefix = f"{alias}." if alias else ""
    embodiment_expr = f"{prefix}{embodiment_column}"
    normalized_active_embodiment = _optional_text(active_embodiment_id)
    if normalized_active_embodiment is None:
        return f"({embodiment_expr} IS NULL)", []
    return (
        f"({embodiment_expr} IS NULL OR {embodiment_expr} = ?)",
        [normalized_active_embodiment],
    )


def candidate_allows_embodiment_boundary(
    candidate: dict[str, Any],
    plan: RetrievalPlan,
) -> bool:
    """Return whether a decoded candidate is visible to the active body."""

    candidate_embodiment_id = _optional_text(
        candidate.get("embodiment_id") or candidate.get("active_embodiment_id")
    )
    active_embodiment_id = _optional_text(plan.active_embodiment_id)
    if active_embodiment_id is None:
        return candidate_embodiment_id is None
    return candidate_embodiment_id is None or candidate_embodiment_id == active_embodiment_id


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "candidate_allows_embodiment_boundary",
    "embodiment_visibility_sql_clause",
    "embodiment_visibility_sql_clause_for_context",
]
