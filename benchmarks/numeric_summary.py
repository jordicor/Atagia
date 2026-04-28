"""Small numeric summary helpers for benchmark artifacts."""

from __future__ import annotations

from collections.abc import Iterable


def summarize_numeric_values(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """Return count/mean/min/max for a finite numeric iterable."""
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": sum(items) / len(items),
        "min": min(items),
        "max": max(items),
    }
