"""Helpers for optional message occurrence timestamps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_optional_timestamp(value: str | None) -> str | None:
    """Normalize optional timestamp strings, treating blanks as missing."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_message_occurred_at(message: Mapping[str, Any]) -> str | None:
    """Return the best-known message occurrence timestamp from a row-like mapping."""
    occurred_at_value = message.get("occurred_at")
    occurred_at = normalize_optional_timestamp(
        str(occurred_at_value) if occurred_at_value is not None else None
    )
    if occurred_at is not None:
        return occurred_at
    created_at_value = message.get("created_at")
    return normalize_optional_timestamp(
        str(created_at_value) if created_at_value is not None else None
    )
