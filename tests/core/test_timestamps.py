"""Tests for timestamp helper utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from atagia.core.timestamps import parse_optional_datetime


def test_parse_optional_datetime_accepts_z_suffix_and_datetime_values() -> None:
    parsed = parse_optional_datetime("2026-05-02T12:00:00Z")

    assert parsed == datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    assert parse_optional_datetime(parsed) is parsed


def test_parse_optional_datetime_returns_none_for_missing_or_invalid_values() -> None:
    assert parse_optional_datetime(None) is None
    assert parse_optional_datetime("not-a-date") is None
