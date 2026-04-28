"""Tests for the centralized LLM output token threshold helper."""

from __future__ import annotations

from atagia.core.llm_output_limits import apply_min_output_threshold


def test_apply_min_output_threshold_none_passes_through() -> None:
    assert apply_min_output_threshold(None) is None


def test_apply_min_output_threshold_below_floor_returns_none() -> None:
    assert apply_min_output_threshold(256) is None
    assert apply_min_output_threshold(100) is None
    assert apply_min_output_threshold(512) is None


def test_apply_min_output_threshold_at_floor_returns_none() -> None:
    # The rule is `<= 512`, so exactly 512 is treated as unset.
    assert apply_min_output_threshold(512) is None


def test_apply_min_output_threshold_above_floor_passes_through() -> None:
    assert apply_min_output_threshold(513) == 513
    assert apply_min_output_threshold(1024) == 1024
    assert apply_min_output_threshold(8192) == 8192
