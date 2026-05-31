"""Tests for the centralized LLM output token threshold helper."""

from __future__ import annotations

from atagia.core.llm_output_limits import apply_min_output_threshold


def test_apply_min_output_threshold_none_passes_through() -> None:
    assert apply_min_output_threshold(None) is None


def test_apply_min_output_threshold_below_floor_raises_to_standard_floor() -> None:
    assert apply_min_output_threshold(100) == 8192
    assert apply_min_output_threshold(256) == 8192
    assert apply_min_output_threshold(512) == 8192
    assert apply_min_output_threshold(1024) == 8192


def test_apply_min_output_threshold_at_floor_passes_through() -> None:
    assert apply_min_output_threshold(8192) == 8192


def test_apply_min_output_threshold_above_floor_passes_through() -> None:
    assert apply_min_output_threshold(16384) == 16384
