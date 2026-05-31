"""Tests for fixed-context answer quality ceiling helpers."""

from __future__ import annotations

from benchmarks.fixed_context_answer_quality import (
    _answer_messages,
    _uses_streaming_answer_generation,
    parse_answer_profile,
)
from benchmarks.broad_list_coverage import AnswerRecord


def test_parse_answer_profile_uses_explicit_token_budget() -> None:
    profile = parse_answer_profile(
        "opus=anthropic/claude-opus-4-7,high:32768"
    )

    assert profile.label == "opus"
    assert profile.model == "anthropic/claude-opus-4-7,high"
    assert profile.max_output_tokens == 32768


def test_parse_answer_profile_defaults_to_baseline_8k_budget() -> None:
    profile = parse_answer_profile("baseline=openrouter/openai/gpt-5.5,high")

    assert profile.max_output_tokens == 8192


def test_parse_answer_profile_raises_small_explicit_budget_to_8k() -> None:
    profile = parse_answer_profile("small=openrouter/openai/gpt-5.5,high:1024")

    assert profile.max_output_tokens == 8192


def test_answer_messages_preserve_selected_context_and_question() -> None:
    record = AnswerRecord(
        question_id="q1",
        question_text="Which places were mentioned?",
        ground_truth="beach, forest",
        prediction="",
        report_source="report.json",
        source_kind="locomo_report",
        selected_context_text="Selected memory: beach and forest.",
    )

    messages = _answer_messages(record)

    assert messages[0].role == "system"
    assert "Use only the selected evidence" in messages[0].content
    assert "Selected memory: beach and forest." in messages[1].content
    assert "Which places were mentioned?" in messages[1].content


def test_large_anthropic_budget_uses_streaming_answer_generation() -> None:
    profile = parse_answer_profile("opus=anthropic/claude-opus-4-7,high:32000")

    assert _uses_streaming_answer_generation(profile) is True


def test_openrouter_large_budget_keeps_non_streaming_path() -> None:
    profile = parse_answer_profile("gpt=openrouter/openai/gpt-5.5,high:32000")

    assert _uses_streaming_answer_generation(profile) is False
