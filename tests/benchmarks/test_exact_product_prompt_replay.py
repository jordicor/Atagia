"""Tests for exact product prompt replay helpers."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.exact_product_prompt_replay import (
    _messages_for_prompt_style,
    load_frozen_cases,
    load_locomo_summary_cases,
)


def test_load_frozen_cases_preserves_stored_system_prompt(tmp_path: Path) -> None:
    path = tmp_path / "frozen-contexts.json"
    path.write_text(
        json.dumps(
            {
                "fixture-conversation:q3": {
                    "question_text": "Which maintenance tasks were completed?",
                    "ground_truth": "filter replacement",
                    "memory_block": "[Retrieved Memories]\n1. filter replacement",
                    "system_prompt": "exact product prompt",
                }
            }
        )
    )

    cases = load_frozen_cases(path, {"fixture-conversation:q3": "fixture_q3_full"})

    assert len(cases) == 1
    assert cases[0].label == "fixture_q3_full"
    assert cases[0].product_system_prompt == "exact product prompt"
    assert cases[0].product_prompt_exact is True


def test_evidence_messages_use_memory_block(tmp_path: Path) -> None:
    path = tmp_path / "frozen-contexts.json"
    path.write_text(
        json.dumps(
            {
                "fixture-conversation:q3": {
                    "question_text": "Which maintenance tasks were completed?",
                    "ground_truth": "filter replacement, sensor calibration",
                    "memory_block": "[Retrieved Memories]\n1. sensor calibration",
                    "system_prompt": "exact product prompt",
                }
            }
        )
    )
    case = load_frozen_cases(
        path,
        {"fixture-conversation:q3": "fixture_q3_full"},
    )[0]

    messages = _messages_for_prompt_style(case, "evidence")

    assert messages[0].role == "system"
    assert "Use only the selected evidence" in messages[0].content
    assert "sensor calibration" in messages[1].content
    assert case.question_text in messages[1].content


def test_load_locomo_summary_cases_marks_reconstructed_prompt(tmp_path: Path) -> None:
    path = tmp_path / "locomo-report.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "results": [
                            {
                                "question": {
                                    "question_id": "conv-x:q1",
                                    "question_text": "What happened?",
                                    "ground_truth": "a filter replacement",
                                },
                                "trace": {
                                    "trusted_evaluation": True,
                                    "selected_memory_summaries": [
                                        {"canonical_preview": "Jon attended a filter replacement."}
                                    ],
                                },
                            }
                        ]
                    }
                ]
            }
        )
    )

    cases = load_locomo_summary_cases([path], {"conv-x:q1": "summary"})

    assert len(cases) == 1
    assert cases[0].product_prompt_exact is False
    assert cases[0].product_prompt_source == "reconstructed_from_selected_summaries"
    assert "Jon attended a filter replacement." in cases[0].memory_block
    assert "Trusted local evaluation context is active" in cases[0].product_system_prompt
