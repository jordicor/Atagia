"""Policy alignment checks for Atagia-bench high-risk questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "atagia_bench" / "data"
requires_private_atagia_bench_data = pytest.mark.skipif(
    not DATA_DIR.is_dir(),
    reason="Private Atagia-bench data is not included in the public checkout.",
)


def _questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*/questions.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    return questions


def _assert_high_risk_questions_expect_secret_withholding(
    questions: list[dict[str, Any]],
) -> None:
    high_risk_questions = [
        question
        for question in questions
        if "high_risk" in question.get("category_tags", [])
    ]

    assert high_risk_questions
    for question in high_risk_questions:
        assert question["answer_type"] == "privacy_check"
        assert question["grader"] == "abstention"
        assert "privacy_check" in question["category_tags"]
        assert "exact_match" not in question["category_tags"]
        assert question["ground_truth"].startswith("ABSTAIN")
        assert question["grader_config"]["private_fact"]
        assert question["grader_config"]["policy"] == "high_risk_secret_literal"
        assert "expected_values" not in question["grader_config"]


def test_synthetic_high_risk_question_expects_secret_withholding() -> None:
    _assert_high_risk_questions_expect_secret_withholding(
        [
            {
                "question_id": "fixture-high-risk",
                "answer_type": "privacy_check",
                "grader": "abstention",
                "category_tags": ["high_risk", "privacy_check"],
                "ground_truth": "ABSTAIN: synthetic fixture",
                "grader_config": {
                    "private_fact": "fixture-only-value",
                    "policy": "high_risk_secret_literal",
                },
            }
        ]
    )


@requires_private_atagia_bench_data
def test_private_high_risk_questions_expect_secret_withholding() -> None:
    _assert_high_risk_questions_expect_secret_withholding(_questions())
