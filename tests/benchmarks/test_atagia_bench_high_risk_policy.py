"""Policy alignment checks for Atagia-bench high-risk questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "atagia_bench" / "data"


def _questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*/questions.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    return questions


def test_high_risk_questions_expect_secret_withholding() -> None:
    high_risk_questions = [
        question
        for question in _questions()
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
