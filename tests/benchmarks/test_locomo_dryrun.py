"""Dry-run validation against the real LoCoMo dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from benchmarks.locomo.adapter import LoCoMoAdapter

_DATA_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "locomo10.json"


@pytest.mark.skipif(not _DATA_PATH.exists(), reason="LoCoMo dataset not downloaded")
def test_locomo_dataset_parses() -> None:
    dataset = LoCoMoAdapter(_DATA_PATH).load()

    category_counts: Counter[int] = Counter()
    conversation_rows: list[tuple[str, int, int]] = []
    for conversation in dataset.conversations:
        scored_questions = conversation.scored_questions
        conversation_rows.append(
            (
                conversation.conversation_id,
                len(conversation.turns),
                len(scored_questions),
            )
        )
        for question in conversation.questions:
            category_counts[question.category] += 1

    print(f"Cat 1 (single-hop): {category_counts[1]} questions")
    print(f"Cat 2 (multi-hop): {category_counts[2]} questions")
    print(f"Cat 3 (temporal): {category_counts[3]} questions")
    print(f"Cat 4 (open-domain): {category_counts[4]} questions")
    print(f"Cat 5 (adversarial, excluded): {category_counts[5]} questions")
    for conversation_id, num_turns, num_scored_questions in conversation_rows:
        print(
            f"{conversation_id}: turns={num_turns}, "
            f"scored_questions={num_scored_questions}"
        )

    assert len(dataset.conversations) == 10
    assert sum(category_counts[category] for category in (1, 2, 3, 4)) == 1540
    assert category_counts == Counter({1: 282, 2: 321, 3: 96, 4: 841, 5: 446})
    assert all(len(conversation.turns) > 0 for conversation in dataset.conversations)
    assert all(
        question.ground_truth.strip()
        for conversation in dataset.conversations
        for question in conversation.scored_questions
    )
