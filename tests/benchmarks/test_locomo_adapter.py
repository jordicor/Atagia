"""Tests for the LoCoMo dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.adapter import LoCoMoAdapter


def _write_dataset(tmp_path: Path, payload: list[dict[str, object]]) -> Path:
    data_path = tmp_path / "locomo.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    return data_path


def test_parse_minimal_dataset(tmp_path: Path) -> None:
    data_path = _write_dataset(
        tmp_path,
        [
            {
                "sample_id": "conv-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice likes red notebooks."},
                        {"speaker": "Bob", "dia_id": "D1:2", "text": "Bob remembers that."},
                    ],
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                },
                "qa": [
                    {
                        "question": "What does Alice like?",
                        "answer": "red notebooks",
                        "evidence": ["D1:1"],
                        "category": 1,
                    }
                ],
            }
        ],
    )

    dataset = LoCoMoAdapter(data_path).load()

    assert dataset.name == "LoCoMo"
    assert len(dataset.conversations) == 1
    conversation = dataset.conversations[0]
    assert conversation.conversation_id == "conv-1"
    assert [turn.role for turn in conversation.turns] == ["user", "assistant"]
    assert conversation.turns[0].timestamp == "2023-05-08T13:56:00"
    assert conversation.turns[0].turn_id == "D1:1"
    assert conversation.questions[0].question_text == "What does Alice like?"
    assert conversation.questions[0].ground_truth == "red notebooks"
    assert conversation.questions[0].evidence_turn_ids == ["D1:1"]


def test_category_filtering(tmp_path: Path) -> None:
    data_path = _write_dataset(
        tmp_path,
        [
            {
                "sample_id": "conv-2",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice likes red notebooks."},
                        {"speaker": "Bob", "dia_id": "D1:2", "text": "Bob remembers that."},
                    ],
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                },
                "qa": [
                    {
                        "question": "Scored question?",
                        "answer": "yes",
                        "evidence": ["D1:1"],
                        "category": 1,
                    },
                    {
                        "question": "Adversarial question?",
                        "answer": "no",
                        "evidence": ["D1:2"],
                        "category": 5,
                    },
                ],
            }
        ],
    )

    conversation = LoCoMoAdapter(data_path).load().conversations[0]

    assert len(conversation.questions) == 2
    assert [question.category for question in conversation.scored_questions] == [1]
    assert [question.question_text for question in conversation.filtered_questions([1, 5])] == [
        "Scored question?"
    ]


def test_speaker_mapping(tmp_path: Path) -> None:
    data_path = _write_dataset(
        tmp_path,
        [
            {
                "sample_id": "conv-3",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Bob", "dia_id": "D1:1", "text": "Bob speaks first."},
                        {"speaker": "Alice", "dia_id": "D1:2", "text": "Alice replies."},
                        {"speaker": "Bob", "dia_id": "D1:3", "text": "Bob closes the loop."},
                    ],
                    "session_1_date_time": "3:14 pm on August 13, 2023",
                },
                "qa": [],
            }
        ],
    )

    conversation = LoCoMoAdapter(data_path).load().conversations[0]

    assert [turn.role for turn in conversation.turns] == ["user", "assistant", "user"]
    assert conversation.turns[0].timestamp == "2023-08-13T15:14:00"


def test_missing_category_fails_fast(tmp_path: Path) -> None:
    data_path = _write_dataset(
        tmp_path,
        [
            {
                "sample_id": "conv-4",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice likes red notebooks."},
                    ],
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                },
                "qa": [
                    {
                        "question": "What does Alice like?",
                        "answer": "red notebooks",
                        "evidence": ["D1:1"],
                    }
                ],
            }
        ],
    )

    with pytest.raises(KeyError, match="category"):
        LoCoMoAdapter(data_path).load()
