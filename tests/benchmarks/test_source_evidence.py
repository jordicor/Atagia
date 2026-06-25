"""Tests for official benchmark source-evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.atagia_bench.adapter import AtagiaBenchQuestion
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.source_evidence import source_evidence_from_turns
from benchmarks.source_evidence import normalize_evidence_turn_ids
from benchmarks.source_evidence import validate_evidence_turn_ids


@dataclass
class _Turn:
    turn_id: str
    role: str
    speaker: str
    timestamp: str
    text: str
    session_id: str = ""
    metadata: dict[str, object] | None = None
    attachments: list[dict[str, object]] | None = None


def test_normalize_evidence_turn_ids_splits_structured_citation_strings() -> None:
    evidence_turn_ids = normalize_evidence_turn_ids(
        ["D8:6; D9:17", "D9:1 D4:4 D4:6", "D:11:26", "D30:05"]
    )

    assert evidence_turn_ids == [
        "D8:6",
        "D9:17",
        "D9:1",
        "D4:4",
        "D4:6",
        "D11:26",
        "D30:5",
    ]


def test_source_evidence_from_turns_uses_official_evidence_order() -> None:
    turns = [
        _Turn(
            turn_id="t2",
            role="user",
            speaker="Rosa",
            timestamp="2025-12-02T11:07:00",
            text="She is due in May.",
        ),
        _Turn(
            turn_id="t1",
            role="assistant",
            speaker="Assistant",
            timestamp="2025-12-02T11:06:00",
            text="Previous context.",
        ),
    ]

    evidence = source_evidence_from_turns(
        evidence_turn_ids=["t1", "t2"],
        turns=turns,
        conversation_id="conv",
    )

    assert [item["turn_id"] for item in evidence] == ["t1", "t2"]
    assert evidence[1]["timestamp"] == "2025-12-02T11:07:00"
    assert evidence[1]["text"] == "She is due in May."


def test_source_evidence_from_turns_includes_caption_attachment_text() -> None:
    turns = [
        _Turn(
            turn_id="D1:1",
            role="user",
            speaker="Rosa",
            timestamp="2025-12-02T11:07:00",
            text="Look at this.",
            session_id="session_1",
            metadata={"blip_caption": "a dog-shaped cup"},
            attachments=[
                {
                    "content_text": (
                        "Visual description of attached image: a dog-shaped cup"
                    )
                }
            ],
        )
    ]

    evidence = source_evidence_from_turns(
        evidence_turn_ids=["D1:1"],
        turns=turns,
        conversation_id="conv",
    )

    assert evidence == [
        {
            "turn_id": "D1:1",
            "conversation_id": "conv",
            "timestamp": "2025-12-02T11:07:00",
            "speaker": "Rosa",
            "role": "user",
            "text": "Look at this.",
            "session_id": "session_1",
            "blip_caption": "a dog-shaped cup",
            "attachment_text": (
                "Visual description of attached image: a dog-shaped cup"
            ),
        }
    ]


def test_validate_evidence_turn_ids_reports_bad_question() -> None:
    turns = [
        _Turn(
            turn_id="D1:1",
            role="user",
            speaker="Rosa",
            timestamp="2025-12-02T11:07:00",
            text="Known evidence.",
        )
    ]

    try:
        validate_evidence_turn_ids(
            evidence_turn_ids=["D1:1", "D1:2"],
            turns=turns,
            dataset_name="TestBench",
            question_id="test-q1",
            conversation_id="conv",
        )
    except ValueError as exc:
        assert "TestBench question test-q1" in str(exc)
        assert "D1:2" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected unresolved evidence validation error")


def test_atagia_bench_grade_context_records_source_evidence_without_memory() -> None:
    question = AtagiaBenchQuestion(
        question_id="rosa-q17",
        question_text="Is Elena expecting a baby, and if so, when?",
        ground_truth="Yes, Elena is expecting a baby in May",
        answer_type="llm_judge",
        evidence_turn_ids=["rosa-04-t07"],
        grader="llm_judge",
    )

    context = AtagiaBenchRunner._grade_context_for_question(
        question,
        {
            "source_evidence": [
                {
                    "turn_id": "rosa-04-t07",
                    "timestamp": "2025-12-02T11:07:00",
                    "text": "She is due in May.",
                }
            ],
            "abstention_kind": None,
        },
    )

    assert context["judge_mode"] == "source_aware_llm_judge"
    assert context["source_evidence_source"] == "official_benchmark_dataset"
    assert context["source_turn_ids"] == ["rosa-04-t07"]
    assert context["source_timestamps"] == ["2025-12-02T11:07:00"]
