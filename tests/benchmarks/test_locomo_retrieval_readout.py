"""Tests for LoCoMo retrieval readout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.base import (
    BenchmarkQuestion,
    BenchmarkReport,
    ConversationReport,
    QuestionResult,
    ScoreResult,
)
from benchmarks.locomo.retrieval_readout import (
    build_retrieval_readout,
    save_retrieval_readout,
)


def test_build_retrieval_readout_summarizes_selected_evidence(
    tmp_path: Path,
) -> None:
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=1.0,
        category_breakdown={1: 1.0},
        conversations=[
            ConversationReport(
                conversation_id="conv-test",
                results=[
                    QuestionResult(
                        question=BenchmarkQuestion(
                            question_text="What did Gina design?",
                            ground_truth="space, furniture, and decor",
                            category=1,
                            evidence_turn_ids=["D1:3"],
                            question_id="conv-test:q1",
                        ),
                        prediction="space, furniture, and decor",
                        score_result=ScoreResult(
                            score=1,
                            reasoning="ok",
                            judge_model="judge",
                        ),
                        memories_used=1,
                        retrieval_time_ms=5.0,
                        trace={
                            "selected_memory_ids": ["mem_literal"],
                            "selected_evidence_memory_ids": ["mem_literal"],
                            "evidence_message_ids": ["msg_1"],
                            "context": {
                                "answer_evidence_memory_ids": ["mem_literal"],
                                "answer_evidence_items": [
                                    {
                                        "memory_id": "mem_literal",
                                        "supporting_quote": (
                                            "I designed the space, furniture, and decor."
                                        ),
                                        "selected_for_answer_pack": True,
                                        "normalization": {
                                            "source_message_ids": ["msg_1"],
                                        },
                                    }
                                ],
                                "answer_evidence_sufficiency": {
                                    "state": "sufficient_direct_quote",
                                    "rendered": True,
                                },
                            },
                            "retrieval_custody": [
                                {
                                    "candidate_id": "mem_literal",
                                    "candidate_kind": "evidence",
                                    "source_kind": "extracted",
                                    "channels": ["fts", "embedding"],
                                    "selected": True,
                                    "selection_rank": 1,
                                    "score_rank": 2,
                                    "scorer": {"final_score": 0.91},
                                }
                            ],
                        },
                    )
                ],
                accuracy=1.0,
                category_breakdown={1: 1.0},
            )
        ],
        total_questions=1,
        total_correct=1,
        timestamp="2026-05-28T00:00:00+00:00",
        duration_seconds=1.0,
    )

    readout = build_retrieval_readout(report, source_report="report.json")
    question = readout["questions"][0]

    assert question["exact_answer_quote_present"] is True
    assert question["exact_answer_quote_rendered"] is True
    assert (
        question["answer_evidence_sufficiency"]["state"]
        == "sufficient_direct_quote"
    )
    assert (
        question["answer_evidence_items"][0]["normalization"]["source_message_ids"]
        == ["msg_1"]
    )
    assert question["selected_counts"]["candidate_kinds"] == {"evidence": 1}
    assert question["selected_counts"]["channels"] == {"embedding": 1, "fts": 1}
    assert question["top_selected_item"]["memory_id"] == "mem_literal"

    output_path = save_retrieval_readout(
        readout,
        tmp_path / "locomo-retrieval-readout.json",
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["total_questions"] == 1
