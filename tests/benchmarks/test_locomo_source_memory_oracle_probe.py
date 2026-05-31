"""Tests for offline LoCoMo source-memory oracle probes."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from benchmarks.locomo.source_memory_oracle_probe import (
    build_source_memory_oracle_probe,
    format_source_memory_oracle_summary,
)


def test_source_memory_oracle_marks_available_memory_missing_from_raw(
    tmp_path: Path,
) -> None:
    data_path = _write_dataset(tmp_path)
    db_path = tmp_path / "benchmark.db"
    _write_db(db_path)
    report_path = _write_report(tmp_path, db_path)

    probe = build_source_memory_oracle_probe([report_path], data_path=data_path)

    assert probe["summary"]["item_count"] == 2
    assert probe["summary"]["oracle_stage_counts"] == {
        "available_not_raw": 1,
        "scored_not_selected": 1,
    }
    by_question = {item["question_id"]: item for item in probe["items"]}
    q1 = by_question["conv-a:q1"]
    assert q1["oracle_stage"] == "available_not_raw"
    assert q1["available_source_memory_count"] == 1
    assert q1["raw_source_memory_count"] == 0
    assert "source_memory_raw_none" in q1["labels"]
    assert "source_memory_available_includes_non_summary" in q1["labels"]
    q2 = by_question["conv-a:q2"]
    assert q2["oracle_stage"] == "scored_not_selected"
    assert q2["raw_source_memory_count"] == 1
    assert q2["scored_source_memory_count"] == 1
    assert q2["selected_source_memory_count"] == 0
    assert "source_memory_available_summary_only" in q2["labels"]
    assert "source_memory_selected_none" in q2["labels"]
    assert format_source_memory_oracle_summary(probe).startswith(
        "LoCoMo source-memory oracle:"
    )
    serialized = json.dumps(probe)
    assert "RAW SOURCE TEXT" not in serialized
    assert "RAW QUESTION TEXT" not in serialized
    assert "RAW GOLD" not in serialized


def _write_dataset(tmp_path: Path) -> Path:
    data_path = tmp_path / "locomo.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-a",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "1:00 PM on May 1, 2026",
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "text": "RAW SOURCE TEXT one",
                                "dia_id": "D1:1",
                            },
                            {
                                "speaker": "Bob",
                                "text": "RAW SOURCE TEXT two",
                                "dia_id": "D1:2",
                            },
                        ],
                    },
                    "qa": [
                        {
                            "question": "RAW QUESTION TEXT 1",
                            "answer": "RAW GOLD 1",
                            "category": 1,
                            "evidence": ["D1:1"],
                        },
                        {
                            "question": "RAW QUESTION TEXT 2",
                            "answer": "RAW GOLD 2",
                            "category": 1,
                            "evidence": ["D1:2"],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_path


def _write_report(tmp_path: Path, db_path: Path) -> Path:
    report_path = tmp_path / "locomo-report.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark_name": "LoCoMo",
                "timestamp": "2026-05-23T00:00:00+00:00",
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "metadata": {"benchmark_db_path": str(db_path)},
                        "results": [
                            _result(
                                "conv-a:q1",
                                ["D1:1"],
                                ["msg_1"],
                                retrieval_custody=[],
                            ),
                            _result(
                                "conv-a:q2",
                                ["D1:2"],
                                ["msg_2"],
                                retrieval_custody=[
                                    {
                                        "candidate_id": "mem_2",
                                        "candidate_kind": "summary_view",
                                        "source_kind": "summarized",
                                        "channels": ["fts"],
                                        "scored": True,
                                        "selected": False,
                                        "drop_stage": "composer",
                                        "drop_reason": "not_selected_after_scoring",
                                        "composer_decision": "not_selected_after_scoring",
                                        "scorer": {"final_score": 0.25},
                                    }
                                ],
                            ),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _result(
    question_id: str,
    evidence_turn_ids: list[str],
    evidence_message_ids: list[str],
    *,
    retrieval_custody: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "question": {
            "question_id": question_id,
            "question_text": "RAW QUESTION TEXT",
            "ground_truth": "RAW GOLD",
            "category": 1,
            "evidence_turn_ids": evidence_turn_ids,
        },
        "prediction": "RAW PREDICTION",
        "score_result": {
            "score": 0,
            "reasoning": "RAW REASON",
            "judge_model": "judge",
        },
        "memories_used": 0,
        "retrieval_time_ms": 1.0,
        "trace": {
            "diagnosis_bucket": "retrieval_or_ranking_miss",
            "sufficiency_diagnostic": "retrieval_insufficient",
            "benchmark_privacy_enforcement": "off",
            "evidence_message_ids": evidence_message_ids,
            "missing_evidence_turn_ids": [],
            "critical_evidence_custody": {
                "counts": {
                    "critical_evidence_count": 1,
                    "raw_candidate_count": 0,
                    "scored_count": 0,
                    "selected_count": 0,
                    "absent_count": 1,
                },
                "items": [],
                "survival_stage_counts": {"absent_from_raw_candidates": 1},
            },
            "retrieval_trace": {"user_id": "benchmark-user"},
            "retrieval_custody": retrieval_custody,
        },
    }


def _write_db(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE messages (
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                text TEXT NOT NULL,
                occurred_at TEXT,
                content_kind TEXT NOT NULL DEFAULT 'text',
                include_raw INTEGER NOT NULL DEFAULT 1,
                skip_by_default INTEGER NOT NULL DEFAULT 0,
                artifact_backed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE memory_objects (
                id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                scope TEXT NOT NULL,
                privacy_level INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            INSERT INTO messages(id, conversation_id, seq, text, occurred_at)
            VALUES
                ('msg_1', 'conv-a', 1, 'RAW SOURCE TEXT one', '2026-05-01T13:00:00'),
                ('msg_2', 'conv-a', 2, 'RAW SOURCE TEXT two', '2026-05-01T13:00:00');
            INSERT INTO memory_objects(
                id, user_id, object_type, source_kind, status, scope,
                privacy_level, payload_json
            )
            VALUES
                (
                    'mem_1', 'benchmark-user', 'evidence', 'extracted',
                    'active', 'chat', 0, '{"source_message_ids":["msg_1"]}'
                ),
                (
                    'mem_2', 'benchmark-user', 'summary_view', 'summarized',
                    'active', 'chat', 0, '{"source_message_ids":["msg_2"]}'
                );
            """
        )
        connection.commit()
    finally:
        connection.close()
