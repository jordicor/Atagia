from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from benchmarks.locomo.source_context_replay_probe import build_source_context_cases


def test_build_source_context_cases_from_retained_db(tmp_path: Path) -> None:
    data_path = tmp_path / "locomo.json"
    db_path = tmp_path / "benchmark.db"
    report_path = tmp_path / "report.json"
    _write_dataset(data_path)
    _write_db(db_path)
    _write_report(report_path, db_path)

    report = build_source_context_cases(
        [report_path],
        data_path=data_path,
        max_cases=4,
        max_cases_per_bucket=4,
    )

    assert report["summary"]["case_count"] == 1
    case = report["cases"][0]
    assert case["question_id"] == "conv-test:q1"
    assert case["probe_bucket"] == "single-hop:non_summary_available_not_raw"
    assert "Selected summary about volunteering." in case["contexts"]["selected_current"]
    assert "Direct source memory: David helped at the shelter." in case["contexts"]["source_memory_all"]
    assert "Direct source memory: David helped at the shelter." in case["contexts"]["source_memory_non_summary"]
    assert "I met David while volunteering at the shelter." in case["contexts"]["gold_source_window"]


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-test",
                    "conversation": {
                        "session_1_date_time": "1:00 PM on January 1, 2026",
                        "session_1": [
                            {
                                "speaker": "Alex",
                                "text": "I met David while volunteering at the shelter.",
                                "dia_id": "D1:1",
                            },
                            {
                                "speaker": "Blair",
                                "text": "That sounds meaningful.",
                                "dia_id": "D1:2",
                            },
                        ],
                    },
                    "qa": [
                        {
                            "question": "Who did Alex meet while volunteering?",
                            "answer": "David",
                            "category": 1,
                            "evidence": ["D1:1"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                seq INTEGER,
                text TEXT,
                occurred_at TEXT,
                content_kind TEXT,
                include_raw INTEGER,
                skip_by_default INTEGER,
                artifact_backed INTEGER
            );
            CREATE TABLE memory_objects (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                object_type TEXT,
                source_kind TEXT,
                status TEXT,
                scope TEXT,
                privacy_level INTEGER,
                canonical_text TEXT,
                payload_json TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, conversation_id, seq, text, occurred_at, content_kind,
                include_raw, skip_by_default, artifact_backed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg-1",
                "conv-test",
                1,
                "I met David while volunteering at the shelter.",
                "2026-01-01T13:00:00",
                "text",
                1,
                0,
                0,
            ),
        )
        rows = [
            (
                "sum-1",
                "benchmark-user",
                "summary_view",
                "summarized",
                "active",
                "chat",
                0,
                "Selected summary about volunteering.",
                {"source_message_ids": ["msg-1"]},
            ),
            (
                "mem-direct",
                "benchmark-user",
                "evidence",
                "extracted",
                "active",
                "chat",
                0,
                "Direct source memory: David helped at the shelter.",
                {"source_message_ids": ["msg-1"]},
            ),
        ]
        for row in rows:
            connection.execute(
                """
                INSERT INTO memory_objects (
                    id, user_id, object_type, source_kind, status, scope,
                    privacy_level, canonical_text, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*row[:-1], json.dumps(row[-1])),
            )
        connection.commit()
    finally:
        connection.close()


def _write_report(path: Path, db_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_name": "LoCoMo",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "model_info": {},
                "conversations": [
                    {
                        "conversation_id": "conv-test",
                        "accuracy": 0.0,
                        "category_breakdown": {"1": 0.0},
                        "metadata": {"benchmark_db_path": str(db_path)},
                        "results": [
                            {
                                "question": {
                                    "question_id": "conv-test:q1",
                                    "question_text": "Who did Alex meet while volunteering?",
                                    "ground_truth": "David",
                                    "category": 1,
                                    "evidence_turn_ids": ["D1:1"],
                                },
                                "prediction": "I do not know.",
                                "score_result": {
                                    "score": 0,
                                    "reasoning": "missing David",
                                    "judge_model": "test",
                                },
                                "memories_used": 1,
                                "retrieval_time_ms": 1.0,
                                "trace": {
                                    "benchmark_privacy_enforcement": "off",
                                    "evidence_message_ids": ["msg-1"],
                                    "evidence_turn_ids": ["D1:1"],
                                    "missing_evidence_turn_ids": [],
                                    "retrieval_trace": {"user_id": "benchmark-user"},
                                    "selected_memory_ids": ["sum-1"],
                                    "critical_evidence_custody": {
                                        "counts": {
                                            "critical_evidence_count": 2,
                                            "raw_candidate_count": 1,
                                            "scored_count": 1,
                                            "selected_count": 1,
                                            "absent_count": 1,
                                        }
                                    },
                                    "retrieval_custody": [
                                        {
                                            "candidate_id": "sum-1",
                                            "candidate_kind": "summary_view",
                                            "channels": ["fts"],
                                            "scored": True,
                                            "selected": True,
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
