"""Tests for offline LoCoMo evidence hydration probes."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from benchmarks.locomo.evidence_hydration_probe import (
    build_evidence_hydration_probe,
    format_evidence_hydration_summary,
)


def test_evidence_hydration_probe_checks_messages_memory_and_artifacts(
    tmp_path: Path,
) -> None:
    data_path = _write_dataset(tmp_path)
    db_path = tmp_path / "benchmark.db"
    _write_db(db_path)
    report_path = _write_report(tmp_path, db_path)

    probe = build_evidence_hydration_probe([report_path], data_path=data_path)

    assert probe["summary"]["item_count"] == 1
    assert probe["summary"]["source_message_found_count"] == 2
    assert probe["summary"]["questions_with_source_memory"] == 1
    assert probe["summary"]["question_status_counts"] == {"source_memory_present": 1}
    item = probe["items"][0]
    assert item["question_id"] == "conv-a:q1"
    assert item["first_loss_stage"] == "critical_custody_unavailable"
    assert item["source_message_found_count"] == 2
    assert item["source_memory_question_union_count"] == 1
    assert item["artifact_chunk_turn_count"] == 1
    assert "critical_trace_zero_but_db_source_memory_present" in item["labels"]
    assert item["turns"][0]["source_memory_object_count"] == 1
    assert item["turns"][1]["artifact_chunk_count"] == 1
    serialized = json.dumps(probe)
    assert "RAW SOURCE TEXT" not in serialized
    assert "RAW QUESTION TEXT" not in serialized
    assert "RAW GOLD" not in serialized
    assert format_evidence_hydration_summary(probe).startswith(
        "LoCoMo evidence hydration probe:"
    )


def test_evidence_hydration_probe_flags_trace_mapping_missing_but_db_found(
    tmp_path: Path,
) -> None:
    data_path = _write_dataset(tmp_path)
    db_path = tmp_path / "benchmark.db"
    _write_db(db_path)
    report_path = _write_report(
        tmp_path,
        db_path,
        trace={
            "evidence_message_ids": [],
            "missing_evidence_turn_ids": ["D1:1", "D1:2"],
        },
    )

    probe = build_evidence_hydration_probe([report_path], data_path=data_path)

    item = probe["items"][0]
    assert item["source_message_found_count"] == 2
    assert "trace_mapping_missing_but_db_message_found" in item["labels"]
    assert item["trace_missing_evidence_turn_count"] == 2


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
                            "question": "RAW QUESTION TEXT",
                            "answer": "RAW GOLD",
                            "category": 1,
                            "evidence": ["D1:1", "D1:2"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_path


def _write_report(
    tmp_path: Path,
    db_path: Path,
    *,
    trace: dict[str, object] | None = None,
) -> Path:
    trace_payload = {
        "diagnosis_bucket": "retrieval_or_ranking_miss",
        "sufficiency_diagnostic": "retrieval_insufficient",
        "benchmark_privacy_enforcement": "off",
        "evidence_message_ids": ["msg_1", "msg_2"],
        "missing_evidence_turn_ids": [],
        "evidence_memory_count": 0,
        "active_evidence_count": 0,
        "critical_evidence_custody": {
            "counts": {
                "critical_evidence_count": 0,
                "raw_candidate_count": 0,
                "scored_count": 0,
                "selected_count": 0,
                "absent_count": 0,
            },
            "items": [],
            "survival_stage_counts": {},
        },
        "retrieval_trace": {"user_id": "benchmark-user"},
    }
    trace_payload.update(trace or {})
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
                            {
                                "question": {
                                    "question_id": "conv-a:q1",
                                    "question_text": "RAW QUESTION TEXT",
                                    "ground_truth": "RAW GOLD",
                                    "category": 1,
                                    "evidence_turn_ids": ["D1:1", "D1:2"],
                                },
                                "prediction": "RAW PREDICTION",
                                "score_result": {
                                    "score": 0,
                                    "reasoning": "RAW REASON",
                                    "judge_model": "judge",
                                },
                                "memories_used": 0,
                                "retrieval_time_ms": 1.0,
                                "trace": trace_payload,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


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
            CREATE TABLE artifacts (
                id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                message_id TEXT,
                status TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE artifact_chunks (
                id TEXT NOT NULL UNIQUE,
                artifact_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            INSERT INTO messages(id, conversation_id, seq, text, occurred_at)
            VALUES
                ('msg_1', 'conv-a', 1, 'RAW SOURCE TEXT one', '2026-05-01T13:00:00'),
                ('msg_2', 'conv-a', 2, 'RAW SOURCE TEXT two', '2026-05-01T13:00:00');
            INSERT INTO memory_objects(
                id, user_id, object_type, source_kind, status, scope,
                privacy_level, payload_json
            )
            VALUES (
                'mem_1', 'benchmark-user', 'evidence', 'extracted',
                'active', 'chat', 0, '{"source_message_ids":["msg_1"]}'
            );
            INSERT INTO artifacts(
                id, user_id, message_id, status, artifact_type, source_kind,
                deleted_at
            )
            VALUES (
                'art_1', 'benchmark-user', 'msg_2', 'ready', 'image',
                'host_embedded', NULL
            );
            INSERT INTO artifact_chunks(
                id, artifact_id, user_id, chunk_index, kind, token_count
            )
            VALUES ('chunk_1', 'art_1', 'benchmark-user', 0, 'summary', 12);
            """
        )
        connection.commit()
    finally:
        connection.close()
