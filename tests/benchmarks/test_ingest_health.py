"""Tests for retained benchmark ingestion health checks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from benchmarks.audit_ingest_health import audit_roots
from benchmarks.ingest_health import audit_benchmark_db, classify_ingest_health
from benchmarks.llm_run_guard import LLMRunGuardConfig


def test_ingest_health_marks_source_backed_memories_without_packets_untrusted(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_objects(id, object_type, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                "mem_1",
                "evidence",
                json.dumps({"source_message_ids": ["msg_1"]}),
            ),
        )

    audit = audit_benchmark_db(db_path)
    health = classify_ingest_health(
        db_audit=audit,
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        require_evidence_packets=True,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert audit["memory_objects_with_source_message_ids"] == 1
    assert audit["memory_objects_with_source_message_ids_without_packets"] == 1
    assert health["trusted_ingest"] is False
    assert any("without evidence packets" in reason for reason in health["reasons"])


def test_ingest_health_trusts_memory_with_source_packet_and_span(tmp_path: Path) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("INSERT INTO messages(id) VALUES (?)", ("msg_1",))
        connection.execute(
            """
            INSERT INTO memory_objects(id, object_type, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                "mem_1",
                "evidence",
                json.dumps({"source_message_ids": ["msg_1"]}),
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_support_edges(id, memory_id, status)
            VALUES (?, ?, ?)
            """,
            ("edge_1", "mem_1", "active"),
        )
        connection.execute(
            """
            INSERT INTO memory_evidence_spans(support_edge_id, memory_id, span_role, message_id)
            VALUES (?, ?, ?, ?)
            """,
            ("edge_1", "mem_1", "source", "msg_1"),
        )

    health = classify_ingest_health(
        db_audit=audit_benchmark_db(db_path),
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        require_evidence_packets=True,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert health["trusted_ingest"] is True
    assert health["reasons"] == []


def test_bulk_ingest_health_requires_rebuild_result_and_summaries(tmp_path: Path) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO messages(id) VALUES (?)",
            [(f"msg_{index}",) for index in range(12)],
        )
        connection.execute(
            """
            INSERT INTO memory_objects(id, object_type, payload_json)
            VALUES (?, ?, ?)
            """,
            ("mem_1", "belief", json.dumps({})),
        )

    health = classify_ingest_health(
        db_audit=audit_benchmark_db(db_path),
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        require_rebuild_result=True,
        require_summary_views=True,
        expected_message_count=12,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert health["trusted_ingest"] is False
    assert any("missing required admin rebuild_result" in reason for reason in health["reasons"])
    assert any("produced no summary_views" in reason for reason in health["reasons"])


def test_bulk_ingest_health_accepts_complete_rebuild_and_summary_views(tmp_path: Path) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO messages(id) VALUES (?)",
            [(f"msg_{index}",) for index in range(12)],
        )
        connection.execute(
            """
            INSERT INTO memory_objects(id, object_type, payload_json)
            VALUES (?, ?, ?)
            """,
            ("mem_1", "belief", json.dumps({})),
        )
        connection.execute("INSERT INTO summary_views(id) VALUES (?)", ("sum_1",))

    health = classify_ingest_health(
        db_audit=audit_benchmark_db(db_path),
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        rebuild_result={"status": "rebuilt", "processed_messages": 12},
        require_rebuild_result=True,
        require_summary_views=True,
        expected_message_count=12,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert health["trusted_ingest"] is True
    assert health["reasons"] == []


def test_bulk_ingest_health_requires_episode_summary_for_long_bulk(tmp_path: Path) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE summary_views ADD COLUMN summary_kind TEXT")
        connection.executemany(
            "INSERT INTO messages(id) VALUES (?)",
            [(f"msg_{index}",) for index in range(60)],
        )
        connection.execute(
            "INSERT INTO summary_views(id, summary_kind) VALUES (?, ?)",
            ("sum_chunk_1", "conversation_chunk"),
        )

    health = classify_ingest_health(
        db_audit=audit_benchmark_db(db_path),
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        rebuild_result={"status": "rebuilt", "processed_messages": 60},
        require_rebuild_result=True,
        require_summary_views=True,
        expected_message_count=60,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert health["trusted_ingest"] is False
    assert any("no episode summary_views" in reason for reason in health["reasons"])


def test_bulk_ingest_health_accepts_chunk_and_episode_summary_kinds(tmp_path: Path) -> None:
    db_path = tmp_path / "benchmark.db"
    _create_health_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE summary_views ADD COLUMN summary_kind TEXT")
        connection.executemany(
            "INSERT INTO messages(id) VALUES (?)",
            [(f"msg_{index}",) for index in range(60)],
        )
        connection.executemany(
            "INSERT INTO summary_views(id, summary_kind) VALUES (?, ?)",
            [
                ("sum_chunk_1", "conversation_chunk"),
                ("sum_episode_1", "episode"),
            ],
        )

    health = classify_ingest_health(
        db_audit=audit_benchmark_db(db_path),
        llm_call_summary={"total_calls": 10, "failed_calls": 0, "by_purpose": {}},
        rebuild_result={"status": "rebuilt", "processed_messages": 60},
        require_rebuild_result=True,
        require_summary_views=True,
        expected_message_count=60,
        llm_guard_config=LLMRunGuardConfig(),
    )

    assert health["trusted_ingest"] is True
    assert health["reasons"] == []


def test_audit_roots_finds_nested_benchmark_dbs(tmp_path: Path) -> None:
    db_dir = tmp_path / "run" / "locomo_conv-test"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "benchmark.db"
    _create_health_db(db_path)

    summary = audit_roots([tmp_path], require_evidence_packets=True)

    assert summary["total"] == 1
    assert summary["trusted"] == 1
    assert summary["entries"][0]["db_path"] == str(db_path)


def _create_health_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY
            );
            CREATE TABLE memory_objects (
                id TEXT PRIMARY KEY,
                object_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE summary_views (
                id TEXT PRIMARY KEY
            );
            CREATE TABLE memory_links (
                id TEXT PRIMARY KEY
            );
            CREATE TABLE memory_support_edges (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE memory_evidence_spans (
                support_edge_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                span_role TEXT NOT NULL,
                message_id TEXT
            );
            """
        )
