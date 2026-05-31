"""Benchmark ingestion health checks for retained SQLite databases."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.llm_run_guard import LLMRunGuardConfig


def audit_benchmark_db(db_path: str | Path) -> dict[str, Any]:
    """Return structural ingestion/provenance counts for a retained benchmark DB."""

    resolved_db_path = Path(db_path).expanduser()
    audit: dict[str, Any] = {
        "db_path": str(resolved_db_path),
        "exists": resolved_db_path.is_file(),
        "tables": {},
        "counts": {},
        "object_type_counts": {},
        "summary_kind_counts": {},
        "memory_objects_with_source_message_ids": 0,
        "memory_objects_with_source_message_ids_without_packets": 0,
        "memory_object_ids_with_source_message_ids_without_packets": [],
        "memory_support_edges_without_source_spans": 0,
        "memory_evidence_spans_with_missing_messages": 0,
    }
    if not resolved_db_path.is_file():
        return audit

    with sqlite3.connect(resolved_db_path) as connection:
        connection.row_factory = sqlite3.Row
        tables = _existing_tables(connection)
        for table_name in (
            "messages",
            "memory_objects",
            "summary_views",
            "memory_links",
            "memory_support_edges",
            "memory_evidence_spans",
        ):
            present = table_name in tables
            audit["tables"][table_name] = present
            audit["counts"][table_name] = _count_rows(connection, table_name) if present else 0

        if "memory_objects" in tables:
            object_type_counts: Counter[str] = Counter()
            memories_with_source_ids: list[str] = []
            for row in connection.execute(
                """
                SELECT id, object_type, payload_json
                FROM memory_objects
                """
            ).fetchall():
                object_type_counts[str(row["object_type"] or "unknown")] += 1
                source_ids = _source_message_ids(row["payload_json"])
                if source_ids:
                    memories_with_source_ids.append(str(row["id"]))
            audit["object_type_counts"] = dict(sorted(object_type_counts.items()))
            audit["memory_objects_with_source_message_ids"] = len(memories_with_source_ids)
            if memories_with_source_ids:
                with_packets = _memory_ids_with_packets(connection, memories_with_source_ids, tables)
                orphan_ids = sorted(set(memories_with_source_ids).difference(with_packets))
                audit["memory_objects_with_source_message_ids_without_packets"] = len(orphan_ids)
                audit["memory_object_ids_with_source_message_ids_without_packets"] = orphan_ids[:20]

        if "memory_support_edges" in tables:
            audit["memory_support_edges_without_source_spans"] = (
                _support_edges_without_source_spans(connection, tables)
            )

        if "memory_evidence_spans" in tables and "messages" in tables:
            audit["memory_evidence_spans_with_missing_messages"] = (
                _spans_with_missing_messages(connection)
            )

        if "summary_views" in tables and _table_has_column(
            connection,
            "summary_views",
            "summary_kind",
        ):
            summary_kind_counts: Counter[str] = Counter()
            for row in connection.execute(
                """
                SELECT summary_kind, COUNT(*) AS count
                FROM summary_views
                GROUP BY summary_kind
                """
            ).fetchall():
                summary_kind_counts[str(row["summary_kind"] or "unknown")] = int(
                    row["count"]
                )
            audit["summary_kind_counts"] = dict(sorted(summary_kind_counts.items()))

    return audit


def classify_ingest_health(
    *,
    db_audit: dict[str, Any],
    llm_call_summary: dict[str, Any] | None = None,
    require_evidence_packets: bool = False,
    llm_guard_config: LLMRunGuardConfig | None = None,
    rebuild_result: dict[str, Any] | None = None,
    require_rebuild_result: bool = False,
    require_summary_views: bool = False,
    expected_message_count: int | None = None,
) -> dict[str, Any]:
    """Classify whether a retained ingestion DB should be trusted for evaluate-only."""

    reasons: list[str] = []
    warnings: list[str] = []

    if not db_audit.get("exists"):
        reasons.append("benchmark DB does not exist")

    counts = db_audit.get("counts") if isinstance(db_audit.get("counts"), dict) else {}
    message_count = int(counts.get("messages") or 0)
    memory_count = int(counts.get("memory_objects") or 0)
    summary_view_count = int(counts.get("summary_views") or 0)
    support_edge_count = int(counts.get("memory_support_edges") or 0)
    evidence_span_count = int(counts.get("memory_evidence_spans") or 0)
    source_backed_memory_count = int(
        db_audit.get("memory_objects_with_source_message_ids") or 0
    )
    orphan_source_memory_count = int(
        db_audit.get("memory_objects_with_source_message_ids_without_packets") or 0
    )
    edges_without_source_spans = int(
        db_audit.get("memory_support_edges_without_source_spans") or 0
    )
    missing_message_spans = int(
        db_audit.get("memory_evidence_spans_with_missing_messages") or 0
    )

    if require_evidence_packets:
        if memory_count > 0 and support_edge_count == 0:
            reasons.append("memory objects exist but memory_support_edges is empty")
        if support_edge_count > 0 and evidence_span_count == 0:
            reasons.append("memory support edges exist but memory_evidence_spans is empty")
        if orphan_source_memory_count:
            reasons.append(
                "source-backed memory objects without evidence packets: "
                f"{orphan_source_memory_count}/{source_backed_memory_count}"
            )
        if edges_without_source_spans:
            reasons.append(
                f"memory support edges without source spans: {edges_without_source_spans}"
            )
    elif memory_count > 0 and support_edge_count == 0:
        warnings.append("legacy DB has memory objects but no evidence packets")

    if missing_message_spans:
        reasons.append(
            f"memory evidence spans point at missing messages: {missing_message_spans}"
        )

    if expected_message_count is not None and message_count != expected_message_count:
        reasons.append(
            "message count does not match expected ingestion turn count: "
            f"{message_count}/{expected_message_count}"
        )

    if require_rebuild_result and not isinstance(rebuild_result, dict):
        reasons.append("bulk ingestion is missing required admin rebuild_result")

    if isinstance(rebuild_result, dict):
        if rebuild_result.get("status") == "rebuilt_partial":
            reasons.append("admin rebuild completed partially")
        if require_rebuild_result and rebuild_result.get("status") != "rebuilt":
            reasons.append(
                "bulk admin rebuild did not report rebuilt status: "
                f"{rebuild_result.get('status')}"
            )
        processed_messages = rebuild_result.get("processed_messages")
        if (
            require_rebuild_result
            and expected_message_count is not None
            and int(processed_messages or 0) != expected_message_count
        ):
            reasons.append(
                "bulk admin rebuild processed an unexpected number of messages: "
                f"{processed_messages}/{expected_message_count}"
            )
        recoverable_failures = int(rebuild_result.get("recoverable_job_failures") or 0)
        if recoverable_failures:
            reasons.append(f"admin rebuild had recoverable job failures: {recoverable_failures}")

    if require_summary_views and message_count >= 10 and summary_view_count == 0:
        reasons.append(
            "bulk compaction produced no summary_views despite enough messages "
            f"for conversation chunking: messages={message_count}"
        )
    summary_kind_counts = (
        db_audit.get("summary_kind_counts")
        if isinstance(db_audit.get("summary_kind_counts"), dict)
        else {}
    )
    if require_summary_views and message_count >= 10 and summary_kind_counts:
        if int(summary_kind_counts.get("conversation_chunk") or 0) == 0:
            reasons.append("bulk compaction produced no conversation_chunk summary_views")
        if message_count >= 50 and int(summary_kind_counts.get("episode") or 0) == 0:
            reasons.append("bulk compaction produced no episode summary_views")

    llm_health = _classify_llm_summary(
        llm_call_summary or {},
        llm_guard_config or LLMRunGuardConfig.disabled(),
    )
    reasons.extend(llm_health["reasons"])

    return {
        "trusted_ingest": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "db_audit": db_audit,
        "llm_health": llm_health,
        "require_evidence_packets": bool(require_evidence_packets),
        "require_rebuild_result": bool(require_rebuild_result),
        "require_summary_views": bool(require_summary_views),
        "expected_message_count": expected_message_count,
    }


def _classify_llm_summary(
    summary: dict[str, Any],
    config: LLMRunGuardConfig,
) -> dict[str, Any]:
    reasons: list[str] = []
    total_calls = int(summary.get("total_calls") or 0)
    failed_calls = int(summary.get("failed_calls") or 0)

    if (
        config.max_total_failed_llm_calls is not None
        and failed_calls > config.max_total_failed_llm_calls
    ):
        reasons.append(
            "total failed LLM calls exceeded "
            f"{config.max_total_failed_llm_calls}: {failed_calls}"
        )

    if (
        config.max_failed_llm_call_ratio is not None
        and total_calls >= config.min_calls_for_failed_ratio
        and total_calls > 0
        and failed_calls / total_calls > config.max_failed_llm_call_ratio
    ):
        reasons.append(
            "LLM failure ratio exceeded "
            f"{config.max_failed_llm_call_ratio:.2%}: {failed_calls}/{total_calls} "
            f"({failed_calls / total_calls:.2%})"
        )

    by_purpose = summary.get("by_purpose") if isinstance(summary.get("by_purpose"), dict) else {}
    for purpose, group in sorted(by_purpose.items()):
        if not isinstance(group, dict):
            continue
        calls = int(group.get("calls") or 0)
        failures = int(group.get("failed_calls") or 0)
        if (
            config.max_failed_calls_per_purpose is not None
            and failures > config.max_failed_calls_per_purpose
        ):
            reasons.append(
                "failed LLM calls for purpose "
                f"{purpose!r} exceeded {config.max_failed_calls_per_purpose}: {failures}"
            )
        if (
            config.max_failed_ratio_per_purpose is not None
            and calls >= config.min_calls_per_purpose_for_failed_ratio
            and calls > 0
            and failures / calls > config.max_failed_ratio_per_purpose
        ):
            reasons.append(
                "LLM failure ratio for purpose "
                f"{purpose!r} exceeded {config.max_failed_ratio_per_purpose:.2%}: "
                f"{failures}/{calls} ({failures / calls:.2%})"
            )

    return {
        "trusted": not reasons,
        "reasons": reasons,
        "total_calls": total_calls,
        "failed_calls": failed_calls,
    }


def _existing_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _table_has_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"] if row is not None else 0)


def _memory_ids_with_packets(
    connection: sqlite3.Connection,
    memory_ids: list[str],
    tables: set[str],
) -> set[str]:
    if "memory_support_edges" not in tables:
        return set()
    ids: set[str] = set()
    for chunk_start in range(0, len(memory_ids), 200):
        chunk = memory_ids[chunk_start : chunk_start + 200]
        placeholders = ", ".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT DISTINCT memory_id
            FROM memory_support_edges
            WHERE memory_id IN ({placeholders})
              AND status = 'active'
            """,
            chunk,
        ).fetchall()
        ids.update(str(row["memory_id"]) for row in rows)
    return ids


def _support_edges_without_source_spans(
    connection: sqlite3.Connection,
    tables: set[str],
) -> int:
    if "memory_evidence_spans" not in tables:
        return _count_rows(connection, "memory_support_edges")
    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memory_support_edges AS edge
        WHERE edge.status = 'active'
          AND NOT EXISTS (
              SELECT 1
              FROM memory_evidence_spans AS span
              WHERE span.support_edge_id = edge.id
                AND span.span_role = 'source'
          )
        """
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _spans_with_missing_messages(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memory_evidence_spans AS span
        WHERE span.message_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM messages AS message
              WHERE message.id = span.message_id
          )
        """
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _source_message_ids(payload_json: Any) -> list[str]:
    if not isinstance(payload_json, str) or not payload_json:
        return []
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return []
    raw_ids = payload.get("source_message_ids") if isinstance(payload, dict) else None
    if not isinstance(raw_ids, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        message_id = str(raw_id).strip()
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)
        normalized.append(message_id)
    return normalized
