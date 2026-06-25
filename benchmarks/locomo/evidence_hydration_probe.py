"""Offline LoCoMo gold-evidence existence and hydration probe."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.output_root import bench_output_root
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.failure_ledger import ReportSpec, parse_report_spec


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "benchmarks" / "data" / "locomo10.json"
_DEFAULT_OUTPUT_DIR = bench_output_root() / "locomo"
_CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "unscored",
}
_CRITICAL_COUNT_KEYS = (
    "critical_evidence_count",
    "raw_candidate_count",
    "scored_count",
    "selected_count",
    "absent_count",
)


@dataclass(frozen=True, slots=True)
class _ConversationDbIndex:
    db_path: Path | None
    db_available: bool
    db_error: str | None
    expected_turn_count: int
    db_message_count: int
    turn_message_ids: dict[str, str]
    messages_by_id: dict[str, dict[str, Any]]


def build_evidence_hydration_probe(
    report_specs: Iterable[ReportSpec | str | Path],
    *,
    data_path: str | Path = _DEFAULT_DATA_PATH,
    include_passing: bool = False,
    first_loss_stages: set[str] | None = None,
) -> dict[str, Any]:
    """Build an offline evidence existence/hydration probe from report JSON."""
    specs = [_normalize_report_spec(spec) for spec in report_specs]
    conversations = {
        conversation.conversation_id: conversation
        for conversation in LoCoMoAdapter(data_path).load().conversations
    }
    source_reports: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    db_cache: dict[tuple[str | None, str], _ConversationDbIndex] = {}

    for spec in specs:
        report = _read_json(spec.path)
        source_reports.append(_source_report_record(spec, report))
        for conversation in _list(report.get("conversations")):
            conversation_dict = _dict(conversation)
            conversation_id = str(conversation_dict.get("conversation_id") or "")
            if spec.conversation_ids is not None and conversation_id not in spec.conversation_ids:
                continue
            dataset_conversation = conversations.get(conversation_id)
            db_path = _conversation_db_path(conversation_dict, report)
            cache_key = (str(db_path) if db_path is not None else None, conversation_id)
            if cache_key not in db_cache:
                db_cache[cache_key] = _build_db_index(
                    db_path=db_path,
                    conversation_id=conversation_id,
                    dataset_turns=list(getattr(dataset_conversation, "turns", [])),
                )
            db_index = db_cache[cache_key]
            for result in _list(conversation_dict.get("results")):
                result_dict = _dict(result)
                if _result_score(result_dict) != 0 and not include_passing:
                    continue
                critical_state, critical_counts = _critical_custody_state_and_counts(result_dict)
                first_loss_stage = _first_loss_stage(critical_counts)
                if first_loss_stages is not None and first_loss_stage not in first_loss_stages:
                    continue
                items.append(
                    _build_question_probe(
                        source_report=spec.path,
                        conversation_id=conversation_id,
                        db_index=db_index,
                        result=result_dict,
                        critical_custody_state=critical_state,
                        critical_counts=critical_counts,
                    )
                )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_passing": include_passing,
        "first_loss_stage_filter": sorted(first_loss_stages)
        if first_loss_stages is not None
        else None,
        "source_reports": source_reports,
        "summary": _summary(items),
        "items": items,
    }


def save_evidence_hydration_probe(probe: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the evidence hydration probe as JSON."""
    return write_json_atomic(Path(output_path).expanduser(), probe)


def format_evidence_hydration_summary(probe: dict[str, Any]) -> str:
    """Return a compact terminal-friendly probe summary."""
    summary = _dict(probe.get("summary"))
    return (
        "LoCoMo evidence hydration probe: "
        f"items={_int(summary.get('item_count'))} "
        f"turns={_int(summary.get('evidence_turn_count'))} "
        f"source_messages_found={_int(summary.get('source_message_found_count'))} "
        f"source_memory_questions={_int(summary.get('questions_with_source_memory'))} "
        f"no_source_memory_questions={_int(summary.get('questions_without_source_memory'))} "
        f"statuses={_format_counts(_dict(summary.get('question_status_counts')))}"
    )


def _normalize_report_spec(spec: ReportSpec | str | Path) -> ReportSpec:
    if isinstance(spec, ReportSpec):
        return spec
    if isinstance(spec, Path):
        return ReportSpec(path=spec.expanduser())
    return parse_report_spec(spec)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in report: {path}")
    return value


def _source_report_record(spec: ReportSpec, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(spec.path),
        "sha256": sha256_file_if_exists(spec.path),
        "conversation_filter": sorted(spec.conversation_ids)
        if spec.conversation_ids is not None
        else None,
        "benchmark_name": str(report.get("benchmark_name") or ""),
        "timestamp": str(report.get("timestamp") or ""),
    }


def _conversation_db_path(
    conversation: dict[str, Any],
    report: dict[str, Any],
) -> Path | None:
    metadata = _dict(conversation.get("metadata"))
    raw_path = metadata.get("benchmark_db_path")
    if not raw_path:
        benchmark_db = _dict(_dict(report.get("model_info")).get("benchmark_db"))
        raw_path = benchmark_db.get("reuse_db")
    if not raw_path:
        return None
    return _resolve_path(Path(str(raw_path)))


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return _PROJECT_ROOT / expanded


def _build_db_index(
    *,
    db_path: Path | None,
    conversation_id: str,
    dataset_turns: list[Any],
) -> _ConversationDbIndex:
    if db_path is None:
        return _ConversationDbIndex(
            db_path=None,
            db_available=False,
            db_error="missing_db_path",
            expected_turn_count=len(dataset_turns),
            db_message_count=0,
            turn_message_ids={},
            messages_by_id={},
        )
    if not db_path.exists():
        return _ConversationDbIndex(
            db_path=db_path,
            db_available=False,
            db_error="db_path_not_found",
            expected_turn_count=len(dataset_turns),
            db_message_count=0,
            turn_message_ids={},
            messages_by_id={},
        )
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, conversation_id, seq, text, occurred_at, content_kind,
                       include_raw, skip_by_default, artifact_backed
                FROM messages
                WHERE conversation_id = ?
                ORDER BY seq ASC
                """,
                (conversation_id,),
            )
        ]
    finally:
        connection.close()
    turn_message_ids = {
        str(turn.turn_id): str(row["id"])
        for turn, row in zip(dataset_turns, rows, strict=False)
        if getattr(turn, "turn_id", None) is not None
    }
    return _ConversationDbIndex(
        db_path=db_path,
        db_available=True,
        db_error=None,
        expected_turn_count=len(dataset_turns),
        db_message_count=len(rows),
        turn_message_ids=turn_message_ids,
        messages_by_id={str(row["id"]): row for row in rows},
    )


def _build_question_probe(
    *,
    source_report: Path,
    conversation_id: str,
    db_index: _ConversationDbIndex,
    result: dict[str, Any],
    critical_custody_state: str,
    critical_counts: dict[str, int],
) -> dict[str, Any]:
    question = _dict(result.get("question"))
    trace = _dict(result.get("trace"))
    evidence_turn_ids = [
        str(turn_id)
        for turn_id in _list(question.get("evidence_turn_ids"))
    ]
    trace_message_ids_by_turn = _trace_message_ids_by_turn(trace, evidence_turn_ids)
    user_id = _trace_user_id(trace)
    turn_records = [
        _turn_probe(
            db_index=db_index,
            user_id=user_id,
            turn_id=turn_id,
            trace_message_id=trace_message_ids_by_turn.get(turn_id),
        )
        for turn_id in evidence_turn_ids
    ]
    source_memory_ids = sorted(
        {
            memory_id
            for turn in turn_records
            for memory_id in _list(turn.get("source_memory_id_sample"))
        }
    )
    source_memory_count = len(source_memory_ids)
    source_message_found_count = sum(
        1 for turn in turn_records if turn.get("db_message_found") is True
    )
    source_memory_turn_count = sum(
        1 for turn in turn_records if _int(turn.get("source_memory_object_count")) > 0
    )
    artifact_chunk_turn_count = sum(
        1 for turn in turn_records if _int(turn.get("artifact_chunk_count")) > 0
    )
    source_memory_object_type_counts = _sum_count_maps(
        turn.get("source_memory_object_type_counts") for turn in turn_records
    )
    source_memory_source_kind_counts = _sum_count_maps(
        turn.get("source_memory_source_kind_counts") for turn in turn_records
    )
    source_memory_summary_view_count = _int(
        source_memory_object_type_counts.get("summary_view")
    )
    source_memory_non_summary_view_count = sum(
        amount
        for kind, amount in source_memory_object_type_counts.items()
        if kind != "summary_view"
    )
    labels = _question_labels(
        evidence_turn_count=len(evidence_turn_ids),
        source_message_found_count=source_message_found_count,
        source_memory_turn_count=source_memory_turn_count,
        source_memory_count=source_memory_count,
        source_memory_non_summary_view_count=source_memory_non_summary_view_count,
        artifact_chunk_turn_count=artifact_chunk_turn_count,
        trace=trace,
        db_index=db_index,
        turns=turn_records,
        critical_custody_state=critical_custody_state,
    )
    return {
        "source_report": str(source_report),
        "conversation_id": conversation_id,
        "question_id": str(question.get("question_id") or ""),
        "category": _optional_int(question.get("category")),
        "category_name": _category_name(_optional_int(question.get("category"))),
        "passed": _result_score(result) != 0,
        "diagnosis_bucket": str(trace.get("diagnosis_bucket") or ""),
        "sufficiency_diagnostic": str(trace.get("sufficiency_diagnostic") or ""),
        "first_loss_stage": _first_loss_stage(critical_counts),
        "critical_custody_state": critical_custody_state,
        "critical_counts": critical_counts,
        "db": {
            "path": str(db_index.db_path) if db_index.db_path is not None else None,
            "available": db_index.db_available,
            "error": db_index.db_error,
            "expected_turn_count": db_index.expected_turn_count,
            "db_message_count": db_index.db_message_count,
            "message_count_matches_dataset": (
                db_index.db_available
                and db_index.expected_turn_count == db_index.db_message_count
            ),
        },
        "evidence_turn_count": len(evidence_turn_ids),
        "source_message_found_count": source_message_found_count,
        "source_memory_turn_count": source_memory_turn_count,
        "source_memory_question_union_count": source_memory_count,
        "source_memory_object_type_counts": source_memory_object_type_counts,
        "source_memory_source_kind_counts": source_memory_source_kind_counts,
        "source_memory_summary_view_count": source_memory_summary_view_count,
        "source_memory_non_summary_view_count": source_memory_non_summary_view_count,
        "active_source_memory_question_union_count": len(
            {
                memory_id
                for turn in turn_records
                for memory_id in _list(turn.get("active_source_memory_id_sample"))
            }
        ),
        "artifact_chunk_turn_count": artifact_chunk_turn_count,
        "trace_evidence_memory_count": _int(trace.get("evidence_memory_count")),
        "trace_active_evidence_count": _int(trace.get("active_evidence_count")),
        "trace_missing_evidence_turn_count": len(
            _list(trace.get("missing_evidence_turn_ids"))
        ),
        "question_status": _question_status(labels),
        "labels": labels,
        "turns": turn_records,
    }


def _trace_message_ids_by_turn(
    trace: dict[str, Any],
    evidence_turn_ids: list[str],
) -> dict[str, str | None]:
    missing_turn_ids = {str(turn_id) for turn_id in _list(trace.get("missing_evidence_turn_ids"))}
    trace_message_ids = [
        str(message_id)
        for message_id in _list(trace.get("evidence_message_ids"))
    ]
    trace_message_iter = iter(trace_message_ids)
    mapping: dict[str, str | None] = {}
    for turn_id in evidence_turn_ids:
        if turn_id in missing_turn_ids:
            mapping[turn_id] = None
            continue
        mapping[turn_id] = next(trace_message_iter, None)
    return mapping


def _turn_probe(
    *,
    db_index: _ConversationDbIndex,
    user_id: str,
    turn_id: str,
    trace_message_id: str | None,
) -> dict[str, Any]:
    expected_message_id = db_index.turn_message_ids.get(turn_id)
    expected_message = (
        db_index.messages_by_id.get(expected_message_id)
        if expected_message_id is not None
        else None
    )
    trace_message = (
        db_index.messages_by_id.get(trace_message_id)
        if trace_message_id is not None
        else None
    )
    support_message_id = expected_message_id or trace_message_id
    memory_rows = (
        _memory_rows_for_source_message(db_index.db_path, user_id, support_message_id)
        if db_index.db_available and support_message_id
        else []
    )
    artifact_record = (
        _artifact_record_for_source_message(db_index.db_path, user_id, support_message_id)
        if db_index.db_available and support_message_id
        else _empty_artifact_record()
    )
    active_memory_rows = [
        row for row in memory_rows if str(row.get("status") or "") == "active"
    ]
    return {
        "turn_id": turn_id,
        "expected_message_id": expected_message_id,
        "trace_message_id": trace_message_id,
        "db_message_found": expected_message is not None,
        "trace_message_found": trace_message is not None if trace_message_id else None,
        "trace_message_matches_expected": (
            trace_message_id == expected_message_id
            if trace_message_id is not None and expected_message_id is not None
            else None
        ),
        "message_text_chars": len(str(expected_message.get("text") or ""))
        if expected_message is not None
        else 0,
        "message_occurred_at_present": bool(expected_message and expected_message.get("occurred_at")),
        "message_content_kind": str(expected_message.get("content_kind") or "")
        if expected_message is not None
        else "",
        "message_include_raw": _optional_bool(expected_message.get("include_raw"))
        if expected_message is not None
        else None,
        "message_skip_by_default": _optional_bool(expected_message.get("skip_by_default"))
        if expected_message is not None
        else None,
        "message_artifact_backed": _optional_bool(expected_message.get("artifact_backed"))
        if expected_message is not None
        else None,
        "source_memory_object_count": len(memory_rows),
        "active_source_memory_object_count": len(active_memory_rows),
        "source_memory_status_counts": _field_counts(memory_rows, "status"),
        "source_memory_object_type_counts": _field_counts(memory_rows, "object_type"),
        "source_memory_source_kind_counts": _field_counts(memory_rows, "source_kind"),
        "source_memory_id_sample": [str(row["id"]) for row in memory_rows[:20]],
        "active_source_memory_id_sample": [
            str(row["id"]) for row in active_memory_rows[:20]
        ],
        **artifact_record,
    }


def _memory_rows_for_source_message(
    db_path: Path | None,
    user_id: str,
    source_message_id: str | None,
) -> list[dict[str, Any]]:
    if db_path is None or source_message_id is None:
        return []
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT DISTINCT mo.id, mo.object_type, mo.source_kind, mo.status,
                                mo.scope, mo.privacy_level
                FROM memory_objects AS mo
                JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
                WHERE mo.user_id = ?
                  AND CAST(source_ids.value AS TEXT) = ?
                ORDER BY mo.id
                """,
                (user_id, source_message_id),
            )
        ]
    finally:
        connection.close()


def _artifact_record_for_source_message(
    db_path: Path | None,
    user_id: str,
    source_message_id: str | None,
) -> dict[str, Any]:
    if db_path is None or source_message_id is None:
        return _empty_artifact_record()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        artifact_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, status, artifact_type, source_kind
                FROM artifacts
                WHERE user_id = ?
                  AND message_id = ?
                  AND deleted_at IS NULL
                ORDER BY id
                """,
                (user_id, source_message_id),
            )
        ]
        artifact_ids = [str(row["id"]) for row in artifact_rows]
        chunk_rows: list[dict[str, Any]] = []
        if artifact_ids:
            placeholders = ",".join("?" for _ in artifact_ids)
            chunk_rows = [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT id, artifact_id, kind, token_count
                    FROM artifact_chunks
                    WHERE user_id = ?
                      AND artifact_id IN ({placeholders})
                    ORDER BY artifact_id, chunk_index
                    """,
                    (user_id, *artifact_ids),
                )
            ]
    finally:
        connection.close()
    return {
        "artifact_count": len(artifact_rows),
        "artifact_chunk_count": len(chunk_rows),
        "artifact_status_counts": _field_counts(artifact_rows, "status"),
        "artifact_type_counts": _field_counts(artifact_rows, "artifact_type"),
        "artifact_chunk_kind_counts": _field_counts(chunk_rows, "kind"),
        "artifact_id_sample": artifact_ids[:20],
        "artifact_chunk_id_sample": [str(row["id"]) for row in chunk_rows[:20]],
    }


def _empty_artifact_record() -> dict[str, Any]:
    return {
        "artifact_count": 0,
        "artifact_chunk_count": 0,
        "artifact_status_counts": {},
        "artifact_type_counts": {},
        "artifact_chunk_kind_counts": {},
        "artifact_id_sample": [],
        "artifact_chunk_id_sample": [],
    }


def _question_labels(
    *,
    evidence_turn_count: int,
    source_message_found_count: int,
    source_memory_turn_count: int,
    source_memory_count: int,
    source_memory_non_summary_view_count: int,
    artifact_chunk_turn_count: int,
    trace: dict[str, Any],
    db_index: _ConversationDbIndex,
    turns: list[dict[str, Any]],
    critical_custody_state: str,
) -> list[str]:
    labels: set[str] = set()
    if not db_index.db_available:
        labels.add("db_unavailable")
    if evidence_turn_count == 0:
        labels.add("no_gold_evidence_turns")
    elif source_message_found_count == evidence_turn_count:
        labels.add("all_source_messages_found")
    elif source_message_found_count == 0:
        labels.add("no_source_messages_found")
    else:
        labels.add("partial_source_messages_found")
    if evidence_turn_count > 0 and source_memory_turn_count == evidence_turn_count:
        labels.add("all_turns_have_source_memory")
    elif source_memory_turn_count == 0:
        labels.add("no_turns_have_source_memory")
    else:
        labels.add("some_turns_have_source_memory")
    if source_memory_count == 0:
        labels.add("question_source_memory_absent")
    else:
        labels.add("question_source_memory_present")
    if source_memory_count > 0 and source_memory_non_summary_view_count == 0:
        labels.add("question_source_memory_summary_only")
    if source_memory_non_summary_view_count > 0:
        labels.add("question_source_memory_includes_non_summary")
    if artifact_chunk_turn_count:
        labels.add("artifact_backed_evidence_present")
    if any(turn.get("trace_message_matches_expected") is False for turn in turns):
        labels.add("trace_message_mismatch")
    if any(
        turn.get("trace_message_id") is None and turn.get("db_message_found") is True
        for turn in turns
    ):
        labels.add("trace_mapping_missing_but_db_message_found")
    if source_memory_count and critical_custody_state == "missing":
        labels.add("critical_custody_missing_but_db_source_memory_present")
    if source_memory_count and critical_custody_state == "zero_critical_evidence":
        labels.add("critical_trace_zero_but_db_source_memory_present")
    if trace.get("benchmark_privacy_enforcement") == "off":
        labels.add("privacy_enforcement_off")
    return sorted(labels)


def _question_status(labels: list[str]) -> str:
    label_set = set(labels)
    if "db_unavailable" in label_set:
        return "db_unavailable"
    if "no_gold_evidence_turns" in label_set:
        return "no_gold_evidence_turns"
    if "no_source_messages_found" in label_set:
        return "source_messages_absent"
    if "partial_source_messages_found" in label_set:
        return "source_messages_partial"
    if "question_source_memory_present" in label_set:
        return "source_memory_present"
    if "artifact_backed_evidence_present" in label_set:
        return "artifact_support_present"
    return "source_messages_present_no_derived_support"


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "item_count": len(items),
        "failed_count": sum(1 for item in items if item.get("passed") is False),
        "evidence_turn_count": sum(_int(item.get("evidence_turn_count")) for item in items),
        "source_message_found_count": sum(
            _int(item.get("source_message_found_count")) for item in items
        ),
        "questions_with_source_memory": sum(
            1 for item in items if _int(item.get("source_memory_question_union_count")) > 0
        ),
        "questions_without_source_memory": sum(
            1 for item in items if _int(item.get("source_memory_question_union_count")) == 0
        ),
        "question_status_counts": _counter_dict(item.get("question_status") for item in items),
        "first_loss_stage_counts": _counter_dict(item.get("first_loss_stage") for item in items),
        "critical_custody_state_counts": _counter_dict(
            item.get("critical_custody_state") for item in items
        ),
        "category_counts": _counter_dict(item.get("category_name") for item in items),
        "label_counts": _label_counts(items),
        "by_category_name": _summary_by(items, "category_name"),
        "by_first_loss_stage": _summary_by(items, "first_loss_stage"),
    }


def _summary_by(items: list[dict[str, Any]], field_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(field_name) or "unknown"), []).append(item)
    return {
        key: {
            "item_count": len(group),
            "evidence_turn_count": sum(_int(item.get("evidence_turn_count")) for item in group),
            "source_message_found_count": sum(
                _int(item.get("source_message_found_count")) for item in group
            ),
            "questions_with_source_memory": sum(
                1
                for item in group
                if _int(item.get("source_memory_question_union_count")) > 0
            ),
            "questions_without_source_memory": sum(
                1
                for item in group
                if _int(item.get("source_memory_question_union_count")) == 0
            ),
            "question_status_counts": _counter_dict(
                item.get("question_status") for item in group
            ),
        }
        for key, group in sorted(grouped.items())
    }


def _trace_user_id(trace: dict[str, Any]) -> str:
    retrieval_trace = _dict(trace.get("retrieval_trace"))
    user_id = str(retrieval_trace.get("user_id") or "").strip()
    return user_id or "benchmark-user"


def _critical_custody_state_and_counts(
    result: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    trace = _dict(result.get("trace"))
    raw_custody = trace.get("critical_evidence_custody")
    custody_available = isinstance(raw_custody, dict)
    custody = _dict(raw_custody)
    counts = _dict(custody.get("counts"))
    parsed_counts = {
        key: _int(counts.get(key))
        for key in _CRITICAL_COUNT_KEYS
    }
    if not custody_available:
        return "missing", parsed_counts
    if parsed_counts.get("critical_evidence_count", 0) <= 0:
        return "zero_critical_evidence", parsed_counts
    return "present", parsed_counts


def _first_loss_stage(counts: dict[str, int]) -> str:
    critical_count = counts.get("critical_evidence_count", 0)
    if critical_count <= 0:
        return "critical_custody_unavailable"
    if counts.get("raw_candidate_count", 0) < critical_count:
        return "raw_absent"
    if counts.get("scored_count", 0) < critical_count:
        return "raw_unscored"
    if counts.get("selected_count", 0) < critical_count:
        return "scored_unselected"
    return "critical_selected_all"


def _result_score(result: dict[str, Any]) -> int:
    score_result = _dict(result.get("score_result"))
    return 1 if _optional_float(score_result.get("score")) not in (None, 0.0) else 0


def _category_name(category: int | None) -> str:
    if category is None:
        return "unknown"
    return _CATEGORY_NAMES.get(category, f"category-{category}")


def _field_counts(records: Iterable[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(field_name)
        if value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _sum_count_maps(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        for key, amount in _dict(value).items():
            counter[str(key)] += _int(amount)
    return dict(sorted(counter.items()))


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter[str(value or "unknown")] += 1
    return dict(sorted(counter.items()))


def _label_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(str(label) for label in _list(item.get("labels")))
    return dict(sorted(counter.items()))


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    parts = [
        f"{key}={_int(amount)}"
        for key, amount in sorted(value.items())
        if _int(amount)
    ]
    return " ".join(parts) if parts else "none"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUTPUT_DIR / f"locomo_evidence_hydration_probe_{timestamp}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help=(
            "LoCoMo report JSON path. Optionally suffix with ::conv-id,conv-id "
            "to keep only those conversations from that report."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=str(_DEFAULT_DATA_PATH),
        help="Path to locomo10.json.",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="Output JSON path. Defaults to docs/tmp with a timestamp.",
    )
    parser.add_argument(
        "--include-passing",
        action="store_true",
        help="Include passing questions instead of failed questions only.",
    )
    parser.add_argument(
        "--first-loss-stage",
        action="append",
        default=[],
        help="Keep only questions with this first_loss_stage. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    first_loss_stages = set(args.first_loss_stage) if args.first_loss_stage else None
    probe = build_evidence_hydration_probe(
        [parse_report_spec(value) for value in args.report],
        data_path=args.data_path,
        include_passing=args.include_passing,
        first_loss_stages=first_loss_stages,
    )
    output = save_evidence_hydration_probe(probe, args.output)
    print(format_evidence_hydration_summary(probe))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
