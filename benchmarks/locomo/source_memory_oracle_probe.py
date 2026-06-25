"""Offline source-memory stage oracle for LoCoMo reports.

This diagnostic compares gold-evidence source memories available in the retained
DB with retrieval custody records already present in a LoCoMo report. It does
not rerun retrieval or answer generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.output_root import bench_output_root
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evidence_hydration_probe import (
    _build_db_index,
    _conversation_db_path,
    _dict,
    _field_counts,
    _first_loss_stage,
    _int,
    _list,
    _memory_rows_for_source_message,
    _optional_float,
    _optional_int,
    _resolve_path,
    _trace_message_ids_by_turn,
    _trace_user_id,
)
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


def build_source_memory_oracle_probe(
    report_specs: Iterable[ReportSpec | str | Path],
    *,
    data_path: str | Path = _DEFAULT_DATA_PATH,
    include_passing: bool = False,
    first_loss_stages: set[str] | None = None,
) -> dict[str, Any]:
    """Build a source-memory stage oracle from existing report and DB artifacts."""
    specs = [_normalize_report_spec(spec) for spec in report_specs]
    conversations = {
        conversation.conversation_id: conversation
        for conversation in LoCoMoAdapter(data_path).load().conversations
    }
    source_reports: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    db_cache: dict[tuple[str | None, str], Any] = {}

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
                critical_state, critical_counts = _critical_custody_state_and_counts(
                    result_dict
                )
                first_loss_stage = _first_loss_stage(critical_counts)
                if first_loss_stages is not None and first_loss_stage not in first_loss_stages:
                    continue
                items.append(
                    _build_item(
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


def save_source_memory_oracle_probe(probe: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a source-memory oracle probe as JSON."""
    return write_json_atomic(Path(output_path).expanduser(), probe)


def format_source_memory_oracle_summary(probe: dict[str, Any]) -> str:
    """Return a compact terminal-friendly oracle summary."""
    summary = _dict(probe.get("summary"))
    return (
        "LoCoMo source-memory oracle: "
        f"items={_int(summary.get('item_count'))} "
        f"available={_int(summary.get('available_source_memory_count'))} "
        f"raw={_int(summary.get('raw_source_memory_count'))} "
        f"scored={_int(summary.get('scored_source_memory_count'))} "
        f"selected={_int(summary.get('selected_source_memory_count'))} "
        f"stage={_format_counts(_dict(summary.get('oracle_stage_counts')))}"
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


def _build_item(
    *,
    source_report: Path,
    conversation_id: str,
    db_index: Any,
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
    user_id = _trace_user_id(trace)
    trace_message_ids_by_turn = _trace_message_ids_by_turn(trace, evidence_turn_ids)
    source_memory_rows = _source_memory_rows_for_question(
        db_index=db_index,
        user_id=user_id,
        evidence_turn_ids=evidence_turn_ids,
        trace_message_ids_by_turn=trace_message_ids_by_turn,
    )
    source_memory_rows_by_id = {
        str(row.get("id") or ""): row
        for row in source_memory_rows
        if str(row.get("id") or "")
    }
    custody_by_id = _retrieval_custody_by_candidate_id(trace)
    row_records = [
        _source_memory_record(row, custody_by_id.get(memory_id))
        for memory_id, row in sorted(source_memory_rows_by_id.items())
    ]
    stage_counts = _stage_counts(row_records)
    kind_counts = _source_kind_counts(row_records)
    labels = _labels(row_records, stage_counts, critical_custody_state)
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
        "evidence_turn_count": len(evidence_turn_ids),
        "available_source_memory_count": len(row_records),
        "active_source_memory_count": sum(
            1 for record in row_records if record.get("status") == "active"
        ),
        "raw_source_memory_count": stage_counts["raw"],
        "scored_source_memory_count": stage_counts["scored"],
        "selected_source_memory_count": stage_counts["selected"],
        "available_non_summary_count": kind_counts["available_non_summary"],
        "raw_non_summary_count": kind_counts["raw_non_summary"],
        "selected_non_summary_count": kind_counts["selected_non_summary"],
        "available_summary_like_count": kind_counts["available_summary_like"],
        "raw_summary_like_count": kind_counts["raw_summary_like"],
        "selected_summary_like_count": kind_counts["selected_summary_like"],
        "oracle_stage": _oracle_stage(stage_counts),
        "labels": labels,
        "source_memory_object_type_counts": _field_counts(row_records, "object_type"),
        "source_memory_source_kind_counts": _field_counts(row_records, "source_kind"),
        "raw_candidate_kind_counts": _field_counts(
            [record for record in row_records if record.get("in_raw_candidates")],
            "candidate_kind",
        ),
        "selected_candidate_kind_counts": _field_counts(
            [record for record in row_records if record.get("selected")],
            "candidate_kind",
        ),
        "drop_stage_counts": _field_counts(
            [record for record in row_records if record.get("drop_stage")],
            "drop_stage",
        ),
        "drop_reason_counts": _field_counts(
            [record for record in row_records if record.get("drop_reason")],
            "drop_reason",
        ),
        "source_memory_records": row_records,
    }


def _source_memory_rows_for_question(
    *,
    db_index: Any,
    user_id: str,
    evidence_turn_ids: list[str],
    trace_message_ids_by_turn: dict[str, str | None],
) -> list[dict[str, Any]]:
    if not getattr(db_index, "db_available", False):
        return []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for turn_id in evidence_turn_ids:
        expected_message_id = db_index.turn_message_ids.get(turn_id)
        trace_message_id = trace_message_ids_by_turn.get(turn_id)
        source_message_id = expected_message_id or trace_message_id
        for row in _memory_rows_for_source_message(
            db_index.db_path,
            user_id,
            source_message_id,
        ):
            memory_id = str(row.get("id") or "")
            if memory_id and memory_id not in rows_by_id:
                rows_by_id[memory_id] = row
    return list(rows_by_id.values())


def _retrieval_custody_by_candidate_id(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in _list(trace.get("retrieval_custody")):
        record_dict = _dict(record)
        candidate_id = str(record_dict.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in records_by_id:
            records_by_id[candidate_id] = record_dict
    return records_by_id


def _source_memory_record(
    row: dict[str, Any],
    custody: dict[str, Any] | None,
) -> dict[str, Any]:
    in_raw = custody is not None
    scored = bool(custody.get("scored")) if custody is not None else False
    selected = bool(custody.get("selected")) if custody is not None else False
    record = {
        "memory_id": str(row.get("id") or ""),
        "object_type": str(row.get("object_type") or ""),
        "source_kind": str(row.get("source_kind") or ""),
        "status": str(row.get("status") or ""),
        "scope": str(row.get("scope") or ""),
        "privacy_level": _int(row.get("privacy_level")),
        "summary_like": _is_summary_like(row),
        "in_raw_candidates": in_raw,
        "scored": scored,
        "selected": selected,
    }
    if custody is not None:
        record.update(
            {
                "candidate_kind": str(custody.get("candidate_kind") or ""),
                "channels": [
                    str(channel)
                    for channel in _list(custody.get("channels"))
                ],
                "score_rank": custody.get("score_rank"),
                "selection_rank": custody.get("selection_rank"),
                "drop_stage": custody.get("drop_stage"),
                "drop_reason": custody.get("drop_reason"),
                "composer_decision": custody.get("composer_decision"),
                "scorer_final_score": _optional_float(
                    _dict(custody.get("scorer")).get("final_score")
                ),
            }
        )
    return record


def _stage_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "available": len(records),
        "raw": sum(1 for record in records if record.get("in_raw_candidates") is True),
        "scored": sum(1 for record in records if record.get("scored") is True),
        "selected": sum(1 for record in records if record.get("selected") is True),
    }


def _source_kind_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    available_non_summary = sum(1 for record in records if not record.get("summary_like"))
    raw_non_summary = sum(
        1
        for record in records
        if record.get("in_raw_candidates") is True and not record.get("summary_like")
    )
    selected_non_summary = sum(
        1
        for record in records
        if record.get("selected") is True and not record.get("summary_like")
    )
    return {
        "available_non_summary": available_non_summary,
        "raw_non_summary": raw_non_summary,
        "selected_non_summary": selected_non_summary,
        "available_summary_like": len(records) - available_non_summary,
        "raw_summary_like": sum(
            1
            for record in records
            if record.get("in_raw_candidates") is True and record.get("summary_like")
        ),
        "selected_summary_like": sum(
            1
            for record in records
            if record.get("selected") is True and record.get("summary_like")
        ),
    }


def _labels(
    records: list[dict[str, Any]],
    stage_counts: dict[str, int],
    critical_custody_state: str,
) -> list[str]:
    labels: set[str] = {f"critical_custody_{critical_custody_state}"}
    available = stage_counts["available"]
    raw = stage_counts["raw"]
    scored = stage_counts["scored"]
    selected = stage_counts["selected"]
    kind_counts = _source_kind_counts(records)
    if available == 0:
        labels.add("source_memory_available_none")
    elif raw == 0:
        labels.add("source_memory_raw_none")
    elif raw < available:
        labels.add("source_memory_raw_partial")
    else:
        labels.add("source_memory_raw_all")
    if raw > 0 and scored == 0:
        labels.add("source_memory_scored_none")
    elif 0 < scored < raw:
        labels.add("source_memory_scored_partial")
    elif raw > 0:
        labels.add("source_memory_scored_all")
    if available > 0 and selected == 0:
        labels.add("source_memory_selected_none")
    elif 0 < selected < available:
        labels.add("source_memory_selected_partial")
    elif available > 0:
        labels.add("source_memory_selected_all")
    if kind_counts["available_non_summary"] == 0 and available > 0:
        labels.add("source_memory_available_summary_only")
    if kind_counts["available_non_summary"] > 0:
        labels.add("source_memory_available_includes_non_summary")
    if kind_counts["available_non_summary"] > 0 and kind_counts["raw_non_summary"] == 0:
        labels.add("non_summary_available_but_not_raw")
    if kind_counts["raw_non_summary"] > 0 and kind_counts["selected_non_summary"] == 0:
        labels.add("non_summary_raw_but_not_selected")
    if selected > 0 and kind_counts["selected_non_summary"] == 0:
        labels.add("selected_source_memory_summary_only")
    return sorted(labels)


def _oracle_stage(stage_counts: dict[str, int]) -> str:
    available = stage_counts["available"]
    if available <= 0:
        return "source_memory_unavailable"
    if stage_counts["raw"] <= 0:
        return "available_not_raw"
    if stage_counts["scored"] <= 0:
        return "raw_not_scored"
    if stage_counts["selected"] <= 0:
        return "scored_not_selected"
    if stage_counts["selected"] < available:
        return "selected_partial"
    return "selected_all"


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


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "item_count": len(items),
        "failed_count": sum(1 for item in items if item.get("passed") is False),
        "available_source_memory_count": sum(
            _int(item.get("available_source_memory_count")) for item in items
        ),
        "raw_source_memory_count": sum(
            _int(item.get("raw_source_memory_count")) for item in items
        ),
        "scored_source_memory_count": sum(
            _int(item.get("scored_source_memory_count")) for item in items
        ),
        "selected_source_memory_count": sum(
            _int(item.get("selected_source_memory_count")) for item in items
        ),
        "available_non_summary_count": sum(
            _int(item.get("available_non_summary_count")) for item in items
        ),
        "raw_non_summary_count": sum(
            _int(item.get("raw_non_summary_count")) for item in items
        ),
        "selected_non_summary_count": sum(
            _int(item.get("selected_non_summary_count")) for item in items
        ),
        "oracle_stage_counts": _counter_dict(item.get("oracle_stage") for item in items),
        "first_loss_stage_counts": _counter_dict(item.get("first_loss_stage") for item in items),
        "critical_custody_state_counts": _counter_dict(
            item.get("critical_custody_state") for item in items
        ),
        "category_counts": _counter_dict(item.get("category_name") for item in items),
        "label_counts": _label_counts(items),
        "by_category_name": _summary_by(items, "category_name"),
        "by_first_loss_stage": _summary_by(items, "first_loss_stage"),
        "by_oracle_stage": _summary_by(items, "oracle_stage"),
    }


def _summary_by(items: list[dict[str, Any]], field_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(field_name) or "unknown"), []).append(item)
    return {
        key: {
            "item_count": len(group),
            "available_source_memory_count": sum(
                _int(item.get("available_source_memory_count")) for item in group
            ),
            "raw_source_memory_count": sum(
                _int(item.get("raw_source_memory_count")) for item in group
            ),
            "scored_source_memory_count": sum(
                _int(item.get("scored_source_memory_count")) for item in group
            ),
            "selected_source_memory_count": sum(
                _int(item.get("selected_source_memory_count")) for item in group
            ),
            "available_non_summary_count": sum(
                _int(item.get("available_non_summary_count")) for item in group
            ),
            "raw_non_summary_count": sum(
                _int(item.get("raw_non_summary_count")) for item in group
            ),
            "selected_non_summary_count": sum(
                _int(item.get("selected_non_summary_count")) for item in group
            ),
            "oracle_stage_counts": _counter_dict(
                item.get("oracle_stage") for item in group
            ),
        }
        for key, group in sorted(grouped.items())
    }


def _category_name(category: int | None) -> str:
    if category is None:
        return "unknown"
    return _CATEGORY_NAMES.get(category, f"category-{category}")


def _is_summary_like(record: dict[str, Any]) -> bool:
    return (
        str(record.get("object_type") or "") == "summary_view"
        or str(record.get("candidate_kind") or "") == "summary_view"
        or str(record.get("source_kind") or "") == "summarized"
    )


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


def _result_score(result: dict[str, Any]) -> int:
    score_result = _dict(result.get("score_result"))
    return 1 if _optional_float(score_result.get("score")) not in (None, 0.0) else 0


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUTPUT_DIR / f"locomo_source_memory_oracle_{timestamp}.json"


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
    probe = build_source_memory_oracle_probe(
        [parse_report_spec(value) for value in args.report],
        data_path=_resolve_path(Path(args.data_path)),
        include_passing=args.include_passing,
        first_loss_stages=first_loss_stages,
    )
    output = save_source_memory_oracle_probe(probe, args.output)
    print(format_source_memory_oracle_summary(probe))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
