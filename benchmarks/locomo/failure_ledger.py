"""Offline LoCoMo failure custody ledger.

This module reads existing LoCoMo JSON reports and emits a behavior-neutral
ledger for failure triage. It intentionally does not call models, databases, or
retrieval code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from atagia.services.answer_postcondition import DEFAULT_ABSTENTION_TEXT
from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "docs" / "tmp"
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
class ReportSpec:
    """One report path plus an optional conversation filter."""

    path: Path
    conversation_ids: frozenset[str] | None = None


def parse_report_spec(raw_value: str) -> ReportSpec:
    """Parse ``path`` or ``path::conv-a,conv-b`` report specs."""
    raw_path = raw_value
    raw_conversations = ""
    if "::" in raw_value:
        raw_path, raw_conversations = raw_value.rsplit("::", 1)
    conversation_ids = frozenset(
        value
        for item in raw_conversations.split(",")
        if (value := item.strip())
    )
    return ReportSpec(
        path=Path(raw_path).expanduser(),
        conversation_ids=conversation_ids or None,
    )


def build_failure_ledger(
    report_specs: Iterable[ReportSpec | str | Path],
    *,
    include_correct: bool = False,
    include_text: bool = False,
) -> dict[str, Any]:
    """Build a custody ledger from one or more LoCoMo report JSON files."""
    specs = [_normalize_report_spec(spec) for spec in report_specs]
    source_reports: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for spec in specs:
        report = _read_json(spec.path)
        source_reports.append(_source_report_record(spec, report))
        for conversation_id, result in _iter_results(report, spec.conversation_ids):
            score = _result_score(result)
            question = _dict(result.get("question"))
            category = _optional_int(question.get("category"))
            all_results.append(
                {
                    "conversation_id": conversation_id,
                    "question_id": str(question.get("question_id") or ""),
                    "category": category,
                    "score": score,
                }
            )
            if score != 0 and not include_correct:
                continue
            items.append(
                _build_item(
                    spec.path,
                    conversation_id,
                    result,
                    include_text=include_text,
                )
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_correct": include_correct,
        "include_text": include_text,
        "source_reports": source_reports,
        "source_result_summary": _source_result_summary(all_results),
        "ledger_summary": _ledger_summary(items),
        "items": items,
    }


def save_failure_ledger(ledger: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a failure ledger as JSON."""
    return write_json_atomic(Path(output_path).expanduser(), ledger)


def format_failure_ledger_summary(ledger: dict[str, Any]) -> str:
    """Return a compact terminal-friendly ledger summary."""
    summary = _dict(ledger.get("ledger_summary"))
    item_count = _int(summary.get("item_count"))
    failed_count = _int(summary.get("failed_count"))
    categories = _format_counts(_dict(summary.get("failed_by_category_name")))
    first_loss = _format_counts(_dict(summary.get("failed_by_first_loss_stage")))
    guards = _format_counts(_dict(summary.get("failed_by_guard_status")))
    return (
        "LoCoMo failure ledger: "
        f"items={item_count} failed={failed_count} "
        f"categories={categories} first_loss={first_loss} guard={guards}"
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
        "total_questions": _int(report.get("total_questions")),
        "total_correct": _int(report.get("total_correct")),
        "overall_accuracy": _optional_float(report.get("overall_accuracy")),
        "model_info": _compact_model_info(_dict(report.get("model_info"))),
    }


def _compact_model_info(model_info: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "provider",
        "answer_model",
        "judge_model",
        "ingest_model",
        "retrieval_model",
        "activation_flags",
        "benchmark_db",
        "parallel_questions",
        "privacy_enforcement",
        "selection",
        "warning_counts",
    )
    return {
        key: model_info[key]
        for key in keep_keys
        if key in model_info
    }


def _iter_results(
    report: dict[str, Any],
    conversation_ids: frozenset[str] | None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    for conversation in _list(report.get("conversations")):
        conversation_dict = _dict(conversation)
        conversation_id = str(conversation_dict.get("conversation_id") or "")
        if conversation_ids is not None and conversation_id not in conversation_ids:
            continue
        for result in _list(conversation_dict.get("results")):
            yield conversation_id, _dict(result)


def _build_item(
    report_path: Path,
    conversation_id: str,
    result: dict[str, Any],
    *,
    include_text: bool,
) -> dict[str, Any]:
    question = _dict(result.get("question"))
    trace = _dict(result.get("trace"))
    category = _optional_int(question.get("category"))
    score = _result_score(result)
    passed = score != 0
    critical = _critical_evidence_record(trace)
    retrieval = _retrieval_record(trace)
    guard = _guard_record(trace, result)
    shadow = _shadow_sufficiency_record(trace, critical)
    runtime = _runtime_record(trace, result)
    labels = _diagnostic_labels(
        passed=passed,
        trace=trace,
        critical=critical,
        retrieval=retrieval,
        guard=guard,
        shadow=shadow,
    )

    item: dict[str, Any] = {
        "source_report": str(report_path),
        "conversation_id": conversation_id,
        "question_id": str(question.get("question_id") or ""),
        "category": category,
        "category_name": _category_name(category),
        "score": score,
        "passed": passed,
        "diagnosis_bucket": _trace_text(trace, "diagnosis_bucket"),
        "sufficiency_diagnostic": _trace_text(trace, "sufficiency_diagnostic"),
        "failure_stage": _trace_text(trace, "failure_stage") or None,
        "evidence_turn_count": len(_list(question.get("evidence_turn_ids"))),
        "trace_evidence_turn_count": len(_list(trace.get("evidence_turn_ids"))),
        "missing_evidence_turn_count": len(_list(trace.get("missing_evidence_turn_ids"))),
        "evidence_memory_count": _int(trace.get("evidence_memory_count")),
        "active_evidence_count": _int(trace.get("active_evidence_count")),
        "selected_memory_count": len(_list(trace.get("selected_memory_ids"))),
        "selected_evidence_memory_count": len(_list(trace.get("selected_evidence_memory_ids"))),
        "memories_used": _int(result.get("memories_used")),
        "retrieval_time_ms": _optional_float(result.get("retrieval_time_ms")),
        "critical_evidence": critical,
        "retrieval": retrieval,
        "guard": guard,
        "shadow_sufficiency": shadow,
        "runtime": runtime,
        "privacy": _privacy_record(trace),
        "diagnostic_labels": labels,
    }
    if include_text:
        item["question_text"] = str(question.get("question_text") or "")
        item["ground_truth"] = str(question.get("ground_truth") or "")
        item["prediction"] = str(result.get("prediction") or "")
        item["judge_reasoning"] = str(_dict(result.get("score_result")).get("reasoning") or "")
    return item


def _critical_evidence_record(trace: dict[str, Any]) -> dict[str, Any]:
    raw_custody = trace.get("critical_evidence_custody")
    custody_available = isinstance(raw_custody, dict)
    custody = _dict(raw_custody)
    counts = _critical_counts(custody)
    items = [_dict(item) for item in _list(custody.get("items"))]
    selected_items = [item for item in items if item.get("selected") is True]
    summary_selected = sum(1 for item in selected_items if _is_summary_like(item))
    non_summary_selected = len(selected_items) - summary_selected
    return {
        "custody_state": _critical_custody_state(custody_available, counts),
        "counts": counts,
        "survival_stage_counts": _int_mapping(custody.get("survival_stage_counts")),
        "first_loss_stage": _first_loss_stage(counts),
        "first_loss_is_partial": _first_loss_is_partial(counts),
        "selected_object_type_counts": _field_counts(selected_items, "object_type"),
        "selected_candidate_kind_counts": _field_counts(selected_items, "candidate_kind"),
        "selected_source_kind_counts": _field_counts(selected_items, "source_kind"),
        "selected_summary_like_count": summary_selected,
        "selected_non_summary_like_count": non_summary_selected,
        "selected_summary_only": bool(selected_items and non_summary_selected == 0),
        "critical_summary_like_count": sum(1 for item in items if _is_summary_like(item)),
        "critical_non_summary_like_count": sum(1 for item in items if not _is_summary_like(item)),
    }


def _critical_custody_state(
    custody_available: bool,
    counts: dict[str, int],
) -> str:
    if not custody_available:
        return "missing"
    if counts.get("critical_evidence_count", 0) <= 0:
        return "zero_critical_evidence"
    return "present"


def _critical_counts(custody: dict[str, Any]) -> dict[str, int]:
    raw_counts = _dict(custody.get("counts"))
    return {
        key: _int(raw_counts.get(key))
        for key in _CRITICAL_COUNT_KEYS
    }


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


def _first_loss_is_partial(counts: dict[str, int]) -> bool:
    stage = _first_loss_stage(counts)
    critical_count = counts.get("critical_evidence_count", 0)
    if critical_count <= 0:
        return False
    if stage == "raw_absent":
        return counts.get("raw_candidate_count", 0) > 0
    if stage == "raw_unscored":
        return counts.get("scored_count", 0) > 0
    if stage == "scored_unselected":
        return counts.get("selected_count", 0) > 0
    return False


def _retrieval_record(trace: dict[str, Any]) -> dict[str, Any]:
    custody = [_dict(record) for record in _list(trace.get("retrieval_custody"))]
    selected_records = [record for record in custody if record.get("selected") is True]
    retrieval_trace = _dict(trace.get("retrieval_trace"))
    custody_trace = _dict(retrieval_trace.get("custody"))
    summary = summarize_retrieval_custody([custody])
    selected_summary = sum(1 for record in selected_records if _is_summary_like(record))
    selected_non_summary = len(selected_records) - selected_summary
    return {
        "candidate_count": _int(summary.get("candidate_count")),
        "selected_count": _int(summary.get("selected_count")),
        "candidate_count_by_channel": _int_mapping(
            custody_trace.get("candidate_count_by_channel")
            or summary.get("channel_counts")
        ),
        "post_user_id_candidate_count": _optional_int(custody_trace.get("post_user_id_candidate_count")),
        "post_scope_coordinate_lifecycle_candidate_count": _optional_int(
            custody_trace.get("post_scope_coordinate_lifecycle_candidate_count")
        ),
        "scored_candidate_count": _optional_int(custody_trace.get("scored_candidate_count")),
        "selected_candidate_count": _optional_int(custody_trace.get("selected_candidate_count")),
        "drop_counts_by_stage": _int_mapping(custody_trace.get("drop_counts_by_stage")),
        "drop_counts_by_reason": _int_mapping(custody_trace.get("drop_counts_by_reason")),
        "selected_source_window_count": len(_list(custody_trace.get("selected_source_window_ids"))),
        "source_window_count": len(_list(custody_trace.get("source_window_ids"))),
        "selected_evidence_id_count": len(_list(custody_trace.get("selected_evidence_ids"))),
        "selected_candidate_kind_counts": _field_counts(selected_records, "candidate_kind"),
        "selected_source_kind_counts": _field_counts(selected_records, "source_kind"),
        "selected_channel_counts": _selected_channel_counts(selected_records),
        "selected_summary_like_count": selected_summary,
        "selected_non_summary_like_count": selected_non_summary,
        "selected_summary_only": bool(selected_records and selected_non_summary == 0),
        "filter_reason_counts": _dict(summary.get("filter_reason_counts")),
    }


def _guard_record(trace: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    guard = _dict(trace.get("answer_postcondition_guard"))
    verdict = _dict(guard.get("verdict"))
    prediction = str(result.get("prediction") or "")
    return {
        "enabled": _optional_bool(guard.get("enabled")),
        "status": str(guard.get("status") or "unavailable"),
        "known_hard_abstention_text": prediction.strip() == DEFAULT_ABSTENTION_TEXT,
        "prediction_chars": len(prediction),
        "retry_count": _int(guard.get("retry_count")),
        "verifier_retry_count": _int(guard.get("verifier_retry_count")),
        "structured_output_failure_count": _int(
            guard.get("verifier_structured_output_failure_count")
        ),
        "evidence_use_repair_count": _int(guard.get("evidence_use_repair_count")),
        "evidence_use_repair_success_count": _int(
            guard.get("evidence_use_repair_success_count")
        ),
        "evidence_use_repair_failure_count": _int(
            guard.get("evidence_use_repair_failure_count")
        ),
        "abstention_reason": str(guard.get("abstention_reason") or ""),
        "abstention_allowed_reason_present": bool(guard.get("abstention_allowed_reason")),
        "abstention_legitimacy_retry_count": _int(
            guard.get("abstention_legitimacy_retry_count")
        ),
        "final_answer_used_required_evidence": _optional_bool(
            guard.get("final_answer_used_required_evidence")
        ),
        "failure_reason_count": len(_list(guard.get("failure_reasons"))),
        "missing_supported_obligation_count": len(
            _list(guard.get("missing_supported_obligations"))
        ),
        "output_limit_retry": bool(guard.get("output_limit_retry")),
        "quality_warning_count": _int(guard.get("quality_warning_count")),
        "verdict": {
            "is_abstention": _optional_bool(verdict.get("is_abstention")),
            "pass_postcondition": _optional_bool(verdict.get("pass_postcondition")),
            "requires_abstention": _optional_bool(verdict.get("requires_abstention")),
            "unsupported_concrete_claims": _optional_bool(
                verdict.get("unsupported_concrete_claims")
            ),
            "covers_requested_facets": _optional_bool(verdict.get("covers_requested_facets")),
            "contains_concrete_claims": _optional_bool(verdict.get("contains_concrete_claims")),
        },
    }


def _shadow_sufficiency_record(
    trace: dict[str, Any],
    critical: dict[str, Any],
) -> dict[str, Any]:
    shadow = _dict(trace.get("shadow_sufficiency_diagnostics"))
    counts = _dict(critical.get("counts"))
    critical_count = _int(counts.get("critical_evidence_count"))
    raw_count = _int(counts.get("raw_candidate_count"))
    selected_count = _int(counts.get("selected_count"))
    state = str(shadow.get("state") or "unavailable")
    sufficient = state == "retrieval_sufficient"
    return {
        "state": state,
        "would_abstain": _optional_bool(shadow.get("would_abstain")),
        "confidence": _optional_float(shadow.get("confidence")),
        "candidate_count": _optional_int(shadow.get("candidate_count")),
        "scored_candidate_count": _optional_int(shadow.get("scored_candidate_count")),
        "shortlist_count": _optional_int(shadow.get("shortlist_count")),
        "direct_evidence_candidate_count": _optional_int(
            shadow.get("direct_evidence_candidate_count")
        ),
        "summary_candidate_count": _optional_int(shadow.get("summary_candidate_count")),
        "artifact_candidate_count": _optional_int(shadow.get("artifact_candidate_count")),
        "verbatim_evidence_search_candidate_count": _optional_int(
            shadow.get("verbatim_evidence_search_candidate_count")
        ),
        "false_positive_zero_raw_critical": bool(
            sufficient and critical_count > 0 and raw_count == 0
        ),
        "false_positive_zero_selected_critical": bool(
            sufficient and critical_count > 0 and selected_count == 0
        ),
        "sufficient_with_partial_selected_critical": bool(
            sufficient and 0 < selected_count < critical_count
        ),
    }


def _runtime_record(trace: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    retrieval_trace = _dict(trace.get("retrieval_trace"))
    runtime = _dict(retrieval_trace.get("runtime_diagnostics"))
    llm_summary = _dict(trace.get("llm_call_summary"))
    token_totals = _dict(llm_summary.get("token_totals"))
    return {
        "retrieval_time_ms": _optional_float(result.get("retrieval_time_ms")),
        "retrieval_total_duration_ms": _optional_float(retrieval_trace.get("total_duration_ms")),
        "db_query_count": _int(runtime.get("db_query_count")),
        "db_query_count_by_operation": _int_mapping(runtime.get("db_query_count_by_operation")),
        "hydration_timings_ms": _float_mapping(runtime.get("hydration_timings_ms")),
        "stage_timings_ms": _float_mapping(runtime.get("stage_timings_ms")),
        "lock_wait_count": _int(runtime.get("lock_wait_count")),
        "sqlite_busy_count": _int(runtime.get("sqlite_busy_count")),
        "llm_total_calls": _int(llm_summary.get("total_calls")),
        "llm_failed_calls": _int(llm_summary.get("failed_calls")),
        "llm_total_tokens": _optional_float(token_totals.get("total_tokens")),
        "llm_input_tokens": _optional_float(token_totals.get("input_tokens")),
        "llm_output_tokens": _optional_float(token_totals.get("output_tokens")),
        "llm_total_latency_ms": _optional_float(llm_summary.get("total_latency_ms")),
        "llm_model_call_counts": _int_mapping(llm_summary.get("model_call_counts")),
    }


def _privacy_record(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_privacy_enforcement": str(
            trace.get("benchmark_privacy_enforcement") or ""
        ),
        "benchmark_answer_privacy_override": _optional_bool(
            trace.get("benchmark_answer_privacy_override")
        ),
        "benchmark_high_risk_secret_redaction_disabled": _optional_bool(
            trace.get("benchmark_high_risk_secret_redaction_disabled")
        ),
    }


def _diagnostic_labels(
    *,
    passed: bool,
    trace: dict[str, Any],
    critical: dict[str, Any],
    retrieval: dict[str, Any],
    guard: dict[str, Any],
    shadow: dict[str, Any],
) -> list[str]:
    labels: set[str] = set()
    if not passed:
        labels.add("failed")
    first_loss_stage = str(critical.get("first_loss_stage") or "")
    if first_loss_stage:
        labels.add(first_loss_stage)
    critical_custody_state = str(critical.get("custody_state") or "")
    if critical_custody_state:
        labels.add(f"critical_custody_{critical_custody_state}")
    if critical.get("first_loss_is_partial") is True:
        labels.add("partial_critical_survival")
    counts = _dict(critical.get("counts"))
    critical_count = _int(counts.get("critical_evidence_count"))
    selected_count = _int(counts.get("selected_count"))
    if critical_count > 0 and selected_count == 0:
        labels.add("critical_selected_none")
    elif critical_count > 0 and selected_count < critical_count:
        labels.add("critical_selected_partial")
    elif critical_count > 0:
        labels.add("critical_selected_all")
    if critical.get("selected_summary_only") is True:
        labels.add("critical_selected_summary_only")
    if retrieval.get("selected_summary_only") is True:
        labels.add("retrieval_selected_summary_only")
    guard_status = str(guard.get("status") or "")
    if guard_status:
        labels.add(f"guard_{guard_status}")
    if guard.get("known_hard_abstention_text") is True:
        labels.add("known_hard_abstention_text")
    if not passed and guard_status == "passed":
        labels.add("guard_passed_but_failed")
    if _int(guard.get("evidence_use_repair_count")):
        labels.add("repair_attempted")
    if _int(guard.get("evidence_use_repair_failure_count")):
        labels.add("repair_failed")
    if _int(guard.get("evidence_use_repair_success_count")):
        labels.add("repair_succeeded")
    if guard.get("final_answer_used_required_evidence") is False:
        labels.add("final_answer_did_not_use_required_evidence")
    if shadow.get("false_positive_zero_raw_critical") is True:
        labels.add("shadow_sufficiency_false_positive_zero_raw")
    if shadow.get("false_positive_zero_selected_critical") is True:
        labels.add("shadow_sufficiency_false_positive_zero_selected")
    if shadow.get("sufficient_with_partial_selected_critical") is True:
        labels.add("shadow_sufficiency_partial_selected_critical")
    if trace.get("benchmark_privacy_enforcement") == "off":
        labels.add("privacy_enforcement_off")
    return sorted(labels)


def _source_result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(_int(result.get("score")) for result in results)
    by_category: dict[str, dict[str, Any]] = {}
    for category, group in _group_by_category(results).items():
        group_total = len(group)
        group_correct = sum(_int(result.get("score")) for result in group)
        by_category[_category_name(category)] = {
            "category": category,
            "correct": group_correct,
            "total": group_total,
            "accuracy": _rate(group_correct, group_total),
        }
    return {
        "total_questions": total,
        "total_correct": correct,
        "overall_accuracy": _rate(correct, total),
        "by_category_name": dict(sorted(by_category.items())),
    }


def _ledger_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    failed_items = [item for item in items if item.get("passed") is False]
    return {
        "item_count": len(items),
        "failed_count": len(failed_items),
        "passed_count": len(items) - len(failed_items),
        "failed_by_category_name": _counter_dict(
            item.get("category_name") for item in failed_items
        ),
        "failed_by_diagnosis_bucket": _counter_dict(
            item.get("diagnosis_bucket") for item in failed_items
        ),
        "failed_by_sufficiency_diagnostic": _counter_dict(
            item.get("sufficiency_diagnostic") for item in failed_items
        ),
        "failed_by_first_loss_stage": _counter_dict(
            _dict(item.get("critical_evidence")).get("first_loss_stage")
            for item in failed_items
        ),
        "failed_by_guard_status": _counter_dict(
            _dict(item.get("guard")).get("status")
            for item in failed_items
        ),
        "failed_label_counts": _label_counts(failed_items),
        "shadow_sufficiency_false_positive_zero_raw": sum(
            1
            for item in failed_items
            if _dict(item.get("shadow_sufficiency")).get(
                "false_positive_zero_raw_critical"
            )
            is True
        ),
        "shadow_sufficiency_false_positive_zero_selected": sum(
            1
            for item in failed_items
            if _dict(item.get("shadow_sufficiency")).get(
                "false_positive_zero_selected_critical"
            )
            is True
        ),
        "guard_passed_but_failed": sum(
            1
            for item in failed_items
            if _dict(item.get("guard")).get("status") == "passed"
        ),
        "guard_abstained_failed": sum(
            1
            for item in failed_items
            if _dict(item.get("guard")).get("status") == "abstained"
        ),
        "known_hard_abstention_failed": sum(
            1
            for item in failed_items
            if _dict(item.get("guard")).get("known_hard_abstention_text") is True
        ),
    }


def _group_by_category(items: list[dict[str, Any]]) -> dict[int | None, list[dict[str, Any]]]:
    grouped: dict[int | None, list[dict[str, Any]]] = {}
    for item in items:
        category = _optional_int(item.get("category"))
        grouped.setdefault(category, []).append(item)
    return grouped


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


def _field_counts(records: Iterable[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(field_name)
        if value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _selected_channel_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for channel in _list(record.get("channels")):
            counter[str(channel)] += 1
    return dict(sorted(counter.items()))


def _label_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(str(label) for label in _list(item.get("diagnostic_labels")))
    return dict(sorted(counter.items()))


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        text = str(value or "unknown")
        counter[text] += 1
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


def _trace_text(trace: dict[str, Any], field_name: str) -> str:
    return str(trace.get(field_name) or "")


def _result_score(result: dict[str, Any]) -> int:
    score_result = _dict(result.get("score_result"))
    return 1 if _optional_float(score_result.get("score")) not in (None, 0.0) else 0


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
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _int_mapping(value: Any) -> dict[str, int]:
    raw = _dict(value)
    result: dict[str, int] = {}
    for key, item in raw.items():
        result[str(key)] = _int(item)
    return dict(sorted(result.items()))


def _float_mapping(value: Any) -> dict[str, float]:
    raw = _dict(value)
    result: dict[str, float] = {}
    for key, item in raw.items():
        parsed = _optional_float(item)
        if parsed is not None:
            result[str(key)] = parsed
    return dict(sorted(result.items()))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUTPUT_DIR / f"locomo_failure_ledger_{timestamp}.json"


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
        "--output",
        default=str(_default_output_path()),
        help="Output JSON path. Defaults to docs/tmp with a timestamp.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include passing questions as controls instead of failures only.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw question, gold, prediction, and judge text in the ledger.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ledger = build_failure_ledger(
        [parse_report_spec(value) for value in args.report],
        include_correct=args.include_correct,
        include_text=args.include_text,
    )
    output = save_failure_ledger(ledger, args.output)
    print(format_failure_ledger_summary(ledger))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
