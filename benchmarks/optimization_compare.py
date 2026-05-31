"""Summarize benchmark runs for local-model optimization experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ReportSummary:
    label: str
    path: str
    benchmark_name: str
    total_questions: int
    total_passed: float
    pass_rate: float
    duration_seconds: float
    failed_question_ids: tuple[str, ...]
    structured_retries: int
    structured_retries_by_purpose: dict[str, int]
    calls_by_purpose: dict[str, int]
    input_tokens_by_purpose: dict[str, float]
    output_tokens_by_purpose: dict[str, float]
    latency_ms_by_purpose: dict[str, float]
    warning_counts: dict[str, int]
    diagnosis_counts: dict[str, int]
    sufficiency_counts: dict[str, int]
    retrieval_time_ms_mean: float | None
    memories_used_mean: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "benchmark_name": self.benchmark_name,
            "total_questions": self.total_questions,
            "total_passed": self.total_passed,
            "pass_rate": self.pass_rate,
            "duration_seconds": self.duration_seconds,
            "failed_question_ids": list(self.failed_question_ids),
            "structured_retries": self.structured_retries,
            "structured_retries_by_purpose": self.structured_retries_by_purpose,
            "calls_by_purpose": self.calls_by_purpose,
            "input_tokens_by_purpose": self.input_tokens_by_purpose,
            "output_tokens_by_purpose": self.output_tokens_by_purpose,
            "latency_ms_by_purpose": self.latency_ms_by_purpose,
            "warning_counts": self.warning_counts,
            "diagnosis_counts": self.diagnosis_counts,
            "sufficiency_counts": self.sufficiency_counts,
            "retrieval_time_ms_mean": self.retrieval_time_ms_mean,
            "memories_used_mean": self.memories_used_mean,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Atagia-bench and LoCoMo report JSON files."
    )
    parser.add_argument("reports", nargs="+", help="Report JSON files or directories")
    parser.add_argument("--label", action="append", default=[], help="Optional label")
    parser.add_argument(
        "--aggregate-label",
        default=None,
        help="Append one aggregate summary over all supplied reports",
    )
    parser.add_argument("--json-output", default=None, help="Write JSON summary")
    parser.add_argument("--markdown-output", default=None, help="Write Markdown summary")
    args = parser.parse_args()

    report_paths = _expand_report_paths(args.reports)
    if len(args.label) > report_paths.__len__():
        raise SystemExit("--label cannot be supplied more times than reports")

    summaries = [
        summarize_report(
            path,
            label=args.label[index] if index < len(args.label) else path.parent.name,
        )
        for index, path in enumerate(report_paths)
    ]
    if args.aggregate_label is not None:
        summaries.append(_aggregate_summaries(summaries, label=args.aggregate_label))

    payload = {
        "reports": [summary.to_json() for summary in summaries],
        "comparisons": _build_pairwise_comparisons(summaries),
    }
    markdown = _to_markdown(summaries)

    if args.json_output:
        _write_json(Path(args.json_output), payload)
    if args.markdown_output:
        Path(args.markdown_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).expanduser().write_text(markdown)

    print(markdown)


def summarize_report(path: Path, *, label: str) -> ReportSummary:
    data = json.loads(path.read_text())
    if "per_question" in data:
        return _summarize_atagia_report(data, path=path, label=label)
    if "conversations" in data:
        return _summarize_generic_benchmark_report(data, path=path, label=label)
    raise ValueError(f"Unsupported benchmark report shape: {path}")


def _summarize_atagia_report(
    data: dict[str, Any], *, path: Path, label: str
) -> ReportSummary:
    per_question = data.get("per_question") or []
    failed = []
    retrieval_times = []
    memories_used = []
    diagnosis = Counter()
    sufficiency = Counter()
    for row in per_question:
        question_id = str(row.get("question_id") or row.get("id") or "")
        passed = _row_passed(row)
        if not passed and question_id:
            failed.append(question_id)
        if isinstance(row.get("retrieval_time_ms"), int | float):
            retrieval_times.append(float(row["retrieval_time_ms"]))
        if isinstance(row.get("memories_used"), int | float):
            memories_used.append(float(row["memories_used"]))
        trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
        _count_optional(diagnosis, trace.get("diagnosis_bucket"))
        _count_optional(sufficiency, trace.get("sufficiency_diagnostic"))

    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    llm_summary = config.get("llm_call_summary") if isinstance(config, dict) else {}
    return ReportSummary(
        label=label,
        path=str(path),
        benchmark_name=str(data.get("benchmark_name") or "atagia_bench"),
        total_questions=int(data.get("total_questions") or len(per_question)),
        total_passed=float(data.get("total_passed") or 0),
        pass_rate=float(data.get("pass_rate") or 0.0),
        duration_seconds=float(data.get("run_duration_seconds") or 0.0),
        failed_question_ids=tuple(sorted(failed)),
        structured_retries=_structured_retry_total(llm_summary),
        structured_retries_by_purpose=_structured_retry_by_purpose(llm_summary),
        calls_by_purpose=_purpose_metric(llm_summary, "calls"),
        input_tokens_by_purpose=_purpose_token_metric(llm_summary, "input_tokens"),
        output_tokens_by_purpose=_purpose_token_metric(llm_summary, "output_tokens"),
        latency_ms_by_purpose=_purpose_metric(llm_summary, "total_latency_ms"),
        warning_counts=_int_dict(config.get("warning_counts")),
        diagnosis_counts=_int_dict(data.get("diagnosis_bucket_counts")) or dict(diagnosis),
        sufficiency_counts=_int_dict(data.get("sufficiency_diagnostic_counts"))
        or dict(sufficiency),
        retrieval_time_ms_mean=_safe_mean(retrieval_times),
        memories_used_mean=_safe_mean(memories_used),
    )


def _summarize_generic_benchmark_report(
    data: dict[str, Any], *, path: Path, label: str
) -> ReportSummary:
    conversations = data.get("conversations") or []
    failed = []
    retrieval_times = []
    memories_used = []
    diagnosis = Counter()
    sufficiency = Counter()
    for conversation in conversations:
        for result in conversation.get("results") or []:
            question = result.get("question") if isinstance(result.get("question"), dict) else {}
            score_result = (
                result.get("score_result")
                if isinstance(result.get("score_result"), dict)
                else {}
            )
            question_id = str(question.get("question_id") or result.get("question_id") or "")
            if float(score_result.get("score") or 0.0) < 1.0 and question_id:
                failed.append(question_id)
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            trace = metadata.get("trace") if isinstance(metadata.get("trace"), dict) else {}
            _count_optional(diagnosis, trace.get("diagnosis_bucket"))
            _count_optional(sufficiency, trace.get("sufficiency_diagnostic"))
            for key, target in (
                ("retrieval_time_ms", retrieval_times),
                ("memories_used", memories_used),
            ):
                value = metadata.get(key) or trace.get(key)
                if isinstance(value, int | float):
                    target.append(float(value))

    model_info = data.get("model_info") if isinstance(data.get("model_info"), dict) else {}
    llm_summary = model_info.get("llm_call_summary") if isinstance(model_info, dict) else {}
    total_questions = int(data.get("total_questions") or 0)
    total_passed = float(data.get("total_correct") or 0.0)
    return ReportSummary(
        label=label,
        path=str(path),
        benchmark_name=str(data.get("benchmark_name") or "benchmark"),
        total_questions=total_questions,
        total_passed=total_passed,
        pass_rate=(total_passed / total_questions if total_questions else 0.0),
        duration_seconds=float(data.get("duration_seconds") or 0.0),
        failed_question_ids=tuple(sorted(failed)),
        structured_retries=_structured_retry_total(llm_summary),
        structured_retries_by_purpose=_structured_retry_by_purpose(llm_summary),
        calls_by_purpose=_purpose_metric(llm_summary, "calls"),
        input_tokens_by_purpose=_purpose_token_metric(llm_summary, "input_tokens"),
        output_tokens_by_purpose=_purpose_token_metric(llm_summary, "output_tokens"),
        latency_ms_by_purpose=_purpose_metric(llm_summary, "total_latency_ms"),
        warning_counts=_int_dict(model_info.get("warning_counts")),
        diagnosis_counts=dict(diagnosis),
        sufficiency_counts=dict(sufficiency),
        retrieval_time_ms_mean=_safe_mean(retrieval_times),
        memories_used_mean=_safe_mean(memories_used),
    )


def _expand_report_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(
                sorted(path.glob("**/*report-*.json"))
                + sorted(path.glob("**/locomo-report-*.json"))
            )
            continue
        raise FileNotFoundError(path)
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _row_passed(row: dict[str, Any]) -> bool:
    if isinstance(row.get("passed"), bool):
        return bool(row["passed"])
    grade = row.get("grade") if isinstance(row.get("grade"), dict) else {}
    if isinstance(grade.get("passed"), bool):
        return bool(grade["passed"])
    if isinstance(row.get("score"), int | float):
        return float(row["score"]) >= 1.0
    return False


def _structured_retry_total(llm_summary: Any) -> int:
    if not isinstance(llm_summary, dict):
        return 0
    repair = llm_summary.get("structured_output_repair")
    if isinstance(repair, dict):
        return int(repair.get("retry_calls") or 0)
    return 0


def _structured_retry_by_purpose(llm_summary: Any) -> dict[str, int]:
    if not isinstance(llm_summary, dict):
        return {}
    repair = llm_summary.get("structured_output_repair")
    by_purpose = repair.get("by_purpose") if isinstance(repair, dict) else {}
    if not isinstance(by_purpose, dict):
        return {}
    return {
        str(purpose): int(row.get("retry_calls") or 0)
        for purpose, row in by_purpose.items()
        if isinstance(row, dict)
    }


def _purpose_metric(llm_summary: Any, key: str) -> dict[str, float]:
    values: dict[str, float] = {}
    if not isinstance(llm_summary, dict):
        return values
    by_purpose = llm_summary.get("by_purpose")
    if not isinstance(by_purpose, dict):
        return values
    for purpose, row in by_purpose.items():
        if isinstance(row, dict):
            values[str(purpose)] = float(row.get(key) or 0.0)
    return values


def _purpose_token_metric(llm_summary: Any, key: str) -> dict[str, float]:
    values: dict[str, float] = {}
    if not isinstance(llm_summary, dict):
        return values
    by_purpose = llm_summary.get("by_purpose")
    if not isinstance(by_purpose, dict):
        return values
    for purpose, row in by_purpose.items():
        tokens = row.get("token_totals") if isinstance(row, dict) else {}
        if isinstance(tokens, dict):
            values[str(purpose)] = float(tokens.get(key) or 0.0)
    return values


def _build_pairwise_comparisons(summaries: list[ReportSummary]) -> list[dict[str, Any]]:
    if len(summaries) < 2:
        return []
    baseline = summaries[0]
    comparisons = []
    for summary in summaries[1:]:
        baseline_failed = set(baseline.failed_question_ids)
        current_failed = set(summary.failed_question_ids)
        comparisons.append(
            {
                "baseline": baseline.label,
                "candidate": summary.label,
                "pass_rate_delta": summary.pass_rate - baseline.pass_rate,
                "duration_seconds_delta": summary.duration_seconds
                - baseline.duration_seconds,
                "structured_retries_delta": summary.structured_retries
                - baseline.structured_retries,
                "new_failures": sorted(current_failed - baseline_failed),
                "recovered_failures": sorted(baseline_failed - current_failed),
            }
        )
    return comparisons


def _aggregate_summaries(
    summaries: list[ReportSummary], *, label: str
) -> ReportSummary:
    if not summaries:
        raise ValueError("Cannot aggregate an empty summary list")
    total_questions = sum(summary.total_questions for summary in summaries)
    total_passed = sum(summary.total_passed for summary in summaries)
    duration_seconds = sum(summary.duration_seconds for summary in summaries)
    failed_ids = tuple(
        sorted(
            question_id
            for summary in summaries
            for question_id in summary.failed_question_ids
        )
    )
    return ReportSummary(
        label=label,
        path=";".join(summary.path for summary in summaries),
        benchmark_name=summaries[0].benchmark_name,
        total_questions=total_questions,
        total_passed=total_passed,
        pass_rate=total_passed / total_questions if total_questions else 0.0,
        duration_seconds=duration_seconds,
        failed_question_ids=failed_ids,
        structured_retries=sum(summary.structured_retries for summary in summaries),
        structured_retries_by_purpose=_sum_dicts(
            summary.structured_retries_by_purpose for summary in summaries
        ),
        calls_by_purpose=_sum_dicts(summary.calls_by_purpose for summary in summaries),
        input_tokens_by_purpose=_sum_float_dicts(
            summary.input_tokens_by_purpose for summary in summaries
        ),
        output_tokens_by_purpose=_sum_float_dicts(
            summary.output_tokens_by_purpose for summary in summaries
        ),
        latency_ms_by_purpose=_sum_float_dicts(
            summary.latency_ms_by_purpose for summary in summaries
        ),
        warning_counts=_sum_dicts(summary.warning_counts for summary in summaries),
        diagnosis_counts=_sum_dicts(summary.diagnosis_counts for summary in summaries),
        sufficiency_counts=_sum_dicts(
            summary.sufficiency_counts for summary in summaries
        ),
        retrieval_time_ms_mean=_weighted_optional_mean(
            (summary.retrieval_time_ms_mean, summary.total_questions)
            for summary in summaries
        ),
        memories_used_mean=_weighted_optional_mean(
            (summary.memories_used_mean, summary.total_questions)
            for summary in summaries
        ),
    )


def _sum_dicts(dicts: Any) -> dict[str, int]:
    total: Counter[str] = Counter()
    for value in dicts:
        total.update({key: int(raw) for key, raw in value.items()})
    return dict(total)


def _sum_float_dicts(dicts: Any) -> dict[str, float]:
    total: defaultdict[str, float] = defaultdict(float)
    for value in dicts:
        for key, raw in value.items():
            total[key] += float(raw)
    return dict(total)


def _weighted_optional_mean(values: Any) -> float | None:
    weighted_total = 0.0
    weight_total = 0
    for value, weight in values:
        if value is None:
            continue
        weighted_total += float(value) * int(weight)
        weight_total += int(weight)
    if weight_total == 0:
        return None
    return weighted_total / weight_total


def _to_markdown(summaries: list[ReportSummary]) -> str:
    lines = [
        "# Benchmark Optimization Summary",
        "",
        "| Label | Benchmark | Passed | Pass rate | Duration | JSON retries | Failed IDs |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        failed = ", ".join(summary.failed_question_ids) or "-"
        lines.append(
            "| {label} | {benchmark} | {passed:g}/{total} | {rate:.4f} | "
            "{duration:.1f}s | {retries} | {failed} |".format(
                label=summary.label,
                benchmark=summary.benchmark_name,
                passed=summary.total_passed,
                total=summary.total_questions,
                rate=summary.pass_rate,
                duration=summary.duration_seconds,
                retries=summary.structured_retries,
                failed=failed,
            )
        )
    lines.append("")
    lines.append("## Hot Purposes")
    lines.append("")
    for summary in summaries:
        lines.append(f"### {summary.label}")
        lines.append("")
        lines.append("| Purpose | Calls | Input tokens | Output tokens | Latency ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        purposes = sorted(
            summary.latency_ms_by_purpose,
            key=lambda purpose: summary.latency_ms_by_purpose[purpose],
            reverse=True,
        )[:8]
        for purpose in purposes:
            lines.append(
                "| {purpose} | {calls:g} | {input:g} | {output:g} | {latency:.1f} |".format(
                    purpose=purpose,
                    calls=summary.calls_by_purpose.get(purpose, 0),
                    input=summary.input_tokens_by_purpose.get(purpose, 0.0),
                    output=summary.output_tokens_by_purpose.get(purpose, 0.0),
                    latency=summary.latency_ms_by_purpose.get(purpose, 0.0),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(raw or 0) for key, raw in value.items()}


def _count_optional(counter: Counter[str], value: Any) -> None:
    if isinstance(value, str) and value:
        counter[value] += 1


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
