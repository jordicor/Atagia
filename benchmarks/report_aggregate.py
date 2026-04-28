"""Combine benchmark report shards into one auditable report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Literal

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.base import BenchmarkReport, ConversationReport, QuestionResult
from benchmarks.custody_report import (
    build_failed_question_custody_report,
    save_failed_question_custody_report,
)
from benchmarks.custody_summary import format_retrieval_custody_summary, summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.numeric_summary import summarize_numeric_values
from benchmarks.report_diff import load_benchmark_report

DuplicateStrategy = Literal["first", "last", "error"]
DEFAULT_WARNING_COUNT_KEYS = (
    "failed_questions",
    "degraded_retrievals",
    "need_detection_degraded",
    "retrieval_no_candidates",
    "composition_selected_none",
    "missing_extraction",
    "memory_not_active",
    "retrieval_or_ranking_miss",
    "answer_policy_or_grading",
    "structured_output_retries",
    "synthesis_preservation",
    "applicability_fallback",
    "provider_rate_limits",
    "tracebacks",
    "failed_worker_jobs",
)


def build_combined_report(
    reports: list[BenchmarkReport],
    *,
    source_paths: list[str],
    source_hashes: list[str | None] | None = None,
    duplicate_strategy: DuplicateStrategy = "last",
) -> BenchmarkReport:
    """Build one report from multiple completed benchmark report shards."""
    if not reports:
        raise ValueError("At least one report is required")
    if len(reports) != len(source_paths):
        raise ValueError("source_paths must match reports length")
    if source_hashes is not None and len(source_hashes) != len(source_paths):
        raise ValueError("source_hashes must match source_paths length")

    conversations: dict[str, dict[str, QuestionResult]] = {}
    conversation_sources: dict[str, list[str]] = {}
    duplicate_questions: list[str] = []

    for report, source_path in zip(reports, source_paths, strict=True):
        for conversation in report.conversations:
            result_map = conversations.setdefault(conversation.conversation_id, {})
            source_list = conversation_sources.setdefault(conversation.conversation_id, [])
            if source_path not in source_list:
                source_list.append(source_path)
            for result in conversation.results:
                question_id = result.question.question_id
                if question_id in result_map:
                    duplicate_questions.append(question_id)
                    if duplicate_strategy == "error":
                        raise ValueError(f"Duplicate benchmark question: {question_id}")
                    if duplicate_strategy == "first":
                        continue
                result_map[question_id] = result

    combined_conversations: list[ConversationReport] = []
    for conversation_id in sorted(conversations):
        results = list(conversations[conversation_id].values())
        total_correct = sum(result.score_result.score for result in results)
        total_questions = len(results)
        combined_conversations.append(
            ConversationReport(
                conversation_id=conversation_id,
                results=results,
                accuracy=_accuracy(total_correct, total_questions),
                category_breakdown=_category_breakdown(results),
                metadata={
                    "source_reports": conversation_sources[conversation_id],
                    "duplicate_question_ids": sorted(set(duplicate_questions)),
                    "duplicate_strategy": duplicate_strategy,
                },
            )
        )

    total_questions = sum(len(conversation.results) for conversation in combined_conversations)
    total_correct = sum(
        result.score_result.score
        for conversation in combined_conversations
        for result in conversation.results
    )
    return BenchmarkReport(
        benchmark_name=reports[0].benchmark_name,
        overall_accuracy=_accuracy(total_correct, total_questions),
        category_breakdown=_category_breakdown(
            result
            for conversation in combined_conversations
            for result in conversation.results
        ),
        conversations=combined_conversations,
        total_questions=total_questions,
        total_correct=total_correct,
        ablation_config=_shared_or_none([report.ablation_config for report in reports]),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_info={
            "aggregate": True,
            "source_reports": source_paths,
            "duplicate_strategy": duplicate_strategy,
            "duplicate_question_ids": sorted(set(duplicate_questions)),
            "source_model_info": [
                report.model_info
                for report in reports
            ],
            "warning_counts": _sum_warning_counts(reports),
            "source_report_sha256": _source_hash_map(source_paths, source_hashes),
            "source_selection": _source_selection_summary(reports, source_paths),
            "source_accuracy_summary": _source_accuracy_summary(reports),
            "retained_db_accuracy_summary": _retained_db_accuracy_summary(
                reports,
                source_paths,
            ),
            "diagnosis_bucket_counts": _trace_field_counts(reports, "diagnosis_bucket"),
            "sufficiency_diagnostic_counts": _trace_field_counts(
                reports,
                "sufficiency_diagnostic",
            ),
            "retrieval_custody_summary": _retrieval_custody_summary(reports),
        },
        duration_seconds=sum(report.duration_seconds for report in reports),
    )


def save_combined_report(report: BenchmarkReport, output_path: str | Path) -> Path:
    """Persist a combined benchmark report as JSON."""
    destination = Path(output_path).expanduser()
    return write_json_atomic(destination, report.model_dump(mode="json"))


def build_combined_run_manifest(
    report: BenchmarkReport,
    *,
    report_path: str | Path,
    custody_path: str | Path | None = None,
) -> dict[str, object]:
    """Build an auditable manifest for one combined benchmark report."""
    report_file = Path(report_path).expanduser()
    custody_file = Path(custody_path).expanduser() if custody_path is not None else None
    model_info = report.model_info
    return {
        "manifest_kind": "combined_benchmark_run_manifest",
        "benchmark_name": report.benchmark_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_timestamp": report.timestamp,
        "report_path": str(report_file),
        "report_sha256": sha256_file_if_exists(report_file),
        "custody_path": str(custody_file) if custody_file is not None else None,
        "custody_sha256": (
            sha256_file_if_exists(custody_file)
            if custody_file is not None
            else None
        ),
        "source_reports": model_info.get("source_reports", []),
        "source_report_sha256": model_info.get("source_report_sha256", {}),
        "source_selection": model_info.get("source_selection", {}),
        "duplicate_strategy": model_info.get("duplicate_strategy"),
        "duplicate_question_ids": model_info.get("duplicate_question_ids", []),
        "warning_counts": model_info.get("warning_counts", {}),
        "source_accuracy_summary": model_info.get("source_accuracy_summary", {}),
        "retained_db_accuracy_summary": model_info.get("retained_db_accuracy_summary", {}),
        "diagnosis_bucket_counts": model_info.get("diagnosis_bucket_counts", {}),
        "sufficiency_diagnostic_counts": model_info.get("sufficiency_diagnostic_counts", {}),
        "retrieval_custody_summary": model_info.get("retrieval_custody_summary", {}),
        "result_summary": {
            "overall_accuracy": report.overall_accuracy,
            "total_correct": report.total_correct,
            "total_questions": report.total_questions,
            "duration_seconds": report.duration_seconds,
            "retrieval_time_ms": _question_result_numeric_summary(
                report,
                "retrieval_time_ms",
            ),
            "memories_used": _question_result_numeric_summary(
                report,
                "memories_used",
            ),
            "conversation_ids": [
                conversation.conversation_id
                for conversation in report.conversations
            ],
        },
        "git": _git_state(),
    }


def save_combined_run_manifest(
    report: BenchmarkReport,
    *,
    report_path: str | Path,
    custody_path: str | Path | None = None,
) -> Path:
    """Persist a combined benchmark run manifest beside the saved report."""
    destination = _default_manifest_output_path(Path(report_path).expanduser())
    return write_json_atomic(
        destination,
        build_combined_run_manifest(
            report,
            report_path=report_path,
            custody_path=custody_path,
        ),
    )


def format_combined_report_summary(
    report: BenchmarkReport,
    *,
    report_path: str | Path,
    custody_path: str | Path,
    manifest_path: str | Path,
) -> str:
    """Return a compact terminal summary for a combined benchmark report."""
    model_info = report.model_info
    source_reports = model_info.get("source_reports", [])
    source_count = len(source_reports) if isinstance(source_reports, list) else 0
    duplicate_question_ids = model_info.get("duplicate_question_ids", [])
    duplicate_count = (
        len(duplicate_question_ids)
        if isinstance(duplicate_question_ids, list)
        else 0
    )
    lines = [
        "=" * 52,
        "Combined Benchmark Report",
        "=" * 52,
        f"Benchmark: {report.benchmark_name}",
        f"Accuracy: {report.overall_accuracy:.1%} ({report.total_correct}/{report.total_questions})",
        f"Source reports: {source_count}",
        f"Duplicate questions: {duplicate_count}",
        _format_warning_counts(model_info.get("warning_counts")),
        format_retrieval_custody_summary(
            model_info.get("retrieval_custody_summary")
        ),
        f"Report saved to: {report_path}",
        f"Failed-question custody saved to: {custody_path}",
        f"Run manifest saved to: {manifest_path}",
        "=" * 52,
    ]
    return "\n".join(lines)


def _default_manifest_output_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}-run-manifest.json")


def _default_custody_output_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}-failed-custody.json")


def _accuracy(total_correct: int, total_questions: int) -> float:
    if total_questions <= 0:
        return 0.0
    return total_correct / total_questions


def _category_breakdown(results: object) -> dict[int, float]:
    stats: dict[int, dict[str, int]] = {}
    for result in results:
        category = result.question.category
        bucket = stats.setdefault(category, {"correct": 0, "total": 0})
        bucket["correct"] += result.score_result.score
        bucket["total"] += 1
    return {
        category: bucket["correct"] / bucket["total"]
        for category, bucket in sorted(stats.items())
        if bucket["total"] > 0
    }


def _shared_or_none(values: list[object]) -> object | None:
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return None


def _sum_warning_counts(reports: list[BenchmarkReport]) -> dict[str, int]:
    totals: dict[str, int] = {
        key: 0
        for key in DEFAULT_WARNING_COUNT_KEYS
    }
    for report in reports:
        warning_counts = report.model_info.get("warning_counts", {})
        has_failed_question_count = False
        if isinstance(warning_counts, dict):
            has_failed_question_count = "failed_questions" in warning_counts
            for key, value in warning_counts.items():
                try:
                    amount = int(value)
                except (TypeError, ValueError):
                    continue
                totals[str(key)] = totals.get(str(key), 0) + amount
        if not has_failed_question_count:
            totals["failed_questions"] += max(
                0,
                int(report.total_questions) - int(report.total_correct),
            )
    return totals


def _format_warning_counts(value: object) -> str:
    if not isinstance(value, dict):
        return "Warning counts: unavailable"
    items: list[str] = []
    for key in sorted(value):
        try:
            amount = int(value[key])
        except (TypeError, ValueError):
            continue
        if amount:
            items.append(f"{key}={amount}")
    if not items:
        return "Warning counts: none"
    return "Warning counts: " + ", ".join(items)


def _source_hash_map(
    source_paths: list[str],
    source_hashes: list[str | None] | None,
) -> dict[str, str]:
    if source_hashes is None:
        return {}
    return {
        source_path: source_hash
        for source_path, source_hash in zip(source_paths, source_hashes, strict=True)
        if source_hash is not None
    }


def _source_selection_summary(
    reports: list[BenchmarkReport],
    source_paths: list[str],
) -> dict[str, dict[str, object]]:
    """Map source reports to their benchmark selection metadata when present."""
    selections: dict[str, dict[str, object]] = {}
    for report, source_path in zip(reports, source_paths, strict=True):
        selection = report.model_info.get("selection")
        if isinstance(selection, dict):
            selections[source_path] = dict(selection)
    return selections


def _score_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _question_result_numeric_summary(
    report: BenchmarkReport,
    field_name: str,
) -> dict[str, float | int | None]:
    return summarize_numeric_values(
        getattr(result, field_name)
        for conversation in report.conversations
        for result in conversation.results
    )


def _source_accuracy_summary(reports: list[BenchmarkReport]) -> dict[str, object]:
    """Summarize source report score variance for replay/model comparisons."""
    return {
        "report_count": len(reports),
        "overall_accuracy": _score_summary([report.overall_accuracy for report in reports]),
        "duration_seconds": _score_summary([report.duration_seconds for report in reports]),
        "total_questions": _score_summary(
            [float(report.total_questions) for report in reports]
        ),
    }


def _retained_db_accuracy_summary(
    reports: list[BenchmarkReport],
    source_paths: list[str],
) -> dict[str, dict[str, object]]:
    """Group replay score variance by retained benchmark DB path when present."""
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for report, source_path in zip(reports, source_paths, strict=True):
        for conversation in report.conversations:
            db_path = conversation.metadata.get("benchmark_db_path")
            if not isinstance(db_path, str) or not db_path:
                continue
            correct = sum(result.score_result.score for result in conversation.results)
            samples[db_path].append(
                {
                    "source_report": source_path,
                    "conversation_id": conversation.conversation_id,
                    "accuracy": conversation.accuracy,
                    "total_correct": correct,
                    "total_questions": len(conversation.results),
                }
            )

    return {
        db_path: {
            "sample_count": len(db_samples),
            "source_reports": sorted(
                {
                    str(sample["source_report"])
                    for sample in db_samples
                }
            ),
            "conversation_ids": sorted(
                {
                    str(sample["conversation_id"])
                    for sample in db_samples
                }
            ),
            "accuracy": _score_summary(
                [float(sample["accuracy"]) for sample in db_samples]
            ),
            "total_correct": _score_summary(
                [float(sample["total_correct"]) for sample in db_samples]
            ),
            "total_questions": _score_summary(
                [float(sample["total_questions"]) for sample in db_samples]
            ),
        }
        for db_path, db_samples in sorted(samples.items())
    }


def _trace_field_counts(reports: list[BenchmarkReport], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        for conversation in report.conversations:
            for result in conversation.results:
                trace = result.trace if isinstance(result.trace, dict) else {}
                raw_value = trace.get(field_name)
                value = str(raw_value).strip() if raw_value is not None else ""
                counts[value or "unknown"] += 1
    return dict(sorted(counts.items()))


def _retrieval_custody_summary(reports: list[BenchmarkReport]) -> dict[str, object]:
    return summarize_retrieval_custody(
        result.trace.get("retrieval_custody", [])
        for report in reports
        for conversation in report.conversations
        for result in conversation.results
        if isinstance(result.trace, dict)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Report JSON files to combine")
    parser.add_argument("--output", required=True, help="Destination combined report JSON")
    parser.add_argument(
        "--duplicate-strategy",
        choices=["first", "last", "error"],
        default="last",
        help="How to handle duplicate question ids across report shards",
    )
    return parser


def main() -> None:
    """Combine benchmark report shards from the command line."""
    args = _build_parser().parse_args()
    reports = [load_benchmark_report(path) for path in args.reports]
    combined = build_combined_report(
        reports,
        source_paths=[str(Path(path).expanduser()) for path in args.reports],
        source_hashes=[sha256_file_if_exists(path) for path in args.reports],
        duplicate_strategy=args.duplicate_strategy,
    )
    output_path = save_combined_report(combined, args.output)
    custody_path = save_failed_question_custody_report(
        build_failed_question_custody_report(combined, source_report=str(output_path)),
        _default_custody_output_path(output_path),
    )
    manifest_path = save_combined_run_manifest(
        combined,
        report_path=output_path,
        custody_path=custody_path,
    )
    print(
        format_combined_report_summary(
            combined,
            report_path=output_path,
            custody_path=custody_path,
            manifest_path=manifest_path,
        ),
        flush=True,
    )


def _git_state() -> dict[str, object]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status or "",
    }


if __name__ == "__main__":
    main()
