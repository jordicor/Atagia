"""Compare LoCoMo benchmark reports produced by different ingest modes."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.base import BenchmarkReport, QuestionResult
from benchmarks.json_artifacts import write_json_atomic


def load_labeled_report(raw_value: str) -> tuple[str, Path, BenchmarkReport]:
    if "=" not in raw_value:
        raise ValueError("--report must use LABEL=PATH")
    label, raw_path = (part.strip() for part in raw_value.split("=", 1))
    if not label:
        raise ValueError("--report label cannot be empty")
    path = Path(raw_path).expanduser()
    return label, path, BenchmarkReport.model_validate_json(
        path.read_text(encoding="utf-8"),
    )


def build_mode_comparison(
    reports: dict[str, BenchmarkReport],
    *,
    report_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    paths = report_paths or {}
    question_maps = {
        label: _question_result_map(report)
        for label, report in reports.items()
    }
    return {
        "comparison_kind": "locomo_mode_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": {
            label: _report_summary(
                report,
                report_path=paths.get(label),
            )
            for label, report in sorted(reports.items())
        },
        "pairwise": [
            _pairwise_comparison(
                before_label,
                after_label,
                reports[before_label],
                reports[after_label],
                question_maps[before_label],
                question_maps[after_label],
            )
            for before_label, after_label in itertools.combinations(sorted(reports), 2)
        ],
    }


def _report_summary(
    report: BenchmarkReport,
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    model_info = report.model_info or {}
    return {
        "report_path": str(report_path) if report_path is not None else None,
        "report_sha256": (
            sha256_file_if_exists(report_path)
            if report_path is not None
            else None
        ),
        "overall_accuracy": report.overall_accuracy,
        "total_correct": report.total_correct,
        "total_questions": report.total_questions,
        "duration_seconds": report.duration_seconds,
        "ingest_mode": model_info.get("ingest_mode"),
        "requested_ingest_mode": model_info.get("requested_ingest_mode"),
        "effective_ingest_mode": model_info.get("effective_ingest_mode"),
        "parallel_conversations": model_info.get("parallel_conversations"),
        "parallel_questions": model_info.get("parallel_questions"),
        "adaptive_parallel_questions": model_info.get("adaptive_parallel_questions"),
        "failure_stage_counts": model_info.get("failure_stage_counts", {}),
        "warning_counts": model_info.get("warning_counts", {}),
    }


def _pairwise_comparison(
    before_label: str,
    after_label: str,
    before_report: BenchmarkReport,
    after_report: BenchmarkReport,
    before_questions: dict[tuple[str, str], QuestionResult],
    after_questions: dict[tuple[str, str], QuestionResult],
) -> dict[str, Any]:
    common_keys = sorted(set(before_questions) & set(after_questions))
    deltas = [
        _question_delta(key, before_questions[key], after_questions[key])
        for key in common_keys
        if _question_changed(before_questions[key], after_questions[key])
    ]
    improved = sum(1 for item in deltas if item["score_delta"] > 0)
    regressed = sum(1 for item in deltas if item["score_delta"] < 0)
    return {
        "before": before_label,
        "after": after_label,
        "accuracy_delta": after_report.overall_accuracy - before_report.overall_accuracy,
        "correct_delta": after_report.total_correct - before_report.total_correct,
        "question_count_delta": after_report.total_questions - before_report.total_questions,
        "common_questions": len(common_keys),
        "missing_from_before": _format_question_keys(
            sorted(set(after_questions) - set(before_questions))
        ),
        "missing_from_after": _format_question_keys(
            sorted(set(before_questions) - set(after_questions))
        ),
        "changed_questions": len(deltas),
        "improved_questions": improved,
        "regressed_questions": regressed,
        "per_question_deltas": deltas,
    }


def _question_result_map(
    report: BenchmarkReport,
) -> dict[tuple[str, str], QuestionResult]:
    return {
        (conversation.conversation_id, result.question.question_id): result
        for conversation in report.conversations
        for result in conversation.results
    }


def _question_changed(before: QuestionResult, after: QuestionResult) -> bool:
    return (
        before.score_result.score != after.score_result.score
        or before.prediction != after.prediction
        or _diagnosis(before) != _diagnosis(after)
    )


def _question_delta(
    key: tuple[str, str],
    before: QuestionResult,
    after: QuestionResult,
) -> dict[str, Any]:
    conversation_id, question_id = key
    return {
        "conversation_id": conversation_id,
        "question_id": question_id,
        "category": after.question.category,
        "question": after.question.question_text,
        "score_before": before.score_result.score,
        "score_after": after.score_result.score,
        "score_delta": after.score_result.score - before.score_result.score,
        "diagnosis_before": _diagnosis(before),
        "diagnosis_after": _diagnosis(after),
        "prediction_changed": before.prediction != after.prediction,
    }


def _diagnosis(result: QuestionResult) -> str | None:
    trace = result.trace if isinstance(result.trace, dict) else {}
    diagnosis = trace.get("diagnosis_bucket")
    return str(diagnosis) if diagnosis is not None else None


def _format_question_keys(keys: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "conversation_id": conversation_id,
            "question_id": question_id,
        }
        for conversation_id, question_id in keys
    ]


def save_mode_comparison(comparison: dict[str, Any], output_path: str | Path) -> Path:
    return write_json_atomic(Path(output_path).expanduser(), comparison)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare LoCoMo mode reports.")
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Labeled report to compare. Repeat at least twice.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if len(args.report) < 2:
        parser.error("at least two --report values are required")
    loaded = [load_labeled_report(raw_value) for raw_value in args.report]
    reports = {label: report for label, _, report in loaded}
    if len(reports) != len(loaded):
        parser.error("--report labels must be unique")
    report_paths = {label: path for label, path, _ in loaded}
    output_path = save_mode_comparison(
        build_mode_comparison(reports, report_paths=report_paths),
        args.output,
    )
    print(json.dumps({"comparison_path": str(output_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
