"""Failure-focused retrieval custody reports for benchmark artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.base import BenchmarkReport, QuestionResult
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.numeric_summary import summarize_numeric_values
from benchmarks.report_diff import load_benchmark_report


class FailedQuestionCustody(BaseModel):
    """One failed benchmark question with retrieval custody context."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    conversation_id: str
    category: int = Field(ge=1)
    question_text: str
    ground_truth: str
    prediction: str
    judge_reasoning: str
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)
    diagnosis_bucket: str
    sufficiency_diagnostic: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_evidence_memory_ids: list[str] = Field(default_factory=list)
    retrieval_custody: list[dict[str, Any]] = Field(default_factory=list)


class FailedQuestionCustodyReport(BaseModel):
    """Aggregated failure-custody report for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    source_report: str
    source_report_sha256: str | None = None
    generated_at: str
    total_failed_questions: int = Field(ge=0)
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)
    sufficiency_counts: dict[str, int] = Field(default_factory=dict)
    memories_used: dict[str, float | int | None] = Field(
        default_factory=lambda: summarize_numeric_values([])
    )
    retrieval_time_ms: dict[str, float | int | None] = Field(
        default_factory=lambda: summarize_numeric_values([])
    )
    retrieval_custody_summary: dict[str, Any] = Field(default_factory=dict)
    failures: list[FailedQuestionCustody] = Field(default_factory=list)


def build_failed_question_custody_report(
    report: BenchmarkReport,
    *,
    source_report: str,
) -> FailedQuestionCustodyReport:
    """Build a failure-focused custody report from a benchmark report."""
    failures: list[FailedQuestionCustody] = []
    diagnosis_counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()

    for conversation in report.conversations:
        for result in conversation.results:
            if result.score_result.score != 0:
                continue
            trace = result.trace or {}
            diagnosis = str(trace.get("diagnosis_bucket") or "unknown")
            sufficiency = str(trace.get("sufficiency_diagnostic") or "unknown")
            diagnosis_counts[diagnosis] += 1
            sufficiency_counts[sufficiency] += 1
            failures.append(
                FailedQuestionCustody(
                    question_id=result.question.question_id,
                    conversation_id=conversation.conversation_id,
                    category=result.question.category,
                    question_text=result.question.question_text,
                    ground_truth=result.question.ground_truth,
                    prediction=result.prediction,
                    judge_reasoning=result.score_result.reasoning,
                    memories_used=result.memories_used,
                    retrieval_time_ms=result.retrieval_time_ms,
                    diagnosis_bucket=diagnosis,
                    sufficiency_diagnostic=sufficiency,
                    evidence_turn_ids=_trace_list(result, "evidence_turn_ids"),
                    evidence_memory_ids=_trace_list(result, "evidence_memory_ids"),
                    selected_memory_ids=_trace_list(result, "selected_memory_ids"),
                    selected_evidence_memory_ids=_trace_list(
                        result,
                        "selected_evidence_memory_ids",
                    ),
                    retrieval_custody=_trace_dict_list(result, "retrieval_custody"),
                )
            )

    return FailedQuestionCustodyReport(
        benchmark_name=report.benchmark_name,
        source_report=source_report,
        source_report_sha256=sha256_file_if_exists(source_report),
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_failed_questions=len(failures),
        diagnosis_counts=dict(diagnosis_counts),
        sufficiency_counts=dict(sufficiency_counts),
        memories_used=summarize_numeric_values(
            failure.memories_used
            for failure in failures
        ),
        retrieval_time_ms=summarize_numeric_values(
            failure.retrieval_time_ms
            for failure in failures
        ),
        retrieval_custody_summary=summarize_retrieval_custody(
            failure.retrieval_custody
            for failure in failures
        ),
        failures=failures,
    )


def save_failed_question_custody_report(
    report: FailedQuestionCustodyReport,
    output_path: str | Path,
) -> Path:
    """Persist a failed-question custody report as JSON."""
    destination = Path(output_path).expanduser()
    return write_json_atomic(destination, report.model_dump(mode="json"))


def _trace_list(result: QuestionResult, field_name: str) -> list[str]:
    value = result.trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _trace_dict_list(result: QuestionResult, field_name: str) -> list[dict[str, Any]]:
    value = result.trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Benchmark report JSON")
    parser.add_argument("--output", required=True, help="Destination custody report JSON")
    return parser


def main() -> None:
    """Generate a failed-question custody report from a benchmark report."""
    args = _build_parser().parse_args()
    source_report = str(Path(args.report).expanduser())
    report = load_benchmark_report(source_report)
    custody = build_failed_question_custody_report(report, source_report=source_report)
    output_path = save_failed_question_custody_report(custody, args.output)
    print(
        f"Saved failed-question custody report to {output_path} "
        f"({custody.total_failed_questions} failures)",
        flush=True,
    )


if __name__ == "__main__":
    main()
