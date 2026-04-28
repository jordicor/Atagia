"""Failure-focused custody reports for Atagia-bench artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaQuestionResult
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.numeric_summary import summarize_numeric_values


class AtagiaBenchFailedQuestionCustody(BaseModel):
    """One failed Atagia-bench question with retrieval custody context."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    conversation_id: str
    persona_id: str
    category_tags: list[str] = Field(default_factory=list)
    question_text: str
    ground_truth: str
    prediction: str
    grader_name: str
    grade_reason: str
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)
    diagnosis_bucket: str
    sufficiency_diagnostic: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_evidence_memory_ids: list[str] = Field(default_factory=list)
    retrieval_custody: list[dict[str, Any]] = Field(default_factory=list)


class AtagiaBenchFailedQuestionCustodyReport(BaseModel):
    """Aggregated failure-custody report for an Atagia-bench run."""

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
    failures: list[AtagiaBenchFailedQuestionCustody] = Field(default_factory=list)


def build_failed_question_custody_report(
    report: AtagiaBenchReport,
    *,
    source_report: str,
) -> AtagiaBenchFailedQuestionCustodyReport:
    """Build a failure-focused custody report from an Atagia-bench report."""
    failures: list[AtagiaBenchFailedQuestionCustody] = []
    diagnosis_counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()

    for result in report.per_question:
        if result.grade.passed:
            continue
        diagnosis = str(result.trace.get("diagnosis_bucket") or "unknown")
        sufficiency = str(result.trace.get("sufficiency_diagnostic") or "unknown")
        diagnosis_counts[diagnosis] += 1
        sufficiency_counts[sufficiency] += 1
        failures.append(
            AtagiaBenchFailedQuestionCustody(
                question_id=result.question_id,
                conversation_id=result.conversation_id,
                persona_id=result.persona_id,
                category_tags=list(result.category_tags),
                question_text=result.question_text,
                ground_truth=result.ground_truth,
                prediction=result.prediction,
                grader_name=result.grade.grader_name,
                grade_reason=result.grade.reason,
                memories_used=result.memories_used,
                retrieval_time_ms=result.retrieval_time_ms,
                diagnosis_bucket=diagnosis,
                sufficiency_diagnostic=sufficiency,
                evidence_turn_ids=list(result.evidence_turn_ids),
                evidence_memory_ids=_trace_list(result, "evidence_memory_ids"),
                selected_memory_ids=_trace_list(result, "selected_memory_ids"),
                selected_evidence_memory_ids=_trace_list(
                    result,
                    "selected_evidence_memory_ids",
                ),
                retrieval_custody=_trace_dict_list(result, "retrieval_custody"),
            )
        )

    return AtagiaBenchFailedQuestionCustodyReport(
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
    report: AtagiaBenchFailedQuestionCustodyReport,
    output_path: str | Path,
) -> Path:
    """Persist a failed-question custody report as JSON."""
    destination = Path(output_path).expanduser()
    return write_json_atomic(destination, report.model_dump(mode="json"))


def _trace_list(result: AtagiaQuestionResult, field_name: str) -> list[str]:
    value = result.trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _trace_dict_list(result: AtagiaQuestionResult, field_name: str) -> list[dict[str, Any]]:
    value = result.trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]
