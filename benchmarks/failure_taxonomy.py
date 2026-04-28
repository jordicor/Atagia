"""Text-free benchmark failure taxonomy sidecar reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaQuestionResult
from benchmarks.base import BenchmarkReport, QuestionResult
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.numeric_summary import summarize_numeric_values

TaxonomyBucket = Literal[
    "evidence_mapping",
    "extraction",
    "governance_or_confirmation",
    "candidate_generation",
    "composition",
    "retrieval_or_ranking",
    "answer_or_grading",
    "unknown",
]

_DIAGNOSIS_BUCKETS: dict[str, TaxonomyBucket] = {
    "evidence_mapping_missing": "evidence_mapping",
    "missing_extraction": "extraction",
    "memory_not_active": "governance_or_confirmation",
    "retrieval_no_candidates": "candidate_generation",
    "composition_selected_none": "composition",
    "retrieval_or_ranking_miss": "retrieval_or_ranking",
    "answer_policy_or_grading": "answer_or_grading",
}

_SUFFICIENCY_BUCKETS: dict[str, TaxonomyBucket] = {
    "missing_memory_extraction": "extraction",
    "unsafe_or_requires_confirmation": "governance_or_confirmation",
    "missing_raw_evidence": "candidate_generation",
    "missing_artifact_support": "candidate_generation",
    "retrieval_insufficient": "retrieval_or_ranking",
    "answer_or_judge_issue": "answer_or_grading",
}

_KNOWN_DIAGNOSIS = frozenset([*_DIAGNOSIS_BUCKETS, "passed"])
_KNOWN_SUFFICIENCY = frozenset([*_SUFFICIENCY_BUCKETS, "retrieval_sufficient"])


class FailureTaxonomyItem(BaseModel):
    """One failed benchmark item classified without raw benchmark text."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    conversation_id: str
    persona_id: str | None = None
    category: int | None = None
    category_tags: list[str] = Field(default_factory=list)
    diagnosis_bucket: str
    sufficiency_diagnostic: str
    taxonomy_bucket: TaxonomyBucket
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)
    evidence_turn_count: int = Field(ge=0)
    evidence_memory_count: int = Field(ge=0)
    selected_memory_count: int = Field(ge=0)
    selected_evidence_memory_count: int = Field(ge=0)
    retrieval_custody_summary: dict[str, Any] = Field(default_factory=dict)


class FailureTaxonomyReport(BaseModel):
    """Aggregated failure taxonomy for one benchmark report."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    source_report: str
    source_report_sha256: str | None = None
    generated_at: str
    total_questions: int = Field(ge=0)
    total_failed_questions: int = Field(ge=0)
    taxonomy_counts: dict[str, int] = Field(default_factory=dict)
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)
    sufficiency_counts: dict[str, int] = Field(default_factory=dict)
    memories_used: dict[str, float | int | None] = Field(
        default_factory=lambda: summarize_numeric_values([])
    )
    retrieval_time_ms: dict[str, float | int | None] = Field(
        default_factory=lambda: summarize_numeric_values([])
    )
    retrieval_custody_summary: dict[str, Any] = Field(default_factory=dict)
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    items: list[FailureTaxonomyItem] = Field(default_factory=list)


class _NormalizedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    question_id: str
    conversation_id: str
    persona_id: str | None = None
    category: int | None = None
    category_tags: list[str] = Field(default_factory=list)
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)
    trace: dict[str, Any] = Field(default_factory=dict)


def build_failure_taxonomy_report(
    report: BenchmarkReport | AtagiaBenchReport,
    *,
    source_report: str,
) -> FailureTaxonomyReport:
    """Build a failure taxonomy report from an existing benchmark report."""
    normalized_results = list(_iter_results(report))
    failed_results = [result for result in normalized_results if not result.passed]
    items = [_build_item(result) for result in failed_results]
    taxonomy_counts = Counter(item.taxonomy_bucket for item in items)
    diagnosis_counts = Counter(item.diagnosis_bucket for item in items)
    sufficiency_counts = Counter(item.sufficiency_diagnostic for item in items)

    return FailureTaxonomyReport(
        benchmark_name=str(getattr(report, "benchmark_name")),
        source_report=source_report,
        source_report_sha256=sha256_file_if_exists(source_report),
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_questions=len(normalized_results),
        total_failed_questions=len(items),
        taxonomy_counts=dict(sorted(taxonomy_counts.items())),
        diagnosis_counts=dict(sorted(diagnosis_counts.items())),
        sufficiency_counts=dict(sorted(sufficiency_counts.items())),
        memories_used=summarize_numeric_values(item.memories_used for item in items),
        retrieval_time_ms=summarize_numeric_values(item.retrieval_time_ms for item in items),
        retrieval_custody_summary=summarize_retrieval_custody(
            _trace_dict_list(result.trace, "retrieval_custody")
            for result in failed_results
        ),
        run_metadata=_run_metadata(report),
        items=items,
    )


def save_failure_taxonomy_report(
    report: FailureTaxonomyReport,
    output_path: str | Path,
) -> Path:
    """Persist a failure taxonomy report as JSON."""
    destination = Path(output_path).expanduser()
    return write_json_atomic(destination, report.model_dump(mode="json"))


def format_failure_taxonomy_summary(report: FailureTaxonomyReport | dict[str, Any] | object) -> str:
    """Return a compact terminal-friendly taxonomy summary."""
    if isinstance(report, FailureTaxonomyReport):
        counts = report.taxonomy_counts
        failed = report.total_failed_questions
    elif isinstance(report, dict):
        counts = report.get("taxonomy_counts", {})
        failed = int(report.get("total_failed_questions") or 0)
    else:
        return "Failure taxonomy: unavailable"

    if not isinstance(counts, dict):
        return "Failure taxonomy: unavailable"
    if not counts:
        return f"Failure taxonomy: failed={failed} buckets=none"
    bucket_text = " ".join(
        f"{bucket}={count}"
        for bucket, count in sorted(counts.items())
    )
    return f"Failure taxonomy: failed={failed} buckets={bucket_text}"


def failure_taxonomy_manifest_summary(report: FailureTaxonomyReport) -> dict[str, Any]:
    """Return the small taxonomy summary stored in run manifests."""
    return {
        "total_failed_questions": report.total_failed_questions,
        "taxonomy_counts": dict(report.taxonomy_counts),
        "diagnosis_counts": dict(report.diagnosis_counts),
        "sufficiency_counts": dict(report.sufficiency_counts),
    }


def _iter_results(report: BenchmarkReport | AtagiaBenchReport) -> Iterable[_NormalizedResult]:
    if isinstance(report, BenchmarkReport):
        for conversation in report.conversations:
            for result in conversation.results:
                yield _from_locomo_result(conversation.conversation_id, result)
        return
    if isinstance(report, AtagiaBenchReport):
        for result in report.per_question:
            yield _from_atagia_result(result)
        return
    raise TypeError(f"Unsupported report type: {type(report)!r}")


def _from_locomo_result(conversation_id: str, result: QuestionResult) -> _NormalizedResult:
    return _NormalizedResult(
        passed=result.score_result.score != 0,
        question_id=result.question.question_id,
        conversation_id=conversation_id,
        category=result.question.category,
        memories_used=result.memories_used,
        retrieval_time_ms=result.retrieval_time_ms,
        trace=dict(result.trace or {}),
    )


def _from_atagia_result(result: AtagiaQuestionResult) -> _NormalizedResult:
    return _NormalizedResult(
        passed=result.grade.passed,
        question_id=result.question_id,
        conversation_id=result.conversation_id,
        persona_id=result.persona_id,
        category_tags=list(result.category_tags),
        memories_used=result.memories_used,
        retrieval_time_ms=result.retrieval_time_ms,
        trace=dict(result.trace or {}),
    )


def _build_item(result: _NormalizedResult) -> FailureTaxonomyItem:
    diagnosis = _known_value(result.trace.get("diagnosis_bucket"), _KNOWN_DIAGNOSIS)
    sufficiency = _known_value(result.trace.get("sufficiency_diagnostic"), _KNOWN_SUFFICIENCY)
    custody = _trace_dict_list(result.trace, "retrieval_custody")
    return FailureTaxonomyItem(
        question_id=result.question_id,
        conversation_id=result.conversation_id,
        persona_id=result.persona_id,
        category=result.category,
        category_tags=list(result.category_tags),
        diagnosis_bucket=diagnosis,
        sufficiency_diagnostic=sufficiency,
        taxonomy_bucket=_taxonomy_bucket(diagnosis, sufficiency),
        memories_used=result.memories_used,
        retrieval_time_ms=result.retrieval_time_ms,
        evidence_turn_count=len(_trace_list(result.trace, "evidence_turn_ids")),
        evidence_memory_count=len(_trace_list(result.trace, "evidence_memory_ids")),
        selected_memory_count=len(_trace_list(result.trace, "selected_memory_ids")),
        selected_evidence_memory_count=len(_trace_list(result.trace, "selected_evidence_memory_ids")),
        retrieval_custody_summary=summarize_retrieval_custody([custody]),
    )


def _taxonomy_bucket(diagnosis: str, sufficiency: str) -> TaxonomyBucket:
    if diagnosis in _DIAGNOSIS_BUCKETS:
        return _DIAGNOSIS_BUCKETS[diagnosis]
    if sufficiency in _SUFFICIENCY_BUCKETS:
        return _SUFFICIENCY_BUCKETS[sufficiency]
    return "unknown"


def _known_value(value: Any, known_values: frozenset[str]) -> str:
    normalized = str(value).strip() if value is not None else ""
    return normalized if normalized in known_values else "unknown"


def _trace_list(trace: dict[str, Any], field_name: str) -> list[str]:
    value = trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _trace_dict_list(trace: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    value = trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]


def _run_metadata(report: BenchmarkReport | AtagiaBenchReport) -> dict[str, Any]:
    if isinstance(report, BenchmarkReport):
        return {
            "ablation_config": report.ablation_config,
            "model_info": dict(report.model_info),
        }
    return {
        "config": dict(report.config),
    }
