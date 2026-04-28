"""Per-question diff tooling for benchmark reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.base import BenchmarkReport, ConversationReport, QuestionResult
from benchmarks.custody_summary import format_retrieval_custody_summary, summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic

DiffStatus = Literal["improved", "regressed", "unchanged", "added", "removed"]


class NumericSummary(BaseModel):
    """Small numeric distribution summary for diff diagnostics."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class QuestionDiff(BaseModel):
    """One per-question change between two benchmark reports."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    conversation_id: str
    category: int = Field(ge=1)
    question_text: str
    status: DiffStatus
    before_score: int | None = Field(default=None, ge=0, le=1)
    after_score: int | None = Field(default=None, ge=0, le=1)
    score_delta: int
    before_prediction: str | None = None
    after_prediction: str | None = None
    before_memories_used: int | None = Field(default=None, ge=0)
    after_memories_used: int | None = Field(default=None, ge=0)
    memories_used_delta: int | None = None
    before_retrieval_time_ms: float | None = Field(default=None, ge=0.0)
    after_retrieval_time_ms: float | None = Field(default=None, ge=0.0)
    retrieval_time_delta_ms: float | None = None
    before_reasoning: str | None = None
    after_reasoning: str | None = None
    before_diagnosis_bucket: str | None = None
    after_diagnosis_bucket: str | None = None
    before_sufficiency_diagnostic: str | None = None
    after_sufficiency_diagnostic: str | None = None
    before_selected_memory_ids: list[str] = Field(default_factory=list)
    after_selected_memory_ids: list[str] = Field(default_factory=list)
    before_selected_evidence_memory_ids: list[str] = Field(default_factory=list)
    after_selected_evidence_memory_ids: list[str] = Field(default_factory=list)


class ConversationDiff(BaseModel):
    """Aggregated diff for one benchmark conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    before_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    after_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy_delta: float | None = None
    improved_questions: int = Field(ge=0)
    regressed_questions: int = Field(ge=0)
    unchanged_questions: int = Field(ge=0)
    added_questions: int = Field(ge=0)
    removed_questions: int = Field(ge=0)
    before_retrieval_time_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    after_retrieval_time_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    retrieval_time_delta_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    before_memories_used: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    after_memories_used: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    memories_used_delta: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    question_diffs: list[QuestionDiff] = Field(default_factory=list)


class BenchmarkDiffReport(BaseModel):
    """Serializable before/after benchmark comparison."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    before_label: str
    after_label: str
    before_report_sha256: str | None = None
    after_report_sha256: str | None = None
    before_timestamp: str
    after_timestamp: str
    generated_at: str
    before_overall_accuracy: float = Field(ge=0.0, le=1.0)
    after_overall_accuracy: float = Field(ge=0.0, le=1.0)
    overall_accuracy_delta: float
    before_total_correct: int = Field(ge=0)
    after_total_correct: int = Field(ge=0)
    total_correct_delta: int
    before_total_questions: int = Field(ge=0)
    after_total_questions: int = Field(ge=0)
    total_questions_delta: int
    before_warning_counts: dict[str, int] = Field(default_factory=dict)
    after_warning_counts: dict[str, int] = Field(default_factory=dict)
    warning_count_deltas: dict[str, int] = Field(default_factory=dict)
    before_diagnosis_bucket_counts: dict[str, int] = Field(default_factory=dict)
    after_diagnosis_bucket_counts: dict[str, int] = Field(default_factory=dict)
    diagnosis_bucket_count_deltas: dict[str, int] = Field(default_factory=dict)
    before_sufficiency_diagnostic_counts: dict[str, int] = Field(default_factory=dict)
    after_sufficiency_diagnostic_counts: dict[str, int] = Field(default_factory=dict)
    sufficiency_diagnostic_count_deltas: dict[str, int] = Field(default_factory=dict)
    before_retrieval_custody_summary: dict[str, object] = Field(
        default_factory=lambda: summarize_retrieval_custody([])
    )
    after_retrieval_custody_summary: dict[str, object] = Field(
        default_factory=lambda: summarize_retrieval_custody([])
    )
    category_deltas: dict[int, float] = Field(default_factory=dict)
    improved_questions: int = Field(ge=0)
    regressed_questions: int = Field(ge=0)
    unchanged_questions: int = Field(ge=0)
    added_questions: int = Field(ge=0)
    removed_questions: int = Field(ge=0)
    before_retrieval_time_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    after_retrieval_time_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    retrieval_time_delta_ms: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    before_memories_used: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    after_memories_used: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    memories_used_delta: NumericSummary = Field(default_factory=lambda: _numeric_summary([]))
    conversations: list[ConversationDiff] = Field(default_factory=list)


def load_benchmark_report(path: str | Path) -> BenchmarkReport:
    """Load a benchmark report JSON artifact."""
    report_path = Path(path).expanduser()
    return BenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))


def build_benchmark_diff(
    before: BenchmarkReport,
    after: BenchmarkReport,
    *,
    before_label: str,
    after_label: str,
    before_report_sha256: str | None = None,
    after_report_sha256: str | None = None,
) -> BenchmarkDiffReport:
    """Return a structured per-question diff between two benchmark reports."""
    before_conversations = {conversation.conversation_id: conversation for conversation in before.conversations}
    after_conversations = {conversation.conversation_id: conversation for conversation in after.conversations}
    conversation_ids = sorted(set(before_conversations) | set(after_conversations))
    conversation_diffs: list[ConversationDiff] = []

    improved_questions = 0
    regressed_questions = 0
    unchanged_questions = 0
    added_questions = 0
    removed_questions = 0

    for conversation_id in conversation_ids:
        before_conversation = before_conversations.get(conversation_id)
        after_conversation = after_conversations.get(conversation_id)
        question_diffs = _question_diffs(
            conversation_id=conversation_id,
            before_conversation=before_conversation,
            after_conversation=after_conversation,
        )
        improved_questions += sum(diff.status == "improved" for diff in question_diffs)
        regressed_questions += sum(diff.status == "regressed" for diff in question_diffs)
        unchanged_questions += sum(diff.status == "unchanged" for diff in question_diffs)
        added_questions += sum(diff.status == "added" for diff in question_diffs)
        removed_questions += sum(diff.status == "removed" for diff in question_diffs)

        before_accuracy = before_conversation.accuracy if before_conversation is not None else None
        after_accuracy = after_conversation.accuracy if after_conversation is not None else None
        conversation_diffs.append(
            ConversationDiff(
                conversation_id=conversation_id,
                before_accuracy=before_accuracy,
                after_accuracy=after_accuracy,
                accuracy_delta=_float_delta(before_accuracy, after_accuracy),
                improved_questions=sum(diff.status == "improved" for diff in question_diffs),
                regressed_questions=sum(diff.status == "regressed" for diff in question_diffs),
                unchanged_questions=sum(diff.status == "unchanged" for diff in question_diffs),
                added_questions=sum(diff.status == "added" for diff in question_diffs),
                removed_questions=sum(diff.status == "removed" for diff in question_diffs),
                before_retrieval_time_ms=_conversation_numeric_summary(
                    before_conversation,
                    "retrieval_time_ms",
                ),
                after_retrieval_time_ms=_conversation_numeric_summary(
                    after_conversation,
                    "retrieval_time_ms",
                ),
                retrieval_time_delta_ms=_question_diff_numeric_summary(
                    question_diffs,
                    "retrieval_time_delta_ms",
                ),
                before_memories_used=_conversation_numeric_summary(
                    before_conversation,
                    "memories_used",
                ),
                after_memories_used=_conversation_numeric_summary(
                    after_conversation,
                    "memories_used",
                ),
                memories_used_delta=_question_diff_numeric_summary(
                    question_diffs,
                    "memories_used_delta",
                ),
                question_diffs=question_diffs,
            )
        )

    category_deltas: dict[int, float] = {}
    for category in sorted(set(before.category_breakdown) | set(after.category_breakdown)):
        category_deltas[category] = round(
            after.category_breakdown.get(category, 0.0) - before.category_breakdown.get(category, 0.0),
            6,
        )
    before_warning_counts = _warning_counts(before)
    after_warning_counts = _warning_counts(after)
    before_diagnosis_counts = _trace_field_counts(before, "diagnosis_bucket")
    after_diagnosis_counts = _trace_field_counts(after, "diagnosis_bucket")
    before_sufficiency_counts = _trace_field_counts(before, "sufficiency_diagnostic")
    after_sufficiency_counts = _trace_field_counts(after, "sufficiency_diagnostic")
    before_retrieval_custody_summary = _retrieval_custody_summary(before)
    after_retrieval_custody_summary = _retrieval_custody_summary(after)

    return BenchmarkDiffReport(
        benchmark_name=after.benchmark_name,
        before_label=before_label,
        after_label=after_label,
        before_report_sha256=before_report_sha256,
        after_report_sha256=after_report_sha256,
        before_timestamp=before.timestamp,
        after_timestamp=after.timestamp,
        generated_at=datetime.now(timezone.utc).isoformat(),
        before_overall_accuracy=before.overall_accuracy,
        after_overall_accuracy=after.overall_accuracy,
        overall_accuracy_delta=round(after.overall_accuracy - before.overall_accuracy, 6),
        before_total_correct=before.total_correct,
        after_total_correct=after.total_correct,
        total_correct_delta=after.total_correct - before.total_correct,
        before_total_questions=before.total_questions,
        after_total_questions=after.total_questions,
        total_questions_delta=after.total_questions - before.total_questions,
        before_warning_counts=before_warning_counts,
        after_warning_counts=after_warning_counts,
        warning_count_deltas=_count_deltas(before_warning_counts, after_warning_counts),
        before_diagnosis_bucket_counts=before_diagnosis_counts,
        after_diagnosis_bucket_counts=after_diagnosis_counts,
        diagnosis_bucket_count_deltas=_count_deltas(
            before_diagnosis_counts,
            after_diagnosis_counts,
        ),
        before_sufficiency_diagnostic_counts=before_sufficiency_counts,
        after_sufficiency_diagnostic_counts=after_sufficiency_counts,
        sufficiency_diagnostic_count_deltas=_count_deltas(
            before_sufficiency_counts,
            after_sufficiency_counts,
        ),
        before_retrieval_custody_summary=before_retrieval_custody_summary,
        after_retrieval_custody_summary=after_retrieval_custody_summary,
        category_deltas=category_deltas,
        improved_questions=improved_questions,
        regressed_questions=regressed_questions,
        unchanged_questions=unchanged_questions,
        added_questions=added_questions,
        removed_questions=removed_questions,
        before_retrieval_time_ms=_report_numeric_summary(before, "retrieval_time_ms"),
        after_retrieval_time_ms=_report_numeric_summary(after, "retrieval_time_ms"),
        retrieval_time_delta_ms=_question_diff_numeric_summary(
            [
                question_diff
                for conversation_diff in conversation_diffs
                for question_diff in conversation_diff.question_diffs
            ],
            "retrieval_time_delta_ms",
        ),
        before_memories_used=_report_numeric_summary(before, "memories_used"),
        after_memories_used=_report_numeric_summary(after, "memories_used"),
        memories_used_delta=_question_diff_numeric_summary(
            [
                question_diff
                for conversation_diff in conversation_diffs
                for question_diff in conversation_diff.question_diffs
            ],
            "memories_used_delta",
        ),
        conversations=conversation_diffs,
    )


def save_benchmark_diff(diff_report: BenchmarkDiffReport, output_path: str | Path) -> Path:
    """Persist a benchmark diff artifact as JSON and return its path."""
    destination = Path(output_path).expanduser()
    return write_json_atomic(destination, diff_report.model_dump(mode="json"))


def format_diff_summary(diff_report: BenchmarkDiffReport) -> str:
    """Return a compact human-readable summary of a benchmark diff."""
    lines = [
        "=" * 50,
        f"{diff_report.benchmark_name} Diff Report",
        "=" * 50,
        f"Before: {diff_report.before_label} ({diff_report.before_timestamp})",
        f"After:  {diff_report.after_label} ({diff_report.after_timestamp})",
        "",
        (
            f"Accuracy: {diff_report.before_overall_accuracy:.1%} -> "
            f"{diff_report.after_overall_accuracy:.1%} "
            f"({diff_report.overall_accuracy_delta:+.1%})"
        ),
        (
            f"Correct: {diff_report.before_total_correct}/{diff_report.before_total_questions} -> "
            f"{diff_report.after_total_correct}/{diff_report.after_total_questions} "
            f"({diff_report.total_correct_delta:+d})"
        ),
        (
            f"Questions: improved={diff_report.improved_questions} "
            f"regressed={diff_report.regressed_questions} "
            f"unchanged={diff_report.unchanged_questions} "
            f"added={diff_report.added_questions} "
            f"removed={diff_report.removed_questions}"
        ),
        "",
        "Retrieval custody:",
        f"  Before: {_format_custody_summary_for_diff(diff_report.before_retrieval_custody_summary)}",
        f"  After:  {_format_custody_summary_for_diff(diff_report.after_retrieval_custody_summary)}",
    ]
    warning_delta_line = _format_count_deltas("Warning deltas", diff_report.warning_count_deltas)
    if warning_delta_line is not None:
        lines.extend(["", warning_delta_line])
    diagnosis_delta_line = _format_count_deltas(
        "Diagnosis deltas",
        diff_report.diagnosis_bucket_count_deltas,
    )
    if diagnosis_delta_line is not None:
        lines.append(diagnosis_delta_line)
    sufficiency_delta_line = _format_count_deltas(
        "Sufficiency deltas",
        diff_report.sufficiency_diagnostic_count_deltas,
    )
    if sufficiency_delta_line is not None:
        lines.append(sufficiency_delta_line)

    significant_category_deltas = [
        (category, delta)
        for category, delta in sorted(diff_report.category_deltas.items())
        if abs(delta) >= 0.01
    ]
    if significant_category_deltas:
        lines.append("")
        lines.append("Category changes (>=1% delta):")
        for category, delta in significant_category_deltas:
            lines.append(f"  Cat {category}: {delta:+.1%}")

    flipped_questions = [
        question_diff
        for conversation_diff in diff_report.conversations
        for question_diff in conversation_diff.question_diffs
        if question_diff.status in ("improved", "regressed")
    ]
    if flipped_questions:
        lines.append("")
        lines.append("Flipped questions:")
        for question_diff in flipped_questions[:20]:
            direction = "PASS" if question_diff.status == "improved" else "FAIL"
            lines.append(
                f"  [{direction}] {question_diff.question_id}: "
                f"{question_diff.before_score} -> {question_diff.after_score} "
                f"(cat {question_diff.category})"
            )
        omitted = len(flipped_questions) - 20
        if omitted > 0:
            lines.append(f"  ... {omitted} more flipped questions")

    lines.append("=" * 50)
    return "\n".join(lines)


def _question_diffs(
    *,
    conversation_id: str,
    before_conversation: ConversationReport | None,
    after_conversation: ConversationReport | None,
) -> list[QuestionDiff]:
    before_results = _result_map(before_conversation)
    after_results = _result_map(after_conversation)
    question_ids = sorted(set(before_results) | set(after_results))
    diffs: list[QuestionDiff] = []

    for question_id in question_ids:
        before_result = before_results.get(question_id)
        after_result = after_results.get(question_id)
        question = (
            before_result.question
            if before_result is not None
            else after_result.question
        )
        before_score = before_result.score_result.score if before_result is not None else None
        after_score = after_result.score_result.score if after_result is not None else None
        status = _diff_status(before_score, after_score)
        memories_used_delta = None
        retrieval_time_delta_ms = None
        if before_result is not None and after_result is not None:
            memories_used_delta = after_result.memories_used - before_result.memories_used
            retrieval_time_delta_ms = round(
                after_result.retrieval_time_ms - before_result.retrieval_time_ms,
                6,
            )
        diffs.append(
            QuestionDiff(
                question_id=question.question_id,
                conversation_id=conversation_id,
                category=question.category,
                question_text=question.question_text,
                status=status,
                before_score=before_score,
                after_score=after_score,
                score_delta=(after_score or 0) - (before_score or 0),
                before_prediction=before_result.prediction if before_result is not None else None,
                after_prediction=after_result.prediction if after_result is not None else None,
                before_memories_used=before_result.memories_used if before_result is not None else None,
                after_memories_used=after_result.memories_used if after_result is not None else None,
                memories_used_delta=memories_used_delta,
                before_retrieval_time_ms=(
                    before_result.retrieval_time_ms if before_result is not None else None
                ),
                after_retrieval_time_ms=(
                    after_result.retrieval_time_ms if after_result is not None else None
                ),
                retrieval_time_delta_ms=retrieval_time_delta_ms,
                before_reasoning=(
                    before_result.score_result.reasoning if before_result is not None else None
                ),
                after_reasoning=(
                    after_result.score_result.reasoning if after_result is not None else None
                ),
                before_diagnosis_bucket=_trace_text(before_result, "diagnosis_bucket"),
                after_diagnosis_bucket=_trace_text(after_result, "diagnosis_bucket"),
                before_sufficiency_diagnostic=_trace_text(
                    before_result,
                    "sufficiency_diagnostic",
                ),
                after_sufficiency_diagnostic=_trace_text(
                    after_result,
                    "sufficiency_diagnostic",
                ),
                before_selected_memory_ids=_trace_list(before_result, "selected_memory_ids"),
                after_selected_memory_ids=_trace_list(after_result, "selected_memory_ids"),
                before_selected_evidence_memory_ids=_trace_list(
                    before_result,
                    "selected_evidence_memory_ids",
                ),
                after_selected_evidence_memory_ids=_trace_list(
                    after_result,
                    "selected_evidence_memory_ids",
                ),
            )
        )
    return diffs


def _result_map(conversation: ConversationReport | None) -> dict[str, QuestionResult]:
    if conversation is None:
        return {}
    return {
        result.question.question_id: result
        for result in conversation.results
    }


def _report_numeric_summary(report: BenchmarkReport, field_name: str) -> NumericSummary:
    return _numeric_summary(
        [
            float(getattr(result, field_name))
            for conversation in report.conversations
            for result in conversation.results
        ]
    )


def _conversation_numeric_summary(
    conversation: ConversationReport | None,
    field_name: str,
) -> NumericSummary:
    if conversation is None:
        return _numeric_summary([])
    return _numeric_summary(
        [float(getattr(result, field_name)) for result in conversation.results]
    )


def _question_diff_numeric_summary(
    question_diffs: list[QuestionDiff],
    field_name: str,
) -> NumericSummary:
    values = [
        float(value)
        for diff in question_diffs
        if (value := getattr(diff, field_name)) is not None
    ]
    return _numeric_summary(values)


def _numeric_summary(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(count=0, mean=None, min=None, max=None)
    return NumericSummary(
        count=len(values),
        mean=sum(values) / len(values),
        min=min(values),
        max=max(values),
    )


def _diff_status(before_score: int | None, after_score: int | None) -> DiffStatus:
    if before_score is None:
        return "added"
    if after_score is None:
        return "removed"
    if after_score > before_score:
        return "improved"
    if after_score < before_score:
        return "regressed"
    return "unchanged"


def _trace_text(result: QuestionResult | None, field_name: str) -> str | None:
    if result is None:
        return None
    value = result.trace.get(field_name)
    if value is None:
        return None
    return str(value)


def _trace_list(result: QuestionResult | None, field_name: str) -> list[str]:
    if result is None:
        return []
    value = result.trace.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _warning_counts(report: BenchmarkReport) -> dict[str, int]:
    value = report.model_info.get("warning_counts", {})
    has_failed_question_count = False
    counts: dict[str, int] = {}
    if not isinstance(value, dict):
        value = {}
    if isinstance(value, dict):
        has_failed_question_count = "failed_questions" in value
        for key, amount in value.items():
            try:
                counts[str(key)] = int(amount)
            except (TypeError, ValueError):
                continue
    if not has_failed_question_count:
        counts["failed_questions"] = max(
            0,
            int(report.total_questions) - int(report.total_correct),
        )
    return counts


def _trace_field_counts(report: BenchmarkReport, field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for conversation in report.conversations:
        for result in conversation.results:
            trace = result.trace if isinstance(result.trace, dict) else {}
            raw_value = trace.get(field_name)
            value = str(raw_value).strip() if raw_value is not None else ""
            key = value or "unknown"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _retrieval_custody_summary(report: BenchmarkReport) -> dict[str, object]:
    return summarize_retrieval_custody(
        result.trace.get("retrieval_custody", [])
        for conversation in report.conversations
        for result in conversation.results
        if isinstance(result.trace, dict)
    )


def _count_deltas(before_counts: dict[str, int], after_counts: dict[str, int]) -> dict[str, int]:
    return {
        key: after_counts.get(key, 0) - before_counts.get(key, 0)
        for key in sorted(set(before_counts) | set(after_counts))
    }


def _format_count_deltas(label: str, value: dict[str, int]) -> str | None:
    parts = [
        f"{key}={amount:+d}"
        for key, amount in sorted(value.items())
        if amount != 0
    ]
    if not parts:
        return None
    return f"{label}: " + ", ".join(parts)


def _format_custody_summary_for_diff(value: object) -> str:
    return format_retrieval_custody_summary(value).removeprefix("Retrieval custody: ")


def _float_delta(before_value: float | None, after_value: float | None) -> float | None:
    if before_value is None or after_value is None:
        return None
    return round(after_value - before_value, 6)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Path to the baseline benchmark report JSON")
    parser.add_argument("--after", required=True, help="Path to the candidate benchmark report JSON")
    parser.add_argument("--output", required=True, help="Path where the diff JSON should be written")
    parser.add_argument("--before-label", default="before", help="Label for the baseline report")
    parser.add_argument("--after-label", default="after", help="Label for the candidate report")
    return parser


def main() -> None:
    """Compare two benchmark reports and save a per-question diff artifact."""
    args = _build_parser().parse_args()
    before_report = load_benchmark_report(args.before)
    after_report = load_benchmark_report(args.after)
    diff_report = build_benchmark_diff(
        before_report,
        after_report,
        before_label=args.before_label,
        after_label=args.after_label,
        before_report_sha256=sha256_file_if_exists(args.before),
        after_report_sha256=sha256_file_if_exists(args.after),
    )
    output_path = save_benchmark_diff(diff_report, args.output)
    print(format_diff_summary(diff_report), flush=True)
    print(f"Saved benchmark diff to {output_path}", flush=True)


if __name__ == "__main__":
    main()
