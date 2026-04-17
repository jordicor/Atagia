"""Per-question diff tooling for benchmark reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.base import BenchmarkReport, ConversationReport, QuestionResult

DiffStatus = Literal["improved", "regressed", "unchanged", "added", "removed"]


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
    question_diffs: list[QuestionDiff] = Field(default_factory=list)


class BenchmarkDiffReport(BaseModel):
    """Serializable before/after benchmark comparison."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    before_label: str
    after_label: str
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
    category_deltas: dict[int, float] = Field(default_factory=dict)
    improved_questions: int = Field(ge=0)
    regressed_questions: int = Field(ge=0)
    unchanged_questions: int = Field(ge=0)
    added_questions: int = Field(ge=0)
    removed_questions: int = Field(ge=0)
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
                question_diffs=question_diffs,
            )
        )

    category_deltas: dict[int, float] = {}
    for category in sorted(set(before.category_breakdown) | set(after.category_breakdown)):
        category_deltas[category] = round(
            after.category_breakdown.get(category, 0.0) - before.category_breakdown.get(category, 0.0),
            6,
        )

    return BenchmarkDiffReport(
        benchmark_name=after.benchmark_name,
        before_label=before_label,
        after_label=after_label,
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
        category_deltas=category_deltas,
        improved_questions=improved_questions,
        regressed_questions=regressed_questions,
        unchanged_questions=unchanged_questions,
        added_questions=added_questions,
        removed_questions=removed_questions,
        conversations=conversation_diffs,
    )


def save_benchmark_diff(diff_report: BenchmarkDiffReport, output_path: str | Path) -> Path:
    """Persist a benchmark diff artifact as JSON and return its path."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            diff_report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination


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
    )
    output_path = save_benchmark_diff(diff_report, args.output)
    print(
        (
            f"Saved benchmark diff to {output_path} "
            f"({diff_report.improved_questions} improved, "
            f"{diff_report.regressed_questions} regressed)"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
