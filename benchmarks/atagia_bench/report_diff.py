"""Per-question diff tooling for Atagia-bench reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaQuestionResult

DiffStatus = Literal["improved", "regressed", "unchanged", "added", "removed"]


class QuestionDiff(BaseModel):
    """One per-question change between two Atagia-bench reports."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    persona_id: str
    category_tags: list[str] = Field(default_factory=list)
    question_text: str
    status: DiffStatus
    before_passed: bool | None = None
    after_passed: bool | None = None
    before_score: float | None = Field(default=None, ge=0.0, le=1.0)
    after_score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_delta: float
    before_prediction: str | None = None
    after_prediction: str | None = None
    before_reason: str | None = None
    after_reason: str | None = None
    before_memories_used: int | None = Field(default=None, ge=0)
    after_memories_used: int | None = Field(default=None, ge=0)
    memories_used_delta: int | None = None


class CategoryDelta(BaseModel):
    """Per-category pass rate delta between two reports."""

    model_config = ConfigDict(extra="forbid")

    category: str
    before_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    after_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    pass_rate_delta: float | None = None
    before_avg_score: float | None = Field(default=None, ge=0.0, le=1.0)
    after_avg_score: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_score_delta: float | None = None


class AtagiaBenchDiffReport(BaseModel):
    """Structured diff between two Atagia-bench reports."""

    model_config = ConfigDict(extra="forbid")

    before_label: str
    after_label: str
    before_timestamp: str
    after_timestamp: str
    generated_at: str
    before_pass_rate: float = Field(ge=0.0, le=1.0)
    after_pass_rate: float = Field(ge=0.0, le=1.0)
    pass_rate_delta: float
    before_avg_score: float = Field(ge=0.0, le=1.0)
    after_avg_score: float = Field(ge=0.0, le=1.0)
    avg_score_delta: float
    before_total: int = Field(ge=0)
    after_total: int = Field(ge=0)
    net_improved: int = Field(ge=0)
    net_regressed: int = Field(ge=0)
    net_unchanged: int = Field(ge=0)
    net_added: int = Field(ge=0)
    net_removed: int = Field(ge=0)
    before_critical_errors: int = Field(ge=0)
    after_critical_errors: int = Field(ge=0)
    category_deltas: list[CategoryDelta] = Field(default_factory=list)
    question_diffs: list[QuestionDiff] = Field(default_factory=list)


def load_atagia_bench_report(path: str | Path) -> AtagiaBenchReport:
    """Load an Atagia-bench report from JSON."""
    report_path = Path(path).expanduser()
    return AtagiaBenchReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )


def build_diff(
    before: AtagiaBenchReport,
    after: AtagiaBenchReport,
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> AtagiaBenchDiffReport:
    """Build a structured diff between two Atagia-bench reports."""
    before_map = _result_map(before)
    after_map = _result_map(after)
    all_ids = sorted(set(before_map) | set(after_map))

    question_diffs: list[QuestionDiff] = []
    improved = 0
    regressed = 0
    unchanged = 0
    added = 0
    removed = 0

    for qid in all_ids:
        before_result = before_map.get(qid)
        after_result = after_map.get(qid)
        diff = _build_question_diff(qid, before_result, after_result)
        question_diffs.append(diff)

        if diff.status == "improved":
            improved += 1
        elif diff.status == "regressed":
            regressed += 1
        elif diff.status == "unchanged":
            unchanged += 1
        elif diff.status == "added":
            added += 1
        elif diff.status == "removed":
            removed += 1

    # Category deltas
    before_cats = {s.category: s for s in before.per_category}
    after_cats = {s.category: s for s in after.per_category}
    all_categories = sorted(set(before_cats) | set(after_cats))
    category_deltas: list[CategoryDelta] = []

    for category in all_categories:
        b_stats = before_cats.get(category)
        a_stats = after_cats.get(category)
        category_deltas.append(
            CategoryDelta(
                category=category,
                before_pass_rate=b_stats.pass_rate if b_stats else None,
                after_pass_rate=a_stats.pass_rate if a_stats else None,
                pass_rate_delta=_float_delta(
                    b_stats.pass_rate if b_stats else None,
                    a_stats.pass_rate if a_stats else None,
                ),
                before_avg_score=b_stats.avg_score if b_stats else None,
                after_avg_score=a_stats.avg_score if a_stats else None,
                avg_score_delta=_float_delta(
                    b_stats.avg_score if b_stats else None,
                    a_stats.avg_score if a_stats else None,
                ),
            )
        )

    return AtagiaBenchDiffReport(
        before_label=before_label,
        after_label=after_label,
        before_timestamp=before.timestamp,
        after_timestamp=after.timestamp,
        generated_at=datetime.now(timezone.utc).isoformat(),
        before_pass_rate=before.pass_rate,
        after_pass_rate=after.pass_rate,
        pass_rate_delta=round(after.pass_rate - before.pass_rate, 6),
        before_avg_score=before.avg_score,
        after_avg_score=after.avg_score,
        avg_score_delta=round(after.avg_score - before.avg_score, 6),
        before_total=before.total_questions,
        after_total=after.total_questions,
        net_improved=improved,
        net_regressed=regressed,
        net_unchanged=unchanged,
        net_added=added,
        net_removed=removed,
        before_critical_errors=before.critical_error_count,
        after_critical_errors=after.critical_error_count,
        category_deltas=category_deltas,
        question_diffs=question_diffs,
    )


def save_diff(
    diff: AtagiaBenchDiffReport,
    output_path: str | Path,
) -> Path:
    """Persist a diff report as JSON and return its path."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            diff.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination


def format_diff_summary(diff: AtagiaBenchDiffReport) -> str:
    """Return a human-readable summary of the diff."""
    lines: list[str] = [
        "=" * 50,
        "Atagia-bench Diff Report",
        "=" * 50,
        f"Before: {diff.before_label} ({diff.before_timestamp})",
        f"After:  {diff.after_label} ({diff.after_timestamp})",
        "",
        f"Pass rate: {diff.before_pass_rate:.1%} -> {diff.after_pass_rate:.1%} "
        f"({diff.pass_rate_delta:+.1%})",
        f"Avg score: {diff.before_avg_score:.3f} -> {diff.after_avg_score:.3f} "
        f"({diff.avg_score_delta:+.3f})",
        f"Critical errors: {diff.before_critical_errors} -> {diff.after_critical_errors}",
        "",
        f"Improved: {diff.net_improved}  Regressed: {diff.net_regressed}  "
        f"Unchanged: {diff.net_unchanged}",
        f"Added: {diff.net_added}  Removed: {diff.net_removed}",
    ]

    # Show flipped questions
    flipped = [
        d for d in diff.question_diffs
        if d.status in ("improved", "regressed")
    ]
    if flipped:
        lines.append("")
        lines.append("Flipped questions:")
        for d in flipped:
            direction = "PASS" if d.status == "improved" else "FAIL"
            lines.append(
                f"  [{direction}] {d.question_id}: "
                f"{d.before_score} -> {d.after_score} "
                f"({', '.join(d.category_tags[:3])})"
            )

    # Category deltas
    significant_cats = [
        c for c in diff.category_deltas
        if c.pass_rate_delta is not None and abs(c.pass_rate_delta) >= 0.01
    ]
    if significant_cats:
        lines.append("")
        lines.append("Category changes (>=1% delta):")
        for c in significant_cats:
            delta_str = f"{c.pass_rate_delta:+.1%}" if c.pass_rate_delta else "N/A"
            lines.append(f"  {c.category}: {delta_str}")

    lines.append("=" * 50)
    return "\n".join(lines)


def _result_map(
    report: AtagiaBenchReport,
) -> dict[str, AtagiaQuestionResult]:
    """Build a question_id -> result mapping."""
    return {r.question_id: r for r in report.per_question}


def _build_question_diff(
    question_id: str,
    before: AtagiaQuestionResult | None,
    after: AtagiaQuestionResult | None,
) -> QuestionDiff:
    """Build a diff for a single question."""
    ref = before or after
    if ref is None:
        raise ValueError(f"Both before and after are None for {question_id}")

    before_score = before.grade.score if before else None
    after_score = after.grade.score if after else None
    before_passed = before.grade.passed if before else None
    after_passed = after.grade.passed if after else None

    status = _diff_status(before_passed, after_passed)

    score_delta = (after_score or 0.0) - (before_score or 0.0)
    memories_delta = None
    if before is not None and after is not None:
        memories_delta = after.memories_used - before.memories_used

    return QuestionDiff(
        question_id=question_id,
        persona_id=ref.persona_id,
        category_tags=ref.category_tags,
        question_text=ref.question_text,
        status=status,
        before_passed=before_passed,
        after_passed=after_passed,
        before_score=before_score,
        after_score=after_score,
        score_delta=round(score_delta, 6),
        before_prediction=before.prediction if before else None,
        after_prediction=after.prediction if after else None,
        before_reason=before.grade.reason if before else None,
        after_reason=after.grade.reason if after else None,
        before_memories_used=before.memories_used if before else None,
        after_memories_used=after.memories_used if after else None,
        memories_used_delta=memories_delta,
    )


def _diff_status(
    before_passed: bool | None,
    after_passed: bool | None,
) -> DiffStatus:
    """Determine the diff status between two pass/fail states."""
    if before_passed is None:
        return "added"
    if after_passed is None:
        return "removed"
    if not before_passed and after_passed:
        return "improved"
    if before_passed and not after_passed:
        return "regressed"
    return "unchanged"


def _float_delta(
    before_value: float | None,
    after_value: float | None,
) -> float | None:
    """Compute a rounded float delta."""
    if before_value is None or after_value is None:
        return None
    return round(after_value - before_value, 6)
