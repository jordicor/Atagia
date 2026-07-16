"""Shared loaders for retained LoCoMo night-run report artifacts.

Both the rejudge tool (`benchmarks.locomo.rejudge`) and the funnel report
(`benchmarks.funnel_report`) read the same retained `locomo-report-*.json`
files. This module centralizes:

- resolving report file paths from files or run directories (deduping to one
  canonical report per conversation),
- iterating per-question records (question text, ground truth, prediction, the
  stored strict verdict, and the full trace), and
- loading the LoCoMo dataset transcripts / official evidence turns needed by the
  `memory_quality` judge protocol.

No engine involvement: these helpers only read existing benchmark artifacts and
the public LoCoMo dataset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.base import BenchmarkConversation
from benchmarks.source_evidence import source_evidence_from_turns

_REPORT_GLOB = "locomo-report-*.json"
_CONV_DIR_PATTERN = re.compile(r"locomo_(conv-\d+)")
_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"


@dataclass(slots=True)
class QuestionRecord:
    """One stored benchmarked question extracted from a night-run report."""

    conversation_id: str
    question_id: str
    category: int
    question_text: str
    ground_truth: str
    evidence_turn_ids: list[str]
    prediction: str
    strict_score: int
    strict_reasoning: str
    strict_judge_model: str
    trace: dict[str, Any] = field(default_factory=dict)


def resolve_report_paths(specs: Iterable[str | Path]) -> list[Path]:
    """Resolve report specs (files or run directories) to canonical report paths.

    A directory is searched recursively for ``locomo-report-*.json`` files. When
    several reports map to the same conversation (for example a base and a
    recovery-eval directory), the lexicographically latest filename wins, which
    corresponds to the most recent run timestamp. Individual report files passed
    directly are always kept.
    """
    candidates: list[Path] = []
    for spec in specs:
        path = Path(spec).expanduser()
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob(_REPORT_GLOB)))
        else:
            raise FileNotFoundError(f"Report spec does not exist: {path}")

    by_conversation: dict[str, Path] = {}
    loose: list[Path] = []
    for path in candidates:
        match = _CONV_DIR_PATTERN.search(path.parent.name)
        if match is None:
            loose.append(path)
            continue
        conversation_id = match.group(1)
        previous = by_conversation.get(conversation_id)
        if previous is None or path.name > previous.name:
            by_conversation[conversation_id] = path

    resolved = sorted(
        set(list(by_conversation.values()) + loose),
        key=str,
    )
    if not resolved:
        raise FileNotFoundError(
            f"No {_REPORT_GLOB} files found under: {list(specs)}"
        )
    return resolved


def load_report(path: str | Path) -> dict[str, Any]:
    """Load one benchmark report JSON as a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_report_records(
    report: dict[str, Any],
    *,
    include_trace: bool = True,
) -> Iterator[QuestionRecord]:
    """Yield per-question records from an already-loaded report dict.

    Fail fast on malformed reports: a result missing its question_id, category,
    or stored score would silently distort a baseline if defaulted, so those
    raise instead of coercing.
    """
    for conversation in report.get("conversations", []):
        conversation_id = str(conversation.get("conversation_id") or "")
        if not conversation_id:
            raise ValueError("Report conversation entry has no conversation_id")
        for result in conversation.get("results", []):
            question = result.get("question") or {}
            score_result = result.get("score_result") or {}
            question_id = question.get("question_id")
            category = question.get("category")
            score = score_result.get("score")
            if not question_id:
                raise ValueError(
                    f"Report result in {conversation_id} has no question_id"
                )
            if category is None:
                raise ValueError(
                    f"Report result {question_id} in {conversation_id} has no "
                    "question category"
                )
            if score is None:
                raise ValueError(
                    f"Report result {question_id} in {conversation_id} has no "
                    "stored score_result.score"
                )
            trace = result.get("trace") or {}
            yield QuestionRecord(
                conversation_id=conversation_id,
                question_id=str(question_id),
                category=int(category),
                question_text=str(question.get("question_text") or ""),
                ground_truth=str(question.get("ground_truth") or ""),
                evidence_turn_ids=list(question.get("evidence_turn_ids") or []),
                prediction=str(result.get("prediction") or ""),
                strict_score=int(score),
                strict_reasoning=str(score_result.get("reasoning") or ""),
                strict_judge_model=str(score_result.get("judge_model") or ""),
                trace=trace if include_trace else {},
            )


def iter_all_records(
    report_paths: Iterable[str | Path],
    *,
    include_trace: bool = True,
) -> Iterator[QuestionRecord]:
    """Load each report one at a time and yield its question records.

    Reports are loaded lazily (one in memory at a time) because retained LoCoMo
    reports can be >100 MB each.
    """
    for path in report_paths:
        report = load_report(path)
        yield from iter_report_records(report, include_trace=include_trace)


def load_conversations(
    data_path: str | Path = _DEFAULT_DATA_PATH,
) -> dict[str, BenchmarkConversation]:
    """Load the LoCoMo dataset and index conversations by id."""
    dataset = LoCoMoAdapter(data_path).load()
    return {
        conversation.conversation_id: conversation
        for conversation in dataset.conversations
    }


def render_conversation_transcript(conversation: BenchmarkConversation) -> str:
    """Render a full LoCoMo conversation as ground-truth transcript text."""
    lines: list[str] = []
    for turn in conversation.turns:
        turn_id = f" {turn.turn_id}" if turn.turn_id else ""
        header = f"[{turn.session_id}{turn_id} {turn.timestamp}] {turn.speaker}:"
        lines.append(f"{header} {turn.text}".rstrip())
        for attachment in turn.attachments:
            content_text = str(attachment.get("content_text") or "").strip()
            if content_text:
                lines.append(f"    (attachment) {content_text}")
    return "\n".join(lines)


def source_evidence_for_record(
    record: QuestionRecord,
    conversation: BenchmarkConversation,
) -> list[dict[str, Any]]:
    """Rebuild the official source-evidence turns for a stored question."""
    return source_evidence_from_turns(
        evidence_turn_ids=record.evidence_turn_ids,
        turns=conversation.turns,
        conversation_id=conversation.conversation_id,
    )
