"""Scoring helpers for the adaptive retrieval gate suite.

This module is pure (no engine, no I/O): it reads the gate classification and
gate action out of an Atagia chat-result debug payload, compares them to the
expected labels, and aggregates accuracy. Keeping it pure makes it directly
unit-testable and reusable from both the runner and ad-hoc trace analysis.

Two observable surfaces are read, in priority order:

1. ``debug["retrieval_plan"]`` and ``debug["retrieval_diagnostics_for_guard"]``
   carry an ``adaptive_gate`` entry once the gate is wired into the pipeline.
   The entry may be a plain status string (fast path reports
   ``"not_applicable"``) or a small object with ``status`` and
   ``classification`` keys.
2. ``debug["retrieval_trace"]["need_detection"]["memory_dependence"]`` always
   records the classification (shadow mode included), so classification
   accuracy is scorable even before the gate action is wired in.

Anything unreadable is reported as ``None``/``unknown`` rather than guessed:
the gate's whole point is that wrong skips are the dangerous direction, so the
scorer never invents a classification or action.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.models.schemas_memory import AdaptiveGateStatus, MemoryDependence
from benchmarks.atagia_bench_gate.dataset import GateProbeQuestion


class GateObservation(BaseModel):
    """What the engine actually did for one probe question.

    All fields are optional because the gate may be in shadow mode (no action),
    unwired (no status surfaced), or degraded (no classification). The runner
    and reports treat missing values explicitly instead of assuming a default.
    """

    model_config = ConfigDict(extra="forbid")

    classification: MemoryDependence | None = None
    gate_status: AdaptiveGateStatus | None = None
    retrieval_skipped: bool | None = None
    selected_memory_count: int | None = None


class GateQuestionScore(BaseModel):
    """Per-question scoring of classification and gate action."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    language: str
    probe_kind: str
    pair_id: str | None
    expected_memory_dependence: MemoryDependence
    expected_skip: bool
    observation: GateObservation
    classification_correct: bool | None
    action_correct: bool | None


class GateSuiteScore(BaseModel):
    """Aggregated scoring across all scored probe questions."""

    model_config = ConfigDict(extra="forbid")

    total_questions: int = Field(ge=0)
    classification_scored: int = Field(ge=0)
    classification_correct: int = Field(ge=0)
    action_scored: int = Field(ge=0)
    action_correct: int = Field(ge=0)
    false_skips: int = Field(ge=0)
    correct_skips: int = Field(ge=0)
    missed_skips: int = Field(ge=0)
    per_question: list[GateQuestionScore] = Field(default_factory=list)
    classification_accuracy_by_language: dict[str, float] = Field(default_factory=dict)

    @property
    def classification_accuracy(self) -> float | None:
        """Return overall classification accuracy, or ``None`` if unscored."""
        if self.classification_scored == 0:
            return None
        return self.classification_correct / self.classification_scored

    @property
    def action_accuracy(self) -> float | None:
        """Return overall gate-action accuracy, or ``None`` if unscored."""
        if self.action_scored == 0:
            return None
        return self.action_correct / self.action_scored


def _coerce_classification(value: Any) -> MemoryDependence | None:
    """Parse a raw classification value into the enum, or ``None``."""
    if value is None:
        return None
    if isinstance(value, MemoryDependence):
        return value
    try:
        return MemoryDependence(str(value))
    except ValueError:
        return None


def _coerce_status(value: Any) -> AdaptiveGateStatus | None:
    """Parse a raw gate-status value into the enum, or ``None``."""
    if value is None:
        return None
    if isinstance(value, AdaptiveGateStatus):
        return value
    try:
        return AdaptiveGateStatus(str(value))
    except ValueError:
        return None


def _adaptive_gate_entry(container: Any) -> dict[str, Any] | str | None:
    """Return the ``adaptive_gate`` entry from a plan/diagnostics dict."""
    if not isinstance(container, dict):
        return None
    entry = container.get("adaptive_gate")
    if isinstance(entry, (dict, str)):
        return entry
    return None


def extract_gate_observation(debug: dict[str, Any] | None) -> GateObservation:
    """Read the gate classification and action from a chat-result debug payload.

    Reads are best-effort and never raise on a missing or unwired surface; an
    absent value stays ``None`` so callers can score only what the engine
    actually reported.
    """
    if not isinstance(debug, dict):
        return GateObservation()

    classification: MemoryDependence | None = None
    gate_status: AdaptiveGateStatus | None = None

    for key in ("retrieval_plan", "retrieval_diagnostics_for_guard"):
        entry = _adaptive_gate_entry(debug.get(key))
        if entry is None:
            continue
        if isinstance(entry, str):
            gate_status = gate_status or _coerce_status(entry)
            continue
        gate_status = gate_status or _coerce_status(entry.get("status"))
        classification = classification or _coerce_classification(
            entry.get("classification")
        )

    if classification is None:
        trace = debug.get("retrieval_trace")
        if isinstance(trace, dict):
            need_detection = trace.get("need_detection")
            if isinstance(need_detection, dict):
                classification = _coerce_classification(
                    need_detection.get("memory_dependence")
                )

    retrieval_skipped: bool | None = None
    if gate_status is AdaptiveGateStatus.SKIPPED:
        retrieval_skipped = True
    elif gate_status is AdaptiveGateStatus.RETRIEVED:
        retrieval_skipped = False

    selected_memory_count = _selected_memory_count(debug)

    return GateObservation(
        classification=classification,
        gate_status=gate_status,
        retrieval_skipped=retrieval_skipped,
        selected_memory_count=selected_memory_count,
    )


def _selected_memory_count(debug: dict[str, Any]) -> int | None:
    """Read the selected-memory count from the debug payload if present."""
    selected = debug.get("selected_memory_ids")
    if isinstance(selected, list):
        return len(selected)
    diagnostics = debug.get("retrieval_diagnostics_for_guard")
    if isinstance(diagnostics, dict):
        count = diagnostics.get("selected_memory_count")
        if isinstance(count, int):
            return count
    return None


def score_question(
    question: GateProbeQuestion,
    observation: GateObservation,
) -> GateQuestionScore:
    """Score one probe question against an engine observation."""
    expected_skip = question.expected_action.skip_retrieval

    classification_correct: bool | None = None
    if observation.classification is not None:
        classification_correct = (
            observation.classification == question.expected_memory_dependence
        )

    action_correct: bool | None = None
    if observation.retrieval_skipped is not None:
        action_correct = observation.retrieval_skipped == expected_skip

    return GateQuestionScore(
        question_id=question.question_id,
        language=question.language,
        probe_kind=question.probe_kind,
        pair_id=question.pair_id,
        expected_memory_dependence=question.expected_memory_dependence,
        expected_skip=expected_skip,
        observation=observation,
        classification_correct=classification_correct,
        action_correct=action_correct,
    )


def aggregate_scores(
    per_question: list[GateQuestionScore],
) -> GateSuiteScore:
    """Aggregate per-question scores into a suite-level score.

    ``false_skips`` (the engine skipped a turn that needed memory) are tracked
    separately because they are the dangerous direction the gate must avoid;
    ``missed_skips`` (the engine retrieved when it could have skipped) are the
    benign, latency-only direction.
    """
    classification_scored = 0
    classification_correct = 0
    action_scored = 0
    action_correct = 0
    false_skips = 0
    correct_skips = 0
    missed_skips = 0

    correct_by_language: dict[str, int] = defaultdict(int)
    scored_by_language: dict[str, int] = defaultdict(int)

    for score in per_question:
        if score.classification_correct is not None:
            classification_scored += 1
            scored_by_language[score.language] += 1
            if score.classification_correct:
                classification_correct += 1
                correct_by_language[score.language] += 1

        if score.observation.retrieval_skipped is not None:
            action_scored += 1
            skipped = score.observation.retrieval_skipped
            if score.action_correct:
                action_correct += 1
            if skipped and not score.expected_skip:
                false_skips += 1
            elif skipped and score.expected_skip:
                correct_skips += 1
            elif not skipped and score.expected_skip:
                missed_skips += 1

    accuracy_by_language = {
        language: correct_by_language[language] / scored_by_language[language]
        for language in sorted(scored_by_language)
    }

    return GateSuiteScore(
        total_questions=len(per_question),
        classification_scored=classification_scored,
        classification_correct=classification_correct,
        action_scored=action_scored,
        action_correct=action_correct,
        false_skips=false_skips,
        correct_skips=correct_skips,
        missed_skips=missed_skips,
        per_question=per_question,
        classification_accuracy_by_language=accuracy_by_language,
    )
