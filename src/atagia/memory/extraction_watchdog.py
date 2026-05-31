"""Internal streaming watchdog for memory extraction output."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from atagia.core.config import Settings
from atagia.services.llm_reliability import (
    analyze_repeated_text_spans,
    destructure_json_for_repetition_analysis,
)
from atagia.services.model_resolution import parse_model_spec
from atagia.services.run_counters import increment_run_counter

logger = logging.getLogger(__name__)

_CONTENT_VALUE_KEYS = frozenset({"canonical_text", "index_text"})
_MAX_ANALYZED_CHARS = 24_000
_MAX_EXCERPT_CHARS = 2_000
_MAX_REPEATED_PHRASES = 8
_REPEAT_ABORT_MIN_COUNT = 3
_REPEAT_ABORT_MIN_RATIO_TOKENS = 0.12
_RUNAWAY_ABORT_MIN_OUTPUT_TOKENS = 1024
_RUNAWAY_ABORT_MIN_OUTPUT_INPUT_RATIO = 12.0
_MECHANICAL_HARD_ABORT_MIN_OUTPUT_TOKENS = 4096
_MECHANICAL_HARD_ABORT_MIN_OUTPUT_INPUT_RATIO = 8.0


class WatchdogVerdict(BaseModel):
    """Verdict-shaped metadata attached to mechanical extraction retries."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["continue", "abort_and_retry_bounded"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    evidence_type: Literal[
        "partial_json_only",
        "repetition_loop",
        "structural_cycle",
        "runaway_growth",
        "other",
    ] = "other"


@dataclass(frozen=True, slots=True)
class WatchdogCheckTelemetry:
    """Why the watchdog inspected the partial stream."""

    gate_trigger: Literal["mechanical_hard_abort"]
    elapsed_seconds: float
    latest_output_excerpt_chars: int


@dataclass(frozen=True, slots=True)
class WatchdogAbortPolicyDecision:
    """Local policy decision applied to mechanical extraction telemetry."""

    allowed: bool
    policy: str
    mechanical_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionWatchdogConfig:
    """Runtime settings for extraction watchdog checks."""

    enabled: bool
    allow_different_provider: bool
    bounded_retry_max_items: int
    bounded_retry_max_output_tokens: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExtractionWatchdogConfig":
        return cls(
            enabled=settings.extraction_watchdog_enabled,
            allow_different_provider=settings.extraction_watchdog_allow_different_provider,
            bounded_retry_max_items=settings.extraction_watchdog_bounded_retry_max_items,
            bounded_retry_max_output_tokens=(
                settings.extraction_watchdog_bounded_retry_max_output_tokens
            ),
        )


@dataclass(frozen=True, slots=True)
class RepeatedPhrase:
    """One repeated n-gram signal."""

    text: str
    n: int
    count: int
    repeat_ratio_tokens: float


@dataclass(frozen=True, slots=True)
class RepetitionSignals:
    """Mechanical repetition and growth telemetry for partial output."""

    output_tokens: int
    source_input_tokens: int
    output_input_ratio: float | None
    max_repeat_count: int
    max_repeat_ratio_tokens: float
    repeated_phrases: tuple[RepeatedPhrase, ...]


class ExtractionWatchdogRetry(RuntimeError):
    """Raised when streamed extraction should be retried with bounded output."""

    def __init__(
        self,
        reason: str,
        *,
        signals: RepetitionSignals,
        verdict: WatchdogVerdict,
        telemetry: WatchdogCheckTelemetry,
        abort_policy: WatchdogAbortPolicyDecision,
    ) -> None:
        super().__init__(reason)
        self.signals = signals
        self.verdict = verdict
        self.telemetry = telemetry
        self.abort_policy = abort_policy


class ExtractionWatchdogObserver:
    """Observes streamed extraction text for mechanical hard-abort signals."""

    def __init__(
        self,
        *,
        config: ExtractionWatchdogConfig,
        source_input_tokens: int,
    ) -> None:
        self._config = config
        self._source_input_tokens = max(1, source_input_tokens)
        self._started_at = perf_counter()

    def reset_for_retry(self, _error: Any | None = None) -> None:
        self._started_at = perf_counter()

    async def on_text(
        self,
        _chunk: str,
        accumulated_text: str,
        _request: Any,
    ) -> None:
        output_tokens = estimate_tokens(accumulated_text)
        elapsed = perf_counter() - self._started_at
        hard_abort = self._mechanical_hard_abort(
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
            accumulated_text=accumulated_text,
        )
        if hard_abort is not None:
            raise hard_abort

    def _mechanical_hard_abort(
        self,
        *,
        output_tokens: int,
        elapsed_seconds: float,
        accumulated_text: str,
    ) -> ExtractionWatchdogRetry | None:
        if output_tokens < _MECHANICAL_HARD_ABORT_MIN_OUTPUT_TOKENS:
            return None
        signals = analyze_repetition_signals(
            accumulated_text,
            source_input_tokens=self._source_input_tokens,
        )
        abort_policy = evaluate_mechanical_hard_abort(signals)
        if not abort_policy.allowed:
            return None
        increment_run_counter(
            "mechanical_runaway_abort_count",
            layer="extraction_watchdog",
            mode="hard_abort",
        )
        telemetry = WatchdogCheckTelemetry(
            gate_trigger="mechanical_hard_abort",
            elapsed_seconds=round(elapsed_seconds, 3),
            latest_output_excerpt_chars=min(len(accumulated_text), _MAX_EXCERPT_CHARS),
        )
        verdict = WatchdogVerdict(
            decision="abort_and_retry_bounded",
            confidence=1.0,
            reason=(
                "Mechanical watchdog detected late runaway extraction output before "
                "the provider output limit."
            ),
            evidence_type="runaway_growth",
        )
        return ExtractionWatchdogRetry(
            f"Extraction watchdog requested bounded retry: {verdict.reason}",
            signals=signals,
            verdict=verdict,
            telemetry=telemetry,
            abort_policy=abort_policy,
        )


def validate_watchdog_provider_policy(
    *,
    extractor_model: str,
    watchdog_model: str,
    allow_different_provider: bool,
) -> None:
    if allow_different_provider:
        return
    extractor = parse_model_spec(extractor_model)
    watchdog = parse_model_spec(watchdog_model)
    if extractor.provider_slug == watchdog.provider_slug:
        return
    raise ValueError(
        "extraction_watchdog uses a different provider than extractor; set "
        "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER=true to allow this"
    )


def evaluate_mechanical_hard_abort(signals: RepetitionSignals) -> WatchdogAbortPolicyDecision:
    """Decide whether local telemetry is already enough to stop a stream."""

    mechanical_evidence = _mechanical_abort_evidence(signals)
    if signals.output_tokens < _MECHANICAL_HARD_ABORT_MIN_OUTPUT_TOKENS:
        return WatchdogAbortPolicyDecision(
            allowed=False,
            policy="blocked_below_mechanical_hard_abort_floor",
            mechanical_evidence=mechanical_evidence,
        )

    hard_evidence = list(mechanical_evidence)
    if (
        signals.output_input_ratio is not None
        and signals.output_input_ratio >= _MECHANICAL_HARD_ABORT_MIN_OUTPUT_INPUT_RATIO
        and "runaway_growth_ratio" not in hard_evidence
    ):
        hard_evidence.append("late_runaway_growth_ratio")

    if "repeated_content" in hard_evidence:
        return WatchdogAbortPolicyDecision(
            allowed=True,
            policy="allowed_mechanical_hard_repetition",
            mechanical_evidence=tuple(hard_evidence),
        )
    if (
        "runaway_growth_ratio" in hard_evidence
        or "late_runaway_growth_ratio" in hard_evidence
    ):
        return WatchdogAbortPolicyDecision(
            allowed=True,
            policy="allowed_mechanical_hard_runaway_growth",
            mechanical_evidence=tuple(hard_evidence),
        )
    return WatchdogAbortPolicyDecision(
        allowed=False,
        policy="blocked_mechanical_hard_without_signal",
        mechanical_evidence=tuple(hard_evidence),
    )


def _mechanical_abort_evidence(signals: RepetitionSignals) -> tuple[str, ...]:
    evidence: list[str] = []
    if (
        signals.max_repeat_count >= _REPEAT_ABORT_MIN_COUNT
        and signals.max_repeat_ratio_tokens >= _REPEAT_ABORT_MIN_RATIO_TOKENS
    ):
        evidence.append("repeated_content")
    if (
        signals.output_tokens >= _RUNAWAY_ABORT_MIN_OUTPUT_TOKENS
        and signals.output_input_ratio is not None
        and signals.output_input_ratio >= _RUNAWAY_ABORT_MIN_OUTPUT_INPUT_RATIO
    ):
        evidence.append("runaway_growth_ratio")
    return tuple(evidence)


def analyze_repetition_signals(text: str, *, source_input_tokens: int) -> RepetitionSignals:
    analyzed_text = _analysis_text(text)
    tokens = _tokenize_mechanical(analyzed_text)
    output_tokens = estimate_tokens(text)
    top_repeated = tuple(
        RepeatedPhrase(
            text=item.text,
            n=item.n,
            count=item.count,
            repeat_ratio_tokens=item.repeat_ratio_tokens,
        )
        for item in analyze_repeated_text_spans(
            analyzed_text,
            n_values=range(5, 13),
            min_count=3,
            max_phrases=_MAX_REPEATED_PHRASES,
            max_phrase_chars=180,
            tokens=tokens,
        )
    )
    max_count = max((item.count for item in top_repeated), default=0)
    max_ratio = max((item.repeat_ratio_tokens for item in top_repeated), default=0.0)
    ratio = output_tokens / max(1, source_input_tokens) if source_input_tokens > 0 else None
    return RepetitionSignals(
        output_tokens=output_tokens,
        source_input_tokens=max(1, source_input_tokens),
        output_input_ratio=ratio,
        max_repeat_count=max_count,
        max_repeat_ratio_tokens=max_ratio,
        repeated_phrases=top_repeated,
    )


def estimate_tokens(text: str) -> int:
    token_count = len(_tokenize_mechanical(text))
    char_estimate = len(text) // 4
    return max(token_count, char_estimate, 1)


def _analysis_text(text: str) -> str:
    return destructure_json_for_repetition_analysis(
        text[-_MAX_ANALYZED_CHARS:],
        value_keys=_CONTENT_VALUE_KEYS,
        min_value_chars=1,
        include_outside_json_text=False,
        fallback_to_original=True,
    )


def _tokenize_mechanical(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    for char in text.casefold():
        if char.isalnum():
            buffer.append(char)
            continue
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens
