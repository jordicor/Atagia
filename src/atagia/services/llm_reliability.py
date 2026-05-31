"""Technical reliability helpers for LLM calls.

This module intentionally avoids semantic answer validation. It only handles
provider/format/runtime failure modes such as output-limit truncation and
mechanically obvious runaway streams.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from atagia.services.run_counters import increment_run_counter


_MAX_ANALYZED_CHARS = 24_000
_MAX_REPEATED_PHRASE_CHARS = 220
_MIN_JSON_VALUE_ANALYSIS_CHARS = 24


@dataclass(frozen=True, slots=True)
class LLMTechnicalRecoveryConfig:
    """Settings for provider-side technical recovery."""

    enabled: bool = False
    output_limit_retry_attempts: int = 0
    runaway_watchdog_enabled: bool = False
    runaway_min_elapsed_seconds: float = 8.0
    runaway_min_output_tokens: int = 2048
    runaway_check_interval_tokens: int = 1024
    runaway_max_checks: int = 2
    runaway_hard_abort_min_output_tokens: int = 4096
    runaway_min_repeat_count: int = 3
    runaway_min_repeat_ratio_tokens: float = 0.12
    runaway_output_input_ratio: float = 12.0
    runaway_hard_output_input_ratio: float = 8.0

    @classmethod
    def disabled(cls) -> "LLMTechnicalRecoveryConfig":
        return cls(
            enabled=False,
            output_limit_retry_attempts=0,
            runaway_watchdog_enabled=False,
        )

    @classmethod
    def default_enabled(cls) -> "LLMTechnicalRecoveryConfig":
        return cls(
            enabled=True,
            output_limit_retry_attempts=1,
            runaway_watchdog_enabled=True,
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "LLMTechnicalRecoveryConfig":
        return cls(
            enabled=bool(settings.llm_technical_recovery_enabled),
            output_limit_retry_attempts=settings.llm_output_limit_retry_attempts,
            runaway_watchdog_enabled=bool(settings.llm_runaway_watchdog_enabled),
            runaway_min_elapsed_seconds=settings.llm_runaway_min_elapsed_seconds,
            runaway_min_output_tokens=settings.llm_runaway_min_output_tokens,
            runaway_check_interval_tokens=settings.llm_runaway_check_interval_tokens,
            runaway_max_checks=settings.llm_runaway_max_checks,
            runaway_hard_abort_min_output_tokens=(
                settings.llm_runaway_hard_abort_min_output_tokens
            ),
            runaway_min_repeat_count=settings.llm_runaway_min_repeat_count,
            runaway_min_repeat_ratio_tokens=(
                settings.llm_runaway_min_repeat_ratio_tokens
            ),
            runaway_output_input_ratio=settings.llm_runaway_output_input_ratio,
            runaway_hard_output_input_ratio=settings.llm_runaway_hard_output_input_ratio,
        )

    def output_limit_retries_enabled(self) -> bool:
        return self.enabled and self.output_limit_retry_attempts > 0

    def runaway_detection_enabled(self) -> bool:
        return self.enabled and self.runaway_watchdog_enabled


@dataclass(frozen=True, slots=True)
class RepeatedTextSpan:
    """One repeated n-gram signal from generated text."""

    text: str
    n: int
    count: int
    repeat_ratio_tokens: float

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "n": self.n,
            "count": self.count,
            "repeat_ratio_tokens": self.repeat_ratio_tokens,
        }


@dataclass(frozen=True, slots=True)
class LLMRunawaySignals:
    """Mechanical signals collected from a partial LLM stream."""

    output_tokens: int
    source_input_tokens: int
    output_input_ratio: float | None
    max_repeat_count: int
    max_repeat_ratio_tokens: float
    repeated_phrases: tuple[RepeatedTextSpan, ...] = ()

    @property
    def repeated_phrase(self) -> str | None:
        return self.repeated_phrases[0].text if self.repeated_phrases else None

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "output_tokens": self.output_tokens,
            "source_input_tokens": self.source_input_tokens,
            "output_input_ratio": self.output_input_ratio,
            "max_repeat_count": self.max_repeat_count,
            "max_repeat_ratio_tokens": self.max_repeat_ratio_tokens,
            "repeated_phrases": [
                phrase.to_diagnostics() for phrase in self.repeated_phrases
            ],
        }


class LLMRunawayAbort(RuntimeError):
    """Raised by technical stream observers when a stream is mechanically runaway."""

    def __init__(
        self,
        reason: str,
        *,
        signals: LLMRunawaySignals,
        accumulated_text: str,
    ) -> None:
        super().__init__(reason)
        self.signals = signals
        self.accumulated_text = accumulated_text


class TechnicalRunawayObserver:
    """Mechanical stream watchdog that never calls another LLM."""

    def __init__(self, config: LLMTechnicalRecoveryConfig) -> None:
        self._config = config
        self._started_at = perf_counter()
        self._checks = 0
        self._next_check_tokens = config.runaway_min_output_tokens

    def reset_for_retry(self, _error: Any | None = None) -> None:
        self._started_at = perf_counter()
        self._checks = 0
        self._next_check_tokens = self._config.runaway_min_output_tokens

    async def on_text(
        self,
        _chunk: str,
        accumulated_text: str,
        request: Any,
    ) -> None:
        if not self._config.runaway_detection_enabled():
            return

        output_tokens = estimate_tokens(accumulated_text)
        source_input_tokens = _estimate_request_input_tokens(request)
        elapsed_seconds = perf_counter() - self._started_at

        if output_tokens >= self._config.runaway_hard_abort_min_output_tokens:
            signals = analyze_runaway_signals(
                accumulated_text,
                source_input_tokens=source_input_tokens,
            )
            if is_mechanical_runaway(signals, self._config, hard_abort=True):
                increment_run_counter(
                    "mechanical_runaway_abort_count",
                    layer="generic_runaway_observer",
                    mode="hard_abort",
                )
                raise LLMRunawayAbort(
                    "Technical watchdog detected runaway LLM output",
                    signals=signals,
                    accumulated_text=accumulated_text,
                )

        if self._checks >= self._config.runaway_max_checks:
            return
        if output_tokens < self._next_check_tokens:
            return
        if elapsed_seconds < self._config.runaway_min_elapsed_seconds:
            return

        self._checks += 1
        self._next_check_tokens = output_tokens + self._config.runaway_check_interval_tokens
        signals = analyze_runaway_signals(
            accumulated_text,
            source_input_tokens=source_input_tokens,
        )
        if is_mechanical_runaway(signals, self._config, hard_abort=False):
            increment_run_counter(
                "mechanical_runaway_abort_count",
                layer="generic_runaway_observer",
                mode="soft_check",
            )
            raise LLMRunawayAbort(
                "Technical watchdog detected runaway LLM output",
                signals=signals,
                accumulated_text=accumulated_text,
            )


class CompositeStreamObserver:
    """Fan out stream text observations to multiple observers in order."""

    def __init__(self, observers: tuple[Any, ...]) -> None:
        self._observers = observers

    async def on_text(
        self,
        chunk: str,
        accumulated_text: str,
        request: Any,
    ) -> None:
        for observer in self._observers:
            await observer.on_text(chunk, accumulated_text, request)

    async def reset_for_retry(self, error: Any | None = None) -> None:
        for observer in self._observers:
            reset = getattr(observer, "reset_for_retry", None)
            if not callable(reset):
                continue
            result = reset(error)
            if hasattr(result, "__await__"):
                await result


def compose_stream_observers(*observers: Any | None) -> Any | None:
    active = tuple(observer for observer in observers if observer is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return CompositeStreamObserver(active)


def analyze_runaway_signals(
    text: str,
    *,
    source_input_tokens: int,
) -> LLMRunawaySignals:
    analyzed = destructure_json_for_repetition_analysis(
        text[-_MAX_ANALYZED_CHARS:],
        min_value_chars=_MIN_JSON_VALUE_ANALYSIS_CHARS,
        include_outside_json_text=True,
        fallback_to_original=True,
    )
    output_tokens = estimate_tokens(text)
    source_tokens = max(1, int(source_input_tokens))
    repeated_phrases = analyze_repeated_text_spans(
        analyzed,
        n_values=(3, 4, 6, 8),
        min_count=2,
        max_phrases=8,
        max_phrase_chars=_MAX_REPEATED_PHRASE_CHARS,
    )
    max_repeat_count = max((item.count for item in repeated_phrases), default=0)
    max_repeat_ratio_tokens = max(
        (item.repeat_ratio_tokens for item in repeated_phrases),
        default=0.0,
    )

    ratio = output_tokens / source_tokens if source_tokens > 0 else None
    return LLMRunawaySignals(
        output_tokens=output_tokens,
        source_input_tokens=source_tokens,
        output_input_ratio=ratio,
        max_repeat_count=max_repeat_count,
        max_repeat_ratio_tokens=max_repeat_ratio_tokens,
        repeated_phrases=repeated_phrases,
    )


def analyze_repeated_text_spans(
    text: str,
    *,
    n_values: range | tuple[int, ...],
    min_count: int,
    max_phrases: int,
    max_phrase_chars: int,
    tokens: list[str] | None = None,
) -> tuple[RepeatedTextSpan, ...]:
    token_list = tokens if tokens is not None else text.split()
    repeated: list[RepeatedTextSpan] = []
    for n in n_values:
        if len(token_list) < n:
            continue
        counts = Counter(
            tuple(token_list[index:index + n])
            for index in range(0, len(token_list) - n + 1)
        )
        for gram, count in counts.items():
            if count < min_count:
                continue
            repeated.append(
                RepeatedTextSpan(
                    text=" ".join(gram)[:max_phrase_chars],
                    n=n,
                    count=count,
                    repeat_ratio_tokens=(n * count) / max(1, len(token_list)),
                )
            )
    repeated.sort(key=lambda item: (-item.repeat_ratio_tokens, -item.count, item.text))
    return tuple(repeated[:max_phrases])


def is_mechanical_runaway(
    signals: LLMRunawaySignals,
    config: LLMTechnicalRecoveryConfig,
    *,
    hard_abort: bool,
) -> bool:
    repeated = (
        signals.max_repeat_count >= config.runaway_min_repeat_count
        and signals.max_repeat_ratio_tokens >= config.runaway_min_repeat_ratio_tokens
    )
    ratio_threshold = (
        config.runaway_hard_output_input_ratio
        if hard_abort
        else config.runaway_output_input_ratio
    )
    runaway_growth = (
        signals.output_input_ratio is not None
        and signals.output_input_ratio >= ratio_threshold
    )
    return repeated or runaway_growth


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def destructure_json_for_repetition_analysis(
    text: str,
    *,
    value_keys: frozenset[str] | None = None,
    min_value_chars: int = _MIN_JSON_VALUE_ANALYSIS_CHARS,
    include_outside_json_text: bool = True,
    fallback_to_original: bool = True,
) -> str:
    """Return long JSON string values plus non-JSON text for mechanical checks."""
    segments = _json_analysis_segments(
        text,
        value_keys=value_keys,
        min_value_chars=min_value_chars,
        include_outside_json_text=include_outside_json_text,
    )
    if segments:
        return "\n".join(segments)[-_MAX_ANALYZED_CHARS:]
    return text[-_MAX_ANALYZED_CHARS:] if fallback_to_original else ""


def _json_analysis_segments(
    text: str,
    *,
    value_keys: frozenset[str] | None,
    min_value_chars: int,
    include_outside_json_text: bool,
) -> list[str]:
    segments: list[str] = []
    outside_buffer: list[str] = []
    pending_key: str | None = None
    depth = 0
    index = 0

    def flush_outside() -> None:
        if not outside_buffer:
            return
        outside = "".join(outside_buffer).strip()
        outside_buffer.clear()
        if outside:
            segments.append(outside)

    while index < len(text):
        char = text[index]
        if char == '"':
            parsed, next_index = _read_json_string(text, index)
            if parsed is None:
                partial = _read_partial_json_string(text, index)
                if depth == 0 and include_outside_json_text:
                    outside_buffer.append(partial)
                    flush_outside()
                elif _json_value_key_allowed(pending_key, value_keys):
                    _append_json_value_segment(
                        segments,
                        partial,
                        min_value_chars=min_value_chars,
                    )
                break

            if depth == 0 and include_outside_json_text:
                outside_buffer.append(parsed)
                index = next_index
                continue

            index = next_index
            cursor = index
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ":":
                pending_key = parsed
                index = cursor + 1
                continue
            if value_keys is None or _json_value_key_allowed(pending_key, value_keys):
                _append_json_value_segment(
                    segments,
                    parsed,
                    min_value_chars=min_value_chars,
                )
            pending_key = None
            continue

        if char in "{[":
            if depth == 0 and include_outside_json_text:
                flush_outside()
            depth += 1
            pending_key = None
            index += 1
            continue
        if char in "}]":
            depth = max(0, depth - 1)
            pending_key = None
            index += 1
            continue
        if depth == 0 and include_outside_json_text:
            outside_buffer.append(char)
        index += 1

    if include_outside_json_text:
        flush_outside()
    return segments


def _json_value_key_allowed(
    pending_key: str | None,
    value_keys: frozenset[str] | None,
) -> bool:
    if value_keys is None:
        return True
    return pending_key in value_keys


def _append_json_value_segment(
    segments: list[str],
    value: str,
    *,
    min_value_chars: int,
) -> None:
    normalized = value.strip()
    if len(normalized) >= min_value_chars:
        segments.append(normalized)


def _read_json_string(text: str, start: int) -> tuple[str | None, int]:
    chars: list[str] = []
    escaped = False
    index = start + 1
    while index < len(text):
        char = text[index]
        if escaped:
            chars.append(_unescape_json_char(char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(chars), index + 1
        else:
            chars.append(char)
        index += 1
    return None, index


def _read_partial_json_string(text: str, start: int) -> str:
    chars: list[str] = []
    escaped = False
    for char in text[start + 1:]:
        if escaped:
            chars.append(_unescape_json_char(char))
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            chars.append(char)
    return "".join(chars)


def _unescape_json_char(char: str) -> str:
    if char in {"n", "r", "t"}:
        return " "
    if char in {'"', "\\", "/"}:
        return char
    return char


def _estimate_request_input_tokens(request: Any) -> int:
    total = 0
    for message in getattr(request, "messages", []) or []:
        total += estimate_tokens(str(getattr(message, "content", "") or ""))
    return max(1, total)
