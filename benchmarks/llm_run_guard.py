"""Fail-fast health gates for benchmark LLM usage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from benchmarks.llm_metrics import summarize_llm_calls


@dataclass(frozen=True)
class LLMRunGuardConfig:
    """Thresholds that make long benchmark runs stop when LLM calls go bad."""

    max_total_llm_calls: int | None = None
    max_total_failed_llm_calls: int | None = None
    max_failed_llm_call_ratio: float | None = 0.20
    min_calls_for_failed_ratio: int = 10
    max_failed_calls_per_purpose: int | None = None
    max_failed_ratio_per_purpose: float | None = 0.30
    min_calls_per_purpose_for_failed_ratio: int = 5
    max_total_tokens: int | None = None
    max_wall_time_seconds: float | None = None
    max_consecutive_failures_per_purpose: int | None = 8

    @classmethod
    def disabled(cls) -> "LLMRunGuardConfig":
        return cls(
            max_total_llm_calls=None,
            max_total_failed_llm_calls=None,
            max_failed_llm_call_ratio=None,
            min_calls_for_failed_ratio=0,
            max_failed_calls_per_purpose=None,
            max_failed_ratio_per_purpose=None,
            min_calls_per_purpose_for_failed_ratio=0,
            max_total_tokens=None,
            max_wall_time_seconds=None,
            max_consecutive_failures_per_purpose=None,
        )


@dataclass(frozen=True)
class LLMRunGuardDecision:
    """Structured result returned by LLM run guard checks."""

    healthy: bool
    violations: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class LLMRunGuardError(RuntimeError):
    """Raised when benchmark LLM health thresholds are exceeded."""

    def __init__(self, decision: LLMRunGuardDecision) -> None:
        self.decision = decision
        message = "; ".join(decision.violations) or "LLM run guard failed"
        super().__init__(message)


def evaluate_llm_records(
    records: list[dict[str, Any]],
    config: LLMRunGuardConfig,
    *,
    elapsed_seconds: float | None = None,
) -> LLMRunGuardDecision:
    """Return a health decision for raw benchmark LLM call records."""

    violations: list[str] = []
    summary = summarize_llm_calls(records)
    total_calls = int(summary.get("total_calls") or 0)
    failed_calls = int(summary.get("failed_calls") or 0)

    if config.max_total_llm_calls is not None and total_calls > config.max_total_llm_calls:
        violations.append(
            f"total LLM calls exceeded {config.max_total_llm_calls}: {total_calls}"
        )

    if (
        config.max_total_failed_llm_calls is not None
        and failed_calls > config.max_total_failed_llm_calls
    ):
        violations.append(
            "total failed LLM calls exceeded "
            f"{config.max_total_failed_llm_calls}: {failed_calls}"
        )

    if (
        config.max_failed_llm_call_ratio is not None
        and total_calls >= config.min_calls_for_failed_ratio
        and total_calls > 0
    ):
        failed_ratio = failed_calls / total_calls
        if failed_ratio > config.max_failed_llm_call_ratio:
            violations.append(
                "LLM failure ratio exceeded "
                f"{config.max_failed_llm_call_ratio:.2%}: {failed_calls}/{total_calls} "
                f"({failed_ratio:.2%})"
            )

    total_tokens = int(((summary.get("token_totals") or {}).get("total_tokens") or 0))
    if config.max_total_tokens is not None and total_tokens > config.max_total_tokens:
        violations.append(
            f"total LLM tokens exceeded {config.max_total_tokens}: {total_tokens}"
        )

    if (
        config.max_wall_time_seconds is not None
        and elapsed_seconds is not None
        and elapsed_seconds > config.max_wall_time_seconds
    ):
        violations.append(
            "LLM benchmark wall time exceeded "
            f"{config.max_wall_time_seconds:.1f}s: {elapsed_seconds:.1f}s"
        )

    _append_purpose_violations(records, config, violations)
    _append_consecutive_failure_violations(records, config, violations)

    return LLMRunGuardDecision(
        healthy=not violations,
        violations=violations,
        summary=summary,
    )


def raise_if_llm_run_unhealthy(
    records: list[dict[str, Any]],
    config: LLMRunGuardConfig,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    decision = evaluate_llm_records(records, config, elapsed_seconds=elapsed_seconds)
    if not decision.healthy:
        raise LLMRunGuardError(decision)


def _append_purpose_violations(
    records: list[dict[str, Any]],
    config: LLMRunGuardConfig,
    violations: list[str],
) -> None:
    calls_by_purpose: Counter[str] = Counter()
    failures_by_purpose: Counter[str] = Counter()
    for record in records:
        purpose = str(record.get("purpose") or "unknown")
        calls_by_purpose[purpose] += 1
        if record.get("error") is not None:
            failures_by_purpose[purpose] += 1

    for purpose in sorted(calls_by_purpose):
        calls = calls_by_purpose[purpose]
        failures = failures_by_purpose[purpose]
        if (
            config.max_failed_calls_per_purpose is not None
            and failures > config.max_failed_calls_per_purpose
        ):
            violations.append(
                "failed LLM calls for purpose "
                f"{purpose!r} exceeded {config.max_failed_calls_per_purpose}: {failures}"
            )
        if (
            config.max_failed_ratio_per_purpose is not None
            and calls >= config.min_calls_per_purpose_for_failed_ratio
            and calls > 0
            and failures / calls > config.max_failed_ratio_per_purpose
        ):
            violations.append(
                "LLM failure ratio for purpose "
                f"{purpose!r} exceeded {config.max_failed_ratio_per_purpose:.2%}: "
                f"{failures}/{calls} ({failures / calls:.2%})"
            )


def _append_consecutive_failure_violations(
    records: list[dict[str, Any]],
    config: LLMRunGuardConfig,
    violations: list[str],
) -> None:
    limit = config.max_consecutive_failures_per_purpose
    if limit is None:
        return

    current_by_purpose: Counter[str] = Counter()
    max_by_purpose: Counter[str] = Counter()
    for record in sorted(records, key=lambda item: int(item.get("sequence") or 0)):
        purpose = str(record.get("purpose") or "unknown")
        if record.get("error") is None:
            current_by_purpose[purpose] = 0
            continue
        current_by_purpose[purpose] += 1
        max_by_purpose[purpose] = max(max_by_purpose[purpose], current_by_purpose[purpose])

    for purpose, consecutive_failures in sorted(max_by_purpose.items()):
        if consecutive_failures > limit:
            violations.append(
                "consecutive failed LLM calls for purpose "
                f"{purpose!r} exceeded {limit}: {consecutive_failures}"
            )
