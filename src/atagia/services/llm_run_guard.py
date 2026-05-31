"""Runtime LLM budget and health guardrails."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class LLMRunGuardConfig:
    """Thresholds that stop unhealthy or runaway LLM usage."""

    enabled: bool = True
    mode: str = "enforce"
    max_total_calls: int | None = None
    max_total_failed_calls: int | None = 40
    max_failed_call_ratio: float | None = 0.50
    min_calls_for_failed_ratio: int = 20
    max_failed_calls_per_purpose: int | None = None
    max_failed_ratio_per_purpose: float | None = 0.50
    min_calls_per_purpose_for_failed_ratio: int = 10
    max_consecutive_failures_per_purpose: int | None = 8
    max_total_tokens: int | None = None
    max_reported_cost_usd: float | None = None
    max_wall_time_seconds: float | None = None

    @classmethod
    def disabled(cls) -> "LLMRunGuardConfig":
        return cls(
            enabled=False,
            mode="off",
            max_total_calls=None,
            max_total_failed_calls=None,
            max_failed_call_ratio=None,
            min_calls_for_failed_ratio=0,
            max_failed_calls_per_purpose=None,
            max_failed_ratio_per_purpose=None,
            min_calls_per_purpose_for_failed_ratio=0,
            max_consecutive_failures_per_purpose=None,
            max_total_tokens=None,
            max_reported_cost_usd=None,
            max_wall_time_seconds=None,
        )

    def normalized_mode(self) -> str:
        mode = self.mode.strip().lower()
        return mode if mode in {"off", "audit", "enforce"} else "enforce"

    def is_enforcing(self) -> bool:
        return self.enabled and self.normalized_mode() == "enforce"


def runtime_llm_run_guard_config(settings: Any) -> LLMRunGuardConfig:
    """Build the default runtime LLM guard config from Settings-like objects."""
    return LLMRunGuardConfig(
        enabled=bool(settings.llm_run_guard_enabled),
        mode=str(settings.llm_run_guard_mode),
        max_total_calls=settings.llm_run_guard_max_total_calls,
        max_total_failed_calls=settings.llm_run_guard_max_total_failed_calls,
        max_failed_call_ratio=settings.llm_run_guard_max_failed_call_ratio,
        min_calls_for_failed_ratio=settings.llm_run_guard_failed_ratio_min_calls,
        max_failed_calls_per_purpose=settings.llm_run_guard_max_failed_calls_per_purpose,
        max_failed_ratio_per_purpose=settings.llm_run_guard_max_failed_ratio_per_purpose,
        min_calls_per_purpose_for_failed_ratio=(
            settings.llm_run_guard_purpose_failure_ratio_min_calls
        ),
        max_consecutive_failures_per_purpose=(
            settings.llm_run_guard_max_consecutive_failures_per_purpose
        ),
        max_total_tokens=settings.llm_run_guard_max_total_tokens,
        max_reported_cost_usd=settings.llm_run_guard_max_reported_cost_usd,
        max_wall_time_seconds=None,
    )


def bulk_ingest_llm_run_guard_config(settings: Any) -> LLMRunGuardConfig:
    """Build the stricter scoped LLM guard used for bulk/admin rebuilds."""
    return LLMRunGuardConfig(
        enabled=bool(settings.bulk_ingest_llm_run_guard_enabled),
        mode=str(settings.llm_run_guard_mode),
        max_total_calls=settings.bulk_ingest_llm_run_guard_max_total_calls,
        max_total_failed_calls=settings.bulk_ingest_llm_run_guard_max_total_failed_calls,
        max_failed_call_ratio=settings.bulk_ingest_llm_run_guard_max_failed_call_ratio,
        min_calls_for_failed_ratio=(
            settings.bulk_ingest_llm_run_guard_failed_ratio_min_calls
        ),
        max_failed_calls_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_failed_calls_per_purpose
        ),
        max_failed_ratio_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose
        ),
        min_calls_per_purpose_for_failed_ratio=(
            settings.bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls
        ),
        max_consecutive_failures_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose
        ),
        max_total_tokens=settings.bulk_ingest_llm_run_guard_max_total_tokens,
        max_reported_cost_usd=settings.bulk_ingest_llm_run_guard_max_reported_cost_usd,
        max_wall_time_seconds=settings.bulk_ingest_llm_run_guard_max_wall_time_seconds,
    )


@dataclass(frozen=True, slots=True)
class LLMRunGuardDecision:
    """Current guard decision plus a JSON-safe state snapshot."""

    healthy: bool
    should_block: bool
    violations: tuple[str, ...] = ()
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PurposeCounters:
    calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0


@dataclass(slots=True)
class LLMRunGuardRun:
    """Mutable counters for one runtime or scoped LLM run."""

    run_id: str
    kind: str
    config: LLMRunGuardConfig
    started_at_monotonic: float
    status: str = "active"
    total_calls: int = 0
    failed_calls: int = 0
    total_tokens: float = 0.0
    reported_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    model_counts: Counter[str] = field(default_factory=Counter)
    error_counts: Counter[str] = field(default_factory=Counter)
    purposes: dict[str, _PurposeCounters] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def elapsed_seconds(self, now: float) -> float:
        return max(0.0, now - self.started_at_monotonic)


_CURRENT_RUN: ContextVar[LLMRunGuardRun | None] = ContextVar(
    "atagia_llm_run_guard_current_run",
    default=None,
)


class LLMRunGuard:
    """In-process guard for LLM call budgets and failure storms."""

    def __init__(
        self,
        default_config: LLMRunGuardConfig | None = None,
        *,
        now: Any = perf_counter,
    ) -> None:
        self._now = now
        self._default_run = LLMRunGuardRun(
            run_id="runtime",
            kind="runtime",
            config=default_config or LLMRunGuardConfig(),
            started_at_monotonic=float(self._now()),
        )
        self._last_scoped_snapshot: dict[str, Any] | None = None

    @contextmanager
    def scoped_run(
        self,
        *,
        run_id: str,
        kind: str,
        config: LLMRunGuardConfig | None = None,
    ) -> Iterator[LLMRunGuardRun]:
        """Apply a separate budget to LLM calls made inside the context."""
        run = LLMRunGuardRun(
            run_id=run_id,
            kind=kind,
            config=config or self._default_run.config,
            started_at_monotonic=float(self._now()),
        )
        token = _CURRENT_RUN.set(run)
        try:
            yield run
        finally:
            self._last_scoped_snapshot = self.snapshot(run)
            _CURRENT_RUN.reset(token)

    def maybe_scoped_run(
        self,
        *,
        run_id: str,
        kind: str,
        config: LLMRunGuardConfig | None = None,
    ) -> Any:
        return self.scoped_run(run_id=run_id, kind=kind, config=config)

    def reset_runtime(self) -> dict[str, Any]:
        """Reset the process-wide guard after an operator resolves the cause."""
        self._default_run = LLMRunGuardRun(
            run_id="runtime",
            kind="runtime",
            config=self._default_run.config,
            started_at_monotonic=float(self._now()),
        )
        return self.snapshot()

    def check_before_call(
        self,
        *,
        call_type: str,
        purpose: str | None,
        request_model: str,
    ) -> LLMRunGuardDecision:
        del call_type, purpose, request_model
        run = self._active_run()
        return self._decision(run, before_next_call=True)

    def record_success(
        self,
        *,
        call_type: str,
        purpose: str | None,
        request_model: str,
        response_model: str | None,
        provider: str | None,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> LLMRunGuardDecision:
        del call_type, response_model, provider
        run = self._active_run()
        self._record_call(
            run,
            purpose=purpose,
            request_model=request_model,
            usage=usage or {},
            latency_ms=latency_ms,
            error_type=None,
        )
        return self._decision(run, before_next_call=False)

    def record_failure(
        self,
        *,
        call_type: str,
        purpose: str | None,
        request_model: str,
        latency_ms: float,
        error_type: str,
    ) -> LLMRunGuardDecision:
        del call_type
        run = self._active_run()
        self._record_call(
            run,
            purpose=purpose,
            request_model=request_model,
            usage={},
            latency_ms=latency_ms,
            error_type=error_type or "UnknownError",
        )
        return self._decision(run, before_next_call=False)

    def snapshot(self, run: LLMRunGuardRun | None = None) -> dict[str, Any]:
        active_run = run or self._active_run()
        return self._snapshot(active_run)

    def runtime_snapshot(self) -> dict[str, Any]:
        snapshot = self._snapshot(self._default_run)
        if self._last_scoped_snapshot is not None:
            snapshot["last_scoped_run"] = dict(self._last_scoped_snapshot)
        return snapshot

    def _active_run(self) -> LLMRunGuardRun:
        return _CURRENT_RUN.get() or self._default_run

    def _record_call(
        self,
        run: LLMRunGuardRun,
        *,
        purpose: str | None,
        request_model: str,
        usage: dict[str, Any],
        latency_ms: float,
        error_type: str | None,
    ) -> None:
        purpose_label = (purpose or "unknown").strip() or "unknown"
        counters = run.purposes.setdefault(purpose_label, _PurposeCounters())
        run.total_calls += 1
        counters.calls += 1
        run.total_latency_ms += max(0.0, float(latency_ms))
        if request_model:
            run.model_counts[request_model] += 1
        run.total_tokens += _total_tokens(usage)
        run.reported_cost_usd += _reported_cost(usage)
        if error_type is None:
            counters.consecutive_failures = 0
            return
        run.failed_calls += 1
        run.error_counts[error_type] += 1
        counters.failed_calls += 1
        counters.consecutive_failures += 1
        counters.max_consecutive_failures = max(
            counters.max_consecutive_failures,
            counters.consecutive_failures,
        )

    def _decision(
        self,
        run: LLMRunGuardRun,
        *,
        before_next_call: bool,
    ) -> LLMRunGuardDecision:
        config = run.config
        if not config.enabled or config.normalized_mode() == "off":
            return LLMRunGuardDecision(
                healthy=True,
                should_block=False,
                snapshot=self._snapshot(run),
            )
        violations = self._violations(run, before_next_call=before_next_call)
        healthy = not violations
        if violations:
            run.violations = violations
            run.status = "failed" if config.is_enforcing() else "degraded"
        elif run.status != "failed":
            run.status = "active"
            run.violations = []
        return LLMRunGuardDecision(
            healthy=healthy,
            should_block=bool(violations and config.is_enforcing()),
            violations=tuple(violations),
            snapshot=self._snapshot(run),
        )

    def _violations(
        self,
        run: LLMRunGuardRun,
        *,
        before_next_call: bool,
    ) -> list[str]:
        config = run.config
        violations: list[str] = []
        total_calls = run.total_calls
        failed_calls = run.failed_calls

        if run.status == "failed" and run.violations:
            return list(run.violations)

        if config.max_total_calls is not None:
            limit = config.max_total_calls
            exceeded = total_calls >= limit if before_next_call else total_calls > limit
            if exceeded:
                violations.append(f"total LLM calls exceeded {limit}: {total_calls}")

        if (
            config.max_total_failed_calls is not None
            and failed_calls > config.max_total_failed_calls
        ):
            violations.append(
                "total failed LLM calls exceeded "
                f"{config.max_total_failed_calls}: {failed_calls}"
            )

        if (
            config.max_failed_call_ratio is not None
            and total_calls >= config.min_calls_for_failed_ratio
            and total_calls > 0
            and failed_calls / total_calls > config.max_failed_call_ratio
        ):
            violations.append(
                "LLM failure ratio exceeded "
                f"{config.max_failed_call_ratio:.2%}: "
                f"{failed_calls}/{total_calls} ({failed_calls / total_calls:.2%})"
            )

        if (
            config.max_total_tokens is not None
            and run.total_tokens > config.max_total_tokens
        ):
            violations.append(
                f"total LLM tokens exceeded {config.max_total_tokens}: "
                f"{int(run.total_tokens)}"
            )

        if (
            config.max_reported_cost_usd is not None
            and run.reported_cost_usd > config.max_reported_cost_usd
        ):
            violations.append(
                "reported LLM cost exceeded "
                f"{config.max_reported_cost_usd:.4f}: {run.reported_cost_usd:.4f}"
            )

        if (
            config.max_wall_time_seconds is not None
            and run.elapsed_seconds(float(self._now())) > config.max_wall_time_seconds
        ):
            violations.append(
                "LLM run wall time exceeded "
                f"{config.max_wall_time_seconds:.1f}s: "
                f"{run.elapsed_seconds(float(self._now())):.1f}s"
            )

        for purpose, counters in sorted(run.purposes.items()):
            if (
                config.max_failed_calls_per_purpose is not None
                and counters.failed_calls > config.max_failed_calls_per_purpose
            ):
                violations.append(
                    "failed LLM calls for purpose "
                    f"{purpose!r} exceeded {config.max_failed_calls_per_purpose}: "
                    f"{counters.failed_calls}"
                )
            if (
                config.max_failed_ratio_per_purpose is not None
                and counters.calls >= config.min_calls_per_purpose_for_failed_ratio
                and counters.calls > 0
                and counters.failed_calls / counters.calls
                > config.max_failed_ratio_per_purpose
            ):
                violations.append(
                    "LLM failure ratio for purpose "
                    f"{purpose!r} exceeded {config.max_failed_ratio_per_purpose:.2%}: "
                    f"{counters.failed_calls}/{counters.calls} "
                    f"({counters.failed_calls / counters.calls:.2%})"
                )
            if (
                config.max_consecutive_failures_per_purpose is not None
                and counters.max_consecutive_failures
                > config.max_consecutive_failures_per_purpose
            ):
                violations.append(
                    "consecutive failed LLM calls for purpose "
                    f"{purpose!r} exceeded "
                    f"{config.max_consecutive_failures_per_purpose}: "
                    f"{counters.max_consecutive_failures}"
                )

        return violations

    def _snapshot(self, run: LLMRunGuardRun) -> dict[str, Any]:
        config = run.config
        return {
            "run_id": run.run_id,
            "kind": run.kind,
            "status": run.status,
            "mode": config.normalized_mode(),
            "enabled": config.enabled,
            "elapsed_seconds": run.elapsed_seconds(float(self._now())),
            "total_calls": run.total_calls,
            "failed_calls": run.failed_calls,
            "failure_ratio": (
                run.failed_calls / run.total_calls if run.total_calls else 0.0
            ),
            "total_tokens": int(run.total_tokens),
            "reported_cost_usd": run.reported_cost_usd,
            "total_latency_ms": run.total_latency_ms,
            "model_call_counts": dict(sorted(run.model_counts.items())),
            "error_class_counts": dict(sorted(run.error_counts.items())),
            "violations": list(run.violations),
            "thresholds": {
                "max_total_calls": config.max_total_calls,
                "max_total_failed_calls": config.max_total_failed_calls,
                "max_failed_call_ratio": config.max_failed_call_ratio,
                "min_calls_for_failed_ratio": config.min_calls_for_failed_ratio,
                "max_failed_calls_per_purpose": config.max_failed_calls_per_purpose,
                "max_failed_ratio_per_purpose": config.max_failed_ratio_per_purpose,
                "min_calls_per_purpose_for_failed_ratio": (
                    config.min_calls_per_purpose_for_failed_ratio
                ),
                "max_consecutive_failures_per_purpose": (
                    config.max_consecutive_failures_per_purpose
                ),
                "max_total_tokens": config.max_total_tokens,
                "max_reported_cost_usd": config.max_reported_cost_usd,
                "max_wall_time_seconds": config.max_wall_time_seconds,
            },
            "by_purpose": {
                purpose: {
                    "calls": counters.calls,
                    "failed_calls": counters.failed_calls,
                    "failure_ratio": (
                        counters.failed_calls / counters.calls
                        if counters.calls
                        else 0.0
                    ),
                    "consecutive_failures": counters.consecutive_failures,
                    "max_consecutive_failures": counters.max_consecutive_failures,
                }
                for purpose, counters in sorted(run.purposes.items())
            },
        }


def _total_tokens(usage: dict[str, Any]) -> float:
    total = _number_at(usage, ("total_tokens",)) or _number_at(usage, ("totalTokenCount",))
    if total is not None:
        return total
    input_tokens = (
        _number_at(usage, ("input_tokens",))
        or _number_at(usage, ("prompt_tokens",))
        or _number_at(usage, ("promptTokenCount",))
        or 0.0
    )
    output_tokens = (
        _number_at(usage, ("output_tokens",))
        or _number_at(usage, ("completion_tokens",))
        or _number_at(usage, ("candidatesTokenCount",))
        or 0.0
    )
    return input_tokens + output_tokens


def _reported_cost(usage: dict[str, Any]) -> float:
    return (
        _number_at(usage, ("cost",))
        or _number_at(usage, ("cost_details", "upstream_inference_cost"))
        or 0.0
    )


def _number_at(source: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = source
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
