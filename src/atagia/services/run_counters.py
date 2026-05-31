"""Run-scoped counters for benchmark-visible internal events."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator


_CURRENT_RUN_COUNTERS: ContextVar["RunCounterAccumulator | None"] = ContextVar(
    "atagia_run_counters",
    default=None,
)


class RunCounterAccumulator:
    """Accumulate low-cardinality counters for the current benchmark run."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._labeled_counts: dict[str, Counter[str]] = {}

    def increment(
        self,
        counter_name: str,
        amount: int = 1,
        **labels: Any,
    ) -> None:
        """Increment a counter and optional label-specific breakdown."""
        if amount <= 0:
            return
        normalized_name = str(counter_name).strip()
        if not normalized_name:
            return
        self._counts[normalized_name] += int(amount)
        label_key = _label_key(labels)
        if label_key is None:
            return
        labeled = self._labeled_counts.setdefault(normalized_name, Counter())
        labeled[label_key] += int(amount)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable counter snapshot."""
        return {
            "counts": dict(sorted(self._counts.items())),
            "labeled_counts": {
                counter_name: dict(sorted(counter.items()))
                for counter_name, counter in sorted(self._labeled_counts.items())
            },
        }


def empty_run_counters() -> dict[str, Any]:
    """Return the canonical empty manifest payload for run counters."""
    return {"counts": {}, "labeled_counts": {}}


def normalize_run_counters(value: Any) -> dict[str, Any]:
    """Normalize arbitrary report payloads into the manifest counter shape."""
    if not isinstance(value, dict):
        return empty_run_counters()
    counts = value.get("counts")
    labeled_counts = value.get("labeled_counts")
    return {
        "counts": _int_mapping(counts),
        "labeled_counts": {
            str(counter_name): _int_mapping(counter)
            for counter_name, counter in (labeled_counts or {}).items()
            if isinstance(counter, dict)
        }
        if isinstance(labeled_counts, dict)
        else {},
    }


def set_run_counter_accumulator(
    accumulator: RunCounterAccumulator,
) -> Token[RunCounterAccumulator | None]:
    """Install an accumulator in the current context and return a reset token."""
    return _CURRENT_RUN_COUNTERS.set(accumulator)


def reset_run_counter_accumulator(
    token: Token[RunCounterAccumulator | None],
) -> None:
    """Reset the current run-counter context."""
    _CURRENT_RUN_COUNTERS.reset(token)


@contextmanager
def use_run_counter_accumulator(
    accumulator: RunCounterAccumulator,
) -> Iterator[None]:
    """Temporarily install an accumulator in the current context."""
    token = set_run_counter_accumulator(accumulator)
    try:
        yield
    finally:
        reset_run_counter_accumulator(token)


def increment_run_counter(
    counter_name: str,
    amount: int = 1,
    **labels: Any,
) -> None:
    """Increment the current run counter when a benchmark context is active."""
    accumulator = _CURRENT_RUN_COUNTERS.get()
    if accumulator is None:
        return
    accumulator.increment(counter_name, amount=amount, **labels)


def current_run_counters() -> dict[str, Any]:
    """Return the active accumulator snapshot or the canonical empty payload."""
    accumulator = _CURRENT_RUN_COUNTERS.get()
    if accumulator is None:
        return empty_run_counters()
    return accumulator.snapshot()


def _label_key(labels: dict[str, Any]) -> str | None:
    normalized = {
        str(key): str(value)
        for key, value in labels.items()
        if value is not None and str(key).strip()
    }
    if not normalized:
        return None
    label_key = "|".join(
        f"{key}={normalized[key]}"
        for key in sorted(normalized)
        if normalized[key]
    )
    return label_key or None


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw_count in value.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            result[str(key)] = count
    return dict(sorted(result.items()))
