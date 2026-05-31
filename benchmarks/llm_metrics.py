"""Lightweight LLM call recording for benchmark runs."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from time import perf_counter
from typing import Any, Iterator

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
)

logger = logging.getLogger(__name__)


_CURRENT_LLM_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "benchmark_llm_context",
    default={},
)


class LLMCallRecorder:
    """Collect timing and usage metadata for benchmark LLM calls."""

    def __init__(self, *, progress_interval: int = 0) -> None:
        self._records: list[dict[str, Any]] = []
        self._next_sequence = 1
        self._progress_interval = max(0, int(progress_interval))

    @contextmanager
    def context(self, **fields: Any) -> Iterator[None]:
        """Attach benchmark context to calls made within this async task."""
        current = dict(_CURRENT_LLM_CONTEXT.get())
        current.update({key: value for key, value in fields.items() if value is not None})
        token = _CURRENT_LLM_CONTEXT.set(current)
        try:
            yield
        finally:
            _CURRENT_LLM_CONTEXT.reset(token)

    def records(self) -> list[dict[str, Any]]:
        """Return recorded calls as JSON-serializable dictionaries."""
        return [dict(record) for record in self._records]

    def records_for_context(self, **fields: Any) -> list[dict[str, Any]]:
        """Return calls whose benchmark context matches all provided fields."""
        expected = {key: value for key, value in fields.items() if value is not None}
        if not expected:
            return self.records()
        matches: list[dict[str, Any]] = []
        for record in self._records:
            context = record.get("context")
            if not isinstance(context, dict):
                continue
            if all(context.get(key) == value for key, value in expected.items()):
                matches.append(dict(record))
        return matches

    def summary(self) -> dict[str, Any]:
        """Return aggregate latency/token metrics for all recorded calls."""
        return summarize_llm_calls(self._records)

    def raise_if_unhealthy(
        self,
        config: Any,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Raise when configured benchmark LLM health thresholds are exceeded."""
        from benchmarks.llm_run_guard import raise_if_llm_run_unhealthy

        raise_if_llm_run_unhealthy(
            self._records,
            config,
            elapsed_seconds=elapsed_seconds,
        )

    def record_completion_success(
        self,
        request: LLMCompletionRequest,
        response: LLMCompletionResponse,
        latency_ms: float,
    ) -> None:
        self._append_record(
            call_type="completion",
            request_model=request.model,
            response_model=response.model,
            provider=response.provider,
            metadata=request.metadata,
            usage=response.usage,
            latency_ms=latency_ms,
            error=None,
        )

    def record_completion_failure(
        self,
        request: LLMCompletionRequest,
        latency_ms: float,
        exc: Exception,
    ) -> None:
        self._append_record(
            call_type="completion",
            request_model=request.model,
            response_model=None,
            provider=None,
            metadata=request.metadata,
            usage={},
            latency_ms=latency_ms,
            error=_error_payload(exc),
        )

    def record_embedding_success(
        self,
        request: LLMEmbeddingRequest,
        response: LLMEmbeddingResponse,
        latency_ms: float,
    ) -> None:
        self._append_record(
            call_type="embedding",
            request_model=request.model,
            response_model=response.model,
            provider=response.provider,
            metadata=request.metadata,
            usage={},
            latency_ms=latency_ms,
            error=None,
        )

    def record_embedding_failure(
        self,
        request: LLMEmbeddingRequest,
        latency_ms: float,
        exc: Exception,
    ) -> None:
        self._append_record(
            call_type="embedding",
            request_model=request.model,
            response_model=None,
            provider=None,
            metadata=request.metadata,
            usage={},
            latency_ms=latency_ms,
            error=_error_payload(exc),
        )

    def _append_record(
        self,
        *,
        call_type: str,
        request_model: str,
        response_model: str | None,
        provider: str | None,
        metadata: dict[str, Any],
        usage: dict[str, Any],
        latency_ms: float,
        error: dict[str, Any] | None,
    ) -> None:
        sequence = self._next_sequence
        self._next_sequence += 1
        purpose = str(metadata.get("purpose") or "")
        record = {
            "sequence": sequence,
            "call_type": call_type,
            "purpose": purpose,
            "request_model": request_model,
            "response_model": response_model,
            "provider": provider,
            "latency_ms": latency_ms,
            "usage": _json_safe(usage),
            "token_counts": _token_counts(usage),
            "cost_counts": _cost_counts(usage),
            "metadata": _metadata_summary(metadata),
            "context": _json_safe(_CURRENT_LLM_CONTEXT.get()),
            "error": error,
        }
        self._records.append(record)
        if (
            self._progress_interval
            and sequence % self._progress_interval == 0
        ):
            summary = self.summary()
            logger.info(
                "benchmark_llm_progress calls=%s failed=%s total_tokens=%s by_purpose=%s",
                summary.get("total_calls"),
                summary.get("failed_calls"),
                int((summary.get("token_totals") or {}).get("total_tokens") or 0),
                {
                    purpose: {
                        "calls": group.get("calls"),
                        "failed_calls": group.get("failed_calls"),
                    }
                    for purpose, group in (summary.get("by_purpose") or {}).items()
                    if isinstance(group, dict)
                },
            )


def install_llm_call_recorder(
    client: LLMClient[Any],
    recorder: LLMCallRecorder,
) -> None:
    """Patch one client instance so all completion/embed calls are recorded."""
    if getattr(client, "_benchmark_llm_recorder_installed", False):
        return

    original_complete = client.complete
    original_complete_streamed = client.complete_streamed
    original_embed = client.embed

    async def complete(request: LLMCompletionRequest) -> LLMCompletionResponse:
        started_at = perf_counter()
        try:
            response = await original_complete(request)
        except Exception as exc:
            recorder.record_completion_failure(
                request,
                (perf_counter() - started_at) * 1000.0,
                exc,
            )
            raise
        recorder.record_completion_success(
            request,
            response,
            (perf_counter() - started_at) * 1000.0,
        )
        return response

    async def complete_streamed(
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
    ) -> LLMCompletionResponse:
        started_at = perf_counter()
        try:
            response = await original_complete_streamed(request, observer=observer)
        except Exception as exc:
            recorder.record_completion_failure(
                request,
                (perf_counter() - started_at) * 1000.0,
                exc,
            )
            raise
        recorder.record_completion_success(
            request,
            response,
            (perf_counter() - started_at) * 1000.0,
        )
        return response

    async def embed(request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        started_at = perf_counter()
        try:
            response = await original_embed(request)
        except Exception as exc:
            recorder.record_embedding_failure(
                request,
                (perf_counter() - started_at) * 1000.0,
                exc,
            )
            raise
        recorder.record_embedding_success(
            request,
            response,
            (perf_counter() - started_at) * 1000.0,
        )
        return response

    client.complete = complete  # type: ignore[method-assign]
    client.complete_streamed = complete_streamed  # type: ignore[method-assign]
    client.embed = embed  # type: ignore[method-assign]
    client._benchmark_llm_recorder_installed = True  # type: ignore[attr-defined]


def summarize_llm_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate recorded LLM calls by purpose and model."""
    total_latency_ms = sum(_number(record.get("latency_ms")) or 0.0 for record in records)
    purpose_records: dict[str, list[dict[str, Any]]] = {}
    model_counter: Counter[str] = Counter()
    for record in records:
        purpose = str(record.get("purpose") or "unknown")
        purpose_records.setdefault(purpose, []).append(record)
        model = str(record.get("request_model") or "")
        if model:
            model_counter[model] += 1
    return {
        "total_calls": len(records),
        "failed_calls": sum(1 for record in records if record.get("error") is not None),
        "first_attempt_failures": _first_attempt_failures(records),
        "output_limit_failures": _error_class_counts(records).get("OutputLimitExceededError", 0),
        "watchdog_bounded_retries": _watchdog_bounded_retries(records),
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": (total_latency_ms / len(records)) if records else None,
        "token_totals": _sum_token_counts(records),
        "cost_totals": _sum_cost_counts(records),
        "model_call_counts": dict(sorted(model_counter.items())),
        "error_class_counts": _error_class_counts(records),
        "sample_errors": _sample_errors(records),
        "retry_success_counts": _retry_success_counts(records),
        "structured_output_repair": _structured_output_repair_summary(records),
        "by_purpose": {
            purpose: _summarize_group(group)
            for purpose, group in sorted(purpose_records.items())
        },
    }


def merge_llm_call_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge conversation-level LLM summaries into a benchmark-level summary."""
    records: list[dict[str, Any]] = []
    for summary in summaries:
        if not summary:
            continue
        for purpose, group in (summary.get("by_purpose") or {}).items():
            if not isinstance(group, dict):
                continue
            records.append(
                {
                    "purpose": purpose,
                    "request_model": "",
                    "latency_ms": group.get("total_latency_ms") or 0.0,
                    "token_counts": group.get("token_totals") or {},
                    "cost_counts": group.get("cost_totals") or {},
                    "error": None,
                    "_summary_count": group.get("calls") or 0,
                    "_summary_failed": group.get("failed_calls") or 0,
                    "_summary_first_attempt_failed": group.get("first_attempt_failures") or 0,
                    "_summary_output_limit_failed": group.get("output_limit_failures") or 0,
                    "_summary_watchdog_bounded_retries": group.get("watchdog_bounded_retries") or 0,
                    "_summary_error_class_counts": group.get("error_class_counts") or {},
                    "_summary_retry_success_counts": group.get("retry_success_counts") or {},
                    "_summary_sample_errors": group.get("sample_errors") or [],
                }
            )
    if not records:
        return summarize_llm_calls([])
    total_calls = sum(int(record.get("_summary_count") or 0) for record in records)
    failed_calls = sum(int(record.get("_summary_failed") or 0) for record in records)
    first_attempt_failures = sum(
        int(record.get("_summary_first_attempt_failed") or 0) for record in records
    )
    output_limit_failures = sum(
        int(record.get("_summary_output_limit_failed") or 0) for record in records
    )
    watchdog_bounded_retries = sum(
        int(record.get("_summary_watchdog_bounded_retries") or 0) for record in records
    )
    total_latency_ms = sum(_number(record.get("latency_ms")) or 0.0 for record in records)
    by_purpose: dict[str, dict[str, Any]] = {}
    for record in records:
        purpose = str(record["purpose"])
        group = by_purpose.setdefault(
            purpose,
            {
                "calls": 0,
                "failed_calls": 0,
                "total_latency_ms": 0.0,
                "mean_latency_ms": None,
                "token_totals": {},
                "cost_totals": {},
                "first_attempt_failures": 0,
                "output_limit_failures": 0,
                "watchdog_bounded_retries": 0,
                "error_class_counts": {},
                "retry_success_counts": {},
                "sample_errors": [],
            },
        )
        group["calls"] += int(record.get("_summary_count") or 0)
        group["failed_calls"] += int(record.get("_summary_failed") or 0)
        group["first_attempt_failures"] += int(record.get("_summary_first_attempt_failed") or 0)
        group["output_limit_failures"] += int(record.get("_summary_output_limit_failed") or 0)
        group["watchdog_bounded_retries"] += int(record.get("_summary_watchdog_bounded_retries") or 0)
        group["total_latency_ms"] += _number(record.get("latency_ms")) or 0.0
        group["token_totals"] = _merge_token_counts(
            group["token_totals"],
            record.get("token_counts") if isinstance(record.get("token_counts"), dict) else {},
        )
        group["cost_totals"] = _merge_numeric_counts(
            group["cost_totals"],
            record.get("cost_counts") if isinstance(record.get("cost_counts"), dict) else {},
        )
        group["error_class_counts"] = _merge_int_counts(
            group["error_class_counts"],
            record.get("_summary_error_class_counts"),
        )
        group["retry_success_counts"] = _merge_int_counts(
            group["retry_success_counts"],
            record.get("_summary_retry_success_counts"),
        )
        if isinstance(record.get("_summary_sample_errors"), list):
            group["sample_errors"].extend(record["_summary_sample_errors"][:3])
            group["sample_errors"] = group["sample_errors"][:3]
    for group in by_purpose.values():
        calls = int(group["calls"])
        group["mean_latency_ms"] = group["total_latency_ms"] / calls if calls else None
    return {
        "total_calls": total_calls,
        "failed_calls": failed_calls,
        "first_attempt_failures": first_attempt_failures,
        "output_limit_failures": output_limit_failures,
        "watchdog_bounded_retries": watchdog_bounded_retries,
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": total_latency_ms / total_calls if total_calls else None,
        "token_totals": _merge_many_token_counts(
            summary.get("token_totals") or {}
            for summary in summaries
            if isinstance(summary.get("token_totals"), dict)
        ),
        "cost_totals": _merge_many_numeric_counts(
            summary.get("cost_totals") or {}
            for summary in summaries
            if isinstance(summary.get("cost_totals"), dict)
        ),
        "model_call_counts": _merge_model_counts(summaries),
        "error_class_counts": _merge_many_int_counts(
            summary.get("error_class_counts") for summary in summaries
        ),
        "sample_errors": _merge_sample_errors(
            summary.get("sample_errors") for summary in summaries
        ),
        "retry_success_counts": _merge_many_int_counts(
            summary.get("retry_success_counts") for summary in summaries
        ),
        "structured_output_repair": _merge_structured_output_repair_summaries(summaries),
        "by_purpose": dict(sorted(by_purpose.items())),
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_latency_ms = sum(_number(record.get("latency_ms")) or 0.0 for record in records)
    return {
        "calls": len(records),
        "failed_calls": sum(1 for record in records if record.get("error") is not None),
        "first_attempt_failures": _first_attempt_failures(records),
        "output_limit_failures": _error_class_counts(records).get("OutputLimitExceededError", 0),
        "watchdog_bounded_retries": _watchdog_bounded_retries(records),
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": total_latency_ms / len(records) if records else None,
        "token_totals": _sum_token_counts(records),
        "cost_totals": _sum_cost_counts(records),
        "error_class_counts": _error_class_counts(records),
        "sample_errors": _sample_errors(records),
        "retry_success_counts": _retry_success_counts(records),
        "structured_output_repair": _structured_output_repair_summary(records),
    }


def _error_class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        error_type = _record_error_type(record)
        if error_type:
            counts[error_type] += 1
    return dict(sorted(counts.items()))


def _first_attempt_failures(records: list[dict[str, Any]]) -> int:
    failures = 0
    for record in records:
        if record.get("error") is None:
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            failures += 1
            continue
        if _truthy(metadata.get("atagia_structured_output_retry")):
            continue
        if _truthy(metadata.get("atagia_structured_output_rescue")):
            continue
        if metadata.get("extraction_retry_mode"):
            continue
        failures += 1
    return failures


def _retry_success_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("error") is not None:
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if _truthy(metadata.get("atagia_structured_output_retry")):
            counts["structured_output_retry"] += 1
        if _truthy(metadata.get("atagia_structured_output_rescue")):
            counts["structured_output_rescue"] += 1
        retry_mode = metadata.get("extraction_retry_mode")
        if isinstance(retry_mode, str) and retry_mode:
            counts[f"extraction_{retry_mode}"] += 1
    return dict(sorted(counts.items()))


def _watchdog_bounded_retries(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("extraction_retry_mode") == "bounded_output":
            count += 1
    return count


def _sample_errors(records: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        error = record.get("error")
        if not isinstance(error, dict):
            continue
        samples.append(
            {
                "sequence": record.get("sequence"),
                "purpose": record.get("purpose") or "unknown",
                "call_type": record.get("call_type"),
                "request_model": record.get("request_model"),
                "error_type": _record_error_type(record),
                "message": _truncate(str(error.get("message") or ""), 300),
                "finish_reason": error.get("finish_reason"),
                "max_output_tokens": error.get("max_output_tokens"),
                "partial_output_chars": error.get("partial_output_chars"),
                "partial_output_excerpt": error.get("partial_output_excerpt"),
                "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
                "context": record.get("context") if isinstance(record.get("context"), dict) else {},
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _record_error_type(record: dict[str, Any]) -> str | None:
    error = record.get("error")
    if not isinstance(error, dict):
        return None
    error_type = error.get("type")
    if isinstance(error_type, str) and error_type:
        return error_type
    return "UnknownError"


def _structured_output_repair_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    retry_calls = 0
    rescue_calls = 0
    rescue_model_counts: Counter[str] = Counter()
    by_purpose: dict[str, dict[str, int]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        purpose = str(record.get("purpose") or "unknown")
        purpose_counts = by_purpose.setdefault(
            purpose,
            {"retry_calls": 0, "rescue_calls": 0},
        )
        if _truthy(metadata.get("atagia_structured_output_retry")):
            retry_calls += 1
            purpose_counts["retry_calls"] += 1
        if _truthy(metadata.get("atagia_structured_output_rescue")):
            rescue_calls += 1
            purpose_counts["rescue_calls"] += 1
            rescue_model = metadata.get("atagia_structured_output_rescue_model")
            if isinstance(rescue_model, str) and rescue_model:
                rescue_model_counts[rescue_model] += 1
    by_purpose = {
        purpose: counts
        for purpose, counts in sorted(by_purpose.items())
        if counts["retry_calls"] or counts["rescue_calls"]
    }
    return {
        "retry_calls": retry_calls,
        "rescue_calls": rescue_calls,
        "rescue_model_call_counts": dict(sorted(rescue_model_counts.items())),
        "by_purpose": by_purpose,
    }


def _merge_structured_output_repair_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    retry_calls = 0
    rescue_calls = 0
    rescue_model_counts: Counter[str] = Counter()
    by_purpose: dict[str, dict[str, int]] = {}
    for summary in summaries:
        repair = summary.get("structured_output_repair")
        if not isinstance(repair, dict):
            continue
        retry_calls += int(repair.get("retry_calls") or 0)
        rescue_calls += int(repair.get("rescue_calls") or 0)
        model_counts = repair.get("rescue_model_call_counts")
        if isinstance(model_counts, dict):
            for model, amount in model_counts.items():
                try:
                    rescue_model_counts[str(model)] += int(amount)
                except (TypeError, ValueError):
                    continue
        purpose_counts = repair.get("by_purpose")
        if isinstance(purpose_counts, dict):
            for purpose, counts in purpose_counts.items():
                if not isinstance(counts, dict):
                    continue
                merged = by_purpose.setdefault(
                    str(purpose),
                    {"retry_calls": 0, "rescue_calls": 0},
                )
                merged["retry_calls"] += int(counts.get("retry_calls") or 0)
                merged["rescue_calls"] += int(counts.get("rescue_calls") or 0)
    return {
        "retry_calls": retry_calls,
        "rescue_calls": rescue_calls,
        "rescue_model_call_counts": dict(sorted(rescue_model_counts.items())),
        "by_purpose": dict(sorted(by_purpose.items())),
    }


def _sum_token_counts(records: list[dict[str, Any]]) -> dict[str, float]:
    return _merge_many_token_counts(
        record.get("token_counts")
        for record in records
        if isinstance(record.get("token_counts"), dict)
    )


def _merge_many_token_counts(values: Any) -> dict[str, float]:
    return _merge_many_numeric_counts(values)


def _sum_cost_counts(records: list[dict[str, Any]]) -> dict[str, float]:
    return _merge_many_numeric_counts(
        record.get("cost_counts")
        for record in records
        if isinstance(record.get("cost_counts"), dict)
    )


def _merge_many_numeric_counts(values: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        totals = _merge_numeric_counts(totals, value)
    return totals


def _merge_token_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return _merge_numeric_counts(left, right)


def _merge_numeric_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    totals = {key: float(value) for key, value in left.items() if _number(value) is not None}
    for key, value in right.items():
        numeric = _number(value)
        if numeric is None:
            continue
        totals[key] = totals.get(key, 0.0) + numeric
    return totals


def _merge_int_counts(left: dict[str, Any], right: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for source in (left, right):
        if not isinstance(source, dict):
            continue
        for key, amount in source.items():
            try:
                counts[str(key)] += int(amount)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _merge_many_int_counts(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, amount in value.items():
            try:
                counts[str(key)] += int(amount)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _merge_sample_errors(values: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                samples.append(item)
                if len(samples) >= limit:
                    return samples
    return samples


def _merge_model_counts(summaries: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for summary in summaries:
        model_counts = summary.get("model_call_counts")
        if not isinstance(model_counts, dict):
            continue
        for model, amount in model_counts.items():
            try:
                counts[str(model)] += int(amount)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _token_counts(usage: dict[str, Any]) -> dict[str, float]:
    counts: dict[str, float] = {}
    input_tokens = _first_number(
        usage,
        ("input_tokens",),
        ("prompt_tokens",),
        ("promptTokenCount",),
    )
    output_tokens = _first_number(
        usage,
        ("output_tokens",),
        ("completion_tokens",),
        ("candidatesTokenCount",),
    )
    thinking_tokens = _first_number(
        usage,
        ("thinking_tokens",),
        ("reasoning_tokens",),
        ("thoughtsTokenCount",),
        ("completion_tokens_details", "reasoning_tokens"),
    )
    total_tokens = _first_number(
        usage,
        ("total_tokens",),
        ("totalTokenCount",),
    )
    cached_input_tokens = _first_number(
        usage,
        ("cache_read_input_tokens",),
        ("prompt_tokens_details", "cached_tokens"),
        ("cachedContentTokenCount",),
    )
    for key, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("thinking_tokens", thinking_tokens),
        ("cached_input_tokens", cached_input_tokens),
    ):
        if value is not None:
            counts[key] = value
    if total_tokens is None:
        total_tokens = sum(counts.get(key, 0.0) for key in ("input_tokens", "output_tokens"))
    if total_tokens:
        counts["total_tokens"] = total_tokens
    return counts


def _cost_counts(usage: dict[str, Any]) -> dict[str, float]:
    """Extract provider-reported monetary fields without estimating prices."""
    counts: dict[str, float] = {}
    for key, path in (
        ("cost", ("cost",)),
        ("upstream_inference_cost", ("cost_details", "upstream_inference_cost")),
        ("upstream_inference_prompt_cost", ("cost_details", "upstream_inference_prompt_cost")),
        ("upstream_inference_completions_cost", ("cost_details", "upstream_inference_completions_cost")),
    ):
        value = _first_cost_number(usage, path)
        if value is not None:
            counts[key] = value
    return counts


def _first_number(usage: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = usage
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        numeric = _number(value)
        if numeric is not None:
            return numeric
    return None


def _first_cost_number(usage: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = usage
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _cost_number(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _cost_number(value: Any) -> float | None:
    numeric = _number(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "purpose",
        "question",
        "atagia_model_spec",
        "atagia_canonical_model",
        "atagia_provider_slug",
        "atagia_structured_output_failure_class",
        "atagia_structured_output_retry",
        "atagia_structured_output_retry_attempt",
        "atagia_structured_output_retry_primary_model",
        "atagia_structured_output_rescue",
        "atagia_structured_output_rescue_model",
        "atagia_structured_output_rescue_original_model",
        "atagia_structured_output_rescue_retry_attempts",
        "extraction_retry_mode",
        "extraction_retry_trigger_class",
        "extraction_watchdog_abort_policy",
        "extraction_watchdog_confidence",
        "extraction_watchdog_elapsed_seconds",
        "extraction_watchdog_evidence_type",
        "extraction_watchdog_gate_trigger",
        "extraction_watchdog_latest_output_excerpt_chars",
        "extraction_watchdog_max_repeat_count",
        "extraction_watchdog_max_repeat_ratio_tokens",
        "extraction_watchdog_mechanical_evidence",
        "extraction_watchdog_output_input_ratio",
        "extraction_watchdog_output_tokens",
        "extraction_watchdog_reason",
        "extraction_watchdog_repeated_phrases",
        "output_limit_finish_reason",
        "output_limit_max_output_tokens",
        "output_limit_partial_output_chars",
        "output_limit_partial_output_excerpt",
    }
    return {
        key: _truncate(value, 300) if isinstance(value, str) else _json_safe(value)
        for key, value in metadata.items()
        if key in allowed
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _error_payload(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": _truncate(str(exc), 500),
    }
    for attr in (
        "provider",
        "finish_reason",
        "max_output_tokens",
        "partial_output_chars",
        "partial_output_excerpt",
    ):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        payload[attr] = _truncate(value, 1000) if isinstance(value, str) else _json_safe(value)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."
