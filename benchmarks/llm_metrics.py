"""Lightweight LLM call recording for benchmark runs."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
)


_CURRENT_LLM_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "benchmark_llm_context",
    default={},
)


class LLMCallRecorder:
    """Collect timing and usage metadata for benchmark LLM calls."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._next_sequence = 1

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
            error={
                "type": type(exc).__name__,
                "message": _truncate(str(exc), 500),
            },
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
            error={
                "type": type(exc).__name__,
                "message": _truncate(str(exc), 500),
            },
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
        error: dict[str, str] | None,
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
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": (total_latency_ms / len(records)) if records else None,
        "token_totals": _sum_token_counts(records),
        "cost_totals": _sum_cost_counts(records),
        "model_call_counts": dict(sorted(model_counter.items())),
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
                }
            )
    if not records:
        return summarize_llm_calls([])
    total_calls = sum(int(record.get("_summary_count") or 0) for record in records)
    failed_calls = sum(int(record.get("_summary_failed") or 0) for record in records)
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
            },
        )
        group["calls"] += int(record.get("_summary_count") or 0)
        group["failed_calls"] += int(record.get("_summary_failed") or 0)
        group["total_latency_ms"] += _number(record.get("latency_ms")) or 0.0
        group["token_totals"] = _merge_token_counts(
            group["token_totals"],
            record.get("token_counts") if isinstance(record.get("token_counts"), dict) else {},
        )
        group["cost_totals"] = _merge_numeric_counts(
            group["cost_totals"],
            record.get("cost_counts") if isinstance(record.get("cost_counts"), dict) else {},
        )
    for group in by_purpose.values():
        calls = int(group["calls"])
        group["mean_latency_ms"] = group["total_latency_ms"] / calls if calls else None
    return {
        "total_calls": total_calls,
        "failed_calls": failed_calls,
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
        "by_purpose": dict(sorted(by_purpose.items())),
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_latency_ms = sum(_number(record.get("latency_ms")) or 0.0 for record in records)
    return {
        "calls": len(records),
        "failed_calls": sum(1 for record in records if record.get("error") is not None),
        "total_latency_ms": total_latency_ms,
        "mean_latency_ms": total_latency_ms / len(records) if records else None,
        "token_totals": _sum_token_counts(records),
        "cost_totals": _sum_cost_counts(records),
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
    }
    return {
        key: _truncate(value, 300) if isinstance(value, str) else _json_safe(value)
        for key, value in metadata.items()
        if key in allowed
    }


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
