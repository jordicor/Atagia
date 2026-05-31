"""Tests for benchmark LLM usage aggregation."""

from __future__ import annotations

import pytest

from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    merge_llm_call_summaries,
)
from benchmarks.llm_run_guard import LLMRunGuardConfig, LLMRunGuardError
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMMessage,
    OutputLimitExceededError,
    LLMProvider,
    LLMStreamEvent,
)


class StreamingMetricsProvider(LLMProvider):
    name = "streaming-metrics"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(provider=self.name, model=request.model, output_text="ok")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(provider=self.name, model=request.model, vectors=[])

    async def stream(self, request: LLMCompletionRequest):
        yield LLMStreamEvent(type="text", content='{"label":"ok"}')
        yield LLMStreamEvent(type="done", payload={"usage": {"completion_tokens": 3}})


def test_llm_call_summary_aggregates_provider_reported_costs() -> None:
    recorder = LLMCallRecorder()
    request = LLMCompletionRequest(
        model="openrouter/test-model",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={"purpose": "benchmark_answer_generation"},
    )

    recorder.record_completion_success(
        request,
        LLMCompletionResponse(
            provider="openrouter",
            model="openrouter/test-model",
            output_text="ok",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "cost": 0.0125,
                "cost_details": {
                    "upstream_inference_cost": "0.0100",
                    "upstream_inference_prompt_cost": 0.004,
                    "upstream_inference_completions_cost": 0.006,
                },
            },
        ),
        latency_ms=42.0,
    )

    summary = recorder.summary()

    assert summary["cost_totals"] == {
        "cost": 0.0125,
        "upstream_inference_cost": 0.01,
        "upstream_inference_prompt_cost": 0.004,
        "upstream_inference_completions_cost": 0.006,
    }
    assert summary["by_purpose"]["benchmark_answer_generation"]["cost_totals"] == summary[
        "cost_totals"
    ]
    assert recorder.records()[0]["cost_counts"] == summary["cost_totals"]


def test_llm_call_summary_counts_structured_output_repair_calls() -> None:
    recorder = LLMCallRecorder()
    retry_request = LLMCompletionRequest(
        model="openrouter/test-model",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={
            "purpose": "need_detection",
            "atagia_structured_output_retry": True,
            "atagia_structured_output_retry_attempt": 1,
        },
    )
    rescue_request = LLMCompletionRequest(
        model="anthropic/claude-opus-4-7",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={
            "purpose": "need_detection",
            "atagia_structured_output_rescue": True,
            "atagia_structured_output_rescue_model": "anthropic/claude-opus-4-7",
        },
    )

    for request in (retry_request, rescue_request):
        recorder.record_completion_success(
            request,
            LLMCompletionResponse(
                provider="test",
                model=request.model,
                output_text="{}",
            ),
            latency_ms=10.0,
        )

    summary = recorder.summary()

    assert summary["structured_output_repair"]["retry_calls"] == 1
    assert summary["structured_output_repair"]["rescue_calls"] == 1
    assert summary["structured_output_repair"]["rescue_model_call_counts"] == {
        "anthropic/claude-opus-4-7": 1
    }
    assert summary["by_purpose"]["need_detection"]["structured_output_repair"] == {
        "retry_calls": 1,
        "rescue_calls": 1,
        "rescue_model_call_counts": {"anthropic/claude-opus-4-7": 1},
        "by_purpose": {"need_detection": {"retry_calls": 1, "rescue_calls": 1}},
    }
    assert recorder.records()[1]["metadata"]["atagia_structured_output_rescue"] is True


def test_llm_call_summary_records_error_classes_and_bounded_retry_success() -> None:
    recorder = LLMCallRecorder()
    first_request = LLMCompletionRequest(
        model="anthropic/claude-sonnet-4-6",
        messages=[LLMMessage(role="user", content="extract")],
        metadata={"purpose": "memory_extraction"},
    )
    bounded_request = first_request.model_copy(
        update={
            "metadata": {
                "purpose": "memory_extraction",
                "extraction_retry_mode": "bounded_output",
            }
        }
    )

    recorder.record_completion_failure(
        first_request,
        latency_ms=100.0,
        exc=OutputLimitExceededError(
            "max output tokens",
            finish_reason="length",
            max_output_tokens=8192,
            partial_output_chars=1200,
            partial_output_excerpt="tail excerpt",
        ),
    )
    recorder.record_completion_success(
        bounded_request,
        LLMCompletionResponse(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-6",
            output_text="{}",
        ),
        latency_ms=50.0,
    )

    summary = recorder.summary()

    assert summary["failed_calls"] == 1
    assert summary["first_attempt_failures"] == 1
    assert summary["output_limit_failures"] == 1
    assert summary["watchdog_bounded_retries"] == 1
    assert summary["error_class_counts"] == {"OutputLimitExceededError": 1}
    assert summary["retry_success_counts"] == {"extraction_bounded_output": 1}
    assert summary["sample_errors"][0]["purpose"] == "memory_extraction"
    assert summary["sample_errors"][0]["finish_reason"] == "length"
    assert summary["sample_errors"][0]["max_output_tokens"] == 8192
    assert summary["sample_errors"][0]["partial_output_chars"] == 1200
    assert summary["sample_errors"][0]["partial_output_excerpt"] == "tail excerpt"
    assert recorder.records()[1]["metadata"]["extraction_retry_mode"] == "bounded_output"


def test_llm_run_guard_fails_on_purpose_failure_ratio() -> None:
    recorder = LLMCallRecorder()
    request = LLMCompletionRequest(
        model="openai/test-model",
        messages=[LLMMessage(role="user", content="extract")],
        metadata={"purpose": "memory_extraction"},
    )

    for index in range(5):
        if index < 3:
            recorder.record_completion_failure(
                request,
                latency_ms=10.0,
                exc=RuntimeError("provider failed"),
            )
        else:
            recorder.record_completion_success(
                request,
                LLMCompletionResponse(
                    provider="test",
                    model=request.model,
                    output_text="{}",
                ),
                latency_ms=10.0,
            )

    with pytest.raises(LLMRunGuardError, match="memory_extraction"):
        recorder.raise_if_unhealthy(
            LLMRunGuardConfig(
                max_failed_llm_call_ratio=None,
                max_failed_ratio_per_purpose=0.30,
                min_calls_per_purpose_for_failed_ratio=5,
                max_consecutive_failures_per_purpose=None,
            )
        )


def test_merge_llm_call_summaries_preserves_cost_totals_by_purpose() -> None:
    merged = merge_llm_call_summaries(
        [
            {
                "total_calls": 1,
                "failed_calls": 0,
                "total_latency_ms": 10.0,
                "mean_latency_ms": 10.0,
                "token_totals": {"input_tokens": 10.0},
                "cost_totals": {"cost": 0.01},
                "model_call_counts": {"openrouter/a": 1},
                "by_purpose": {
                    "need_detection": {
                        "calls": 1,
                        "failed_calls": 0,
                        "total_latency_ms": 10.0,
                        "mean_latency_ms": 10.0,
                        "token_totals": {"input_tokens": 10.0},
                        "cost_totals": {"cost": 0.01},
                    }
                },
            },
            {
                "total_calls": 2,
                "failed_calls": 0,
                "total_latency_ms": 30.0,
                "mean_latency_ms": 15.0,
                "token_totals": {"input_tokens": 20.0},
                "cost_totals": {"cost": 0.02},
                "model_call_counts": {"openrouter/a": 2},
                "by_purpose": {
                    "need_detection": {
                        "calls": 2,
                        "failed_calls": 0,
                        "total_latency_ms": 30.0,
                        "mean_latency_ms": 15.0,
                        "token_totals": {"input_tokens": 20.0},
                        "cost_totals": {"cost": 0.02},
                    }
                },
            },
        ]
    )

    assert merged["total_calls"] == 3
    assert merged["cost_totals"] == {"cost": 0.03}
    assert merged["by_purpose"]["need_detection"]["cost_totals"] == {"cost": 0.03}


@pytest.mark.asyncio
async def test_llm_call_recorder_records_streamed_completions() -> None:
    provider = StreamingMetricsProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    request = LLMCompletionRequest(
        model="openai/test-model",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={"purpose": "memory_extraction"},
    )

    await client.complete_streamed(request)

    summary = recorder.summary()
    assert summary["by_purpose"]["memory_extraction"]["calls"] == 1
    assert summary["by_purpose"]["memory_extraction"]["token_totals"]["output_tokens"] == 3.0
