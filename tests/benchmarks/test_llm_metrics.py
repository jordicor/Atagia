"""Tests for benchmark LLM usage aggregation."""

from __future__ import annotations

import pytest

from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    merge_llm_call_summaries,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMMessage,
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
