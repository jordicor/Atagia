"""Tests for runtime LLM run guard integration."""

from __future__ import annotations

import pytest

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMMessage,
    LLMProvider,
    LLMRunGuardError,
    RetryPolicy,
    TransientLLMError,
)
from atagia.services.llm_run_guard import LLMRunGuard, LLMRunGuardConfig


class StaticProvider(LLMProvider):
    name = "guard-tests"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="ok",
            usage={"input_tokens": 7, "output_tokens": 3},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embedding is not used in this test")


class AlwaysTransientProvider(StaticProvider):
    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        raise TransientLLMError("temporary outage")


class SequenceProvider(StaticProvider):
    def __init__(self, outcomes: list[str]) -> None:
        super().__init__()
        self.outcomes = list(outcomes)

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome == "fail":
            raise TransientLLMError("temporary outage")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="ok",
        )


def _request(purpose: str = "extractor") -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="model-a",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={"purpose": purpose},
    )


def _client(
    provider: LLMProvider,
    config: LLMRunGuardConfig,
    *,
    attempts: int = 1,
) -> LLMClient[object]:
    return LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=RetryPolicy(
            attempts=attempts,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        llm_run_guard=LLMRunGuard(config),
    )


@pytest.mark.asyncio
async def test_run_guard_counts_retry_attempts_and_stops_failure_storms() -> None:
    provider = AlwaysTransientProvider()
    client = _client(
        provider,
        LLMRunGuardConfig(
            max_total_failed_calls=1,
            max_failed_call_ratio=None,
            max_failed_ratio_per_purpose=None,
            max_consecutive_failures_per_purpose=None,
        ),
        attempts=3,
    )

    with pytest.raises(LLMRunGuardError) as error:
        await client.complete(_request())

    assert provider.calls == 2
    assert error.value.decision.snapshot["failed_calls"] == 2
    assert error.value.decision.snapshot["error_class_counts"] == {
        "TransientLLMError": 2
    }


@pytest.mark.asyncio
async def test_run_guard_blocks_before_call_when_total_call_budget_is_spent() -> None:
    provider = StaticProvider()
    client = _client(
        provider,
        LLMRunGuardConfig(
            max_total_calls=1,
            max_total_failed_calls=None,
            max_failed_call_ratio=None,
            max_failed_ratio_per_purpose=None,
            max_consecutive_failures_per_purpose=None,
        ),
    )

    await client.complete(_request())
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())

    assert provider.calls == 1


@pytest.mark.asyncio
async def test_run_guard_enforces_failure_ratio_per_purpose() -> None:
    provider = SequenceProvider(["fail", "ok"])
    client = _client(
        provider,
        LLMRunGuardConfig(
            max_total_failed_calls=None,
            max_failed_call_ratio=None,
            max_failed_ratio_per_purpose=0.49,
            min_calls_per_purpose_for_failed_ratio=2,
            max_consecutive_failures_per_purpose=None,
        ),
    )

    with pytest.raises(TransientLLMError):
        await client.complete(_request("topic_working_set_update"))
    with pytest.raises(LLMRunGuardError) as error:
        await client.complete(_request("topic_working_set_update"))

    by_purpose = error.value.decision.snapshot["by_purpose"]
    assert by_purpose["topic_working_set_update"]["calls"] == 2
    assert by_purpose["topic_working_set_update"]["failed_calls"] == 1


@pytest.mark.asyncio
async def test_scoped_bulk_guard_does_not_spend_runtime_budget() -> None:
    provider = StaticProvider()
    client = _client(
        provider,
        LLMRunGuardConfig(
            max_total_calls=1,
            max_total_failed_calls=None,
            max_failed_call_ratio=None,
            max_failed_ratio_per_purpose=None,
            max_consecutive_failures_per_purpose=None,
        ),
    )

    scoped_config = LLMRunGuardConfig(
        max_total_calls=2,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    with client.llm_run_guard_scope(
        run_id="bulk-1",
        kind="admin_rebuild",
        config=scoped_config,
    ):
        await client.complete(_request())
        await client.complete(_request())

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 0
    assert snapshot["last_scoped_run"]["total_calls"] == 2

    await client.complete(_request())
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())
