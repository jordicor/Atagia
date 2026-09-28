"""Provider dispatch limits across independent callers of one engine client."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMMessage,
    LLMProvider,
    LLMStreamEvent,
    RetryPolicy,
    TransientLLMError,
)
from atagia.services.llm_run_guard import LLMRunGuard, LLMRunGuardConfig


def _completion() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="stub-model",
        messages=[LLMMessage(role="user", content="hello")],
    )


class BlockingProvider(LLMProvider):
    name = "dispatch-test"

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.active = 0
        self.peak = 0
        self.started = 0
        self.at_capacity = asyncio.Event()
        self.release = asyncio.Event()

    async def _block(self) -> None:
        self.started += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == self.capacity:
            self.at_capacity.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        await self._block()
        return LLMCompletionResponse(provider=self.name, model=request.model)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        await self._block()
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[1.0])],
        )

    async def stream(self, request: LLMCompletionRequest):
        self.started += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            yield LLMStreamEvent(type="text", content="first")
            await self.release.wait()
            yield LLMStreamEvent(type="done", payload={"usage": {}})
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_dispatch_limit_is_shared_by_completions_and_embeddings() -> None:
    provider = BlockingProvider(capacity=2)
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        max_concurrent_requests_per_provider=2,
    )
    tasks = [asyncio.create_task(client.complete(_completion())) for _ in range(5)]
    tasks += [
        asyncio.create_task(
            client.embed(LLMEmbeddingRequest(model="stub-model", input_texts=["x"]))
        )
        for _ in range(3)
    ]
    try:
        await asyncio.wait_for(provider.at_capacity.wait(), 1)
        await asyncio.sleep(0)
        assert provider.started == 2
        assert provider.peak == 2
        provider.release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert provider.started == 8
        assert provider.peak == 2
        assert provider.active == 0
    finally:
        provider.release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_waiting_and_active_cancellation_release_capacity() -> None:
    provider = BlockingProvider(capacity=1)
    guard = LLMRunGuard(LLMRunGuardConfig())
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        max_concurrent_requests_per_provider=1,
        llm_run_guard=guard,
    )
    active = asyncio.create_task(client.complete(_completion()))
    await asyncio.wait_for(provider.at_capacity.wait(), 1)
    waiting = asyncio.create_task(client.complete(_completion()))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert provider.started == 1
    assert guard.runtime_snapshot()["total_calls"] == 0
    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert provider.active == 0
    assert guard.runtime_snapshot()["cancelled_calls"] == 1
    provider.release.set()
    await asyncio.wait_for(client.complete(_completion()), 1)
    assert provider.started == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed_completion", [False, True])
async def test_stream_releases_capacity_when_cancelled(streamed_completion: bool) -> None:
    provider = BlockingProvider(capacity=1)
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        max_concurrent_requests_per_provider=1,
    )
    if streamed_completion:
        task = asyncio.create_task(client.complete_streamed(_completion()))
        await asyncio.sleep(0)
        assert provider.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        async with aclosing(client.stream(_completion())) as events:
            assert (await anext(events)).content == "first"
            assert provider.active == 1
    assert provider.active == 0
    provider.release.set()
    await asyncio.wait_for(client.complete(_completion()), 1)


class RetryProvider(LLMProvider):
    name = "retry-dispatch"

    def __init__(self) -> None:
        self.failed = asyncio.Event()
        self.first_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "first":
            self.first_calls += 1
            if self.first_calls == 1:
                self.failed.set()
                raise TransientLLMError("temporary")
        return LLMCompletionResponse(provider=self.name, model=request.model)


@pytest.mark.asyncio
async def test_retry_backoff_does_not_occupy_dispatch_slot() -> None:
    provider = RetryProvider()
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        max_concurrent_requests_per_provider=1,
        retry_policy=RetryPolicy(
            attempts=2,
            base_delay_seconds=0.2,
            max_delay_seconds=0.2,
            jitter_fraction=0,
        ),
    )
    first = asyncio.create_task(
        client.complete(_completion().model_copy(update={"metadata": {"purpose": "first"}}))
    )
    await asyncio.wait_for(provider.failed.wait(), 1)
    await asyncio.wait_for(client.complete(_completion()), 0.1)
    await asyncio.wait_for(first, 1)
    assert provider.first_calls == 2
