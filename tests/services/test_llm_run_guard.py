"""Tests for runtime LLM run guard integration.

THE GUARD BLOCKS BEFORE A CALL, NEVER AFTER ONE.

This file used to encode two contradictory contracts: one test asserted the
guard refuses a call BEFORE the provider is reached, another asserted it raises
AFTER a provider round-trip that succeeded. The pre-call contract is the one
kept, everywhere. A violation can only be discovered by recording a call, and
the call that reveals it is frequently a SUCCESS (a success is non-decreasing on
a failure ratio, but it is the sample that finally reaches the ratio's minimum
call count). Raising on it would discard an answer the provider already
generated and the operator already paid for -- and, since the runtime guard is
process-wide, it would discard it for whichever user happened to arrive after
somebody else's outage. Recording therefore only latches the verdict; the block
lands on the next call.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRunGuardError,
    LLMStreamEvent,
    RetryPolicy,
    TransientLLMError,
)
from atagia.services.llm_run_guard import (
    LLMRunGuard,
    LLMRunGuardConfig,
    LLMRunGuardDecision,
    begin_llm_call_meter,
    end_llm_call_meter,
)


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


class SilentUsageProvider(StaticProvider):
    """Succeeds and reports no usage at all, like most providers report no cost."""

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="ok",
            usage={},
        )


class BrokenStreamProvider(StaticProvider):
    """Streams one chunk, then raises an error the adapter never normalized."""

    async def stream(
        self,
        request: LLMCompletionRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls += 1
        yield LLMStreamEvent(type="text", content="partial")
        raise ValueError("adapter bug")


class GatedProvider(StaticProvider):
    """Blocks inside the first provider call until released, then fails it.

    Lets a test hold one call in flight while the guard is reset underneath it,
    which is the shape of an operator resolving an outage while a straggler from
    that outage has not returned yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_first = True

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        if self.fail_first:
            self.fail_first = False
            self.entered.set()
            await self.release.wait()
            raise TransientLLMError("outage straggler")
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


class _FakeClock:
    """Monotonic clock the test advances by hand.

    The guard's recovery window is a real duration, so the only way to prove a
    process survives a failure storm without sleeping through it in the test
    suite is to control the clock the guard reads.
    """

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def _client(
    provider: LLMProvider,
    config: LLMRunGuardConfig,
    *,
    attempts: int = 1,
    guard: LLMRunGuard | None = None,
) -> LLMClient[object]:
    return LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=RetryPolicy(
            attempts=attempts,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        llm_run_guard=guard or LLMRunGuard(config),
    )


@pytest.mark.asyncio
async def test_run_guard_counts_retry_attempts_and_stops_failure_storms() -> None:
    provider = AlwaysTransientProvider()
    client = _client(
        provider,
        LLMRunGuardConfig(
            # A budget of 2 permits 2: the third attempt is refused before the
            # provider, which is what makes the retry attempts observable as
            # separate charges against the run.
            max_total_failed_calls=2,
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
    """The success that TRIPS the ratio is delivered; the next call is refused.

    The second round-trip succeeds at the provider and is also the call that
    pushes the purpose to ``min_calls_per_purpose_for_failed_ratio``, so the
    ratio is evaluated for the first time and trips on it. The answer belongs to
    the caller that paid for it; the verdict belongs to the next call.
    """
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

    response = await client.complete(_request("topic_working_set_update"))
    assert response.output_text == "ok"
    assert provider.calls == 2

    # The next call is refused BEFORE the provider is reached -- which the
    # provider proves by having no scripted outcome left to pop.
    with pytest.raises(LLMRunGuardError) as error:
        await client.complete(_request("topic_working_set_update"))
    assert provider.calls == 2

    by_purpose = error.value.decision.snapshot["by_purpose"]
    assert by_purpose["topic_working_set_update"]["calls"] == 2
    assert by_purpose["topic_working_set_update"]["failed_calls"] == 1


@pytest.mark.asyncio
async def test_one_streamed_round_trip_is_metered_exactly_once() -> None:
    """A stream that trips the guard must not be counted twice.

    ``LLMRunGuardError`` is an ``LLMError``, so a post-success raise from inside
    the stream's own ``try`` was caught by its ``except LLMError`` and recorded a
    SECOND, failed round-trip for the same provider call -- inflating the very
    per-turn metric CS-1.4 exists to measure, and worsening the ratio that kept
    the guard tripped.
    """
    provider = SequenceProvider(["fail", "ok"])
    client = _client(
        provider,
        LLMRunGuardConfig(
            max_total_failed_calls=None,
            max_failed_call_ratio=0.49,
            min_calls_for_failed_ratio=2,
            max_failed_ratio_per_purpose=None,
            max_consecutive_failures_per_purpose=None,
        ),
    )

    meter = begin_llm_call_meter()
    delivered = ""
    try:
        with pytest.raises(TransientLLMError):
            await client.complete(_request("chat_reply"))
        async for event in client.stream(_request("chat_reply")):
            if event.type == "text":
                delivered += event.content or ""
    finally:
        end_llm_call_meter(meter)

    # The stream was delivered in full and raised nothing after delivery.
    assert delivered == "ok"
    assert provider.calls == 2
    # One failed completion + one successful stream. Not three, not two failures.
    assert meter.total_calls == 2
    assert meter.failed_calls == 1
    assert meter.by_purpose["chat_reply"].calls == 2

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 2
    assert snapshot["failed_calls"] == 1
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request("chat_reply"))
    assert provider.calls == 2


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


# ---------------------------------------------------------------------------
# Budget verdicts are final; health verdicts expire.
#
# The process-wide run never ends, so every absolute cumulative counter on it
# eventually crosses any fixed threshold no matter how healthy the process is.
# Its health signals therefore have to be bounded (a window, a current streak)
# AND recoverable -- and recoverable means the block has to expire, because
# blocking starves the very counters that would clear it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_survives_a_transient_failure_storm() -> None:
    """A bad minute blocks while it lasts, then the process works again."""
    clock = _FakeClock()
    provider = SequenceProvider(["fail", "fail", "fail", "fail", "ok"])
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=4,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    # Four real provider failures. The fourth reaches the streak threshold, and
    # its caller still gets the true cause -- the provider error -- not the
    # guard's verdict on it.
    for _ in range(4):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    assert provider.calls == 4

    # Blocked, and the provider is genuinely not reached.
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())
    assert provider.calls == 4

    clock.advance(61.0)

    # An idle process must not report itself dead once its window has elapsed:
    # the next call goes through as the probe, so the snapshot has to say so --
    # and it must not say "active" either, because a recovering run serves that
    # one call and refuses every other. The violations stay visible: they are
    # what the run is recovering FROM, and nothing has contradicted them yet.
    idle = client.llm_run_guard_snapshot()
    assert idle is not None
    assert idle["status"] == "recovering"
    assert idle["violations"] != []
    assert idle["probe_in_flight"] is False
    assert idle["health_block_streak"] == 1
    assert idle["blocked_seconds_remaining"] is None

    response = await client.complete(_request())
    assert response.output_text == "ok"
    assert provider.calls == 5

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "active"
    assert snapshot["violations"] == []
    assert snapshot["recovery_count"] == 1
    assert snapshot["health_block_streak"] == 0
    assert snapshot["blocked_seconds_remaining"] is None
    # The ledger survives the recovery: only the health signals are cleared.
    assert snapshot["total_calls"] == 5
    assert snapshot["failed_calls"] == 4


@pytest.mark.asyncio
async def test_budget_verdict_never_expires() -> None:
    """A spent budget does not un-spend, however long you wait."""
    clock = _FakeClock()
    provider = AlwaysTransientProvider()
    config = LLMRunGuardConfig(
        max_total_failed_calls=2,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    # The second failure spends the budget. Its caller sees the provider error
    # that caused it; the block lands on the call after it.
    for _ in range(2):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())

    clock.advance(3600.0)

    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())
    assert provider.calls == 2

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["verdict_is_final"] is True
    assert snapshot["recovery_count"] == 0


@pytest.mark.asyncio
async def test_failure_ratio_reads_the_recent_window_not_the_whole_run() -> None:
    """A live outage trips even after a long healthy history.

    Under lifetime accounting the same sequence is 11 failures in 51 calls --
    21.6%, comfortably under a 50% threshold -- which is exactly how a ratio
    computed over a long-lived process stops detecting outages.
    """
    clock = _FakeClock()
    provider = SequenceProvider(["ok"] * 40 + ["fail"] * 11)
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=0.50,
        min_calls_for_failed_ratio=10,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
        health_window_calls=20,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    for _ in range(40):
        await client.complete(_request())
    for _ in range(11):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    # The 11th failure crosses the windowed ratio, so the next call is refused
    # before the provider -- which has no scripted outcome left to give it.
    with pytest.raises(LLMRunGuardError) as error:
        await client.complete(_request())
    assert provider.calls == 51

    snapshot = error.value.decision.snapshot
    assert snapshot["recent_calls"] == 20
    assert snapshot["recent_failed_calls"] == 11
    assert snapshot["recent_failure_ratio"] == pytest.approx(0.55)
    assert snapshot["failure_ratio"] == pytest.approx(11 / 51)
    assert snapshot["failure_ratio"] < 0.50


@pytest.mark.asyncio
async def test_consecutive_failures_read_the_current_streak() -> None:
    """A historic burst must not condemn a run that is failing intermittently."""
    clock = _FakeClock()
    provider = SequenceProvider(["fail", "fail", "fail"] + ["ok", "fail"] * 6)
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=3,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    for _ in range(3):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())
    assert provider.calls == 3

    clock.advance(61.0)

    for _ in range(6):
        await client.complete(_request())
        with pytest.raises(TransientLLMError):
            await client.complete(_request())

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "active"
    # The worst streak ever seen is still reported -- as diagnostics, which is
    # all it ever was. It just no longer decides anything.
    assert snapshot["by_purpose"]["extractor"]["max_consecutive_failures"] == 3
    assert snapshot["by_purpose"]["extractor"]["consecutive_failures"] == 1


@pytest.mark.asyncio
async def test_bounded_scoped_run_keeps_its_verdict() -> None:
    """A bulk run ends, so its verdict is final even for a health violation."""
    clock = _FakeClock()
    provider = AlwaysTransientProvider()
    runtime_config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
        recovery_seconds=60.0,
    )
    client = _client(
        provider,
        runtime_config,
        guard=LLMRunGuard(runtime_config, now=clock),
    )

    scoped_config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=2,
        recovery_seconds=None,
    )
    with client.llm_run_guard_scope(
        run_id="bulk-final",
        kind="admin_rebuild",
        config=scoped_config,
    ):
        for _ in range(2):
            with pytest.raises(TransientLLMError):
                await client.complete(_request())
        with pytest.raises(LLMRunGuardError):
            await client.complete(_request())

        clock.advance(3600.0)

        with pytest.raises(LLMRunGuardError):
            await client.complete(_request())
        assert provider.calls == 2

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["last_scoped_run"]["verdict_is_final"] is True
    assert snapshot["last_scoped_run"]["recovery_count"] == 0


@pytest.mark.asyncio
async def test_audit_mode_clears_degraded_when_health_returns() -> None:
    """Audit mode never blocks, so its verdict must track live health."""
    provider = SequenceProvider(["fail", "fail", "ok"])
    config = LLMRunGuardConfig(
        mode="audit",
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
    )
    client = _client(provider, config)

    for _ in range(2):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    degraded = client.llm_run_guard_snapshot()
    assert degraded is not None
    assert degraded["status"] == "degraded"

    await client.complete(_request())
    recovered = client.llm_run_guard_snapshot()
    assert recovered is not None
    assert recovered["status"] == "active"


# ---------------------------------------------------------------------------
# Spend that nobody consumed is still spend.
#
# A proxy client disconnect cancels the streaming generator after the provider
# has already produced. The round-trip is paid for, so it must be counted; the
# provider was working, so it must not move any health signal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_abandoned_stream_is_counted_as_spend_not_as_a_failure() -> None:
    provider = StaticProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    meter = begin_llm_call_meter()
    try:
        events = client.stream(_request("chat_reply"))
        async for _event in events:
            break  # the client read one chunk and went away
        await events.aclose()
    finally:
        end_llm_call_meter(meter)

    assert provider.calls == 1
    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 1
    assert snapshot["cancelled_calls"] == 1
    assert snapshot["failed_calls"] == 0
    # No health evidence in either direction: the window stays empty, so a
    # disconnect storm can neither trip the breaker nor certify the provider.
    assert snapshot["recent_calls"] == 0
    assert snapshot["by_purpose"]["chat_reply"]["cancelled_calls"] == 1
    assert snapshot["by_purpose"]["chat_reply"]["calls"] == 1
    assert meter.total_calls == 1
    assert meter.failed_calls == 0


@pytest.mark.asyncio
async def test_a_cancelled_completion_is_counted_and_leaves_the_streak_alone() -> None:
    """A cancellation neither extends nor clears the current failure streak."""
    provider = GatedProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    # One real failure first, so there is a streak for the cancellation to leave
    # alone. GatedProvider fails its first call once released.
    provider.release.set()
    with pytest.raises(TransientLLMError):
        await client.complete(_request())
    provider.release.clear()
    provider.fail_first = True
    provider.entered.clear()

    meter = begin_llm_call_meter()
    try:
        inflight = asyncio.create_task(client.complete(_request()))
        await provider.entered.wait()
        inflight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await inflight
    finally:
        end_llm_call_meter(meter)

    assert provider.calls == 2
    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 2
    assert snapshot["cancelled_calls"] == 1
    assert snapshot["failed_calls"] == 1
    assert snapshot["recent_calls"] == 1
    assert snapshot["by_purpose"]["extractor"]["consecutive_failures"] == 1
    assert meter.total_calls == 1
    assert meter.failed_calls == 0


@pytest.mark.asyncio
async def test_an_unnormalized_provider_error_is_a_failure_not_a_cancellation() -> None:
    """The split is Exception vs not-an-Exception, not "everything unexpected".

    A provider adapter that leaks a raw error still failed the round-trip, and
    the breaker must see it. Only CancelledError/GeneratorExit-class unwinds --
    which are not Exceptions -- mean the caller went away.
    """
    provider = BrokenStreamProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    with pytest.raises(ValueError):
        async for _event in client.stream(_request("chat_reply")):
            pass

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 1
    assert snapshot["failed_calls"] == 1
    assert snapshot["cancelled_calls"] == 0
    assert snapshot["error_class_counts"] == {"ValueError": 1}


@pytest.mark.asyncio
async def test_an_in_flight_call_is_charged_to_the_run_it_was_checked_against() -> None:
    """An operator reset must not be re-tripped by a straggler from the outage.

    ``reset_runtime`` replaces the process-wide run wholesale. A call that was
    checked against the old run and returns after the reset belongs to the old
    run's ledger: charging it to the fresh one condemns a run that has served no
    traffic at all.
    """
    provider = GatedProvider()
    config = LLMRunGuardConfig(
        # One failed call spends the whole budget, so "where did this failure
        # land" is directly observable as "is the next call blocked".
        max_total_failed_calls=1,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    guard = LLMRunGuard(config)
    client = _client(provider, config, guard=guard)

    inflight = asyncio.create_task(client.complete(_request()))
    await provider.entered.wait()
    stale_run_snapshot = guard.reset_runtime()
    assert stale_run_snapshot["total_calls"] == 0

    provider.release.set()
    with pytest.raises(TransientLLMError):
        await inflight

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 0
    assert snapshot["failed_calls"] == 0
    assert snapshot["violations"] == []

    # The fresh run serves traffic instead of being blocked by the old outage.
    response = await client.complete(_request())
    assert response.output_text == "ok"
    assert provider.calls == 2


# ---------------------------------------------------------------------------
# Recovery is HALF-OPEN: one probe, not a fresh diagnosis.
#
# Reopening fully clears the health window and every streak, so after each cycle
# the run has to re-earn a whole ``min_calls_for_failed_ratio`` sample, or a
# whole fresh streak, before it can trip again. Against a provider that is still
# down that is 9-20 failing calls per cycle, at the recovery cadence, for the
# whole outage. These tests pin the alternative: one call per cycle, and the
# cycles get longer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sustained_outage_costs_one_probe_per_cycle() -> None:
    """Ten cycles against a dead provider cost ten calls, not ten diagnoses."""
    clock = _FakeClock()
    provider = AlwaysTransientProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=8,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    for _ in range(8):
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
    assert provider.calls == 8

    # Ten recovery cycles with traffic arriving throughout. Exactly one caller
    # per cycle reaches the provider; the other two are refused.
    for _ in range(10):
        clock.advance(100_000.0)
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
        for _ in range(2):
            with pytest.raises(LLMRunGuardError):
                await client.complete(_request())

    assert provider.calls == 18
    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    # The streak is the trip plus one per failed probe, and the evidence was
    # never thrown away: the run knows it has seen 18 consecutive failures.
    assert snapshot["health_block_streak"] == 11
    assert snapshot["by_purpose"]["extractor"]["consecutive_failures"] == 18
    assert snapshot["recovery_count"] == 0


@pytest.mark.asyncio
async def test_each_failed_probe_doubles_the_block_up_to_the_ceiling() -> None:
    """The cadence backs off, so a long outage is not probed at a fixed rate."""
    clock = _FakeClock()
    provider = AlwaysTransientProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    client = _client(provider, config, guard=LLMRunGuard(config, now=clock))

    with pytest.raises(TransientLLMError):
        await client.complete(_request())

    blocks = []
    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    blocks.append(snapshot["blocked_seconds_remaining"])
    for _ in range(5):
        clock.advance(100_000.0)
        with pytest.raises(TransientLLMError):
            await client.complete(_request())
        snapshot = client.llm_run_guard_snapshot()
        assert snapshot is not None
        blocks.append(snapshot["blocked_seconds_remaining"])

    assert blocks == [60.0, 120.0, 240.0, 480.0, 480.0, 480.0]


@pytest.mark.asyncio
async def test_only_one_call_is_admitted_while_the_run_is_half_open() -> None:
    """Concurrent callers arriving at the reopening moment do not all get in."""
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")
    clock.advance(61.0)

    probe = guard.begin_call(purpose="extractor", request_model="model-a")
    assert probe.is_probe is True
    assert probe.decision.should_block is False
    # ``begin_call`` is synchronous, so by the time the next caller in the same
    # loop reads the run, the window has already been re-armed.
    for _ in range(3):
        blocked = guard.begin_call(purpose="extractor", request_model="model-a")
        assert blocked.is_probe is False
        assert blocked.decision.should_block is True

    snapshot = guard.snapshot()
    assert snapshot["status"] == "recovering"
    assert snapshot["probe_in_flight"] is True


@pytest.mark.asyncio
async def test_the_probe_window_can_elapse_with_a_probe_still_in_flight() -> None:
    """Admission is timer-driven, so two probe tickets CAN be out at once.

    This is the case the same-instant test above cannot reach: the clock is
    advanced past the re-armed window while the first probe is still out. Nothing
    here is a defect -- a provider round-trip may legally outlive the window (the
    stock config pairs a 60s recovery with a 120s request timeout, and a streamed
    proxy reply holds its ticket for the whole stream) -- which is exactly why the
    verdict has to be decided by generation rather than by arrival.
    """
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")

    clock.advance(61.0)
    probe_a = guard.begin_call(purpose="extractor", request_model="model-a")
    clock.advance(61.0)
    probe_b = guard.begin_call(purpose="extractor", request_model="model-a")

    assert probe_a.is_probe is True
    assert probe_b.is_probe is True
    assert probe_b.probe_generation == probe_a.probe_generation + 1


@pytest.mark.asyncio
async def test_a_probe_that_returns_after_a_newer_one_decides_nothing() -> None:
    """The probe admitted last decides, not the one that happens to return last.

    Probe A is admitted, the window elapses while it is still in flight, probe B
    is admitted and FAILS, and only then does A come back with a success. Applying
    A would clear the violations, null the deadline and wipe the health window --
    reopening general traffic in the middle of an outage on evidence B has already
    contradicted.
    """
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")

    clock.advance(61.0)
    probe_a = guard.begin_call(purpose="extractor", request_model="model-a")
    clock.advance(61.0)
    probe_b = guard.begin_call(purpose="extractor", request_model="model-a")

    guard.record_failure(probe_b, latency_ms=1.0, error_type="TransientLLMError")
    blocked_after_b = guard.snapshot()
    assert blocked_after_b["status"] == "failed"

    guard.record_success(probe_a, usage={"total_tokens": 5}, latency_ms=90_000.0)

    snapshot = guard.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["violations"] == blocked_after_b["violations"]
    assert snapshot["recovery_count"] == 0
    assert snapshot["health_block_streak"] == blocked_after_b["health_block_streak"]
    assert snapshot["blocked_seconds_remaining"] is not None
    # General traffic stays refused: B's failure is still the run's verdict.
    assert guard.begin_call(
        purpose="extractor",
        request_model="model-a",
    ).decision.should_block is True


@pytest.mark.asyncio
async def test_a_stale_probe_is_still_charged_and_still_counted_as_evidence() -> None:
    """Deciding nothing is not the same as costing nothing.

    The round-trip happened and was paid for, and its outcome says something real
    about the provider. Only the probe VERDICT is discarded.
    """
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")

    clock.advance(61.0)
    probe_a = guard.begin_call(purpose="extractor", request_model="model-a")
    clock.advance(61.0)
    probe_b = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(probe_b, latency_ms=1.0, error_type="TransientLLMError")

    before = guard.snapshot()
    guard.record_failure(probe_a, latency_ms=7.0, error_type="TransientLLMError")
    after = guard.snapshot()

    assert after["total_calls"] == before["total_calls"] + 1
    assert after["failed_calls"] == before["failed_calls"] + 1
    assert after["recent_calls"] == before["recent_calls"] + 1
    assert after["total_latency_ms"] == before["total_latency_ms"] + 7.0
    # The backoff did NOT escalate a second time: only B's failure moved it.
    assert after["health_block_streak"] == before["health_block_streak"]


@pytest.mark.asyncio
async def test_a_late_probe_failure_cannot_condemn_a_healthy_run() -> None:
    """A stale FAILURE must not arm a recovery window on a run that recovered.

    The reverse ordering, and the worse one. A window armed on a healthy run is
    not a block -- ``status`` is active and there are no violations, so traffic
    flows -- but the next call after it elapses is treated as a PROBE: it skips
    the pre-call decision, reports itself unhealthy with an empty violation tuple,
    and on success runs the reopen path, which wipes the health ring of a run that
    was never unhealthy and inflates its recovery count. That destroys the very
    evidence the breaker would need for the next real outage.
    """
    clock = _FakeClock()
    # The stock streak threshold, not 1: a single late failure on a healthy run
    # is ordinary evidence and must be free to be exactly that. What is under
    # test is whether it also arms a recovery window.
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=8,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    for _ in range(8):
        failing = guard.begin_call(purpose="extractor", request_model="model-a")
        guard.record_failure(failing, latency_ms=1.0, error_type="TransientLLMError")

    clock.advance(61.0)
    probe_a = guard.begin_call(purpose="extractor", request_model="model-a")
    clock.advance(61.0)
    probe_b = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_success(probe_b, usage={"total_tokens": 5}, latency_ms=1.0)
    assert guard.snapshot()["status"] == "active"

    for _ in range(120):
        healthy = guard.begin_call(purpose="extractor", request_model="model-a")
        assert healthy.is_probe is False
        guard.record_success(healthy, usage={"total_tokens": 5}, latency_ms=1.0)

    clock.advance(30.0)
    guard.record_failure(probe_a, latency_ms=1.0, error_type="TransientLLMError")

    healthy_snapshot = guard.snapshot()
    assert healthy_snapshot["status"] == "active"
    assert healthy_snapshot["blocked_seconds_remaining"] is None
    assert healthy_snapshot["probe_in_flight"] is False

    clock.advance(600.0)
    following = guard.begin_call(purpose="extractor", request_model="model-a")
    assert following.is_probe is False
    assert following.decision.healthy is True
    guard.record_success(following, usage={"total_tokens": 5}, latency_ms=1.0)

    after = guard.snapshot()
    # The ring was never cleared: it still holds everything it held before, plus
    # the one call recorded since.
    assert after["recent_calls"] == healthy_snapshot["recent_calls"] + 1
    assert after["recovery_count"] == healthy_snapshot["recovery_count"]


@pytest.mark.asyncio
async def test_a_cancelled_probe_counts_as_neither_outcome() -> None:
    """A probe whose caller went away proves nothing, so it decides nothing."""
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")
    clock.advance(61.0)

    probe = guard.begin_call(purpose="extractor", request_model="model-a")
    assert probe.is_probe is True
    guard.record_cancellation(probe, usage=None, latency_ms=1.0)

    snapshot = guard.snapshot()
    # Not a recovery, and not a reason to back off further.
    assert snapshot["recovery_count"] == 0
    assert snapshot["health_block_streak"] == 1
    assert snapshot["probe_in_flight"] is False
    assert snapshot["status"] == "failed"

    # The next call is still refused until the re-armed window elapses, and then
    # becomes a new probe -- the run neither reopened nor gave up on probing.
    assert guard.begin_call(
        purpose="extractor",
        request_model="model-a",
    ).decision.should_block is True
    clock.advance(61.0)
    assert guard.begin_call(purpose="extractor", request_model="model-a").is_probe


@pytest.mark.asyncio
async def test_a_successful_probe_reopens_the_run_for_everyone() -> None:
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=1,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    first = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_failure(first, latency_ms=1.0, error_type="TransientLLMError")
    clock.advance(61.0)

    probe = guard.begin_call(purpose="extractor", request_model="model-a")
    guard.record_success(probe, usage={"input_tokens": 4, "output_tokens": 2}, latency_ms=1.0)

    snapshot = guard.snapshot()
    assert snapshot["status"] == "active"
    assert snapshot["violations"] == []
    assert snapshot["recovery_count"] == 1
    assert snapshot["health_block_streak"] == 0
    assert snapshot["probe_in_flight"] is False
    for _ in range(3):
        assert guard.begin_call(
            purpose="extractor",
            request_model="model-a",
        ).decision.should_block is False


# ---------------------------------------------------------------------------
# "max" means one thing across the whole config block.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("max_total_calls", 3),
        ("max_total_failed_calls", 3),
        ("max_failed_calls_per_purpose", 3),
        ("max_consecutive_failures_per_purpose", 3),
    ],
)
async def test_every_max_count_blocks_at_the_number_it_names(
    field: str,
    limit: int,
) -> None:
    """A budget of N permits N, on every count dimension, not N on some and N+1
    on others.

    This is the off-by-one pin: before, only ``max_total_calls`` blocked at the
    number in its own name, and the three failure dimensions each tolerated one
    more than they said.
    """
    thresholds: dict[str, int | float | None] = {
        "max_total_calls": None,
        "max_total_failed_calls": None,
        "max_failed_calls_per_purpose": None,
        "max_consecutive_failures_per_purpose": None,
        "max_failed_call_ratio": None,
        "max_failed_ratio_per_purpose": None,
    }
    thresholds[field] = limit
    # ``max_total_calls`` counts every round-trip; the other three count only
    # failed ones, so each needs the provider that moves its own counter.
    counts_every_call = field == "max_total_calls"
    provider: StaticProvider = (
        StaticProvider() if counts_every_call else AlwaysTransientProvider()
    )
    client = _client(provider, LLMRunGuardConfig(**thresholds))

    for _ in range(limit):
        if counts_every_call:
            await client.complete(_request())
        else:
            with pytest.raises(TransientLLMError):
                await client.complete(_request())
    assert provider.calls == limit

    with pytest.raises(LLMRunGuardError):
        await client.complete(_request())
    # The counter the threshold names never went past the name.
    assert provider.calls == limit


@pytest.mark.asyncio
async def test_a_spent_budget_refuses_the_probe_instead_of_funding_it() -> None:
    """A health block must not become the hole a spent budget walks through.

    Ten tickets are checked out while the run is healthy (the documented
    non-atomic check-then-act), four failures trip the health block, and the six
    round-trips already in flight are recorded afterwards -- putting the call
    budget exactly on its limit. Nothing latches it: a standing block
    short-circuits the record-time decision, and a count on its limit is not a
    violation at record time by design. The probe path was the only remaining
    evaluation point and it built its decision by hand, so every recovery cycle
    funded one more call past a ``max_*`` count for the length of the outage.
    """
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=10,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=4,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    tickets = [
        guard.begin_call(purpose="extractor", request_model="model-a")
        for _ in range(10)
    ]
    assert all(ticket.decision.should_block is False for ticket in tickets)
    for ticket in tickets:
        guard.record_failure(ticket, latency_ms=1.0, error_type="TransientLLMError")

    spent = guard.snapshot()
    assert spent["total_calls"] == 10
    assert spent["verdict_is_final"] is False

    clock.advance(61.0)
    refused = guard.begin_call(purpose="extractor", request_model="model-a")

    assert refused.is_probe is False
    assert refused.decision.should_block is True
    assert any(
        violation.startswith("total LLM calls budget 10 is spent")
        for violation in refused.decision.violations
    )
    # A budget verdict is final: no later window ever reopens it for a probe.
    final = guard.snapshot()
    assert final["verdict_is_final"] is True
    assert final["blocked_seconds_remaining"] is None
    assert final["total_calls"] == 10
    clock.advance(100_000.0)
    assert guard.begin_call(
        purpose="extractor",
        request_model="model-a",
    ).is_probe is False


@pytest.mark.asyncio
async def test_a_probe_in_flight_cannot_reopen_a_run_that_spent_its_budget() -> None:
    """A ticket issued before the budget closed the run must not undo that.

    The probe is admitted while only a health verdict stands, the budget is then
    crossed and latched by an in-flight round-trip, and the probe finally returns
    a success. Reopening on it would clear a FINAL verdict.
    """
    clock = _FakeClock()
    config = LLMRunGuardConfig(
        max_total_calls=5,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=2,
        recovery_seconds=60.0,
    )
    guard = LLMRunGuard(config, now=clock)

    tickets = [
        guard.begin_call(purpose="extractor", request_model="model-a")
        for _ in range(5)
    ]
    for ticket in tickets[:2]:
        guard.record_failure(ticket, latency_ms=1.0, error_type="TransientLLMError")
    assert guard.snapshot()["status"] == "failed"

    clock.advance(61.0)
    probe = guard.begin_call(purpose="extractor", request_model="model-a")
    assert probe.is_probe is True

    # The three round-trips still in flight when the block landed come back and
    # put the call budget exactly on its limit while the probe is out.
    for ticket in tickets[2:]:
        guard.record_failure(ticket, latency_ms=1.0, error_type="TransientLLMError")
    assert guard.snapshot()["total_calls"] == 5

    clock.advance(61.0)
    refused = guard.begin_call(purpose="extractor", request_model="model-a")
    assert refused.decision.should_block is True
    assert guard.snapshot()["verdict_is_final"] is True

    guard.record_success(probe, usage={"total_tokens": 5}, latency_ms=1.0)

    snapshot = guard.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["verdict_is_final"] is True
    assert snapshot["recovery_count"] == 0
    assert guard.begin_call(
        purpose="extractor",
        request_model="model-a",
    ).decision.should_block is True


@pytest.mark.asyncio
async def test_a_run_that_stops_exactly_on_its_budget_reports_clean() -> None:
    """Spending the whole allowance is not a violation; asking for more is.

    This is why an absolute budget is enforced at the next ``begin_call`` rather
    than latched the moment the counter reaches it: a bulk rebuild that used all
    of its calls and finished did exactly what it was configured to do.
    """
    provider = StaticProvider()
    config = LLMRunGuardConfig(
        max_total_calls=2,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    for _ in range(2):
        await client.complete(_request())

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "active"
    assert snapshot["violations"] == []
    assert snapshot["total_calls"] == 2


# ---------------------------------------------------------------------------
# Audit mode has to log the trip; observing it is the whole point of the mode.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_mode_logs_the_trip_it_exists_to_observe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The moment the guard WOULD have blocked has to reach the logs.

    Keyed on ``should_block``, this log never fired in audit mode -- the one mode
    whose entire purpose is to evaluate thresholds before enforcing them.
    """
    provider = AlwaysTransientProvider()
    config = LLMRunGuardConfig(
        mode="audit",
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=2,
    )
    client = _client(provider, config)

    with caplog.at_level("ERROR", logger="atagia.services.llm_client"):
        for _ in range(5):
            with pytest.raises(TransientLLMError):
                await client.complete(_request())

    trips = [
        record
        for record in caplog.records
        if record.getMessage().startswith("LLM run guard tripped")
    ]
    # Exactly one trip EVENT, not one line per degraded call and not silence.
    assert len(trips) == 1
    assert "audit mode" in trips[0].getMessage()
    assert provider.calls == 5
    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "degraded"


@pytest.mark.asyncio
async def test_enforce_mode_logs_the_trip_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = AlwaysTransientProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=2,
    )
    client = _client(provider, config)

    with caplog.at_level("ERROR", logger="atagia.services.llm_client"):
        for _ in range(2):
            with pytest.raises(TransientLLMError):
                await client.complete(_request())
        with pytest.raises(LLMRunGuardError):
            await client.complete(_request())

    trips = [
        record
        for record in caplog.records
        if record.getMessage().startswith("LLM run guard tripped")
    ]
    assert len(trips) == 1
    assert "the next provider call will be blocked" in trips[0].getMessage()


# ---------------------------------------------------------------------------
# What the operator reads has to describe the run it is reading.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellations_do_not_dilute_the_reported_failure_ratio() -> None:
    """A disconnect storm must not make an outage read milder than it is."""
    provider = StaticProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    guard = LLMRunGuard(config)
    client = _client(provider, config, guard=guard)

    call = guard.begin_call(purpose="chat_reply", request_model="model-a")
    guard.record_failure(call, latency_ms=1.0, error_type="TransientLLMError")
    for _ in range(9):
        cancelled = guard.begin_call(purpose="chat_reply", request_model="model-a")
        guard.record_cancellation(cancelled, usage=None, latency_ms=1.0)

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_calls"] == 10
    assert snapshot["cancelled_calls"] == 9
    # One failure out of one evidence-bearing call, not one out of ten.
    assert snapshot["failure_ratio"] == pytest.approx(1.0)
    assert snapshot["by_purpose"]["chat_reply"]["failure_ratio"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_budget_measured_in_a_quantity_nobody_reports_is_visible() -> None:
    """A token or cost cap that can never fire must not look like a healthy run."""
    provider = SilentUsageProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_total_tokens=1000,
        max_reported_cost_usd=10.0,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    for _ in range(4):
        await client.complete(_request())

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_tokens"] == 0
    assert snapshot["successful_calls_without_token_usage"] == 4
    assert snapshot["successful_calls_without_reported_cost"] == 4
    assert snapshot["status"] == "active"


@pytest.mark.asyncio
async def test_a_provider_that_reports_usage_is_not_counted_as_blind() -> None:
    provider = StaticProvider()
    config = LLMRunGuardConfig(
        max_total_calls=None,
        max_total_failed_calls=None,
        max_failed_call_ratio=None,
        max_failed_ratio_per_purpose=None,
        max_consecutive_failures_per_purpose=None,
    )
    client = _client(provider, config)

    await client.complete(_request())

    snapshot = client.llm_run_guard_snapshot()
    assert snapshot is not None
    assert snapshot["total_tokens"] == 10
    assert snapshot["successful_calls_without_token_usage"] == 0
    # StaticProvider reports tokens but no cost, which is the normal case for
    # every provider that does not price its own responses.
    assert snapshot["successful_calls_without_reported_cost"] == 1


# ---------------------------------------------------------------------------
# A guard block is a LOCAL refusal, and the hierarchy has to keep saying so.
# ---------------------------------------------------------------------------


def test_a_guard_block_is_an_llm_error_so_degradation_sites_stay_correct() -> None:
    """Deliberate, not accidental. See ``LLMRunGuardError``'s docstring.

    Every reachable ``except LLMError`` in this repo responds to "no answer" by
    degrading -- 503, abstention, skipped background refresh -- and that is the
    right answer for "stop spending" too. Reparenting off ``LLMError`` turns
    each of those into an uncaught ``RuntimeError`` unless it is individually
    re-taught this class, which trades a routing risk for a crash risk.
    """
    error = LLMRunGuardError(
        LLMRunGuardDecision(healthy=False, should_block=True, violations=("stop",))
    )
    assert isinstance(error, LLMError)


def test_a_guard_block_never_reroutes_to_another_model() -> None:
    """The one path that answers an error by calling a provider again refuses.

    A guard block is about the RUN, so re-issuing the same request through any
    fallback cannot satisfy it -- the fallback's own ``begin_call`` refuses it
    too, at the cost of a second error and a misleading log line.
    """
    guard_error = LLMRunGuardError(
        LLMRunGuardDecision(
            healthy=False,
            should_block=True,
            # Worded exactly like the substring test it must not trip.
            violations=("the LLM run guard blocked the response",),
        )
    )
    assert LLMClient._is_policy_blocked_error(guard_error) is False


def test_config_rejects_thresholds_that_could_never_fire() -> None:
    with pytest.raises(ValueError, match="min_calls_for_failed_ratio"):
        LLMRunGuardConfig(min_calls_for_failed_ratio=500, health_window_calls=200)
    with pytest.raises(ValueError, match="min_calls_per_purpose_for_failed_ratio"):
        LLMRunGuardConfig(
            min_calls_per_purpose_for_failed_ratio=500,
            health_window_calls=200,
        )
    with pytest.raises(ValueError, match="recovery_seconds must be positive"):
        LLMRunGuardConfig(recovery_seconds=0.0)
    with pytest.raises(ValueError, match="health_window_calls must be positive"):
        LLMRunGuardConfig(health_window_calls=0)
    # A budget of N permits N, so a budget of 0 permits nothing: it would close
    # the run before its first call rather than stop it at the first failure.
    for field in (
        "max_total_calls",
        "max_total_failed_calls",
        "max_failed_calls_per_purpose",
        "max_consecutive_failures_per_purpose",
        "max_total_tokens",
        "max_reported_cost_usd",
        "max_wall_time_seconds",
    ):
        with pytest.raises(ValueError, match=f"{field} must be positive when set"):
            LLMRunGuardConfig(**{field: 0})
