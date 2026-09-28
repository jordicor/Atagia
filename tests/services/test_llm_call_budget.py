"""Per-turn LLM call budget (ratchet) test and call-meter unit tests (CS-1.4).

The budget test is the ratchet that Fase 2/3 latency work squeezes: it runs a
full engine turn against a fake provider and asserts BOTH how many synchronous
LLM provider calls a turn makes and WHICH stages made them. Adding an
unconditional card breaks the ceiling; dropping a card or a whole stage breaks
the pinned composition. Either direction can only move with an explicit edit
here, in the same change that moved the engine.
"""

from __future__ import annotations

import pytest

from atagia import Atagia
from atagia.memory.applicability_scorer import (
    _DEFAULT_APPLICABILITY_CARD_CANDIDATES_PER_BATCH,
)
from atagia.core.repositories import (
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
)
from atagia.services.llm_run_guard import (
    LLMCallMeter,
    LLMCallOutcome,
    begin_isolated_llm_call_meter,
    begin_llm_call_meter,
    bind_llm_call_meter,
    end_llm_call_meter,
    record_call_on_active_meter,
)

from tests.test_engine import EngineProvider, _install_stub_client

# ---------------------------------------------------------------------------
# Budget for one full turn on the fixture below: a CEILING and a COMPOSITION.
#
# A ceiling alone is a one-way ratchet. `total_calls <= N` fails when a card is
# added and passes when one is deleted, so a capability regression -- a
# need-detection card silently dropped, a whole stage removed -- ships green,
# and F2/F3 latency work can "improve" the number by deleting a stage instead of
# making one cheaper. The composition below closes that direction: the exact
# per-purpose breakdown is pinned, so a removal fails as loudly as an addition.
#
# HOW THIS WAS MEASURED
#   Run this module's `test_full_turn_llm_call_budget` and read the
#   `[CS-1.4 budget]` line it prints (total plus the full by_purpose map). The
#   fixture stores _CANDIDATE_COUNT candidate memories that all match the fake
#   need-detection search words, so applicability scoring runs in batches (see
#   applicability_scorer batch size = 4).
#
# WHY THE PURPOSES ARE SPELLED OUT INSTEAD OF DERIVED
#   Deriving the expected set from the engine (importing the card list, counting
#   registered stages) would make the pin follow the code: delete a card and the
#   expectation deletes itself. The literal list is the point -- it is an
#   independent statement of what a turn is supposed to do.
#
# HOW TO CHANGE IT IN FASE 2/3
#   Both directions need an explicit, reviewed edit here, and that is the whole
#   mechanism:
#     * Cheaper (a stage batched better, a card collapsed into another): re-run,
#       read the printed map, lower the ceiling AND update the composition.
#     * More expensive: never raise the ceiling to make a regression pass. More
#       sync LLM calls is the thing F2/F3 exists to reduce.
#     * A capability deliberately dropped: remove its key here, in the same
#       change that removes it from the engine, so the diff shows both.
#   Only the applicability relevance entry is derived, from the candidate count and
#   the batch size: those scale with the pool by design, and a change in batching
#   is a cost change the ceiling already governs, not a capability change.
# ---------------------------------------------------------------------------
_CANDIDATE_COUNT = 12
# 12 candidates / batch size 4 = 3 batches, dates are persisted at ingest.
_EXPECTED_APPLICABILITY_BATCHES = -(
    -_CANDIDATE_COUNT // _DEFAULT_APPLICABILITY_CARD_CANDIDATES_PER_BATCH
)
# Measured 2026-07-24 on the deterministic fixture below.
_EXPECTED_CALLS_BY_PURPOSE: dict[str, int] = {
    # The nine need-detection cards (memory/need_detector.py). Each fires
    # exactly once per turn; none of them is covered by any other assertion.
    "need_detection_query_language_card": 1,
    "need_detection_answer_language_card": 1,
    "need_detection_needs_card": 1,
    "need_detection_memory_card": 1,
    "need_detection_exact_card": 1,
    "need_detection_shape_card": 1,
    "need_detection_facets_card": 1,
    "need_detection_callback_card": 1,
    "need_detection_search_words_card": 1,
    # Applicability scoring: one relevance call per candidate batch.
    "applicability_relevance_card": _EXPECTED_APPLICABILITY_BATCHES,
    # The answer.
    "chat_reply": 1,
}
_MEASURED_TOTAL_CALLS = sum(_EXPECTED_CALLS_BY_PURPOSE.values())
# Zero slack: the fixture is fully deterministic (fixed candidate count, fixed
# batch size, fake provider), so the measured total does not vary run to run.
# The ceiling therefore equals the measured value and ANY new unconditional sync
# LLM call -- even a single one -- trips this test. That strictness is the point:
# F2/F3 must be able to see a regression the moment it lands. Do not add slack to
# quiet a jittery total; if the total ever jitters, fix the nondeterminism.
_BUDGET_SLACK = 0
_TURN_LLM_CALL_CEILING = _MEASURED_TOTAL_CALLS + _BUDGET_SLACK


async def _seed_candidates(engine: Atagia, *, user_id: str, conversation_id: str) -> None:
    connection = await engine.runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, engine.runtime.clock)
        surfaces = MemoryRetrievalSurfaceRepository(connection, engine.runtime.clock)
        for index in range(_CANDIDATE_COUNT):
            memory_id = f"mem_retry_{index:02d}"
            await memories.create_memory_object(
                user_id=user_id,
                conversation_id=conversation_id,
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text=(
                    f"Retry loop guidance {index}: back off before the next retry "
                    f"loop attempt to avoid a tight retry loop."
                ),
                source_kind=MemorySourceKind.VERBATIM,
                confidence=0.9,
                privacy_level=0,
                memory_id=memory_id,
                payload={"writer_kind": "manual"},
            )
            await surfaces.upsert_surface(
                user_id=user_id,
                memory_id=memory_id,
                surface_type="alias",
                surface_text=f"retry loop {index}",
            )
    finally:
        await connection.close()


async def _run_full_turn(monkeypatch: pytest.MonkeyPatch) -> dict:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    try:
        await engine.create_user("usr_budget")
        await engine.create_conversation(
            "usr_budget", "cnv_budget", assistant_mode_id="coding_debug"
        )
        await _seed_candidates(
            engine, user_id="usr_budget", conversation_id="cnv_budget"
        )
        result = await engine.chat(
            user_id="usr_budget",
            conversation_id="cnv_budget",
            mode="coding_debug",
            message="How do I fix this retry loop?",
            debug=True,
        )
    finally:
        await engine.close()
    assert result.debug is not None
    return result.debug


# ---------------------------------------------------------------------------
# LLMCallMeter unit tests
# ---------------------------------------------------------------------------


def test_call_meter_records_totals_and_per_purpose_and_latency() -> None:
    meter = LLMCallMeter()
    meter.record(purpose="need_detection_needs_card", latency_ms=10.0, outcome=LLMCallOutcome.SUCCESS)
    meter.record(purpose="applicability_relevance_card", latency_ms=20.0, outcome=LLMCallOutcome.SUCCESS)
    meter.record(purpose="applicability_relevance_card", latency_ms=5.0, outcome=LLMCallOutcome.FAILURE)
    meter.record(purpose=None, latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS)

    assert meter.total_calls == 4
    assert meter.failed_calls == 1
    assert meter.total_latency_ms == pytest.approx(36.0)
    assert meter.by_purpose["applicability_relevance_card"].calls == 2
    assert meter.by_purpose["need_detection_needs_card"].calls == 1
    # A missing purpose is bucketed under "unknown", never dropped.
    assert meter.by_purpose["unknown"].calls == 1
    # CS-1.5: latency is attributed per purpose, not only in the total, and a
    # purpose's latency accumulates across its calls (failed attempts included,
    # because a failed round-trip still spends wall time).
    assert meter.by_purpose["applicability_relevance_card"].latency_ms == pytest.approx(
        25.0
    )
    assert meter.by_purpose["need_detection_needs_card"].latency_ms == pytest.approx(
        10.0
    )
    assert meter.by_purpose["unknown"].latency_ms == pytest.approx(1.0)
    assert sum(
        usage.latency_ms for usage in meter.by_purpose.values()
    ) == pytest.approx(meter.total_latency_ms)


def test_call_meter_counts_a_cancelled_round_trip_as_spend_not_as_a_failure() -> None:
    """An abandoned call cost money and wall time; it was not a provider error.

    A proxy client that disconnects after the first chunk still made the provider
    generate. Dropping it would make disconnect storms invisible; calling it a
    failure would report the provider as broken because a network went away.
    """
    meter = LLMCallMeter()
    meter.record(purpose="chat_reply", latency_ms=30.0, outcome=LLMCallOutcome.CANCELLED)

    assert meter.total_calls == 1
    assert meter.failed_calls == 0
    assert meter.total_latency_ms == pytest.approx(30.0)
    assert meter.by_purpose["chat_reply"].calls == 1
    assert meter.by_purpose["chat_reply"].latency_ms == pytest.approx(30.0)


def test_zero_call_meter_reports_empty() -> None:
    # A meter that recorded nothing is the zero-call baseline: a full engine turn
    # cannot be zero-call because it always generates the chat reply, so the
    # zero-call state is exercised here directly rather than via a cache-hit turn.
    meter = LLMCallMeter()
    assert meter.total_calls == 0
    assert meter.failed_calls == 0
    assert meter.total_latency_ms == 0.0
    assert meter.by_purpose == {}


def test_record_on_active_meter_is_noop_without_active_meter() -> None:
    # No active meter bound -> the recorder must not raise or count anywhere.
    record_call_on_active_meter(purpose="chat_reply", latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS)


def test_begin_end_call_meter_binds_records_and_unbinds() -> None:
    meter = begin_llm_call_meter()
    try:
        record_call_on_active_meter(
            purpose="need_detection_needs_card", latency_ms=2.0, outcome=LLMCallOutcome.SUCCESS
        )
        assert meter.total_calls == 1
    finally:
        end_llm_call_meter(meter)
    assert meter.by_purpose["need_detection_needs_card"].calls == 1
    assert meter.by_purpose["need_detection_needs_card"].latency_ms == pytest.approx(2.0)
    # After unbinding, further records must not reach this meter.
    record_call_on_active_meter(purpose="chat_reply", latency_ms=3.0, outcome=LLMCallOutcome.SUCCESS)
    assert meter.total_calls == 1


def test_ending_an_outer_meter_leaves_the_inner_one_bound() -> None:
    """Removal is by identity, so unbind order cannot resurrect a dropped meter.

    A proxy turn binds its whole-turn meter and the sidecar retrieval it calls
    binds another inside it. Whichever finishes first must take only its own
    meter with it.
    """
    outer = begin_llm_call_meter()
    inner = begin_llm_call_meter()
    try:
        record_call_on_active_meter(purpose="planner", latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS)
        # A call made while both are bound lands on both, not just the innermost.
        assert outer.total_calls == 1
        assert inner.total_calls == 1

        end_llm_call_meter(outer)
        record_call_on_active_meter(
            purpose="applicability", latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS
        )
        assert outer.total_calls == 1
        assert inner.total_calls == 2
    finally:
        end_llm_call_meter(inner)
    record_call_on_active_meter(purpose="chat_reply", latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS)
    assert inner.total_calls == 2
    assert outer.total_calls == 1


# ---------------------------------------------------------------------------
# Full-turn budget / ratchet test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_turn_llm_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    debug = await _run_full_turn(monkeypatch)

    metrics = debug["llm_call_metrics"]
    assert metrics is not None, "per-turn LLM call metrics missing from debug payload"
    # The same metrics object must also ride on the retrieval trace.
    trace_metrics = debug["retrieval_trace"]["llm_call_metrics"]
    assert trace_metrics == metrics

    total_calls = metrics["total_calls"]
    by_purpose = {
        purpose: usage["calls"] for purpose, usage in metrics["by_purpose"].items()
    }

    # Print so a ceiling update is a copy/paste, and so a failure shows the shape.
    print(
        "\n[CS-1.4 budget] total_calls=%s ceiling=%s by_purpose=%s latency_ms=%.1f"
        % (
            total_calls,
            _TURN_LLM_CALL_CEILING,
            by_purpose,
            metrics["total_latency_ms"],
        )
    )

    # 1) Ratchet, cost direction: a new unconditional card tomorrow pushes this
    #    over the ceiling.
    assert total_calls <= _TURN_LLM_CALL_CEILING, (
        f"per-turn LLM calls {total_calls} exceeded ceiling "
        f"{_TURN_LLM_CALL_CEILING}; if this is an intended, reviewed increase, "
        f"update _EXPECTED_CALLS_BY_PURPOSE. by_purpose={by_purpose}"
    )

    # 2) Ratchet, capability direction: the exact composition is pinned, so a
    #    stage that stops firing fails here instead of quietly making the cost
    #    number look better. A missing key IS the regression.
    assert by_purpose == _EXPECTED_CALLS_BY_PURPOSE, (
        "per-turn LLM call composition changed. Missing purposes are a "
        "capability regression (a card or stage stopped running); new ones are "
        "new sync LLM cost. Either way, update _EXPECTED_CALLS_BY_PURPOSE in the "
        "same change that moved the engine. "
        f"expected={_EXPECTED_CALLS_BY_PURPOSE} actual={by_purpose}"
    )

    # 3) Breakdown sanity: the answer always fires exactly once, and the total is
    #    the sum of the per-purpose counts (nothing is silently uncounted).
    assert by_purpose.get("chat_reply") == 1
    assert sum(by_purpose.values()) == total_calls
    assert metrics["total_latency_ms"] >= 0.0
    # CS-1.5: every counted purpose also reports the wall time it spent, and the
    # per-purpose latency never exceeds the turn's measured provider latency.
    assert set(metrics["by_purpose"]) == set(by_purpose)
    assert all(usage["latency_ms"] >= 0.0 for usage in metrics["by_purpose"].values())
    assert sum(
        usage["latency_ms"] for usage in metrics["by_purpose"].values()
    ) == pytest.approx(metrics["total_latency_ms"])

    # Relevance scales with the candidate pool; dates never dispatch here.
    relevance_calls = by_purpose.get("applicability_relevance_card", 0)
    assert relevance_calls >= 2
    assert by_purpose.get("applicability_date_card", 0) == 0
    assert by_purpose.get("memory_date_resolution", 0) == 0
    assert relevance_calls == max(by_purpose.values())
    assert relevance_calls == _EXPECTED_APPLICABILITY_BATCHES



def test_binding_the_same_meter_twice_in_one_context_raises() -> None:
    """A double bind is a caller bug, not a tolerable state.

    ``record_call_on_active_meter`` fans out over the whole stack, so a meter
    bound twice would count every subsequent round-trip twice. The streaming
    proxy is correct today only because the generator's context is created after
    the setup scope unbinds -- a property of how the scopes happen to nest, not
    something the API enforced. Now it does.
    """
    meter = begin_llm_call_meter()
    try:
        with pytest.raises(RuntimeError, match="already bound in this context"):
            bind_llm_call_meter(meter)
        # The failed bind must not have grown the stack: one record, one count.
        record_call_on_active_meter(purpose="chat_reply", latency_ms=1.0, outcome=LLMCallOutcome.SUCCESS)
        assert meter.total_calls == 1
    finally:
        end_llm_call_meter(meter)


def test_binding_a_second_distinct_meter_is_still_allowed() -> None:
    """Nesting is the supported case and must keep working.

    A proxy turn binds a whole-turn meter and the sidecar retrieval it calls
    binds another inside it; both must see the retrieval round-trips.
    """
    outer = begin_llm_call_meter()
    inner = LLMCallMeter()
    bind_llm_call_meter(inner)
    try:
        record_call_on_active_meter(purpose="planner", latency_ms=2.0, outcome=LLMCallOutcome.SUCCESS)
        assert outer.total_calls == 1
        assert inner.total_calls == 1
    finally:
        end_llm_call_meter(inner)
        end_llm_call_meter(outer)


def test_isolated_meter_detaches_background_work_from_the_spawning_turn() -> None:
    """Background work must not accumulate on the turn that spawned it.

    ``asyncio.create_task`` copies the spawning context, meters included, so a
    background task that merely PUSHES its own meter still charges every
    round-trip to the turn it came from -- a turn whose telemetry row is already
    written. This matters beyond the chat surface: a proxy turn keeps its own
    meter bound across the sidecar retrieval that schedules the smart_fast warm,
    so the scheduler unbinding first is not enough on its own.
    """
    turn_meter = begin_llm_call_meter()
    try:
        # Stand in for the copied context a spawned task runs in.
        background_meter = begin_isolated_llm_call_meter()
        record_call_on_active_meter(purpose="planner", latency_ms=4.0, outcome=LLMCallOutcome.SUCCESS)
        assert background_meter.total_calls == 1
        assert turn_meter.total_calls == 0
        end_llm_call_meter(background_meter)
    finally:
        end_llm_call_meter(turn_meter)
