"""Offline CI coverage for the card 2 (per-range summary) harness.

These tests run the harness with the in-process fake provider only -- no
network, no API keys -- and assert the reliability/concurrency invariants the
harness is meant to measure: 100% range coverage, correct retry accounting, and
observed max concurrency equal to the configured cap.
"""

from __future__ import annotations

import pytest

from benchmarks.card2_summary.__main__ import main as card2_main
from benchmarks.card2_summary.cases import (
    FROZEN_RANGE_FIXTURE_ID,
    Card2Case,
    load_cases,
)
from benchmarks.card2_summary.fake_provider import FakeSummaryProvider
from benchmarks.card2_summary.runner import run_harness


def test_selftest_cli_exits_zero() -> None:
    # The offline --selftest path runs the full pipeline and self-asserts.
    assert card2_main(["--selftest"]) == 0


def test_load_cases_partitions_cover_every_seq() -> None:
    for case in load_cases("all"):
        covered: list[int] = []
        for start_seq, end_seq in case.ranges:
            covered.extend(range(start_seq, end_seq + 1))
        message_seqs = sorted(int(message["seq"]) for message in case.messages)
        assert covered == message_seqs, case.case_id


def test_load_cases_rejects_unknown_set() -> None:
    with pytest.raises(ValueError):
        load_cases("bogus")


async def test_harness_full_coverage_retry_and_concurrency_cap() -> None:
    cases = load_cases("all")
    retry_range = cases[0].ranges[0]
    inner = FakeSummaryProvider(summary_delay_seconds=0.02, retry_range=retry_range)

    report = await run_harness(
        cases=cases,
        model="openrouter/google/test-fake",
        concurrency=4,
        runs=1,
        case_set="all",
        inner_provider=inner,
        provider_name="card2-fake",
        card_examples_enabled=True,
        selftest=True,
    )

    aggregate = report.aggregate
    assert report.frozen_range_fixture_id == FROZEN_RANGE_FIXTURE_ID
    assert aggregate["overall_range_coverage"] == 1.0
    assert aggregate["hard_failure_case_count"] == 0
    # Exactly one injected empty-then-valid retry across the run.
    assert aggregate["total_retry_count"] == 1
    # One LLM call per range plus the single retry.
    assert aggregate["total_llm_call_count"] == aggregate["total_ranges"] + 1
    # The saturation stress case alone has >= cap single-message ranges.
    assert aggregate["observed_max_concurrency"] == 4


async def test_harness_concurrency_one_serializes() -> None:
    # With the cap at 1, the engine runs ranges sequentially: max concurrency 1.
    cases = load_cases("stress")
    inner = FakeSummaryProvider(summary_delay_seconds=0.01)

    report = await run_harness(
        cases=cases,
        model="minimax/MiniMax-M3",
        concurrency=1,
        runs=1,
        case_set="stress",
        inner_provider=inner,
        provider_name="card2-fake",
        card_examples_enabled=True,
        selftest=True,
    )

    assert report.aggregate["overall_range_coverage"] == 1.0
    assert report.aggregate["observed_max_concurrency"] == 1


async def test_harness_hard_failure_when_summary_never_recovers() -> None:
    # A provider that always returns empty exhausts the per-range retries and
    # raises -- the harness must record a hard failure, not silently pass.
    class AlwaysEmptyProvider(FakeSummaryProvider):
        async def complete(self, request):  # type: ignore[override]
            response = await super().complete(request)
            return response.model_copy(update={"output_text": ""})

    case = Card2Case(
        case_id="always_empty",
        family="stress",
        reference_time_utc="2026-05-04T12:00:00+00:00",
        messages=[
            {"seq": 1, "role": "user", "occurred_at": None, "text": "Only message."}
        ],
        ranges=[(1, 1)],
    )
    report = await run_harness(
        cases=[case],
        model="minimax/MiniMax-M3",
        concurrency=1,
        runs=1,
        case_set="stress",
        inner_provider=AlwaysEmptyProvider(),
        provider_name="card2-fake",
        card_examples_enabled=True,
        selftest=True,
    )

    metrics = report.per_case[0]
    assert metrics.hard_failure is True
    assert metrics.range_coverage == 0.0
    assert report.aggregate["hard_failure_case_count"] == 1
