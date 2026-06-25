"""CLI for the card 2 (per-range summary) reliability + concurrency harness.

The harness drives the real engine method
``Compactor._summarize_message_ranges_card`` over frozen card-1 ranges. It never
copies the card 2 prompt (prompt fidelity is satisfied by calling the engine).

Examples
--------
Offline self-test (no network, no keys; exits 0 on success):

    python -m benchmarks.card2_summary --selftest

Phase A (OpenRouter, concurrency 4) -- Gemini Flash-Lite, then minimax-m3:

    python -m benchmarks.card2_summary --model openrouter/google/gemini-3.1-flash-lite --concurrency 4
    python -m benchmarks.card2_summary --model openrouter/minimax/minimax-m3 --concurrency 4

Phase B (direct providers, concurrency sweep):

    python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 1
    python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 2
    python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 4
    python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 8

Live runs need the relevant provider key and incur cost.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from atagia.services.model_resolution import OPENROUTER_FLASH_LITE_MODEL

from benchmarks.card2_summary.cases import load_cases
from benchmarks.card2_summary.fake_provider import FakeSummaryProvider
from benchmarks.card2_summary.report import format_markdown, write_report
from benchmarks.card2_summary.runner import RunReport, run_harness
from benchmarks.output_root import bench_output_root, resolve_output_dir

_DEFAULT_OUTPUT_PARENT = ("card2_summary",)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.card2_summary",
        description=(
            "Reliability + concurrency harness for the compactor per-range "
            "summary card (card 2). Drives the real engine method."
        ),
    )
    parser.add_argument(
        "--model",
        default=OPENROUTER_FLASH_LITE_MODEL,
        help=(
            "Compactor model spec, e.g. openrouter/google/gemini-3.1-flash-lite, "
            "openrouter/minimax/minimax-m3, minimax/MiniMax-M3 (default: "
            f"{OPENROUTER_FLASH_LITE_MODEL})."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Summary card concurrency cap (>= 1; 1 runs sequentially).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repeated runs over the case set.",
    )
    parser.add_argument(
        "--cases",
        choices=("all", "realistic", "stress"),
        default="all",
        help="Which case family to run.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Disable the card's few-shot examples (card_examples_enabled=False).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Report directory (must be OUTSIDE the repo). Default: a UTC-stamped "
            "dir under the benchmark output root / card2_summary."
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Offline mode: swap in a fake in-process provider (no network/keys), "
            "run the full pipeline, and assert coverage/retry/concurrency."
        ),
    )
    return parser.parse_args(argv)


def _resolve_output_dir(out: Path | None) -> Path:
    if out is not None:
        return resolve_output_dir(out, require_utc=False)
    return resolve_output_dir(*_DEFAULT_OUTPUT_PARENT, root=bench_output_root())


async def _run_selftest() -> int:
    """Run the harness with a fake provider and assert the core invariants."""
    concurrency = 4
    cases = load_cases("all")
    # Pick a real range present in the case set to force one retry on.
    retry_range = cases[0].ranges[0]
    inner = FakeSummaryProvider(
        summary_delay_seconds=0.02,  # force overlap so the cap is observable
        retry_range=retry_range,
    )
    report = await run_harness(
        cases=cases,
        model="openrouter/google/selftest-fake",
        concurrency=concurrency,
        runs=1,
        case_set="all",
        inner_provider=inner,
        provider_name="card2-fake",  # single-provider passthrough
        card_examples_enabled=True,
        selftest=True,
    )

    aggregate = report.aggregate
    failures: list[str] = []
    if aggregate.get("overall_range_coverage") != 1.0:
        failures.append(
            f"coverage != 100% (got {aggregate.get('overall_range_coverage')})"
        )
    if aggregate.get("hard_failure_case_count", 1) != 0:
        failures.append(
            f"hard failures present ({aggregate.get('hard_failure_case_count')})"
        )
    # Exactly one injected empty-then-valid retry across the whole run.
    if aggregate.get("total_retry_count") != 1:
        failures.append(
            f"expected exactly 1 retry, got {aggregate.get('total_retry_count')}"
        )
    # The saturation stress case has >= cap single-message ranges, so the
    # observed max concurrency must equal the configured cap.
    if aggregate.get("observed_max_concurrency") != concurrency:
        failures.append(
            "observed max concurrency != cap "
            f"({aggregate.get('observed_max_concurrency')} != {concurrency})"
        )

    print(format_markdown(report))
    if failures:
        print("\nSELFTEST FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nSELFTEST PASSED: coverage 100%, retry counted, max concurrency == cap.")
    return 0


async def _run_live(args: argparse.Namespace) -> int:
    # Import here so the offline self-test never depends on provider SDKs/keys.
    from benchmarks.card2_summary.live_provider import build_live_provider

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    cases = load_cases(args.cases)
    inner = build_live_provider(args.model)
    print(
        f"Live card 2 harness: model={args.model} concurrency={args.concurrency} "
        f"runs={args.runs} cases={args.cases} (incurs API cost)"
    )
    report: RunReport = await run_harness(
        cases=cases,
        model=args.model,
        concurrency=args.concurrency,
        runs=args.runs,
        case_set=args.cases,
        inner_provider=inner,
        provider_name=None,  # route by model-id prefix, like production
        card_examples_enabled=not args.no_examples,
        selftest=False,
    )
    output_dir = _resolve_output_dir(args.out)
    json_path, markdown_path = write_report(report, output_dir)
    print(format_markdown(report))
    print(f"\nJSON report:     {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.selftest:
        return asyncio.run(_run_selftest())
    return asyncio.run(_run_live(args))


if __name__ == "__main__":
    sys.exit(main())
