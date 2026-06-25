# Card 2 (per-range summary) harness

A focused reliability + concurrency harness for the compactor's conversation-chunk
**summary card** ("card 2"), which issues **one LLM call per confirmed range**,
concurrently, bounded by a configurable cap.

## What it measures

The harness drives the real engine method
`Compactor._summarize_message_ranges_card` over **frozen card-1 ranges** (a fixed
partition of each case's messages), so card 2 is exercised in isolation. Per
`(model, concurrency, run, case)` it records:

- **range_coverage** — every frozen range returns a non-empty summary (target 100%).
- **retry_count** — total summary LLM calls minus number of ranges (≥ 0).
- **hard_failures** — ranges that raised after retries were exhausted.
- **wall_clock_ms** — total time for the case.
- **llm_call_count** and **observed max concurrency**.
- **rate_limit / transient / request error counts** — captured exception types/messages.

To count calls, time them, observe concurrency, and capture errors **without
touching the engine prompt**, every provider is wrapped in a `_CountingProvider`
(`runner.py`) that instruments only the per-range summary purpose and delegates
`complete()` unchanged.

## Prompt fidelity

This harness **drives the engine method directly** — it does NOT copy or
re-implement the card 2 prompt. The prompt it measures is always the production
prompt. (`tests/benchmarks/test_benchmark_prompt_fidelity.py` enforces that
shadow benchmarks never copy engine prompts; this harness satisfies that by
calling the engine, so it has no champion-prompt copy to drift.)

## Cases and frozen ranges

- **realistic** — the 5 conversational cases from
  `benchmarks/model_casting/inputs/compactor.json`, each with a hand-authored,
  contiguous, non-overlapping partition (the card-1 stand-in).
- **stress** — two synthetic cases with realistic fictional dialogue (no
  benchmark entities): `stress_many_single_ranges` (~20 single-message ranges to
  saturate concurrency) and `stress_long_single_range` (one ~30-message range).

The frozen partition is versioned by `FROZEN_RANGE_FIXTURE_ID` in `cases.py`, and
that id is recorded in every report so metrics can be matched to the exact fixture.

## Offline self-test (no network, no keys)

```bash
python -m benchmarks.card2_summary --selftest
```

This swaps in an in-process fake provider (canned non-empty summary per range,
one injected empty-then-valid response to exercise retry, and a small sleep to
force concurrency overlap), runs the full pipeline, and asserts: 100% coverage,
exactly one counted retry, and observed max concurrency equal to the cap. Exits 0
on success. CI covers the same path via
`tests/benchmarks/test_card2_summary_harness.py`.

## Live runs (incur API cost)

Live runs need the relevant provider key:
`OPENROUTER_API_KEY` (or `ATAGIA_OPENROUTER_API_KEY`) for OpenRouter models,
`GEMINI_KEY` / `ATAGIA_GOOGLE_API_KEY` for direct Gemini,
`MINIMAX_API_KEY` / `ATAGIA_MINIMAX_API_KEY` for direct MiniMax.

The compactor model is selected with `--model`; provider routing is by model-id
prefix, exactly as production (`google/<id>`, `minimax/MiniMax-M3` direct;
`openrouter/google/<id>`, `openrouter/minimax/minimax-m3` via OpenRouter).

### Phase A — OpenRouter, concurrency 4

```bash
python -m benchmarks.card2_summary --model openrouter/google/gemini-3.1-flash-lite --concurrency 4
python -m benchmarks.card2_summary --model openrouter/minimax/minimax-m3       --concurrency 4
```

### Phase B — direct providers, concurrency sweep {1,2,4,8}

```bash
python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 1
python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 2
python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 4
python -m benchmarks.card2_summary --model minimax/MiniMax-M3 --concurrency 8
```

(Swap `--model google/gemini-3.1-flash-lite` for the direct-Gemini sweep.)

## Options

- `--model` — compactor model spec (default `openrouter/google/gemini-3.1-flash-lite`).
- `--concurrency` — summary card cap, `>= 1` (`1` runs sequentially). Default 4.
- `--runs` — repeated runs over the case set. Default 3.
- `--cases` — `all` | `realistic` | `stress`. Default `all`.
- `--no-examples` — disable the card's few-shot examples (`card_examples_enabled=False`).
- `--out` — report directory; **must be outside the repo** (the shared
  `benchmarks/output_root` guard fails fast on a repo-internal path). Default: a
  UTC-stamped dir under the external benchmark output root
  (`$ATAGIA_BENCH_OUTPUT_ROOT` or `../atagia-benchmarks`) `/card2_summary/`.
- `--selftest` — offline mode (above).

## Output

Each live invocation writes `report.json` (run config + per-case + aggregate
metrics) and `report.md` (a short markdown summary table) under `--out`. Cases
are fictional, so reports carry no PII.
