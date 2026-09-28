"""Interleaved before/after comparison using the production applicability scorer.

Only relevance routing changes. Date cards, candidates, policy, and ranking stay
identical. Artifacts include every case and call, not just aggregate wins.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from atagia.core.config import Settings
from atagia.services.llm_client import RetryPolicy
from atagia.services.model_resolution import parse_model_spec, resolve_component_model
from atagia.services.providers import build_llm_client

from benchmarks.applicability_cards.compare import (
    _DEFAULT_CASES_PATH,
    _estimate_records_cost_usd,
    _latency_summary,
    load_cases,
    run_one_variant,
    write_jsonl_atomic,
)
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder, summarize_llm_calls
from benchmarks.output_root import assert_outside_repo


def summarize_arm(rows: list[dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, Any]:
    relevance = [call for call in calls if call["purpose"] == "applicability_relevance_card"]
    return {
        "trials": len(rows),
        "top_hits": sum(row["score"]["top_hit"] for row in rows),
        "exact_matches": sum(row["score"]["exact_match"] for row in rows),
        "mean_useful_recall": sum(row["score"]["expected_useful_recall"] for row in rows) / len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "wall_time_ms": _latency_summary([row["wall_time_ms"] for row in rows]),
        "relevance_latency_ms": _latency_summary([call["latency_ms"] for call in relevance]),
        "all_calls": summarize_llm_calls(calls),
        "relevance_calls": summarize_llm_calls(relevance),
        "estimated_relevance_cost_usd": _estimate_records_cost_usd(relevance),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Repetitions and limit must be positive")
    cases_path = Path(args.cases)
    cases = load_cases(cases_path, limit=args.limit)
    if not cases:
        raise ValueError("The comparison requires at least one case")
    output = assert_outside_repo(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    base = Settings.from_env()
    baseline = args.baseline_model or resolve_component_model(base, "applicability_relevance")
    date_model = resolve_component_model(base, "applicability_scorer")
    settings = replace(
        base,
        llm_forced_global_model=None,
        llm_finite_decisions_enabled=(
            base.llm_finite_decisions_enabled
            or parse_model_spec(args.challenger_model).provider_slug == "typesafe"
        ),
        llm_component_models={
            **base.llm_component_models,
            "applicability_scorer": date_model,
            "applicability_relevance": args.challenger_model,
        },
        llm_request_timeout_seconds=30.0,
        llm_output_limit_retry_attempts=0,
        llm_run_guard_enabled=True,
        llm_run_guard_mode="enforce",
        llm_run_guard_max_total_calls=args.max_calls,
        llm_run_guard_max_total_tokens=args.max_tokens,
        llm_run_guard_max_reported_cost_usd=args.max_reported_cost,
        llm_run_guard_max_total_failed_calls=1,
    )
    client = build_llm_client(settings, retry_policy=RetryPolicy(attempts=1))
    # Check both routes before the first billable call.
    client._provider(parse_model_spec(baseline).provider_name)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    variant = "cards_batch_4_no_date" if args.no_date else "cards_batch_4"
    started = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    try:
        for repetition in range(1, args.repetitions + 1):
            for index, case in enumerate(cases):
                arms = [("before", baseline), ("after", args.challenger_model)]
                if (index + repetition) % 2 == 0:
                    arms.reverse()
                for arm, model in arms:
                    with recorder.context(arm=arm, case_id=case.case_id, repetition=repetition):
                        row = await run_one_variant(
                            client=client, base_settings=settings, case=case, variant=variant,
                            card_model=date_model, relevance_model=model, repetition=repetition,
                        )
                    row["arm"] = arm
                    rows.append(row)
                    write_jsonl_atomic(output / "per_case.jsonl", rows)
                    write_json_atomic(output / "llm_calls.json", recorder.records())
                    print(
                        f"{arm} {case.case_id} rep={repetition} "
                        f"top_hit={row['score']['top_hit']} "
                        f"wall_ms={row['wall_time_ms']:.0f} error={row['error'] is not None}",
                        flush=True,
                    )
                    if row["error"]:
                        raise RuntimeError(f"Stopped on first failed trial; inspect {output / 'per_case.jsonl'}")
    finally:
        await client.aclose()

    pairs = []
    for repetition in range(1, args.repetitions + 1):
        for case in cases:
            pair = {row["arm"]: row for row in rows
                    if row["case_id"] == case.case_id and row["repetition"] == repetition}
            pairs.append({
                "case_id": case.case_id,
                "repetition": repetition,
                "before": pair["before"]["score"],
                "after": pair["after"]["score"],
                "rank_changed": pair["before"]["score"]["ranked_ids"] != pair["after"]["score"]["ranked_ids"],
            })
    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "baseline_relevance_model": baseline,
        "challenger_relevance_model": args.challenger_model,
        "date_model": date_model,
        "variant": variant,
        "repetitions": args.repetitions,
        "privacy_enforcement": "off",
        "arms": {
            arm: summarize_arm([row for row in rows if row["arm"] == arm],
                               recorder.records_for_context(arm=arm))
            for arm in ("before", "after")
        },
        "per_case_comparison": pairs,
        "scope": "Synthetic fixed-candidate scorer test, not full retrieval or answer quality.",
    }
    write_json_atomic(output / "summary.json", summary)
    print(json.dumps({"output_dir": str(output), "status": "complete"}))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(_DEFAULT_CASES_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-model")
    parser.add_argument("--challenger-model", default="typesafe/jev-latest")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-date", action="store_true")
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=500000)
    parser.add_argument("--max-reported-cost", type=float, default=2.5)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
