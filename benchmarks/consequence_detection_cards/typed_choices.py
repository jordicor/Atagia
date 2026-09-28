"""Bounded paired comparison of production consequence-decision cards.

The only arm difference is the model used by the gate, sentiment, and link
cards. Action, outcome, and language remain on the same generative model.
This synthetic component comparison is not evidence of end-to-end memory quality.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv

from atagia.core.clock import SystemClock
from atagia.core.config import Settings
from atagia.memory.consequence_detector import ConsequenceDetector
from atagia.services.llm_client import RetryPolicy
from atagia.services.model_resolution import resolve_component_model
from atagia.services.providers import build_llm_client

from benchmarks.consequence_detection_cards.compare import (
    _DEFAULT_CASES_PATH,
    BenchmarkCase,
    conversation_context,
    load_cases,
    normalize_signal,
    score_output,
)
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    summarize_llm_calls,
)
from benchmarks.output_root import assert_outside_repo

ArmName = Literal["before", "after"]
_DECISION_COMPONENT_IDS = (
    "consequence_gate",
    "consequence_sentiment",
    "consequence_link",
)


def _arm_order(case_index: int) -> tuple[ArmName, ArmName]:
    return ("before", "after") if case_index % 2 == 0 else ("after", "before")


async def _run_arm(
    *,
    arm: ArmName,
    case: BenchmarkCase,
    detector: ConsequenceDetector,
) -> dict[str, Any]:
    started = perf_counter()
    output: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    try:
        signal = await detector.detect(
            message_text=case.message,
            role=case.role,
            conversation_context=conversation_context(case),
            recent_assistant_messages=[dict(item) for item in case.recent_assistant_messages],
        )
        output = normalize_signal(signal)
    except Exception as exc:  # noqa: BLE001 - benchmark records and stops on failure
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed_ms = (perf_counter() - started) * 1000.0
    return {
        "case_id": case.case_id,
        "arm": arm,
        "elapsed_ms": elapsed_ms,
        "output": output,
        "score": score_output(output, case, error=error),
        "error": error,
    }


def _summarize_arm(
    rows: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_calls = [
        call
        for call in calls
        if call.get("purpose")
        in {
            "consequence_gate_card",
            "consequence_sentiment_card",
            "consequence_link_card",
        }
    ]
    return {
        "cases": len(rows),
        "exact_matches": sum(row["score"]["exact_match"] for row in rows),
        "detection_matches": sum(row["score"]["detection_match"] for row in rows),
        "sentiment_matches": sum(row["score"]["sentiment_match"] for row in rows),
        "link_matches": sum(row["score"]["link_match"] for row in rows),
        "technical_failures": sum(row["score"]["technical_failure"] for row in rows),
        "latency_p50_ms": median(row["elapsed_ms"] for row in rows),
        "all_calls": summarize_llm_calls(calls),
        "decision_calls": summarize_llm_calls(decision_calls),
    }


def _recorded_call_failure(calls: list[dict[str, Any]]) -> dict[str, str] | None:
    failures = [call for call in calls if call.get("error") is not None]
    if not failures:
        return None
    purposes = sorted({str(call.get("purpose") or "unknown") for call in failures})
    return {
        "type": "RecordedLLMFailure",
        "message": (
            f"{len(failures)} LLM call(s) failed in recorded purposes: "
            + ", ".join(purposes)
        ),
    }


def _apply_recorded_call_failure(
    row: dict[str, Any],
    *,
    case: BenchmarkCase,
    calls: list[dict[str, Any]],
) -> None:
    if row["error"] is not None:
        return
    recorded_error = _recorded_call_failure(calls)
    if recorded_error is None:
        return
    row["error"] = recorded_error
    row["score"] = score_output(
        row["output"],
        case,
        error=recorded_error,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)[: args.limit]
    if not cases:
        raise ValueError("The paired comparison requires at least one case")
    output_dir = assert_outside_repo(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_settings = Settings.from_env()
    baseline_model = args.baseline_model or resolve_component_model(
        base_settings,
        "consequence_detector",
    )
    settings = replace(
        base_settings,
        llm_forced_global_model=None,
        llm_finite_decisions_enabled=True,
        llm_component_models={
            **base_settings.llm_component_models,
            "consequence_detector": baseline_model,
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
    client._interactive_retry_policy = RetryPolicy(attempts=1)
    client._extraction_retry_policy = RetryPolicy(attempts=1)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        for case_index, case in enumerate(cases):
            for arm in _arm_order(case_index):
                decision_model = (
                    baseline_model if arm == "before" else args.challenger_model
                )
                arm_settings = replace(
                    settings,
                    llm_component_models={
                        **settings.llm_component_models,
                        **{
                            component_id: decision_model
                            for component_id in _DECISION_COMPONENT_IDS
                        },
                    },
                )
                detector = ConsequenceDetector(
                    llm_client=client,
                    clock=SystemClock(),
                    settings=arm_settings,
                    card_concurrency=args.card_concurrency,
                )
                with recorder.context(arm=arm, case_id=case.case_id):
                    row = await _run_arm(arm=arm, case=case, detector=detector)
                _apply_recorded_call_failure(
                    row,
                    case=case,
                    calls=recorder.records_for_context(
                        arm=arm,
                        case_id=case.case_id,
                    ),
                )
                rows.append(row)
                write_json_atomic(output_dir / "per_case.json", rows)
                write_json_atomic(output_dir / "llm_calls.json", recorder.records())
                print(
                    f"{arm} {case.case_id} exact={row['score']['exact_match']} "
                    f"elapsed_ms={row['elapsed_ms']:.0f} error={row['error'] is not None}",
                    flush=True,
                )
                if row["error"] is not None:
                    raise RuntimeError(
                        "Stopped on the first failed arm; inspect retained artifacts"
                    )
    finally:
        write_json_atomic(output_dir / "per_case.json", rows)
        write_json_atomic(output_dir / "llm_calls.json", recorder.records())
        write_json_atomic(output_dir / "run_guard.json", client.llm_run_guard_snapshot())
        await client.aclose()

    pairs = []
    for case in cases:
        pair = {row["arm"]: row for row in rows if row["case_id"] == case.case_id}
        pairs.append(
            {
                "case_id": case.case_id,
                "expected": asdict(case),
                "before": pair["before"],
                "after": pair["after"],
            }
        )
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "baseline_model": baseline_model,
        "challenger_model": args.challenger_model,
        "case_count": len(cases),
        "decision_components": list(_DECISION_COMPONENT_IDS),
        "arms": {
            arm: _summarize_arm(
                [row for row in rows if row["arm"] == arm],
                recorder.records_for_context(arm=arm),
            )
            for arm in ("before", "after")
        },
        "pairs": pairs,
        "scope": (
            "Synthetic production consequence-card comparison. Only gate, sentiment, "
            "and link routing changes; this is not end-to-end memory-quality evidence."
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "status": "complete"}))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(_DEFAULT_CASES_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-model")
    parser.add_argument("--challenger-model", default="typesafe/jev-latest")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--card-concurrency", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--max-reported-cost", type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 12:
        parser.error("limit must be between 1 and 12")
    if args.card_concurrency < 1:
        parser.error("card concurrency must be positive")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
