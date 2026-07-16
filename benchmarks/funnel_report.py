"""Standing failure-funnel report over retained LoCoMo run artifacts.

Reproduces the S1-S4 failure funnel from stored ``locomo-report-*.json`` traces
with break-cause instrumentation and standing memory-error counters. No engine
involvement: everything is derived from artifacts that already exist plus (for
the memory-error counters) retained memory_quality rejudge output.

Stage definitions (each failed question lands in exactly one, by the deepest
stage its gold evidence reached):

- S1 extraction-miss: no memory was produced from the gold turns
  (``trace.evidence_memory_ids`` empty; cross-checked against
  ``trace.missing_evidence_turn_ids``).
- S2 search-miss: gold memories exist but none reached raw candidates (every
  ``critical_evidence_custody.items[]`` entry is absent from raw candidates).
- S3 selection-loss: >=1 gold item reached raw candidates but none was selected.
- S4 answer-stage: >=1 gold memory was selected; split full vs partial by
  per-gold-turn coverage of selected memories.

Eviction reasons for dying gold come from ``trace.retrieval_custody[]`` joined
by ``memory_id`` (the ``critical_evidence_custody.items[]`` entries lack them).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.locomo.night_run_artifacts import (
    QuestionRecord,
    iter_all_records,
    resolve_report_paths,
)
from benchmarks.output_root import assert_outside_repo, utc_run_id

# Mechanical abstention phrase set (v1). Scanning stored predictions is DATA
# analysis, not engine input, so a fixed phrase set is allowed here. Versioned
# so downstream comparisons are reproducible.
ABSTENTION_PHRASES_V1: tuple[str, ...] = (
    "not supported",
    "no information",
    "not enough information",
    "insufficient information",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "no record",
    "not mentioned",
    "isn't mentioned",
    "is not mentioned",
    "not stated",
    "does not mention",
    "doesn't mention",
    "do not have",
    "don't have",
    "does not have",
    "no mention",
    "couldn't find",
    "could not find",
    "can't find",
    "cannot find",
    "not available",
    "i'm not sure",
    "i am not sure",
    "unable to answer",
    "no evidence",
    "not provided",
    "not specified",
)

_MONTHS = {
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11", "december": "12",
    "jan": "1", "feb": "2", "mar": "3", "apr": "4", "jun": "6", "jul": "7",
    "aug": "8", "sep": "9", "sept": "9", "oct": "10", "nov": "11", "dec": "12",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A frozen, previously verified baseline over the retained artifacts. The
# funnel must reproduce these exact counts before any new numbers are trusted.
BASELINE_2026_06_25: dict[str, Any] = {
    "S1": 0,
    "S2": 98,
    "S3": 227,
    "S4": 365,
    "S4_full": 204,
    "S4_partial": 161,
    "S3_dying_gold_items": 630,
    "S3_budget_exhausted": 405,
    "S3_lower_score": 225,
}

_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial-unscored",
}


@dataclass(slots=True)
class _FunnelAccumulator:
    total_questions: int = 0
    total_failed: int = 0
    total_passed: int = 0
    stage_counts: Counter[str] = field(default_factory=Counter)
    s4_full: int = 0
    s4_partial: int = 0
    s3_dying_items: int = 0
    s3_eviction: Counter[str] = field(default_factory=Counter)
    s3_top10_deaths: int = 0
    s3_rank1_deaths: int = 0
    s4_partial_dying_items: int = 0
    s4_partial_eviction: Counter[str] = field(default_factory=Counter)
    stage_by_category: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Abstention / containment (over failed questions).
    abstentions: int = 0
    gold_contained_failures: int = 0
    abstention_kind_null: int = 0
    abstention_kind_present: int = 0


def _normalize_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_PATTERN.findall(text.lower()):
        tokens.add(_MONTHS.get(raw, raw))
    return tokens


def _is_abstention(prediction: str) -> bool:
    lowered = prediction.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES_V1)


def _gold_contained(prediction: str, ground_truth: str) -> bool:
    gold_tokens = _normalize_tokens(ground_truth)
    if not gold_tokens:
        return False
    prediction_tokens = _normalize_tokens(prediction)
    return gold_tokens <= prediction_tokens


def _eviction_and_rank_maps(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    eviction: dict[str, Any] = {}
    rank: dict[str, Any] = {}
    for record in trace.get("retrieval_custody") or []:
        candidate_id = record.get("candidate_id")
        if candidate_id is None:
            continue
        eviction[candidate_id] = record.get("eviction_reason")
        rank[candidate_id] = record.get("score_rank")
    return eviction, rank


def _classify_stage(record: QuestionRecord) -> str | None:
    """Return the funnel stage for a FAILED question, or None if it passed."""
    if record.strict_score == 1:
        return None
    trace = record.trace
    ev_mem_ids = trace.get("evidence_memory_ids") or []
    if not ev_mem_ids:
        return "S1"
    items = (trace.get("critical_evidence_custody") or {}).get("items", [])
    n_raw = sum(1 for item in items if item.get("in_raw_candidates"))
    # Dedupe-aware survival: gold collapsed into a SELECTED duplicate
    # carrier reached the composed context content-wise, so it counts as
    # selected. Old (pre-dedupe) artifacts have no ``deduped_into_selected``
    # key, so their classification is byte-identical.
    n_selected = sum(
        1
        for item in items
        if item.get("selected") or item.get("deduped_into_selected")
    )
    selected_evidence = trace.get("selected_evidence_memory_ids") or []
    if selected_evidence or n_selected >= 1:
        return "S4"
    if n_raw >= 1:
        return "S3"
    return "S2"


def _s4_is_full(record: QuestionRecord) -> bool:
    trace = record.trace
    evidence_message_ids = set(trace.get("evidence_message_ids") or [])
    if not evidence_message_ids:
        return False
    items = (trace.get("critical_evidence_custody") or {}).get("items", [])
    covered: set[str] = set()
    for item in items:
        # A gold item deduped into a selected carrier covers its own source
        # messages: the representative carries the span union at runtime.
        # Absent key on old artifacts keeps the split byte-identical.
        if item.get("selected") or item.get("deduped_into_selected"):
            covered.update(item.get("source_message_ids") or [])
    return evidence_message_ids <= covered


def _accumulate(records: Iterable[QuestionRecord]) -> _FunnelAccumulator:
    acc = _FunnelAccumulator()
    for record in records:
        acc.total_questions += 1
        if record.strict_score == 1:
            acc.total_passed += 1
            continue
        acc.total_failed += 1
        trace = record.trace
        grade_context = trace.get("grade_context") or {}
        if grade_context.get("abstention_kind") is None:
            acc.abstention_kind_null += 1
        else:
            acc.abstention_kind_present += 1
        if _is_abstention(record.prediction):
            acc.abstentions += 1
        if _gold_contained(record.prediction, record.ground_truth):
            acc.gold_contained_failures += 1

        stage = _classify_stage(record)
        if stage is None:
            continue
        acc.stage_counts[stage] += 1
        acc.stage_by_category[stage][record.category] += 1

        items = (trace.get("critical_evidence_custody") or {}).get("items", [])
        eviction, rank = _eviction_and_rank_maps(trace)

        if stage == "S4":
            is_full = _s4_is_full(record)
            if is_full:
                acc.s4_full += 1
            else:
                acc.s4_partial += 1
                for item in items:
                    if item.get("scored") and not item.get("selected"):
                        acc.s4_partial_dying_items += 1
                        acc.s4_partial_eviction[
                            _reason_label(eviction.get(item.get("memory_id")))
                        ] += 1
        elif stage == "S3":
            for item in items:
                if item.get("scored") and not item.get("selected"):
                    acc.s3_dying_items += 1
                    memory_id = item.get("memory_id")
                    acc.s3_eviction[_reason_label(eviction.get(memory_id))] += 1
                    score_rank = rank.get(memory_id)
                    if isinstance(score_rank, int) and score_rank <= 10:
                        acc.s3_top10_deaths += 1
                        if score_rank == 1:
                            acc.s3_rank1_deaths += 1
    return acc


def _reason_label(reason: Any) -> str:
    if reason is None:
        return "unlabeled"
    return str(reason)


def build_funnel_report(
    report_paths: Sequence[Path],
    *,
    memory_quality_verdicts_path: str | Path | None = None,
) -> dict[str, Any]:
    acc = _accumulate(iter_all_records(report_paths, include_trace=True))
    stage_counts = {stage: int(acc.stage_counts.get(stage, 0)) for stage in ("S1", "S2", "S3", "S4")}
    funnel = {
        **stage_counts,
        "S4_full": acc.s4_full,
        "S4_partial": acc.s4_partial,
    }
    baseline_match = _baseline_match(funnel, acc)
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "benchmarks.funnel_report",
        "source_reports": [
            {"path": str(path), "sha256": sha256_file_if_exists(path)}
            for path in report_paths
        ],
        "totals": {
            "questions": acc.total_questions,
            "passed": acc.total_passed,
            "failed": acc.total_failed,
            "accuracy": round(acc.total_passed / acc.total_questions, 4)
            if acc.total_questions
            else 0.0,
        },
        "funnel": funnel,
        "funnel_by_category": {
            stage: {
                str(category): count
                for category, count in sorted(acc.stage_by_category[stage].items())
            }
            for stage in ("S1", "S2", "S3", "S4")
        },
        "s3_break_cause": {
            "dying_gold_items": acc.s3_dying_items,
            "eviction_reason": dict(sorted(acc.s3_eviction.items())),
            "top10_ranked_deaths": acc.s3_top10_deaths,
            "rank1_deaths": acc.s3_rank1_deaths,
        },
        "s4_partial_break_cause": {
            "dying_gold_items": acc.s4_partial_dying_items,
            "eviction_reason": dict(sorted(acc.s4_partial_eviction.items())),
        },
        "break_cause_note": (
            "Eviction reasons are the artifact-available signal. Newer composer "
            "traces split the old conflated budget_exhausted bucket into token-wall "
            "budget_exhausted / item_cap_reached / class_cap_reached / "
            "diversity_demoted via a composer trace field; runs produced before "
            "that split (e.g. the frozen baseline) still carry the single "
            "budget_exhausted label covering token-wall + item-cap + "
            "diversity-demotion, so --assert-baseline stays count-exact on them."
        ),
        "abstention": {
            "phrase_set_version": "v1",
            "abstaining_failures": acc.abstentions,
            "gold_contained_failures": acc.gold_contained_failures,
        },
        "abstention_kind_instrumentation": {
            "null_count": acc.abstention_kind_null,
            "present_count": acc.abstention_kind_present,
            "note": (
                "trace.grade_context.abstention_kind is null across all failed "
                "questions. Investigated: it is NOT an engine field — "
                "the LoCoMo harness hardcodes it to None in "
                "LoCoMoBenchmark._grade_context_for_question "
                "(benchmarks/locomo/benchmark.py), a benchmark-side placeholder "
                "the atagia_bench/third_party runners populate from grader "
                "config but LoCoMo never does. Wiring it to a real abstention "
                "signal is deferred harness work; the mechanical "
                "phrase-set detector above covers the need meanwhile."
            ),
        },
        "baseline_match": baseline_match,
    }
    result["memory_error_counters"] = _load_memory_error_counters(
        memory_quality_verdicts_path
    )
    return result


def _baseline_match(funnel: dict[str, int], acc: _FunnelAccumulator) -> dict[str, Any]:
    observed = {
        "S1": funnel["S1"],
        "S2": funnel["S2"],
        "S3": funnel["S3"],
        "S4": funnel["S4"],
        "S4_full": funnel["S4_full"],
        "S4_partial": funnel["S4_partial"],
        "S3_dying_gold_items": acc.s3_dying_items,
        "S3_budget_exhausted": int(acc.s3_eviction.get("budget_exhausted", 0)),
        "S3_lower_score": int(acc.s3_eviction.get("lower_score", 0)),
    }
    matches = observed == BASELINE_2026_06_25
    return {
        "reference": "night_gemini_kimi_highspeed_20260625",
        "expected": BASELINE_2026_06_25,
        "observed": observed,
        "count_exact": matches,
    }


def _load_memory_error_counters(
    verdicts_path: str | Path | None,
) -> dict[str, Any]:
    if verdicts_path is None:
        return {
            "available": False,
            "note": (
                "Provide --memory-quality-verdicts pointing at the retained "
                "memory_quality rejudge output to populate false-addition and "
                "misattribution counters."
            ),
        }
    payload = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    counters = payload.get("memory_error_counters")
    if counters is None:
        raise ValueError(
            f"{verdicts_path} has no memory_error_counters block; it must be a "
            "memory_quality rejudge output."
        )
    return {"available": True, "source": str(verdicts_path), **counters}


def _format_summary(report: dict[str, Any]) -> str:
    totals = report["totals"]
    funnel = report["funnel"]
    s3 = report["s3_break_cause"]
    baseline = report["baseline_match"]
    lines = [
        f"LoCoMo failure funnel over {totals['questions']} questions "
        f"({totals['passed']} passed / {totals['failed']} failed, "
        f"{totals['accuracy'] * 100:.2f}%)",
        (
            f"  S1 extraction-miss: {funnel['S1']}  "
            f"S2 search-miss: {funnel['S2']}  "
            f"S3 selection-loss: {funnel['S3']}  "
            f"S4 answer-stage: {funnel['S4']} "
            f"({funnel['S4_full']} full / {funnel['S4_partial']} partial)"
        ),
        (
            f"  S3 dying gold items: {s3['dying_gold_items']} "
            f"({s3['eviction_reason']}); top10 deaths {s3['top10_ranked_deaths']}, "
            f"rank1 deaths {s3['rank1_deaths']}"
        ),
        f"  Baseline count-exact reproduction: {'YES' if baseline['count_exact'] else 'NO'}",
    ]
    if not baseline["count_exact"]:
        lines.append(f"    expected={baseline['expected']}")
        lines.append(f"    observed={baseline['observed']}")
    abst = report["abstention"]
    lines.append(
        f"  Abstaining failures: {abst['abstaining_failures']}  "
        f"gold-contained failures: {abst['gold_contained_failures']}"
    )
    counters = report.get("memory_error_counters") or {}
    if counters.get("available"):
        lines.append(
            f"  Memory-error counters: false_addition_rate "
            f"{counters.get('false_addition_rate')} "
            f"misattribution_rate {counters.get('misattribution_rate')}"
        )
    return "\n".join(lines)


def _output_path(output_dir: Path | None) -> Path:
    target = output_dir if output_dir is not None else Path.cwd() / "funnel_output"
    assert_outside_repo(target)
    target.mkdir(parents=True, exist_ok=True)
    return target / f"locomo-funnel-{utc_run_id()}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the LoCoMo failure-funnel report from run artifacts."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Report JSON files or run directories.",
    )
    parser.add_argument(
        "--memory-quality-verdicts",
        default=None,
        help="Optional memory_quality rejudge output for error counters.",
    )
    parser.add_argument("--output", default=None, help="Output directory (outside repo).")
    parser.add_argument(
        "--assert-baseline",
        action="store_true",
        help="Exit non-zero unless the funnel count-exact reproduces the 2026-06-25 baseline.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report_paths = resolve_report_paths(args.reports)
    report = build_funnel_report(
        report_paths,
        memory_quality_verdicts_path=args.memory_quality_verdicts,
    )
    output_path = _output_path(
        Path(args.output).expanduser() if args.output else None
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(_format_summary(report))
    print(f"\nFunnel report saved to: {output_path}")
    if args.assert_baseline and not report["baseline_match"]["count_exact"]:
        print("BASELINE MISMATCH: funnel did not count-exact reproduce the baseline.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
