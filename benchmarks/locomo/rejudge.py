"""Re-judge retained LoCoMo night-run predictions under a chosen protocol.

This tool re-scores every stored ``(question, ground_truth, prediction)`` from
existing ``locomo-report-*.json`` artifacts with one of the three judge
protocols and emits a parallel report plus a per-question diff versus the stored
strict verdicts. There is NO engine involvement: predictions are read from disk;
only the judge LLM is called.

Protocols:

- ``source_aware_strict`` — reuses the stored verdicts (they were produced by the
  same judge model + strict prompt), so this is the honest strict baseline with
  zero new LLM calls.
- ``gold_only_lenient`` — Mem0-parity re-judge of every question.
- ``memory_quality`` — two-stage: a stored strict PASS is a memory_quality pass
  by construction, so only stored strict FAILURES are re-judged (with the full
  conversation transcript as ground truth for extras). The E1b extras-truth
  audit falls out of these structured verdicts.

Cost safety: the judge is pinned to a Kimi model. Real token counts (including
provider-reported cached-input tokens) are captured per call and priced with the
documented Kimi rates. A live budget guard extrapolates observed cost at
per-conversation granularity and halts (writing a partial report) once the
projection exceeds ``--max-cost-usd``; because it checks between conversations,
it can overshoot by up to one conversation's batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder
from benchmarks.locomo.night_run_artifacts import (
    QuestionRecord,
    iter_all_records,
    load_conversations,
    render_conversation_transcript,
    resolve_report_paths,
    source_evidence_for_record,
)
from benchmarks.output_root import assert_outside_repo, utc_run_id
from benchmarks.scorer import JudgeProtocol, JudgeVerdict, LLMJudgeScorer
from atagia.core.config import Settings
from atagia.services.providers import build_llm_client

# Load .env before any Settings.from_env() call resolves provider keys.
load_dotenv()

_DEFAULT_JUDGE_MODEL = "kimi/kimi-k2.7-code-highspeed"
_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
# Documented Kimi K2.7 Code rates (USD per 1M tokens): openrouter.ai/moonshotai
# /kimi-k2.7-code and platform.kimi.ai/docs/pricing/chat.
_KIMI_INPUT_PRICE = 0.95
_KIMI_OUTPUT_PRICE = 4.00
_KIMI_CACHED_INPUT_PRICE = 0.19
# Rough char-per-token ratio for the pre-run projection only (real token counts
# come from the provider during the run).
_CHARS_PER_TOKEN = 4.0
_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial-unscored",
}


@dataclass(slots=True)
class RejudgeConfig:
    report_specs: list[str]
    protocol: JudgeProtocol
    judge_model: str = _DEFAULT_JUDGE_MODEL
    data_path: Path = _DEFAULT_DATA_PATH
    output_dir: Path | None = None
    concurrency: int = 8
    max_questions: int | None = None
    max_cost_usd: float = 10.0
    dry_run: bool = False
    input_price: float = _KIMI_INPUT_PRICE
    output_price: float = _KIMI_OUTPUT_PRICE
    cached_input_price: float = _KIMI_CACHED_INPUT_PRICE


@dataclass(slots=True)
class _Verdict:
    """Per-question rejudge outcome (structured, JSON-serializable)."""

    question_id: str
    conversation_id: str
    category: int
    strict_score: int
    protocol: str
    score: int
    reasoning: str
    failure_reason: str | None = None
    missing_info: bool | None = None
    false_addition: bool | None = None
    misattribution: bool | None = None
    true_addition_only: bool | None = None
    rejudged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "conversation_id": self.conversation_id,
            "category": self.category,
            "strict_score": self.strict_score,
            "protocol": self.protocol,
            "score": self.score,
            "rejudged": self.rejudged,
            "failure_reason": self.failure_reason,
            "missing_info": self.missing_info,
            "false_addition": self.false_addition,
            "misattribution": self.misattribution,
            "true_addition_only": self.true_addition_only,
            "reasoning": self.reasoning,
        }


def _record_input_chars(
    record: QuestionRecord,
    protocol: JudgeProtocol,
    transcript: str,
) -> int:
    base = len(record.question_text) + len(record.ground_truth) + len(record.prediction)
    if protocol is JudgeProtocol.MEMORY_QUALITY:
        return base + len(transcript) + 600
    return base + 400


def _project_cost(
    records: Sequence[QuestionRecord],
    protocol: JudgeProtocol,
    transcripts: dict[str, str],
    config: RejudgeConfig,
) -> dict[str, Any]:
    """Pre-run, cache-free cost projection using mechanical token estimates."""
    to_call = [
        record
        for record in records
        if _needs_llm(record, protocol)
    ]
    input_tokens = sum(
        _estimate_tokens_for_call(record, protocol, transcripts) for record in to_call
    )
    output_tokens = 200 * len(to_call)
    cost = (
        input_tokens / 1_000_000 * config.input_price
        + output_tokens / 1_000_000 * config.output_price
    )
    return {
        "llm_calls_planned": len(to_call),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd_no_cache": round(cost, 4),
        "price_assumption": {
            "input_per_mtok": config.input_price,
            "output_per_mtok": config.output_price,
            "cached_input_per_mtok": config.cached_input_price,
        },
    }


def _estimate_tokens_for_call(
    record: QuestionRecord,
    protocol: JudgeProtocol,
    transcripts: dict[str, str],
) -> int:
    transcript = transcripts.get(record.conversation_id, "")
    input_chars = _record_input_chars(record, protocol, transcript)
    return int(input_chars / _CHARS_PER_TOKEN) + 1


def _needs_llm(record: QuestionRecord, protocol: JudgeProtocol) -> bool:
    if protocol is JudgeProtocol.SOURCE_AWARE_STRICT:
        return False
    if protocol is JudgeProtocol.MEMORY_QUALITY:
        # Two-stage: strict passes are memory_quality passes by construction.
        return record.strict_score == 0
    return True  # gold_only_lenient re-judges everything.


def _stored_verdict(record: QuestionRecord, protocol: JudgeProtocol) -> _Verdict:
    """Build a verdict that needs no LLM call (strict pass / strict reuse)."""
    if protocol is JudgeProtocol.MEMORY_QUALITY:
        # A stored strict PASS: all requested info present and no unsupported
        # extras (strict already rejects those), so memory_quality passes.
        return _Verdict(
            question_id=record.question_id,
            conversation_id=record.conversation_id,
            category=record.category,
            strict_score=record.strict_score,
            protocol=protocol.value,
            score=1,
            reasoning="Strict PASS -> memory_quality PASS by construction (no extras).",
            missing_info=False,
            false_addition=False,
            misattribution=False,
            true_addition_only=False,
            rejudged=False,
        )
    # source_aware_strict: reuse the stored verdict verbatim.
    return _Verdict(
        question_id=record.question_id,
        conversation_id=record.conversation_id,
        category=record.category,
        strict_score=record.strict_score,
        protocol=protocol.value,
        score=record.strict_score,
        reasoning=record.strict_reasoning,
        failure_reason=None if record.strict_score == 1 else "other",
        rejudged=False,
    )


def _verdict_from_judge(
    record: QuestionRecord,
    judge_verdict: JudgeVerdict,
) -> _Verdict:
    return _Verdict(
        question_id=record.question_id,
        conversation_id=record.conversation_id,
        category=record.category,
        strict_score=record.strict_score,
        protocol=judge_verdict.protocol,
        score=judge_verdict.score,
        reasoning=judge_verdict.reasoning,
        failure_reason=judge_verdict.failure_reason,
        missing_info=judge_verdict.missing_info,
        false_addition=judge_verdict.false_addition,
        misattribution=judge_verdict.misattribution,
        true_addition_only=judge_verdict.true_addition_only,
        rejudged=True,
    )


def _actual_cost(records: list[dict[str, Any]], config: RejudgeConfig) -> dict[str, Any]:
    """Compute honest cost from provider-reported token counts."""
    input_tokens = 0.0
    cached_input_tokens = 0.0
    output_tokens = 0.0
    for record in records:
        counts = record.get("token_counts") or {}
        input_tokens += float(counts.get("input_tokens") or 0.0)
        cached_input_tokens += float(counts.get("cached_input_tokens") or 0.0)
        output_tokens += float(counts.get("output_tokens") or 0.0)
    non_cached_input = max(0.0, input_tokens - cached_input_tokens)
    cost = (
        non_cached_input / 1_000_000 * config.input_price
        + cached_input_tokens / 1_000_000 * config.cached_input_price
        + output_tokens / 1_000_000 * config.output_price
    )
    return {
        "llm_calls": len(records),
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": round(cost, 4),
    }


async def run_rejudge(config: RejudgeConfig) -> dict[str, Any]:
    report_paths = resolve_report_paths(config.report_specs)
    records = list(iter_all_records(report_paths, include_trace=False))
    if config.max_questions is not None:
        records = records[: config.max_questions]

    transcripts: dict[str, str] = {}
    need_transcripts = config.protocol is JudgeProtocol.MEMORY_QUALITY
    conversations: dict[str, Any] = {}
    if need_transcripts:
        conversations = load_conversations(config.data_path)
        transcripts = {
            conversation_id: render_conversation_transcript(conversation)
            for conversation_id, conversation in conversations.items()
        }

    strict_baseline = _aggregate(
        [
            _Verdict(
                question_id=r.question_id,
                conversation_id=r.conversation_id,
                category=r.category,
                strict_score=r.strict_score,
                protocol=JudgeProtocol.SOURCE_AWARE_STRICT.value,
                score=r.strict_score,
                reasoning=r.strict_reasoning,
            )
            for r in records
        ]
    )

    projection = _project_cost(records, config.protocol, transcripts, config)
    print("Rejudge cost projection (cache-free upper bound):", flush=True)
    print(json.dumps(projection, indent=2), flush=True)
    if projection["estimated_cost_usd_no_cache"] > config.max_cost_usd:
        print(
            f"NOTE: cache-free projection "
            f"${projection['estimated_cost_usd_no_cache']} exceeds "
            f"--max-cost-usd ${config.max_cost_usd}. Kimi prefix-caching of the "
            f"shared transcript prefix typically reduces this well below the "
            f"cap; a live per-conversation budget guard will halt the run if the "
            f"observed cost trends above the cap.",
            flush=True,
        )

    if config.dry_run:
        return {
            "dry_run": True,
            "report_paths": [str(path) for path in report_paths],
            "total_questions": len(records),
            "strict_baseline": strict_baseline,
            "projection": projection,
        }

    settings = Settings.from_env()
    client = build_llm_client(settings)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    scorer = LLMJudgeScorer(client, config.judge_model, config.protocol)

    verdicts: list[_Verdict] = []
    stopped_early = False
    stop_reason: str | None = None

    # Group by conversation so the shared transcript prefix stays cache-warm and
    # the budget guard can extrapolate after each conversation completes.
    by_conversation: dict[str, list[QuestionRecord]] = defaultdict(list)
    for record in records:
        by_conversation[record.conversation_id].append(record)

    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def judge_one(record: QuestionRecord) -> _Verdict:
        transcript = transcripts.get(record.conversation_id)
        source_evidence = None
        if need_transcripts:
            conversation = conversations.get(record.conversation_id)
            if conversation is None or transcript is None:
                raise ValueError(
                    f"No dataset conversation for {record.conversation_id}"
                )
            source_evidence = source_evidence_for_record(record, conversation)
        async with semaphore:
            judge_verdict = await scorer.evaluate(
                question=record.question_text,
                prediction=record.prediction,
                ground_truth=record.ground_truth,
                source_evidence=source_evidence,
                conversation_transcript=transcript,
            )
        return _verdict_from_judge(record, judge_verdict)

    llm_calls_done = 0
    for conversation_id in sorted(by_conversation):
        convo_records = by_conversation[conversation_id]
        to_call = [r for r in convo_records if _needs_llm(r, config.protocol)]
        for record in convo_records:
            if not _needs_llm(record, config.protocol):
                verdicts.append(_stored_verdict(record, config.protocol))

        if to_call:
            # Warm the transcript prefix cache with the first call, then fan out.
            first = await judge_one(to_call[0])
            verdicts.append(first)
            if len(to_call) > 1:
                rest = await asyncio.gather(*(judge_one(r) for r in to_call[1:]))
                verdicts.extend(rest)
            llm_calls_done += len(to_call)

            # Live budget guard: extrapolate from observed cost.
            cost_so_far = _actual_cost(recorder.records(), config)
            if llm_calls_done > 0:
                total_planned = projection["llm_calls_planned"]
                avg = cost_so_far["cost_usd"] / llm_calls_done
                projected_total = avg * total_planned
                if projected_total > config.max_cost_usd:
                    stopped_early = True
                    stop_reason = (
                        f"Halted after {llm_calls_done}/{total_planned} judge "
                        f"calls: observed ${cost_so_far['cost_usd']} implies a "
                        f"projected ${round(projected_total, 2)} total, above "
                        f"--max-cost-usd ${config.max_cost_usd}."
                    )
                    print(stop_reason, flush=True)
                    break

    result = _build_result(
        config=config,
        report_paths=report_paths,
        records=records,
        verdicts=verdicts,
        strict_baseline=strict_baseline,
        projection=projection,
        recorder=recorder,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )
    return result


def _build_result(
    *,
    config: RejudgeConfig,
    report_paths: list[Path],
    records: list[QuestionRecord],
    verdicts: list[_Verdict],
    strict_baseline: dict[str, Any],
    projection: dict[str, Any],
    recorder: LLMCallRecorder,
    stopped_early: bool,
    stop_reason: str | None,
) -> dict[str, Any]:
    protocol = config.protocol
    aggregate = _aggregate(verdicts)
    flips = _verdict_flips(verdicts)
    cost = _actual_cost(recorder.records(), config)
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "benchmarks.locomo.rejudge",
        "judge_protocol": protocol.value,
        "judge_model": config.judge_model,
        "source_reports": [
            {"path": str(path), "sha256": sha256_file_if_exists(path)}
            for path in report_paths
        ],
        "run_config": {
            "data_path": str(config.data_path),
            "concurrency": config.concurrency,
            "max_questions": config.max_questions,
            "max_cost_usd": config.max_cost_usd,
            "privacy_enforcement": "off",
        },
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "coverage": {
            "total_questions": len(records),
            "verdicts_produced": len(verdicts),
        },
        "strict_baseline": strict_baseline,
        "protocol_result": aggregate,
        "expected_ordering_check": _expected_ordering(strict_baseline, aggregate, protocol),
        "verdict_flips": flips,
        "cost": {"projection": projection, "actual": cost},
        "verdicts": [verdict.as_dict() for verdict in verdicts],
    }
    if protocol is JudgeProtocol.MEMORY_QUALITY:
        result["e1b_extras_truth_audit"] = _e1b_decomposition(verdicts)
        result["memory_error_counters"] = _memory_error_counters(verdicts, len(records))
    return result


def _aggregate(verdicts: Sequence[_Verdict]) -> dict[str, Any]:
    total = len(verdicts)
    passed = sum(1 for verdict in verdicts if verdict.score == 1)
    per_category: dict[str, dict[str, Any]] = {}
    by_category: dict[int, list[_Verdict]] = defaultdict(list)
    for verdict in verdicts:
        by_category[verdict.category].append(verdict)
    for category in sorted(by_category):
        items = by_category[category]
        cat_passed = sum(1 for verdict in items if verdict.score == 1)
        per_category[str(category)] = {
            "name": _CATEGORY_NAMES.get(category, str(category)),
            "passed": cat_passed,
            "total": len(items),
            "accuracy": round(cat_passed / len(items), 4) if items else 0.0,
        }
    return {
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "category_breakdown": per_category,
    }


def _expected_ordering(
    strict_baseline: dict[str, Any],
    aggregate: dict[str, Any],
    protocol: JudgeProtocol,
) -> dict[str, Any]:
    strict_acc = strict_baseline["accuracy"]
    protocol_acc = aggregate["accuracy"]
    # Both gold_only_lenient (parity) and memory_quality should be >= strict.
    satisfied = protocol_acc + 1e-9 >= strict_acc
    return {
        "rule": f"{protocol.value} accuracy should be >= source_aware_strict accuracy",
        "strict_accuracy": strict_acc,
        "protocol_accuracy": protocol_acc,
        "satisfied": bool(satisfied),
    }


def _verdict_flips(verdicts: Sequence[_Verdict]) -> dict[str, Any]:
    strict_fail_now_pass: list[dict[str, Any]] = []
    strict_pass_now_fail: list[dict[str, Any]] = []
    for verdict in verdicts:
        if verdict.strict_score == 0 and verdict.score == 1:
            strict_fail_now_pass.append(_flip_entry(verdict))
        elif verdict.strict_score == 1 and verdict.score == 0:
            strict_pass_now_fail.append(_flip_entry(verdict))
    return {
        "strict_fail_to_protocol_pass": {
            "count": len(strict_fail_now_pass),
            "items": strict_fail_now_pass,
        },
        "strict_pass_to_protocol_fail": {
            "count": len(strict_pass_now_fail),
            "items": strict_pass_now_fail,
        },
    }


def _flip_entry(verdict: _Verdict) -> dict[str, Any]:
    return {
        "question_id": verdict.question_id,
        "category": verdict.category,
        "category_name": _CATEGORY_NAMES.get(verdict.category, str(verdict.category)),
        "false_addition": verdict.false_addition,
        "misattribution": verdict.misattribution,
        "true_addition_only": verdict.true_addition_only,
        "reasoning": verdict.reasoning,
    }


def _e1b_decomposition(verdicts: Sequence[_Verdict]) -> dict[str, Any]:
    """Decompose the strict-failure addition-flagged pool (E1b audit).

    The pool is every strict FAILURE whose memory_quality re-judge found all
    requested information present (``missing_info`` is False), i.e. strict failed
    it for the extras rather than for omission. Each pool member is classified as
    a TRUE extra (``true_addition_only``), a FALSE addition (hallucination), or a
    MISATTRIBUTION (wrong person / crossed reference).
    """
    pool = [
        verdict
        for verdict in verdicts
        if verdict.strict_score == 0
        and verdict.rejudged
        and verdict.missing_info is False
    ]
    overall = _classify_extras(pool)
    per_category: dict[str, dict[str, int]] = {}
    by_category: dict[int, list[_Verdict]] = defaultdict(list)
    for verdict in pool:
        by_category[verdict.category].append(verdict)
    for category in sorted(by_category):
        counts = _classify_extras(by_category[category])
        counts["name"] = _CATEGORY_NAMES.get(category, str(category))
        per_category[str(category)] = counts
    return {
        "definition": (
            "strict failures whose memory_quality re-judge found no missing info "
            "(failed for extras), decomposed into true / false / misattributed; "
            "buckets can overlap (multi_flagged) and some members carry no flag "
            "(none_flagged), so buckets do not sum to pool"
        ),
        "addition_flagged_pool": len(pool),
        "overall": overall,
        "per_category": per_category,
    }


def _classify_extras(verdicts: Sequence[_Verdict]) -> dict[str, int]:
    """Count E1b buckets plus their relationship to the pool.

    The three flag buckets neither partition the pool nor are disjoint, so the
    counts alone cannot be reconciled against ``pool``. ``none_flagged`` (pool
    members carrying no flag — e.g. a strict verdict the memory_quality judge
    simply disagreed with, or a non-addition failure reason) and
    ``multi_flagged`` (members counted in more than one bucket) make the
    arithmetic auditable.
    """
    true_extra = sum(1 for verdict in verdicts if verdict.true_addition_only is True)
    false_extra = sum(1 for verdict in verdicts if verdict.false_addition is True)
    misattributed = sum(1 for verdict in verdicts if verdict.misattribution is True)
    none_flagged = 0
    multi_flagged = 0
    for verdict in verdicts:
        flag_count = sum(
            1
            for flag in (
                verdict.true_addition_only,
                verdict.false_addition,
                verdict.misattribution,
            )
            if flag is True
        )
        if flag_count == 0:
            none_flagged += 1
        elif flag_count > 1:
            multi_flagged += 1
    return {
        "pool": len(verdicts),
        "true_addition_only": true_extra,
        "false_addition": false_extra,
        "misattribution": misattributed,
        "none_flagged": none_flagged,
        "multi_flagged": multi_flagged,
    }


def _memory_error_counters(
    verdicts: Sequence[_Verdict],
    total_questions: int,
) -> dict[str, Any]:
    """Standing false-addition and misattribution counters (overall + category).

    Rates use the total scored question count as the denominator so they are
    comparable across phases regardless of how many were re-judged.
    """
    denom = total_questions or 1
    false_add = sum(1 for verdict in verdicts if verdict.false_addition is True)
    misattr = sum(1 for verdict in verdicts if verdict.misattribution is True)
    per_category: dict[str, dict[str, Any]] = {}
    by_category: dict[int, list[_Verdict]] = defaultdict(list)
    total_by_category: dict[int, int] = defaultdict(int)
    for verdict in verdicts:
        total_by_category[verdict.category] += 1
        by_category[verdict.category].append(verdict)
    for category in sorted(by_category):
        items = by_category[category]
        cat_denom = total_by_category[category] or 1
        cat_false = sum(1 for verdict in items if verdict.false_addition is True)
        cat_misattr = sum(1 for verdict in items if verdict.misattribution is True)
        per_category[str(category)] = {
            "name": _CATEGORY_NAMES.get(category, str(category)),
            "false_addition_count": cat_false,
            "false_addition_rate": round(cat_false / cat_denom, 4),
            "misattribution_count": cat_misattr,
            "misattribution_rate": round(cat_misattr / cat_denom, 4),
        }
    return {
        "denominator_questions": total_questions,
        "false_addition_count": false_add,
        "false_addition_rate": round(false_add / denom, 4),
        "misattribution_count": misattr,
        "misattribution_rate": round(misattr / denom, 4),
        "per_category": per_category,
    }


def _output_path(config: RejudgeConfig) -> Path:
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = Path.cwd() / "rejudge_output"
    assert_outside_repo(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"locomo-rejudge-{config.protocol.value}-{utc_run_id()}.json"


def _format_summary(result: dict[str, Any]) -> str:
    protocol = result["judge_protocol"]
    strict = result["strict_baseline"]
    proto = result["protocol_result"]
    lines = [
        f"Rejudge protocol: {protocol}  judge_model: {result['judge_model']}",
        (
            f"Strict baseline:   {strict['passed']}/{strict['total']} = "
            f"{strict['accuracy'] * 100:.2f}%"
        ),
        (
            f"{protocol}: {proto['passed']}/{proto['total']} = "
            f"{proto['accuracy'] * 100:.2f}%"
        ),
    ]
    ordering = result.get("expected_ordering_check") or {}
    lines.append(
        f"Ordering ({ordering.get('rule', '')}): "
        f"{'OK' if ordering.get('satisfied') else 'VIOLATED'}"
    )
    for category in sorted(proto.get("category_breakdown", {})):
        entry = proto["category_breakdown"][category]
        lines.append(
            f"  cat {category} ({entry['name']}): "
            f"{entry['passed']}/{entry['total']} = {entry['accuracy'] * 100:.2f}%"
        )
    if "e1b_extras_truth_audit" in result:
        audit = result["e1b_extras_truth_audit"]["overall"]
        lines.append(
            "E1b extras-truth audit (strict-fail, no-missing-info pool "
            f"= {audit['pool']}): true={audit['true_addition_only']} "
            f"false={audit['false_addition']} "
            f"misattributed={audit['misattribution']} "
            f"(none_flagged={audit['none_flagged']}, "
            f"multi_flagged={audit['multi_flagged']})"
        )
        counters = result["memory_error_counters"]
        lines.append(
            f"Memory-error counters: false_addition={counters['false_addition_count']} "
            f"(rate {counters['false_addition_rate']}) "
            f"misattribution={counters['misattribution_count']} "
            f"(rate {counters['misattribution_rate']})"
        )
    cost = result["cost"]["actual"]
    lines.append(
        f"Actual cost: ${cost['cost_usd']} over {cost['llm_calls']} calls "
        f"(input {cost['input_tokens']} tok, cached {cost['cached_input_tokens']}, "
        f"output {cost['output_tokens']})"
    )
    if result.get("stopped_early"):
        lines.append(f"STOPPED EARLY: {result.get('stop_reason')}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-judge retained LoCoMo predictions under a judge protocol."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Report JSON files or run directories to re-judge.",
    )
    parser.add_argument(
        "--judge-protocol",
        choices=[protocol.value for protocol in JudgeProtocol],
        default=JudgeProtocol.MEMORY_QUALITY.value,
        help="Judge protocol to apply.",
    )
    parser.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--data-path", default=str(_DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=None, help="Output directory (outside repo).")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Print projection and exit.")
    parser.add_argument("--input-price", type=float, default=_KIMI_INPUT_PRICE)
    parser.add_argument("--output-price", type=float, default=_KIMI_OUTPUT_PRICE)
    parser.add_argument("--cached-input-price", type=float, default=_KIMI_CACHED_INPUT_PRICE)
    return parser


def _config_from_args(args: argparse.Namespace) -> RejudgeConfig:
    return RejudgeConfig(
        report_specs=list(args.reports),
        protocol=JudgeProtocol(args.judge_protocol),
        judge_model=args.judge_model,
        data_path=Path(args.data_path).expanduser(),
        output_dir=Path(args.output).expanduser() if args.output else None,
        concurrency=args.concurrency,
        max_questions=args.max_questions,
        max_cost_usd=args.max_cost_usd,
        dry_run=args.dry_run,
        input_price=args.input_price,
        output_price=args.output_price,
        cached_input_price=args.cached_input_price,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    result = asyncio.run(run_rejudge(config))
    if config.dry_run:
        print(json.dumps(result, indent=2))
        return
    output_path = _output_path(config)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(_format_summary(result))
    print(f"\nRejudge report saved to: {output_path}")


if __name__ == "__main__":
    main()
