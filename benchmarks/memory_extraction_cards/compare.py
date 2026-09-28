"""Compare the production memory-extraction graph at concurrency 1, 2, or 8.

Prompts, parsing, dependencies, and retries belong to the real engine. This
shadow benchmark observes complete client calls and scores synthetic cases.
Provider retry attempts remain available through the diagnostic recorder.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Literal
import unicodedata

from dotenv import load_dotenv

from atagia.core.config import Settings
from atagia.memory.extraction_cards import extract_lean_with_cards
from atagia.memory.extraction_mapping import lean_result_to_extraction_result
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
    LeanExtractionResult,
    MemoryEvidenceSupportKind,
)
from atagia.services.llm_client import LLMClient, LLMCompletionRequest
from atagia.services.model_resolution import examples_enabled_for_component
from atagia.services.prompt_authority import (
    prompt_authority_metadata,
)
from atagia.services.providers import build_llm_client

from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_delay,
    install_llm_call_recorder,
    summarize_llm_calls,
)
from benchmarks.output_root import assert_outside_repo, resolve_output_dir

load_dotenv()

VariantName = Literal[
    "cards_parallel",
    "cards_bounded_2",
    "cards_serial",
]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PROJECT_ROOT / "src" / "atagia" / "resources" / "manifests"
_DEFAULT_CASES_PATH = _PROJECT_ROOT / "benchmarks" / "memory_extraction_cards" / "cases.jsonl"
_DIRECT_GEMINI_FLASH_LITE_MODEL = "google/gemini-3.1-flash-lite"
_DIRECT_MINIMAX_M3_MODEL = "minimax/MiniMax-M3"
_DEFAULT_VARIANTS: tuple[VariantName, ...] = (
    "cards_parallel",
    "cards_bounded_2",
)
_ALLOWED_VARIANTS: tuple[VariantName, ...] = (
    "cards_parallel",
    "cards_bounded_2",
    "cards_serial",
)
_VARIANT_CONCURRENCY: dict[VariantName, int] = {
    "cards_serial": 1,
    "cards_bounded_2": 2,
    "cards_parallel": 8,
}
_MODEL_PRICE_PER_MILLION = {
    "google/gemini-3.1-flash-lite": {
        "input_tokens": 0.25,
        "output_tokens": 1.50,
        "cached_input_tokens": 0.25,
        "source": "Google Gemini API public pricing, checked 2026-06-18",
    },
    "minimax/MiniMax-M3": {
        "input_tokens": 0.30,
        "output_tokens": 1.20,
        "cached_input_tokens": 0.06,
        "source": "MiniMax M3 standard pay-as-you-go <=512k pricing, checked 2026-06-18",
    },
    "openrouter/openai/gpt-5.6-luna": {
        "input_tokens": 0.10,
        "output_tokens": 0.60,
        "cached_input_tokens": 0.01,
        "source": (
            "OpenRouter GPT-5.6 Luna promo pricing, checked 2026-07-31 "
            "(OpenAI direct list after 2026-07-30 cut: 0.20/1.20, cached 0.02)"
        ),
    },
}
@dataclass(frozen=True, slots=True)
class ExpectedCandidate:
    label: str
    must_include: tuple[str, ...]
    kind: str | None = None
    kind_any: tuple[str, ...] = ()
    scope: str | None = None
    any_include: tuple[str, ...] = ()
    any_include_groups: tuple[tuple[str, ...], ...] = ()
    source_must_include: tuple[str, ...] = ()
    preserve_verbatim: bool | None = None
    support_kind: str | None = None
    language_codes: tuple[str, ...] = ()
    temporal_type: str | None = None
    temporal_type_any: tuple[str, ...] = ()
    valid_from_date: str | None = None
    claim_key: str | None = None
    allow_extra_candidates: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    message: str
    expected_candidates: tuple[ExpectedCandidate, ...]
    forbidden_must_include: tuple[str, ...] = ()
    forbidden_unless_include: tuple[str, ...] = ()
    role: str = "user"
    mode: str = "general_qa"
    occurred_at: str = "2026-06-17T12:00:00+00:00"
    recent_context: tuple[dict[str, Any], ...] = ()
    notes: str = ""


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(_DEFAULT_CASES_PATH))
    parser.add_argument(
        "--variants",
        default=",".join(_DEFAULT_VARIANTS),
        help="Comma-separated variants: cards_parallel,cards_bounded_2,cards_serial",
    )
    parser.add_argument("--card-model", default=_DIRECT_GEMINI_FLASH_LITE_MODEL)
    parser.add_argument(
        "--model",
        default=None,
        help="Convenience override that sets --card-model.",
    )
    parser.add_argument(
        "--examples",
        choices=("default", "on", "off"),
        default="default",
        help=(
            "Few-shot examples in card prompts. 'default' uses the resolved "
            "extractor setting (production behavior). 'on'/'off' set the global "
            "card_examples_enabled; a per-component override "
            "(llm_component_examples['extractor']), if configured, still takes "
            "precedence, exactly as in production."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case ids to run after loading --cases/--limit.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm-progress-every", type=int, default=0)
    parser.add_argument("--parallel-trials", type=int, default=1)
    parser.add_argument(
        "--llm-call-delay-ms",
        type=int,
        default=0,
        help=(
            "Serialize benchmark LLM calls and sleep this many milliseconds "
            "before each call. Useful for direct providers with tight rate limits."
        ),
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=float,
        default=60.0,
        help="Per case/variant timeout. Timed-out trials are recorded as technical failures.",
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(Path(args.cases), limit=args.limit)
    if str(args.case_ids).strip():
        cases = _filter_cases(cases, str(args.case_ids))
    variants = _parse_variants(args.variants)
    repetitions = max(1, int(args.repetitions))
    card_model = str(args.model or args.card_model)

    output_dir = (
        resolve_output_dir("memory_extraction_cards")
        if args.output_dir is None
        else assert_outside_repo(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = replace(
        Settings.from_env(),
        llm_forced_global_model=card_model,
        extraction_watchdog_enabled=False,
        llm_run_guard_enabled=False,
        llm_run_guard_mode="off",
    )
    if args.examples != "default":
        settings = replace(settings, card_examples_enabled=(args.examples == "on"))
    include_examples = examples_enabled_for_component(settings, "extractor")
    client = build_llm_client(settings)
    recorder = LLMCallRecorder(progress_interval=args.llm_progress_every)
    install_llm_call_recorder(client, recorder)
    llm_call_delay_ms = max(0, int(args.llm_call_delay_ms))
    if llm_call_delay_ms:
        install_llm_call_delay(client, delay_seconds=llm_call_delay_ms / 1000.0)

    started_at = datetime.now(timezone.utc)
    trial_specs = [
        (repetition, case, variant)
        for repetition in range(repetitions)
        for case in cases
        for variant in variants
    ]
    parallel_trials = max(1, int(args.parallel_trials))

    async def run_trial(
        repetition: int,
        case: BenchmarkCase,
        variant: VariantName,
    ) -> dict[str, Any]:
        with recorder.context(
            benchmark="memory_extraction_cards",
            case_id=case.case_id,
            variant=variant,
            repetition=repetition + 1,
        ):
            started = perf_counter()
            card_calls: list[dict[str, Any]] = []
            try:
                row = await asyncio.wait_for(
                    run_one_variant(
                        client=client,
                        case=case,
                        variant=variant,
                        card_model=card_model,
                        repetition=repetition + 1,
                        include_examples=include_examples,
                        call_sink=card_calls,
                    ),
                    timeout=max(1.0, float(args.trial_timeout_seconds)),
                )
            except TimeoutError:
                row = timeout_row(
                    case=case,
                    variant=variant,
                    card_model=card_model,
                    repetition=repetition + 1,
                    wall_time_ms=(perf_counter() - started) * 1000.0,
                    timeout_seconds=max(1.0, float(args.trial_timeout_seconds)),
                    card_calls=card_calls,
                )
        print(
            f"{variant} {case.case_id} rep={repetition + 1} "
            f"technical_ok={row['technical_ok']} "
            f"recall={row['score']['expected_recall']:.2f} "
            f"wall_ms={row['wall_time_ms']:.0f}",
            flush=True,
        )
        return row

    if parallel_trials == 1:
        rows = []
        for repetition, case, variant in trial_specs:
            rows.append(await run_trial(repetition, case, variant))
    else:
        semaphore = asyncio.Semaphore(parallel_trials)

        async def run_bounded_trial(
            repetition: int,
            case: BenchmarkCase,
            variant: VariantName,
        ) -> dict[str, Any]:
            async with semaphore:
                return await run_trial(repetition, case, variant)

        rows = await asyncio.gather(
            *(
                run_bounded_trial(repetition, case, variant)
                for repetition, case, variant in trial_specs
            )
        )

    finished_at = datetime.now(timezone.utc)
    summary = summarize_run(
        rows,
        recorder=recorder,
        variants=variants,
        started_at=started_at,
        finished_at=finished_at,
        card_model=card_model,
        cases_path=Path(args.cases),
    )
    summary_path = write_json_atomic(output_dir / "summary.json", summary)
    per_case_path = write_jsonl_atomic(output_dir / "per_case.jsonl", rows)
    calls_path = write_json_atomic(output_dir / "llm_calls.json", recorder.records())
    summary["artifacts"] = {
        "summary": str(summary_path),
        "per_case": str(per_case_path),
        "llm_calls": str(calls_path),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary["artifacts"], indent=2, sort_keys=True))
    return summary


async def run_one_variant(
    *,
    client: LLMClient[Any],
    case: BenchmarkCase,
    variant: VariantName,
    card_model: str,
    repetition: int,
    include_examples: bool,
    call_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    card_calls = [] if call_sink is None else call_sink
    error: dict[str, Any] | None = None
    assembly_repairs: list[str] = []
    try:
        result, card_calls, assembly_repairs = await run_cards_variant(
            client=client,
            case=case,
            model=card_model,
            variant=variant,
            include_examples=include_examples,
            call_sink=card_calls,
        )
    except Exception as exc:  # noqa: BLE001
        result = LeanExtractionResult(nothing_durable=True)
        error = _error_payload(exc)
    wall_time_ms = (perf_counter() - started) * 1000.0
    output = normalize_result(result)
    score = score_output(output, case, error=error)
    technical_ok = error is None
    passed = technical_ok and bool(score["exact_match"])
    return {
        "case_id": case.case_id,
        "variant": variant,
        "repetition": repetition,
        "model": card_model,
        "role": case.role,
        "mode": case.mode,
        "message": case.message,
        "notes": case.notes,
        "wall_time_ms": wall_time_ms,
        "technical_ok": technical_ok,
        "passed": passed,
        "output": output,
        "score": score,
        "error": error,
        "assembly_repairs": assembly_repairs,
        "card_calls": card_calls,
        "card_concurrency": _VARIANT_CONCURRENCY[variant],
    }


def timeout_row(
    *,
    case: BenchmarkCase,
    variant: VariantName,
    card_model: str,
    repetition: int,
    wall_time_ms: float,
    timeout_seconds: float,
    card_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    error = {
        "type": "TimeoutError",
        "message": f"trial exceeded {timeout_seconds:.1f}s",
        "reason": "trial_timeout",
        "details": [],
    }
    output = {
        "nothing_durable": True,
        "candidate_count": 0,
        "candidates": [],
    }
    return {
        "case_id": case.case_id,
        "variant": variant,
        "repetition": repetition,
        "model": card_model,
        "role": case.role,
        "mode": case.mode,
        "message": case.message,
        "notes": case.notes,
        "wall_time_ms": wall_time_ms,
        "technical_ok": False,
        "passed": False,
        "output": output,
        "score": score_output(output, case, error=error),
        "error": error,
        "assembly_repairs": [],
        "card_calls": [] if card_calls is None else card_calls,
    }


class _CapturedClient:
    """Observe production client calls without changing prompts or retry behavior."""

    def __init__(self, client: LLMClient[Any], calls: list[dict[str, Any]]) -> None:
        self._client = client
        self.calls = calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def complete_choice_questions(self, **kwargs: Any) -> dict[str, str]:
        return await LLMClient.complete_choice_questions(self, **kwargs)

    async def complete_score_questions(self, **kwargs: Any) -> dict[str, Any]:
        return await LLMClient.complete_score_questions(self, **kwargs)

    async def complete(self, request: LLMCompletionRequest) -> Any:
        call = {
            "purpose": request.metadata.get("purpose"),
            "request": request.model_dump(mode="json"),
            "response": None,
            "error": None,
        }
        self.calls.append(call)
        started = perf_counter()
        try:
            response = await self._client.complete(request)
            call["response"] = response.model_dump(mode="json")
            return response
        except BaseException as exc:
            call["error"] = _error_payload(exc)
            raise
        finally:
            call["wall_time_ms"] = (perf_counter() - started) * 1000.0


async def run_cards_variant(
    *,
    client: LLMClient[Any],
    case: BenchmarkCase,
    model: str,
    variant: VariantName,
    include_examples: bool,
    call_sink: list[dict[str, Any]] | None = None,
) -> tuple[LeanExtractionResult, list[dict[str, Any]], list[str]]:
    """Run the real extraction graph with only its concurrency varied."""
    context = _context_for_case(case)
    calls = [] if call_sink is None else call_sink
    result, repairs = await extract_lean_with_cards(
        llm_client=_CapturedClient(client, calls),
        model=model,
        evidence_model=model,
        temporal_type_model=model,
        date_model=model,
        classification_models={
            "memory_kind": model,
            "memory_scope": model,
            "memory_confidence": model,
        },
        message_text=case.message,
        role=case.role,
        context=context,
        resolved_policy=_resolved_policy(case.mode),
        allowed_write_scopes=tuple(MemoryExtractor._allowed_write_scopes(context)),
        occurred_at=case.occurred_at,
        prior_chunk_context=None,
        metadata=prompt_authority_metadata(
            _authority_context(context, purpose="memory_extraction"),
            prompt_authority_kind="process_metadata",
        ),
        card_concurrency=_VARIANT_CONCURRENCY[variant],
        include_examples=include_examples,
    )
    return result, calls, repairs


def normalize_result(result: LeanExtractionResult) -> dict[str, Any]:
    rich = lean_result_to_extraction_result(result)
    rows: list[dict[str, Any]] = []
    buckets = (
        ("evidence", rich.evidences),
        ("belief", rich.beliefs),
        ("contract_signal", rich.contract_signals),
        ("state_update", rich.state_updates),
    )
    for kind, items in buckets:
        for item in items:
            row = {
                "kind": kind,
                "canonical_text": item.canonical_text,
                "index_text": item.index_text,
                "subject_scope": item.scope.value,
                "confidence": item.confidence,
                "language_codes": list(item.language_codes),
                "preserve_verbatim": item.preserve_verbatim,
                "source_span": item.source_quote,
                "support_kind": (item.support_kind or MemoryEvidenceSupportKind.DIRECT).value,
                "temporal_type": item.temporal_type,
                "valid_from_iso": item.valid_from_iso,
                "valid_to_iso": item.valid_to_iso,
            }
            if kind == "belief":
                row["claim_key"] = getattr(item, "claim_key", None)
                row["claim_value"] = getattr(item, "claim_value", None)
            rows.append(row)
    return {
        "nothing_durable": result.nothing_durable,
        "candidate_count": len(rows),
        "candidates": rows,
    }


def score_output(
    output: dict[str, Any],
    case: BenchmarkCase,
    *,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = list(output.get("candidates") or [])
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    missing: list[str] = []
    missing_details: list[dict[str, Any]] = []
    for expected in case.expected_candidates:
        matched_index, candidate_checks = _find_expected_match(rows, expected, used)
        if matched_index is None:
            split_indices, split_checks = _find_expected_split_match(
                rows,
                expected,
                used,
                candidate_checks,
            )
            if split_indices is None:
                missing.append(expected.label)
                missing_details.append(
                    {
                        "label": expected.label,
                        "candidate_checks": split_checks,
                    }
                )
                continue
            used.update(split_indices)
            matches.append(
                {
                    "label": expected.label,
                    "candidate_indices": list(split_indices),
                    "canonical_text": " | ".join(
                        str(rows[index].get("canonical_text") or "")
                        for index in split_indices
                    ),
                }
            )
            continue
        used.add(matched_index)
        matches.append(
            {
                "label": expected.label,
                "candidate_index": matched_index,
                "canonical_text": rows[matched_index].get("canonical_text"),
            }
        )
    unmatched_candidates = [
        {
            "candidate_index": index,
            "kind": row.get("kind"),
            "subject_scope": row.get("subject_scope"),
            "canonical_text": row.get("canonical_text"),
        }
        for index, row in enumerate(rows)
        if index not in used
    ]
    forbidden_hits: list[str] = []
    forbidden_unless = tuple(item.casefold() for item in case.forbidden_unless_include)
    for forbidden in case.forbidden_must_include:
        forbidden_norm = forbidden.casefold()
        for row in rows:
            search_text = _candidate_search_text(row)
            if forbidden_norm not in search_text:
                continue
            if forbidden_unless and any(allowed in search_text for allowed in forbidden_unless):
                continue
            forbidden_hits.append(forbidden)
            break
    expected_count = len(case.expected_candidates)
    recall = 1.0 if expected_count == 0 else (len(matches) / expected_count)
    exact_match = (
        error is None
        and not missing
        and not forbidden_hits
        and (expected_count > 0 or not rows)
    )
    return {
        "exact_match": exact_match,
        "expected_recall": recall,
        "matched_labels": [match["label"] for match in matches],
        "matched_candidates": matches,
        "missing_labels": missing,
        "missing_details": missing_details,
        "forbidden_hits": forbidden_hits,
        "extra_candidate_count": max(0, len(rows) - len(used)),
        "unmatched_candidates": unmatched_candidates,
    }


def summarize_run(
    rows: list[dict[str, Any]],
    *,
    recorder: LLMCallRecorder,
    variants: tuple[VariantName, ...],
    started_at: datetime,
    finished_at: datetime,
    card_model: str,
    cases_path: Path,
) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        latencies = [float(row["wall_time_ms"]) for row in variant_rows]
        exact = sum(1 for row in variant_rows if row["score"]["exact_match"])
        passed = sum(1 for row in variant_rows if row.get("passed"))
        technical_ok = sum(1 for row in variant_rows if row["technical_ok"])
        failed = sum(1 for row in variant_rows if row.get("error"))
        recall_values = [float(row["score"].get("expected_recall") or 0.0) for row in variant_rows]
        llm_records = recorder.records_for_context(variant=variant)
        llm_summary = summarize_llm_calls(llm_records)
        by_variant[variant] = {
            "cases": len(variant_rows),
            "technical_ok_count": technical_ok,
            "technical_ok_rate": _safe_div(technical_ok, len(variant_rows)),
            "pass_count": passed,
            "pass_rate": _safe_div(passed, len(variant_rows)),
            "exact_match_count": exact,
            "exact_match_rate": _safe_div(exact, len(variant_rows)),
            "mean_expected_recall": (
                sum(recall_values) / len(recall_values) if recall_values else 1.0
            ),
            "failed_trials": failed,
            "wall_time_ms": _latency_summary(latencies),
            "estimated_cost_usd": _estimate_cost_usd(
                card_model,
                llm_summary.get("token_totals") or {},
            ),
            "llm_call_summary": llm_summary,
            "mismatch_case_ids": [
                row["case_id"]
                for row in variant_rows
                if not row["score"]["exact_match"]
            ],
            "technical_failure_case_ids": [
                row["case_id"]
                for row in variant_rows
                if not row["technical_ok"]
            ],
            "failed_case_ids": [
                row["case_id"]
                for row in variant_rows
                if not row.get("passed")
            ],
        }
    llm_call_summary = recorder.summary()
    return {
        "benchmark": "memory_extraction_cards",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "cases_path": str(cases_path),
        "card_model": card_model,
        "variants": by_variant,
        "pairwise": pairwise_disagreements(rows),
        "pricing_assumptions": _pricing_assumptions(card_model),
        "estimated_cost_usd": sum(
            float(row.get("estimated_cost_usd") or 0.0)
            for row in by_variant.values()
        ),
        "llm_call_summary": llm_call_summary,
    }


def pairwise_disagreements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case_id"]), int(row["repetition"]))
        by_case.setdefault(key, {})[str(row["variant"])] = row
    pairs: list[dict[str, Any]] = []
    for (case_id, repetition), variants in sorted(by_case.items()):
        baseline = variants.get("cards_parallel")
        if baseline is None:
            continue
        for variant, row in sorted(variants.items()):
            if variant == "cards_parallel":
                continue
            if row["score"] != baseline["score"]:
                pairs.append(
                    {
                        "case_id": case_id,
                        "repetition": repetition,
                        "variant": variant,
                        "baseline_score": baseline["score"],
                        "variant_score": row["score"],
                        "baseline_technical_ok": baseline["technical_ok"],
                        "variant_technical_ok": row["technical_ok"],
                    }
                )
    return pairs


def load_cases(path: Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    if limit is not None and limit <= 0:
        return []
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw = json.loads(stripped)
            cases.append(_case_from_raw(raw))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _filter_cases(cases: list[BenchmarkCase], raw_case_ids: str) -> list[BenchmarkCase]:
    requested = [item.strip() for item in raw_case_ids.split(",") if item.strip()]
    if not requested:
        return cases
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {', '.join(missing)}")
    return [by_id[case_id] for case_id in requested]


def _case_from_raw(raw: dict[str, Any]) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=str(raw["case_id"]),
        message=str(raw["message"]),
        role=str(raw.get("role") or "user"),
        mode=str(raw.get("mode") or "general_qa"),
        occurred_at=str(raw.get("occurred_at") or "2026-06-17T12:00:00+00:00"),
        recent_context=tuple(
            dict(message)
            for message in raw.get("recent_context", [])
            if isinstance(message, dict)
        ),
        expected_candidates=tuple(
            _expected_candidate_from_raw(item)
            for item in raw.get("expected_candidates", [])
            if isinstance(item, dict)
        ),
        forbidden_must_include=tuple(
            str(item).casefold() for item in raw.get("forbidden_must_include", [])
        ),
        forbidden_unless_include=tuple(
            str(item).casefold() for item in raw.get("forbidden_unless_include", [])
        ),
        notes=str(raw.get("notes") or ""),
    )


def _expected_candidate_from_raw(raw: dict[str, Any]) -> ExpectedCandidate:
    return ExpectedCandidate(
        label=str(raw["label"]),
        kind=str(raw["kind"]) if raw.get("kind") is not None else None,
        kind_any=tuple(str(item) for item in raw.get("kind_any", [])),
        scope=str(raw["scope"]) if raw.get("scope") is not None else None,
        must_include=tuple(str(item).casefold() for item in raw.get("must_include", [])),
        any_include=tuple(str(item).casefold() for item in raw.get("any_include", [])),
        any_include_groups=tuple(
            tuple(str(piece).casefold() for piece in group)
            for group in raw.get("any_include_groups", [])
            if isinstance(group, list)
        ),
        source_must_include=tuple(
            str(item).casefold() for item in raw.get("source_must_include", [])
        ),
        preserve_verbatim=(
            bool(raw["preserve_verbatim"])
            if raw.get("preserve_verbatim") is not None
            else None
        ),
        support_kind=str(raw["support_kind"]) if raw.get("support_kind") is not None else None,
        language_codes=tuple(str(item).lower() for item in raw.get("language_codes", [])),
        temporal_type=str(raw["temporal_type"]) if raw.get("temporal_type") is not None else None,
        temporal_type_any=tuple(str(item) for item in raw.get("temporal_type_any", [])),
        valid_from_date=str(raw["valid_from_date"]) if raw.get("valid_from_date") is not None else None,
        claim_key=str(raw["claim_key"]) if raw.get("claim_key") is not None else None,
        allow_extra_candidates=bool(raw.get("allow_extra_candidates", True)),
    )


def _context_for_case(case: BenchmarkCase) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="bench_user",
        conversation_id=f"cnv_{case.case_id}",
        source_message_id=f"msg_{case.case_id}",
        assistant_mode_id=case.mode,
        mode=case.mode,
        privacy_enforcement="off",
        recent_messages=[
            ExtractionContextMessage(
                id=str(message.get("id") or f"ctx_{index + 1}"),
                role=str(message.get("role") or "user"),
                content=str(message.get("content") or ""),
                seq=int(message.get("seq", index + 1)),
                occurred_at=message.get("occurred_at"),
            )
            for index, message in enumerate(case.recent_context)
        ],
    )


def _resolved_policy(mode: str) -> Any:
    manifests = ManifestLoader(_MANIFESTS_DIR).load_all()
    manifest = manifests[mode]
    return PolicyResolver().resolve(manifest, None, None)


def _authority_context(
    context: ExtractionConversationContext,
    *,
    purpose: str,
) -> Any:
    from atagia.services.prompt_authority import process_authority_context

    return process_authority_context(
        privacy_enforcement=context.privacy_enforcement,
        user_id=context.user_id,
        privilege_level=context.authenticated_user_privilege_level,
        is_atagia_master=context.authenticated_user_is_atagia_master,
        purpose=purpose,
    )


def _find_expected_match(
    rows: list[dict[str, Any]],
    expected: ExpectedCandidate,
    used: set[int],
) -> tuple[int | None, list[dict[str, Any]]]:
    candidate_checks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in used:
            continue
        reasons = _row_mismatch_reasons(row, expected)
        candidate_checks.append(
            {
                "candidate_index": index,
                "canonical_text": row.get("canonical_text"),
                "kind": row.get("kind"),
                "subject_scope": row.get("subject_scope"),
                "reasons": reasons,
            }
        )
        if reasons:
            continue
        return index, candidate_checks
    return None, candidate_checks


def _find_expected_split_match(
    rows: list[dict[str, Any]],
    expected: ExpectedCandidate,
    used: set[int],
    candidate_checks: list[dict[str, Any]],
) -> tuple[tuple[int, ...] | None, list[dict[str, Any]]]:
    eligible_indices: list[int] = []
    for index, row in enumerate(rows):
        if index in used:
            continue
        reasons = _row_mismatch_reasons(row, expected)
        if reasons and all(_is_split_content_reason(reason) for reason in reasons):
            eligible_indices.append(index)
    if len(eligible_indices) < 2:
        return None, candidate_checks

    combined = dict(rows[eligible_indices[0]])
    combined["canonical_text"] = " ".join(
        str(rows[index].get("canonical_text") or "")
        for index in eligible_indices
    )
    combined["source_span"] = " ".join(
        str(rows[index].get("source_span") or "")
        for index in eligible_indices
    )
    languages: list[str] = []
    seen_languages: set[str] = set()
    for index in eligible_indices:
        for language in rows[index].get("language_codes") or []:
            normalized = str(language).lower()
            if normalized in seen_languages:
                continue
            seen_languages.add(normalized)
            languages.append(normalized)
    combined["language_codes"] = languages

    split_reasons = _row_mismatch_reasons(combined, expected)
    split_check = {
        "split_candidate_indices": list(eligible_indices),
        "canonical_text": combined.get("canonical_text"),
        "kind": combined.get("kind"),
        "subject_scope": combined.get("subject_scope"),
        "reasons": split_reasons,
    }
    checks = [*candidate_checks, split_check]
    if split_reasons:
        return None, checks
    return tuple(eligible_indices), checks


def _is_split_content_reason(reason: str) -> bool:
    return reason.startswith("canonical_missing") or reason.startswith("source_missing")


def _row_matches_expected(row: dict[str, Any], expected: ExpectedCandidate) -> bool:
    return not _row_mismatch_reasons(row, expected)


def _row_mismatch_reasons(row: dict[str, Any], expected: ExpectedCandidate) -> list[str]:
    reasons: list[str] = []
    canonical = _match_text(row.get("canonical_text") or "")
    source = _match_text(row.get("source_span") or "")
    if expected.kind_any and row.get("kind") not in set(expected.kind_any):
        reasons.append(f"kind:{row.get('kind')} not in {list(expected.kind_any)}")
    elif expected.kind is not None and row.get("kind") != expected.kind:
        reasons.append(f"kind:{row.get('kind')}!={expected.kind}")
    if expected.scope is not None and row.get("subject_scope") != expected.scope:
        reasons.append(f"scope:{row.get('subject_scope')}!={expected.scope}")
    if expected.preserve_verbatim is not None and bool(row.get("preserve_verbatim")) is not expected.preserve_verbatim:
        reasons.append(
            f"preserve_verbatim:{row.get('preserve_verbatim')}!={expected.preserve_verbatim}"
        )
    if expected.support_kind is not None and row.get("support_kind") != expected.support_kind:
        reasons.append(f"support_kind:{row.get('support_kind')}!={expected.support_kind}")
    if expected.temporal_type_any and row.get("temporal_type") not in set(expected.temporal_type_any):
        reasons.append(
            f"temporal_type:{row.get('temporal_type')} not in {list(expected.temporal_type_any)}"
        )
    elif expected.temporal_type is not None and row.get("temporal_type") != expected.temporal_type:
        reasons.append(f"temporal_type:{row.get('temporal_type')}!={expected.temporal_type}")
    if expected.claim_key is not None and row.get("claim_key") != expected.claim_key:
        reasons.append(f"claim_key:{row.get('claim_key')}!={expected.claim_key}")
    if expected.valid_from_date is not None:
        valid_from = str(row.get("valid_from_iso") or "")
        if not valid_from.startswith(expected.valid_from_date):
            reasons.append(f"valid_from:{valid_from}!~{expected.valid_from_date}")
    languages = {str(item).lower() for item in row.get("language_codes") or []}
    if expected.language_codes and not set(expected.language_codes).issubset(languages):
        reasons.append(f"language_codes:{sorted(languages)} missing {list(expected.language_codes)}")
    must_include = tuple(_match_text(needle) for needle in expected.must_include)
    if any(needle not in canonical for needle in must_include):
        missing = [
            raw
            for raw, normalized in zip(expected.must_include, must_include, strict=True)
            if normalized not in canonical
        ]
        reasons.append(f"canonical_missing:{missing}")
    any_include = tuple(_match_text(needle) for needle in expected.any_include)
    if any_include and not any(needle in canonical for needle in any_include):
        reasons.append(f"canonical_missing_any:{list(expected.any_include)}")
    for group in expected.any_include_groups:
        normalized_group = tuple(_match_text(needle) for needle in group)
        if normalized_group and not any(needle in canonical for needle in normalized_group):
            reasons.append(f"canonical_missing_any_group:{list(group)}")
    source_must_include = tuple(_match_text(needle) for needle in expected.source_must_include)
    if any(needle not in source for needle in source_must_include):
        missing_source = [
            raw
            for raw, needle in zip(
                expected.source_must_include,
                source_must_include,
                strict=True,
            )
            if needle not in source
        ]
        reasons.append(f"source_missing:{missing_source}")
    return reasons


def _candidate_search_text(row: dict[str, Any]) -> str:
    return _match_text(
        " ".join(
            str(row.get(key) or "")
            for key in ("canonical_text", "index_text", "claim_key", "claim_value")
        )
    )


def _match_text(value: Any) -> str:
    text = " ".join(str(value or "").casefold().split())
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


















def _norm(value: str) -> str:
    return " ".join(value.casefold().split())




def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "reason": getattr(exc, "reason", None),
        "details": list(getattr(exc, "details", ()) or ()),
    }




def _parse_variants(value: str) -> tuple[VariantName, ...]:
    raw_values = tuple(item.strip() for item in value.split(",") if item.strip())
    variants: list[VariantName] = []
    for raw in raw_values:
        if raw not in _ALLOWED_VARIANTS:
            raise ValueError(f"Unknown variant {raw!r}; expected one of {_ALLOWED_VARIANTS}")
        variants.append(raw)  # type: ignore[arg-type]
    return tuple(variants)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _estimate_cost_usd(model: str, token_totals: dict[str, Any]) -> float | None:
    pricing = _MODEL_PRICE_PER_MILLION.get(model)
    if pricing is None:
        return None
    input_tokens = float(token_totals.get("input_tokens") or 0.0)
    cached_input_tokens = float(token_totals.get("cached_input_tokens") or 0.0)
    output_tokens = float(token_totals.get("output_tokens") or 0.0)
    uncached_input_tokens = max(0.0, input_tokens - cached_input_tokens)
    input_cost = uncached_input_tokens * float(pricing["input_tokens"]) / 1_000_000.0
    cache_cost = cached_input_tokens * float(pricing["cached_input_tokens"]) / 1_000_000.0
    output_cost = output_tokens * float(pricing["output_tokens"]) / 1_000_000.0
    return input_cost + cache_cost + output_cost


def _pricing_assumptions(*models: str) -> dict[str, dict[str, Any]]:
    assumptions: dict[str, dict[str, Any]] = {}
    for model in models:
        pricing = _MODEL_PRICE_PER_MILLION.get(model)
        if pricing is None:
            continue
        assumptions[model] = {
            "input_usd_per_million_tokens": pricing["input_tokens"],
            "output_usd_per_million_tokens": pricing["output_tokens"],
            "cached_input_usd_per_million_tokens": pricing["cached_input_tokens"],
            "source": pricing["source"],
            "note": "Benchmark estimate from observed token counters; provider invoices remain authoritative.",
        }
    return assumptions


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            tmp.write("\n")
    tmp_path.replace(path)
    return path
