"""Compare plain-text card memory-extraction variants.

This is a shadow benchmark. It does not change production ingestion. The card
variants (``cards_parallel``, ``cards_bounded_2``, ``cards_serial``) ask simpler
line-based questions and assemble the lean ``LeanExtractionResult`` contract
locally, then score the result against synthetic extraction cases.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Literal
import unicodedata

from dotenv import load_dotenv

from atagia.core.config import Settings
from atagia.core.language_codes import normalize_optional_iso_639_1_code
from atagia.core.text_utils import truncate_inline
from atagia.memory.extraction_cards import (
    CandidateDraft,
    build_candidate_prompt,
    build_enrichment_prompt,
    _CARD_SYSTEM_PROMPTS,
)
from atagia.memory.extraction_mapping import lean_result_to_extraction_result
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    CoverageMember,
    ExtractionContextMessage,
    ExtractionConversationContext,
    LeanExtractionCandidate,
    LeanExtractionResult,
    LeanTemporalStatus,
    MemoryEvidenceSupportKind,
)
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
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
CardName = Literal[
    "candidate",
    "kind_scope",
    "evidence",
    "index",
    "temporal",
    "belief",
    "coverage_members",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
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
_CARD_ENRICHMENT_NAMES: tuple[CardName, ...] = (
    "kind_scope",
    "evidence",
    "index",
    "temporal",
    "belief",
    "coverage_members",
)
_CARD_PURPOSES: dict[CardName, str] = {
    "candidate": "memory_extraction_candidate_card",
    "kind_scope": "memory_extraction_kind_scope_card",
    "evidence": "memory_extraction_evidence_card",
    "index": "memory_extraction_index_card",
    "temporal": "memory_extraction_temporal_card",
    "belief": "memory_extraction_belief_card",
    "coverage_members": "memory_extraction_coverage_members_card",
}
_CARD_MAX_OUTPUT_TOKENS: dict[CardName, int] = {
    "candidate": 1024,
    "kind_scope": 512,
    "evidence": 1024,
    "index": 1024,
    "temporal": 512,
    "belief": 512,
    "coverage_members": 1024,
}
_COVERAGE_DISPLAY_TEXT_MAX_CHARS = 160
_CARD_TRANSIENT_RETRY_DELAYS_SECONDS: tuple[float, ...] = (3.0, 10.0, 30.0, 60.0)
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
}
_VALID_KINDS = {"evidence", "belief", "contract_signal", "state_update"}
_VALID_SCOPES = {"chat", "character", "user"}
_VALID_SUPPORT_KINDS = {item.value for item in MemoryEvidenceSupportKind}
_VALID_TEMPORAL_TYPES = {
    "permanent",
    "bounded",
    "event_triggered",
    "ephemeral",
    "unknown",
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


@dataclass(frozen=True, slots=True)
class CardResult:
    card_name: CardName
    raw_output: str | None
    parsed: Any
    parse_valid: bool
    malformed_count: int = 0
    error: dict[str, Any] | None = None
    retry_count: int = 0


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
            try:
                row = await asyncio.wait_for(
                    run_one_variant(
                        client=client,
                        case=case,
                        variant=variant,
                        card_model=card_model,
                        repetition=repetition + 1,
                        include_examples=include_examples,
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
                )
        print(
            f"{variant} {case.case_id} rep={repetition + 1} "
            f"technical_ok={row['technical_ok']} "
            f"recall={row['score']['expected_recall']:.2f} "
            f"parse_invalid={row['parse_invalid_count']} "
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
) -> dict[str, Any]:
    started = perf_counter()
    card_results: list[CardResult] = []
    error: dict[str, Any] | None = None
    assembly_repairs: list[str] = []
    try:
        result, card_results, assembly_repairs = await run_cards_variant(
            client=client,
            case=case,
            model=card_model,
            variant=variant,
            include_examples=include_examples,
        )
    except Exception as exc:  # noqa: BLE001
        result = LeanExtractionResult(nothing_durable=True)
        error = _error_payload(exc)
    wall_time_ms = (perf_counter() - started) * 1000.0
    output = normalize_result(result)
    score = score_output(output, case, error=error)
    parse_invalid_count = sum(1 for card in card_results if not card.parse_valid)
    malformed_count = sum(card.malformed_count for card in card_results)
    card_retry_count = sum(card.retry_count for card in card_results)
    technical_ok = error is None and parse_invalid_count == 0
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
        "parse_invalid_count": parse_invalid_count,
        "malformed_count": malformed_count,
        "card_retry_count": card_retry_count,
        "assembly_repairs": assembly_repairs,
        "card_results": [
            {
                "card_name": card.card_name,
                "parse_valid": card.parse_valid,
                "malformed_count": card.malformed_count,
                "error": card.error,
                "retry_count": card.retry_count,
                "raw_output": card.raw_output,
                "parsed": _json_safe(card.parsed),
            }
            for card in card_results
        ],
    }


def timeout_row(
    *,
    case: BenchmarkCase,
    variant: VariantName,
    card_model: str,
    repetition: int,
    wall_time_ms: float,
    timeout_seconds: float,
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
        "parse_invalid_count": 0,
        "malformed_count": 0,
        "card_retry_count": 0,
        "assembly_repairs": [],
        "card_results": [],
    }


async def run_cards_variant(
    *,
    client: LLMClient[Any],
    case: BenchmarkCase,
    model: str,
    variant: VariantName,
    include_examples: bool,
) -> tuple[LeanExtractionResult, list[CardResult], list[str]]:
    context = _context_for_case(case)
    policy = _resolved_policy(case.mode)
    allowed_write_scopes = tuple(MemoryExtractor._allowed_write_scopes(context))
    results: list[CardResult] = []
    candidate_card = await run_card(
        client=client,
        model=model,
        card_name="candidate",
        case=case,
        context=context,
        prompt=build_candidate_prompt(
            message_text=case.message,
            role=case.role,
            context=context,
            resolved_policy=policy,
            allowed_write_scopes=allowed_write_scopes,
            occurred_at=case.occurred_at,
            prior_chunk_context=None,
            max_candidate_count=None,
            include_examples=include_examples,
        ),
    )
    results.append(candidate_card)
    candidates = tuple(candidate_card.parsed or ())
    if candidate_card.error is not None or not candidates:
        return LeanExtractionResult(nothing_durable=True), results, []

    async def enrichment(card_name: CardName) -> CardResult:
        return await run_card(
            client=client,
            model=model,
            card_name=card_name,
            case=case,
            context=context,
            prompt=build_enrichment_prompt(
                card_name,
                message_text=case.message,
                role=case.role,
                context=context,
                resolved_policy=policy,
                allowed_write_scopes=allowed_write_scopes,
                occurred_at=case.occurred_at,
                prior_chunk_context=None,
                candidates=candidates,
                include_examples=include_examples,
            ),
        )

    if variant == "cards_serial":
        enrichment_results = []
        for card_name in _CARD_ENRICHMENT_NAMES:
            enrichment_results.append(await enrichment(card_name))
    else:
        limit = 2 if variant == "cards_bounded_2" else len(_CARD_ENRICHMENT_NAMES)
        semaphore = asyncio.Semaphore(limit)

        async def bounded(card_name: CardName) -> CardResult:
            async with semaphore:
                return await enrichment(card_name)

        enrichment_results = await asyncio.gather(
            *(bounded(card_name) for card_name in _CARD_ENRICHMENT_NAMES)
        )
    results.extend(enrichment_results)
    result, repairs = assemble_card_result(candidates, enrichment_results)
    return result, results, repairs


async def run_card(
    *,
    client: LLMClient[Any],
    model: str,
    card_name: CardName,
    case: BenchmarkCase,
    context: ExtractionConversationContext,
    prompt: str,
) -> CardResult:
    request = LLMCompletionRequest(
        model=model,
        messages=[
            LLMMessage(
                role="system",
                content=_CARD_SYSTEM_PROMPTS[card_name],
            ),
            LLMMessage(role="user", content=prompt),
        ],
        max_output_tokens=_CARD_MAX_OUTPUT_TOKENS[card_name],
        metadata={
            "user_id": context.user_id,
            "conversation_id": context.conversation_id,
            "assistant_mode_id": context.assistant_mode_id,
            "purpose": _CARD_PURPOSES[card_name],
            "memory_extraction_card": card_name,
            **prompt_authority_metadata(
                _authority_context(context, purpose=_CARD_PURPOSES[card_name]),
                prompt_authority_kind="process_metadata",
            ),
        },
    )
    retry_count = 0
    while True:
        try:
            response = await client.complete(request)
            break
        except Exception as exc:  # noqa: BLE001
            if (
                retry_count >= len(_CARD_TRANSIENT_RETRY_DELAYS_SECONDS)
                or not _is_retryable_card_error(exc)
            ):
                return CardResult(
                    card_name=card_name,
                    raw_output=None,
                    parsed=() if card_name == "candidate" else {},
                    parse_valid=False,
                    error=_error_payload(exc),
                    retry_count=retry_count,
                )
            delay = _CARD_TRANSIENT_RETRY_DELAYS_SECONDS[retry_count]
            retry_count += 1
            await asyncio.sleep(delay)
    parsed, malformed_count = parse_card_output(card_name, response.output_text)
    return CardResult(
        card_name=card_name,
        raw_output=response.output_text,
        parsed=parsed,
        parse_valid=malformed_count == 0,
        malformed_count=malformed_count,
        retry_count=retry_count,
    )


def parse_card_output(card_name: CardName, text: str) -> tuple[Any, int]:
    if card_name == "candidate":
        return parse_candidate_card_output(text)
    if card_name == "kind_scope":
        return parse_kind_scope_card_output(text)
    if card_name == "evidence":
        return parse_evidence_card_output(text)
    if card_name == "index":
        return parse_index_card_output(text)
    if card_name == "temporal":
        return parse_temporal_card_output(text)
    if card_name == "coverage_members":
        return parse_coverage_members_card_output(text)
    return parse_belief_card_output(text)


def parse_candidate_card_output(text: str) -> tuple[tuple[CandidateDraft, ...], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return (), 0
    candidates: list[CandidateDraft] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    malformed = 0
    for line in lines:
        if "|" in line:
            raw_id, raw_text = line.split("|", 1)
            candidate_id = _clean_candidate_id(raw_id) or f"cand_{len(candidates) + 1:03d}"
            canonical_text = _clean_text_value(raw_text)
        else:
            candidate_id = f"cand_{len(candidates) + 1:03d}"
            canonical_text = _clean_text_value(line)
            malformed += 1
        if not canonical_text:
            malformed += 1
            continue
        text_key = _norm(canonical_text)
        if candidate_id in seen_ids or text_key in seen_texts:
            continue
        seen_ids.add(candidate_id)
        seen_texts.add(text_key)
        candidates.append(
            CandidateDraft(
                candidate_id=candidate_id,
                canonical_text=canonical_text,
            )
        )
    return tuple(candidates), malformed


def parse_kind_scope_card_output(text: str) -> tuple[dict[str, dict[str, Any]], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, dict[str, Any]] = {}
    malformed = 0
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 3:
            malformed += 1
            continue
        candidate_id = _clean_candidate_id(tokens[0])
        kind = _clean_atom(tokens[1])
        scope = _clean_atom(tokens[2])
        confidence = _float_or_none(tokens[3] if len(tokens) >= 4 else None)
        if candidate_id is None or kind not in _VALID_KINDS or scope not in _VALID_SCOPES:
            malformed += 1
            continue
        parsed[candidate_id] = {
            "kind": kind,
            "subject_scope": scope,
            "confidence": _clamp_confidence(confidence, default=0.75),
        }
    return parsed, malformed


def parse_evidence_card_output(text: str) -> tuple[dict[str, dict[str, Any]], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, dict[str, Any]] = {}
    malformed = 0
    for line in lines:
        left, raw_span = _split_optional_pipe(line)
        tokens = _line_tokens(left)
        if len(tokens) < 4:
            malformed += 1
            continue
        candidate_id = _clean_candidate_id(tokens[0])
        support_kind = _clean_atom(tokens[1])
        preserve_verbatim = _bool_or_none(tokens[2])
        language_codes = _language_codes_from_token(",".join(tokens[3:]))
        if (
            candidate_id is None
            or support_kind not in _VALID_SUPPORT_KINDS
            or preserve_verbatim is None
        ):
            malformed += 1
            continue
        parsed[candidate_id] = {
            "support_kind": support_kind,
            "preserve_verbatim": preserve_verbatim,
            "language_codes": language_codes or ("en",),
            "source_span": _none_or_text(raw_span),
        }
    return parsed, malformed


def parse_index_card_output(text: str) -> tuple[dict[str, str | None], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, str | None] = {}
    malformed = 0
    for line in lines:
        if "|" not in line:
            malformed += 1
            continue
        raw_id, raw_value = line.split("|", 1)
        candidate_id = _clean_candidate_id(raw_id)
        if candidate_id is None:
            malformed += 1
            continue
        parsed[candidate_id] = _none_or_text(raw_value)
    return parsed, malformed


def parse_temporal_card_output(text: str) -> tuple[dict[str, dict[str, str | None]], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, dict[str, str | None]] = {}
    malformed = 0
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 2:
            malformed += 1
            continue
        candidate_id = _clean_candidate_id(tokens[0])
        temporal_type = _clean_atom(tokens[1])
        if candidate_id is None:
            malformed += 1
            continue
        if temporal_type == "none":
            parsed[candidate_id] = {
                "temporal_type": None,
                "valid_from_iso": None,
                "valid_to_iso": None,
            }
            continue
        if temporal_type not in _VALID_TEMPORAL_TYPES:
            malformed += 1
            continue
        valid_from = _none_or_text(tokens[2] if len(tokens) >= 3 else None)
        valid_to = _none_or_text(tokens[3] if len(tokens) >= 4 else None)
        parsed[candidate_id] = {
            "temporal_type": temporal_type,
            "valid_from_iso": valid_from,
            "valid_to_iso": valid_to,
        }
    return parsed, malformed


def parse_belief_card_output(text: str) -> tuple[dict[str, dict[str, str | None]], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, dict[str, str | None]] = {}
    malformed = 0
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 2:
            malformed += 1
            continue
        candidate_id = _clean_candidate_id(tokens[0])
        if candidate_id is None:
            malformed += 1
            continue
        claim_key = _none_or_text(tokens[1])
        if claim_key is None:
            parsed[candidate_id] = {"claim_key": None, "claim_value": None}
            continue
        claim_value = "_".join(tokens[2:]) if len(tokens) >= 3 else None
        parsed[candidate_id] = {
            "claim_key": _normalize_claim_key(claim_key),
            "claim_value": _none_or_text(claim_value) or "true",
        }
    return parsed, malformed


def parse_coverage_members_card_output(
    text: str,
) -> tuple[dict[str, list[CoverageMember]], int]:
    lines = _card_lines(text)
    if _lines_are_none(lines):
        return {}, 0
    parsed: dict[str, list[CoverageMember]] = {}
    malformed = 0
    for line in lines:
        if "|" not in line:
            malformed += 1
            continue
        raw_id, raw_members = line.split("|", 1)
        candidate_id = _clean_candidate_id(raw_id)
        if candidate_id is None:
            malformed += 1
            continue
        members, line_malformed = _coverage_members_from_json(raw_members)
        malformed += line_malformed
        parsed[candidate_id] = members
    return parsed, malformed


def _coverage_members_from_json(raw_members: str) -> tuple[list[CoverageMember], int]:
    stripped = raw_members.strip()
    if not stripped or _clean_atom(stripped) in {"none", "null", "[]"}:
        return [], 0
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return [], 1
    if not isinstance(decoded, list):
        return [], 1
    members: list[CoverageMember] = []
    seen_keys: set[str] = set()
    malformed = 0
    for entry in decoded:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        member_key = _clean_text_value(entry.get("member_key"))
        display_text = truncate_inline(
            str(entry.get("display_text") or ""),
            _COVERAGE_DISPLAY_TEXT_MAX_CHARS,
        )
        if not member_key or not display_text:
            malformed += 1
            continue
        if member_key in seen_keys:
            continue
        seen_keys.add(member_key)
        members.append(CoverageMember(member_key=member_key, display_text=display_text))
    return members, malformed


def assemble_card_result(
    candidates: tuple[CandidateDraft, ...],
    card_results: list[CardResult],
) -> tuple[LeanExtractionResult, list[str]]:
    by_card = {card.card_name: card.parsed for card in card_results if card.error is None}
    kind_scope = dict(by_card.get("kind_scope") or {})
    evidence = dict(by_card.get("evidence") or {})
    index = dict(by_card.get("index") or {})
    temporal = dict(by_card.get("temporal") or {})
    belief = dict(by_card.get("belief") or {})
    coverage_members = dict(by_card.get("coverage_members") or {})
    repairs: list[str] = []
    lean_candidates: list[LeanExtractionCandidate] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        kind_scope_row = kind_scope.get(candidate_id) or {}
        evidence_row = evidence.get(candidate_id) or {}
        temporal_row = temporal.get(candidate_id) or {}
        belief_row = belief.get(candidate_id) or {}
        kind = str(kind_scope_row.get("kind") or candidate.kind)
        claim_key = _none_or_text(belief_row.get("claim_key"))
        claim_value = _none_or_text(belief_row.get("claim_value"))
        if kind == "belief" and (claim_key is None or claim_value is None):
            repairs.append(f"{candidate_id}: belief_without_claim_fields_downgraded")
            kind = "evidence"
        language_codes = tuple(evidence_row.get("language_codes") or candidate.language_codes)
        if not language_codes:
            repairs.append(f"{candidate_id}: missing_language_defaulted_en")
            language_codes = ("en",)
        temporal_status = _temporal_status_from_row(
            temporal_row,
            repairs=repairs,
            candidate_id=candidate_id,
        )
        member_list = list(coverage_members.get(candidate_id) or ())
        try:
            lean_candidates.append(
                LeanExtractionCandidate(
                    canonical_text=candidate.canonical_text,
                    kind=kind,
                    subject_scope=str(kind_scope_row.get("subject_scope") or candidate.subject_scope),
                    confidence=_clamp_confidence(
                        _float_or_none(kind_scope_row.get("confidence")),
                        default=candidate.confidence,
                    ),
                    language_codes=list(language_codes),
                    index_text=index.get(candidate_id) or candidate.index_text,
                    preserve_verbatim=bool(
                        evidence_row.get("preserve_verbatim", candidate.preserve_verbatim)
                    ),
                    source_span=evidence_row.get("source_span") or candidate.source_span,
                    temporal_status=temporal_status,
                    support_kind=str(evidence_row.get("support_kind") or candidate.support_kind),
                    claim_key=claim_key,
                    claim_value=claim_value,
                    coverage_members=member_list,
                )
            )
        except Exception as exc:  # noqa: BLE001
            repairs.append(f"{candidate_id}: dropped_after_validation:{exc.__class__.__name__}")
    return LeanExtractionResult(
        nothing_durable=not lean_candidates,
        candidates=lean_candidates,
    ), repairs


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
        parse_invalid = sum(int(row.get("parse_invalid_count") or 0) for row in variant_rows)
        malformed = sum(int(row.get("malformed_count") or 0) for row in variant_rows)
        card_retries = sum(int(row.get("card_retry_count") or 0) for row in variant_rows)
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
            "parse_invalid_count": parse_invalid,
            "malformed_line_count": malformed,
            "card_retry_count": card_retries,
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


def _temporal_status_from_row(
    row: dict[str, str | None],
    *,
    repairs: list[str],
    candidate_id: str,
) -> LeanTemporalStatus | None:
    temporal_type = row.get("temporal_type") if row else None
    if temporal_type is None:
        return None
    try:
        return LeanTemporalStatus(
            type=temporal_type,
            valid_from_iso=row.get("valid_from_iso"),
            valid_to_iso=row.get("valid_to_iso"),
        )
    except Exception as exc:  # noqa: BLE001
        repairs.append(f"{candidate_id}: temporal_status_dropped:{exc.__class__.__name__}")
        return None


def _split_optional_pipe(line: str) -> tuple[str, str | None]:
    if "|" not in line:
        return line, None
    left, right = line.split("|", 1)
    return left, right


def _card_lines(text: str) -> list[str]:
    stripped = (
        text.strip()
        .replace("<TAB>", " ")
        .replace("<tab>", " ")
        .replace("\\t", " ")
        .replace("\t", " ")
    )
    if not stripped:
        return []
    lines: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        line = line.strip("`")
        if line.startswith("- "):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return lines


def _lines_are_none(lines: list[str]) -> bool:
    return not lines or all(_clean_atom(line) in {"none", "no", "nothing"} for line in lines)


def _line_tokens(line: str) -> list[str]:
    return [token.strip() for token in line.replace(",", " ").split() if token.strip()]


def _clean_atom(value: Any) -> str:
    return str(value or "").strip().strip("`*_.,;:[](){}\"'").casefold()


def _clean_candidate_id(value: Any) -> str | None:
    cleaned = _clean_atom(value)
    if not cleaned:
        return None
    if cleaned.startswith("candidate_"):
        cleaned = "cand_" + cleaned.removeprefix("candidate_")
    if cleaned.startswith("cand") and not cleaned.startswith("cand_"):
        suffix = cleaned.removeprefix("cand").strip("_-")
        cleaned = f"cand_{suffix}"
    if not cleaned.startswith("cand_"):
        return None
    return cleaned


def _clean_text_value(value: Any) -> str:
    return " ".join(str(value or "").strip().strip("`").split())


def _none_or_text(value: Any) -> str | None:
    cleaned = _clean_text_value(value)
    if not cleaned or _clean_atom(cleaned) in {"none", "null", "na", "n/a", "-"}:
        return None
    return cleaned


def _bool_or_none(value: Any) -> bool | None:
    cleaned = _clean_atom(value)
    if cleaned in {"true", "yes", "y", "1"}:
        return True
    if cleaned in {"false", "no", "n", "0"}:
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _clamp_confidence(value: float | None, *, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, value))


def _language_codes_from_token(value: Any) -> tuple[str, ...]:
    raw = str(value or "")
    pieces = [
        piece.strip()
        for piece in raw.replace("/", ",").replace("+", ",").replace(";", ",").split(",")
        if piece.strip()
    ]
    codes: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        normalized = normalize_optional_iso_639_1_code(piece)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        codes.append(normalized)
    return tuple(codes)


def _normalize_claim_key(value: str) -> str:
    cleaned = _clean_atom(value).replace("-", "_")
    parts = [part for part in cleaned.split(".") if part]
    return ".".join(parts) if parts else "memory.claim"


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_retryable_card_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "LLMRunGuardError":
        return False
    text = f"{exc.__class__.__name__} {exc}".casefold()
    retryable_markers = (
        "transient",
        "overload",
        "overloaded",
        "temporarily",
        "high load",
        "rate limit",
        "rate_limit",
        "timeout",
        "timed out",
        "529",
        "503",
    )
    return any(marker in text for marker in retryable_markers)


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "reason": getattr(exc, "reason", None),
        "details": list(getattr(exc, "details", ()) or ()),
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


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
