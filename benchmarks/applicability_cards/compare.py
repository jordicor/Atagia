"""Benchmark applicability plain-text card scoring."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExactFacet,
    ExtractionContextMessage,
    ExtractionConversationContext,
    NeedTrigger,
    RetrievalPlan,
    RetrievalTrace,
    ScoredCandidate,
)
from atagia.services.llm_client import LLMClient
from atagia.services.providers import build_llm_client

from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    summarize_llm_calls,
)
from benchmarks.output_root import assert_outside_repo, resolve_output_dir

load_dotenv()

VariantName = Literal[
    "cards_single",
    "cards_batch_4",
    "cards_batch_8",
    "cards_batch_4_no_date",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES_PATH = _PROJECT_ROOT / "benchmarks" / "applicability_cards" / "cases.jsonl"
_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DIRECT_GEMINI_FLASH_LITE_MODEL = "google/gemini-3.1-flash-lite"
_MINIMAX_M3_MODEL = "minimax/MiniMax-M3"
_DEFAULT_VARIANTS: tuple[VariantName, ...] = ("cards_batch_4",)
_ALLOWED_VARIANTS: tuple[VariantName, ...] = (
    "cards_single",
    "cards_batch_4",
    "cards_batch_8",
    "cards_batch_4_no_date",
)
_FIXED_CLOCK = datetime(2026, 6, 17, tzinfo=timezone.utc)
_MODEL_PRICE_PER_MILLION = {
    "google/gemini-3.1-flash-lite": {
        "input_tokens": 0.25,
        "output_tokens": 1.50,
        "cached_input_tokens": 0.25,
        "source": "Google Gemini 3.1 Flash-Lite public pricing",
    },
    "minimax/MiniMax-M3": {
        "input_tokens": 0.30,
        "output_tokens": 1.20,
        "cached_input_tokens": 0.06,
        "source": "MiniMax M3 standard pay-as-you-go <=512k input pricing",
    },
}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    query: str
    candidates: list[dict[str, Any]]
    expected_top_ids: tuple[str, ...]
    expected_useful_ids: tuple[str, ...]
    expected_drop_ids: tuple[str, ...] = ()
    expected_resolved_dates: dict[str, str | None] | None = None
    mode: str = "general_qa"
    query_type: str = "default"
    exact_recall_mode: bool = False
    exact_facets: tuple[str, ...] = ()
    detected_needs: tuple[str, ...] = ()
    recent_context: tuple[dict[str, str], ...] = ()
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
        help=(
            "Comma-separated variants: cards_single,cards_batch_4,"
            "cards_batch_8,cards_batch_4_no_date"
        ),
    )
    parser.add_argument("--card-model", default=_DIRECT_GEMINI_FLASH_LITE_MODEL)
    parser.add_argument(
        "--model",
        default=None,
        help="Convenience override for --card-model.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expand-to",
        type=int,
        default=None,
        help=(
            "Deterministically expand the fixture set to this many synthetic "
            "cases before applying --limit."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm-progress-every", type=int, default=0)
    parser.add_argument("--parallel-trials", type=int, default=1)
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(Path(args.cases))
    if args.expand_to is not None:
        cases = expand_cases(cases, args.expand_to)
    if args.limit is not None:
        cases = cases[: args.limit]
    variants = _parse_variants(args.variants)
    repetitions = max(1, int(args.repetitions))
    card_model = str(args.model or args.card_model)

    output_dir = (
        resolve_output_dir("applicability_cards")
        if args.output_dir is None
        else assert_outside_repo(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = replace(
        Settings.from_env(),
        llm_forced_global_model=card_model,
    )
    client = build_llm_client(settings)
    recorder = LLMCallRecorder(progress_interval=args.llm_progress_every)
    install_llm_call_recorder(client, recorder)

    started_at = datetime.now(timezone.utc)
    trial_specs = [
        (repetition, case, variant)
        for repetition in range(repetitions)
        for case in cases
        for variant in variants
    ]
    rows: list[dict[str, Any]]
    parallel_trials = max(1, int(args.parallel_trials))

    async def run_trial(
        repetition: int,
        case: BenchmarkCase,
        variant: VariantName,
    ) -> dict[str, Any]:
        with recorder.context(
            benchmark="applicability_cards",
            case_id=case.case_id,
            variant=variant,
            repetition=repetition + 1,
        ):
            row = await run_one_variant(
                client=client,
                base_settings=settings,
                case=case,
                variant=variant,
                card_model=card_model,
                repetition=repetition + 1,
            )
        print(
            f"{variant} {case.case_id} rep={repetition + 1} "
            f"top_hit={row['score']['top_hit']} "
            f"recall={row['score']['expected_useful_recall']:.2f} "
            f"parse_invalid={row['parse_invalid_count']} "
            f"wall_ms={row['wall_time_ms']:.0f}"
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
    base_settings: Settings,
    case: BenchmarkCase,
    variant: VariantName,
    card_model: str,
    repetition: int,
) -> dict[str, Any]:
    model = card_model
    settings = replace(base_settings, llm_forced_global_model=model)
    scorer = ApplicabilityScorer(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        settings=settings,
    )
    context = _context_for_case(case)
    resolved_policy = _resolved_policy(case.mode)
    retrieval_plan = _plan_for_case(case, resolved_policy.privacy_ceiling)
    detected_needs = _detected_needs(case.detected_needs)
    trace = RetrievalTrace(
        query_text=case.query,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        requested_mode=case.mode,
        effective_mode=case.mode,
        timestamp_iso=_FIXED_CLOCK.isoformat(),
        privacy_enforcement="off",
    )
    started = perf_counter()
    try:
        scored = await scorer.score_shortlist(
            case.candidates,
            message_text=case.query,
            conversation_context=context,
            resolved_policy=resolved_policy,
            detected_needs=detected_needs,
            retrieval_plan=retrieval_plan,
            trace=trace,
            card_batch_size=_card_batch_size_for_variant(variant),
            date_card_enabled=(variant != "cards_batch_4_no_date"),
        )
        error = None
    except Exception as exc:  # noqa: BLE001
        scored = []
        error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "reason": getattr(exc, "reason", None),
            "details": list(getattr(exc, "details", ()) or ()),
        }
    wall_time_ms = (perf_counter() - started) * 1000.0
    output = normalize_scored(scored)
    diagnostics = [
        diagnostic.model_dump(mode="json")
        for diagnostic in trace.structured_output_diagnostics
    ]
    score = score_output(output, case)
    return {
        "case_id": case.case_id,
        "query": case.query,
        "notes": case.notes,
        "variant": variant,
        "repetition": repetition,
        "model": model,
        "wall_time_ms": wall_time_ms,
        "expected": {
            "top_ids": list(case.expected_top_ids),
            "useful_ids": list(case.expected_useful_ids),
            "drop_ids": list(case.expected_drop_ids),
            "resolved_dates": case.expected_resolved_dates or {},
        },
        "output": output,
        "score": score,
        "error": error,
        "parse_invalid_count": _parse_invalid_count(diagnostics, error=error),
        "diagnostics": diagnostics,
    }


def load_cases(path: Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw = json.loads(stripped)
            case = BenchmarkCase(
                case_id=str(raw["case_id"]),
                query=str(raw["query"]),
                mode=str(raw.get("mode") or "general_qa"),
                query_type=str(raw.get("query_type") or "default"),
                exact_recall_mode=bool(raw.get("exact_recall_mode", False)),
                exact_facets=tuple(str(item) for item in raw.get("exact_facets", [])),
                detected_needs=tuple(str(item) for item in raw.get("detected_needs", [])),
                recent_context=tuple(
                    {
                        "role": str(message.get("role") or "user"),
                        "content": str(message.get("content") or ""),
                    }
                    for message in raw.get("recent_context", [])
                    if isinstance(message, dict)
                ),
                candidates=[
                    _normalize_candidate(candidate, index)
                    for index, candidate in enumerate(raw["candidates"])
                ],
                expected_top_ids=tuple(str(item) for item in raw["expected_top_ids"]),
                expected_useful_ids=tuple(
                    str(item)
                    for item in raw.get("expected_useful_ids", raw["expected_top_ids"])
                ),
                expected_drop_ids=tuple(str(item) for item in raw.get("expected_drop_ids", [])),
                expected_resolved_dates={
                    str(key): (None if value is None else str(value))
                    for key, value in dict(raw.get("expected_resolved_dates", {})).items()
                },
                notes=str(raw.get("notes") or ""),
            )
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    return cases


def expand_cases(cases: list[BenchmarkCase], target_count: int) -> list[BenchmarkCase]:
    """Return base cases plus deterministic synthetic variants up to target_count."""

    if target_count <= len(cases):
        return cases[:target_count]
    expanded = list(cases)
    for raw_case in _iter_synthetic_raw_cases():
        if len(expanded) >= target_count:
            break
        expanded.append(_case_from_raw(raw_case))
    return expanded


def _case_from_raw(raw: dict[str, Any]) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=str(raw["case_id"]),
        query=str(raw["query"]),
        mode=str(raw.get("mode") or "general_qa"),
        query_type=str(raw.get("query_type") or "default"),
        exact_recall_mode=bool(raw.get("exact_recall_mode", False)),
        exact_facets=tuple(str(item) for item in raw.get("exact_facets", [])),
        detected_needs=tuple(str(item) for item in raw.get("detected_needs", [])),
        recent_context=tuple(
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
            for message in raw.get("recent_context", [])
            if isinstance(message, dict)
        ),
        candidates=[
            _normalize_candidate(candidate, index)
            for index, candidate in enumerate(raw["candidates"])
        ],
        expected_top_ids=tuple(str(item) for item in raw["expected_top_ids"]),
        expected_useful_ids=tuple(
            str(item)
            for item in raw.get("expected_useful_ids", raw["expected_top_ids"])
        ),
        expected_drop_ids=tuple(str(item) for item in raw.get("expected_drop_ids", [])),
        expected_resolved_dates={
            str(key): (None if value is None else str(value))
            for key, value in dict(raw.get("expected_resolved_dates", {})).items()
        },
        notes=str(raw.get("notes") or ""),
    )


def _iter_synthetic_raw_cases():
    people = [
        "Mara",
        "Leo",
        "Nadia",
        "Owen",
        "Iris",
        "Ravi",
        "Elena",
        "Sam",
        "Clara",
        "Noah",
    ]
    cities = [
        ("Valencia", "Granada"),
        ("Porto", "Lisbon"),
        ("Berlin", "Munich"),
        ("Dublin", "Cork"),
        ("Kyoto", "Osaka"),
        ("Austin", "Denver"),
        ("Lyon", "Marseille"),
        ("Sevilla", "Madrid"),
        ("Prague", "Vienna"),
        ("Chicago", "Boston"),
    ]
    offices = [
        ("North Campus", "West Annex"),
        ("Berlin hub", "Madrid hub"),
        ("Room 4B", "Room 2A"),
        ("Harbor desk", "Station desk"),
        ("Green floor", "Blue floor"),
    ]
    supplements = [
        ("magnesium glycinate", "200 mg"),
        ("vitamin D", "1000 IU"),
        ("omega-3", "1200 mg"),
        ("melatonin", "1 mg"),
        ("zinc", "15 mg"),
    ]
    appointments = [
        ("dentist", "Dr. Ramos"),
        ("physio", "Dr. Chen"),
        ("passport renewal", "the civic office"),
        ("tax review", "Marta"),
        ("portfolio review", "Dana"),
    ]
    codes = [
        ("wifi", "luna-4182"),
        ("safe box", "delta-9031"),
        ("garage keypad", "7426"),
        ("guest laptop", "nebula-55"),
        ("alarm panel", "1938"),
    ]
    projects = [
        ("billing deploy", "freeze writes, shift traffic back, verify error rates"),
        ("mobile release", "pause rollout, restore the previous build, watch crashes"),
        ("search migration", "disable the new index, replay writes, compare counts"),
        ("CRM import", "stop imports, restore the snapshot, rerun validation"),
        ("docs publish", "revert the page bundle, clear cache, check links"),
    ]
    preferences = [
        ("restaurants", "quiet corner tables"),
        ("meetings", "a written agenda before the call"),
        ("hotel rooms", "a desk near natural light"),
        ("work sessions", "a 25 minute focus timer"),
        ("train seats", "forward-facing aisle seats"),
    ]
    absolute_events = [
        ("launch review", "2026-07-12"),
        ("budget sync", "2026-08-03"),
        ("workshop", "2026-09-21"),
        ("renewal deadline", "2026-10-05"),
        ("team offsite", "2026-11-18"),
    ]
    languages = [
        (
            "es",
            "Cual es el codigo de {thing} de {person}?",
            "El codigo de {thing} de {person} es {code}.",
        ),
        (
            "ca",
            "Quin es el codi de {thing} de {person}?",
            "El codi de {thing} de {person} es {code}.",
        ),
        (
            "fr",
            "Quel est le code {thing} de {person} ?",
            "Le code {thing} de {person} est {code}.",
        ),
    ]

    for index in range(500):
        person = people[index % len(people)]
        city, old_city = cities[index % len(cities)]
        suffix = f"synth_{index:03d}"
        pattern = index % 10
        if pattern == 0:
            yield {
                "case_id": f"{suffix}_current_city",
                "query": f"What city does {person} currently live in?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "exact_facets": ["location"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_city_current",
                        f"{person} currently lives in {city}.",
                        rrf=0.44,
                    ),
                    _raw_candidate(
                        f"{suffix}_city_old",
                        f"{person} used to live in {old_city} before moving.",
                        status="superseded",
                        rrf=0.42,
                    ),
                    _raw_candidate(
                        f"{suffix}_travel_pref",
                        f"{person} likes walkable neighborhoods when traveling.",
                        object_type="interaction_contract",
                        rrf=0.36,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_city_current"],
                "expected_useful_ids": [f"{suffix}_city_current"],
                "expected_drop_ids": [f"{suffix}_travel_pref"],
            }
        elif pattern == 1:
            lang, query_template, answer_template = languages[index % len(languages)]
            thing, code = codes[index % len(codes)]
            yield {
                "case_id": f"{suffix}_{lang}_code",
                "query": query_template.format(thing=thing, person=person),
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "exact_facets": ["code"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_code_exact",
                        answer_template.format(
                            thing=thing,
                            person=person,
                            code=code,
                        ),
                        rrf=0.45,
                    ),
                    _raw_candidate(
                        f"{suffix}_code_related",
                        f"{person} reset the {thing} after a maintenance visit.",
                        rrf=0.40,
                    ),
                    _raw_candidate(
                        f"{suffix}_code_noise",
                        f"{person} prefers short morning planning notes.",
                        object_type="interaction_contract",
                        rrf=0.35,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_code_exact"],
                "expected_useful_ids": [f"{suffix}_code_exact"],
                "expected_drop_ids": [f"{suffix}_code_noise"],
            }
        elif pattern == 2:
            supplement, dose = supplements[index % len(supplements)]
            yield {
                "case_id": f"{suffix}_dose",
                "query": f"What {supplement} dose did {person} save?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "exact_facets": ["quantity", "medication"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_dose_exact",
                        f"{person} takes {supplement} {dose} at night.",
                        rrf=0.45,
                    ),
                    _raw_candidate(
                        f"{suffix}_dose_generic",
                        f"{person} mentioned {supplement} while discussing routines.",
                        object_type="summary_view",
                        rrf=0.42,
                    ),
                    _raw_candidate(
                        f"{suffix}_dose_other",
                        f"{person} takes a multivitamin with breakfast.",
                        rrf=0.36,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_dose_exact"],
                "expected_useful_ids": [f"{suffix}_dose_exact"],
                "expected_drop_ids": [f"{suffix}_dose_other"],
            }
        elif pattern == 3:
            appointment, with_whom = appointments[index % len(appointments)]
            anchor = datetime(2026, 6, 10 + (index % 5), 9, 0, tzinfo=timezone.utc)
            expected_date = (anchor.date() - timedelta(days=1)).isoformat()
            yield {
                "case_id": f"{suffix}_relative_yesterday",
                "query": f"What appointment did {person} say they had yesterday?",
                "query_type": "temporal",
                "exact_recall_mode": True,
                "exact_facets": ["date", "person_name"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_yesterday_exact",
                        (
                            f"Yesterday {person} had a {appointment} appointment "
                            f"with {with_whom}."
                        ),
                        rrf=0.44,
                        payload_json={
                            "source_message_window_start_occurred_at": anchor.isoformat(),
                            "source_message_window_end_occurred_at": anchor.isoformat(),
                        },
                    ),
                    _raw_candidate(
                        f"{suffix}_next_week",
                        f"Next week {person} has a planning call.",
                        rrf=0.39,
                    ),
                    _raw_candidate(
                        f"{suffix}_appointment_pref",
                        f"{person} prefers morning appointments.",
                        object_type="interaction_contract",
                        rrf=0.35,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_yesterday_exact"],
                "expected_useful_ids": [f"{suffix}_yesterday_exact"],
                "expected_drop_ids": [f"{suffix}_appointment_pref"],
                "expected_resolved_dates": {f"{suffix}_yesterday_exact": expected_date},
            }
        elif pattern == 4:
            yield {
                "case_id": f"{suffix}_broad_cities",
                "query": f"Which cities has {person} visited?",
                "query_type": "broad_list",
                "exact_recall_mode": True,
                "exact_facets": ["location"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_visit_a",
                        f"{person} visited {city} during a spring trip.",
                        rrf=0.43,
                    ),
                    _raw_candidate(
                        f"{suffix}_visit_b",
                        f"During a September trip, {person} spent three days in {old_city}.",
                        rrf=0.42,
                    ),
                    _raw_candidate(
                        f"{suffix}_current_city",
                        f"{person} currently lives in {city}.",
                        rrf=0.38,
                    ),
                    _raw_candidate(
                        f"{suffix}_travel_noise",
                        f"{person} prefers direct trains for travel.",
                        object_type="interaction_contract",
                        rrf=0.36,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_visit_a", f"{suffix}_visit_b"],
                "expected_useful_ids": [f"{suffix}_visit_a", f"{suffix}_visit_b"],
                "expected_drop_ids": [f"{suffix}_travel_noise"],
            }
        elif pattern == 5:
            office, old_office = offices[index % len(offices)]
            yield {
                "case_id": f"{suffix}_office_current",
                "query": f"What is {person}'s current office location?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "exact_facets": ["location"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_office_current",
                        f"{person}'s current office location is {office}.",
                        object_type="state_snapshot",
                        rrf=0.43,
                        vitality=0.7,
                    ),
                    _raw_candidate(
                        f"{suffix}_office_old",
                        f"{person}'s office location used to be {old_office}.",
                        object_type="state_snapshot",
                        status="superseded",
                        rrf=0.42,
                        vitality=0.3,
                    ),
                    _raw_candidate(
                        f"{suffix}_office_pref",
                        f"{person} likes offices with natural light.",
                        object_type="interaction_contract",
                        rrf=0.36,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_office_current"],
                "expected_useful_ids": [f"{suffix}_office_current"],
                "expected_drop_ids": [f"{suffix}_office_pref"],
            }
        elif pattern == 6:
            project, plan = projects[index % len(projects)]
            yield {
                "case_id": f"{suffix}_assistant_plan",
                "query": f"What rollback plan did you suggest for the {project}?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_plan_exact",
                        f"The assistant suggested this rollback plan for the {project}: {plan}.",
                        rrf=0.44,
                    ),
                    _raw_candidate(
                        f"{suffix}_plan_context",
                        f"{person} said the {project} had errors after deployment.",
                        rrf=0.40,
                    ),
                    _raw_candidate(
                        f"{suffix}_plan_generic",
                        "Rollback plans should include a backup and a postmortem note.",
                        object_type="belief",
                        rrf=0.37,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_plan_exact"],
                "expected_useful_ids": [f"{suffix}_plan_exact"],
                "expected_drop_ids": [f"{suffix}_plan_generic"],
            }
        elif pattern == 7:
            domain, preference = preferences[index % len(preferences)]
            yield {
                "case_id": f"{suffix}_preference",
                "query": f"What does {person} prefer for {domain}?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_pref_exact",
                        f"{person} prefers {preference} for {domain}.",
                        object_type="interaction_contract",
                        rrf=0.45,
                    ),
                    _raw_candidate(
                        f"{suffix}_pref_related",
                        f"{person} discussed {domain} while planning next month.",
                        object_type="summary_view",
                        rrf=0.41,
                    ),
                    _raw_candidate(
                        f"{suffix}_pref_noise",
                        f"{person} currently lives in {city}.",
                        rrf=0.36,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_pref_exact"],
                "expected_useful_ids": [f"{suffix}_pref_exact"],
                "expected_drop_ids": [f"{suffix}_pref_noise"],
            }
        elif pattern == 8:
            event, date_value = absolute_events[index % len(absolute_events)]
            yield {
                "case_id": f"{suffix}_absolute_date",
                "query": f"When is {person}'s {event}?",
                "query_type": "temporal",
                "exact_recall_mode": True,
                "exact_facets": ["date"],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_event_exact",
                        f"{person}'s {event} is on {date_value}.",
                        rrf=0.45,
                    ),
                    _raw_candidate(
                        f"{suffix}_event_related",
                        f"{person} prepared notes for the {event}.",
                        rrf=0.40,
                    ),
                    _raw_candidate(
                        f"{suffix}_event_noise",
                        f"{person} likes concise calendar reminders.",
                        object_type="interaction_contract",
                        rrf=0.35,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_event_exact"],
                "expected_useful_ids": [f"{suffix}_event_exact"],
                "expected_drop_ids": [f"{suffix}_event_noise"],
                "expected_resolved_dates": {f"{suffix}_event_exact": None},
            }
        else:
            item = [
                "standing desk",
                "mechanical keyboard",
                "blue notebook",
                "travel adapter",
                "studio microphone",
            ][index % 5]
            yield {
                "case_id": f"{suffix}_recent_context",
                "query": "Which item did they ask me to remember?",
                "query_type": "slot_fill",
                "exact_recall_mode": True,
                "recent_context": [
                    {
                        "role": "user",
                        "content": (
                            f"We were just talking about {person}'s workspace setup."
                        ),
                    }
                ],
                "candidates": [
                    _raw_candidate(
                        f"{suffix}_item_exact",
                        f"{person} asked the assistant to remember the {item}.",
                        rrf=0.44,
                    ),
                    _raw_candidate(
                        f"{suffix}_item_related",
                        f"{person} reorganized the workspace last Friday.",
                        rrf=0.40,
                    ),
                    _raw_candidate(
                        f"{suffix}_item_noise",
                        f"{person} currently lives in {city}.",
                        rrf=0.35,
                    ),
                ],
                "expected_top_ids": [f"{suffix}_item_exact"],
                "expected_useful_ids": [f"{suffix}_item_exact"],
                "expected_drop_ids": [f"{suffix}_item_noise"],
            }


def _raw_candidate(
    memory_id: str,
    canonical_text: str,
    *,
    object_type: str = "evidence",
    status: str = "active",
    rrf: float = 0.40,
    vitality: float = 0.5,
    payload_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": memory_id,
        "canonical_text": canonical_text,
        "object_type": object_type,
        "status": status,
        "rrf_score": rrf,
        "vitality": vitality,
        "payload_json": payload_json or {},
    }


def normalize_scored(scored: list[ScoredCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": item.memory_id,
            "llm_applicability": item.llm_applicability,
            "retrieval_score": item.retrieval_score,
            "final_score": item.final_score,
            "resolved_date": item.resolved_date,
            "object_type": item.memory_object.get("object_type"),
            "canonical_text": item.memory_object.get("canonical_text"),
        }
        for item in scored
    ]


def score_output(output: list[dict[str, Any]], case: BenchmarkCase) -> dict[str, Any]:
    ranked_ids = [str(item["memory_id"]) for item in output]
    top_id = ranked_ids[0] if ranked_ids else None
    top_hit = top_id in set(case.expected_top_ids)
    top3 = set(ranked_ids[:3])
    expected_useful = set(case.expected_useful_ids)
    expected_drop = set(case.expected_drop_ids)
    useful_hits = sorted(expected_useful & top3)
    drop_cutoff = max(1, len(expected_useful))
    drop_hits = sorted(expected_drop & set(ranked_ids[:drop_cutoff]))
    expected_date_matches: dict[str, bool] = {}
    output_by_id = {str(item["memory_id"]): item for item in output}
    for memory_id, expected_date in (case.expected_resolved_dates or {}).items():
        expected_date_matches[memory_id] = (
            output_by_id.get(memory_id, {}).get("resolved_date") == expected_date
        )
    useful_recall = _safe_div(len(useful_hits), len(expected_useful))
    exact_match = (
        top_hit
        and useful_recall >= 1.0
        and not drop_hits
        and all(expected_date_matches.values())
    )
    return {
        "exact_match": exact_match,
        "top_hit": top_hit,
        "top_id": top_id,
        "ranked_ids": ranked_ids,
        "expected_useful_recall": useful_recall,
        "expected_useful_hits": useful_hits,
        "expected_drop_top3_hits": drop_hits,
        "expected_date_matches": expected_date_matches,
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
        top_hits = sum(1 for row in variant_rows if row["score"]["top_hit"])
        recall_values = [
            float(row["score"].get("expected_useful_recall") or 0.0)
            for row in variant_rows
        ]
        failed = sum(1 for row in variant_rows if row.get("error"))
        parse_invalid = sum(int(row.get("parse_invalid_count") or 0) for row in variant_rows)
        llm_records = recorder.records_for_context(variant=variant)
        llm_summary = summarize_llm_calls(llm_records)
        model = card_model
        by_variant[variant] = {
            "cases": len(variant_rows),
            "exact_match_count": exact,
            "exact_match_rate": _safe_div(exact, len(variant_rows)),
            "top_hit_count": top_hits,
            "top_hit_rate": _safe_div(top_hits, len(variant_rows)),
            "mean_expected_useful_recall": (
                sum(recall_values) / len(recall_values) if recall_values else 1.0
            ),
            "failed_trials": failed,
            "parse_invalid_count": parse_invalid,
            "wall_time_ms": _latency_summary(latencies),
            "estimated_cost_usd": _estimate_cost_usd(
                model,
                llm_summary.get("token_totals") or {},
            ),
            "llm_call_summary": llm_summary,
            "mismatch_case_ids": [
                row["case_id"]
                for row in variant_rows
                if not row["score"]["exact_match"]
            ],
        }
    llm_call_summary = recorder.summary()
    return {
        "benchmark": "applicability_cards",
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
        baseline = variants.get("cards_batch_4")
        if baseline is None:
            continue
        for variant, row in sorted(variants.items()):
            if variant == "cards_batch_4":
                continue
            if row["score"]["ranked_ids"] != baseline["score"]["ranked_ids"]:
                pairs.append(
                    {
                        "case_id": case_id,
                        "repetition": repetition,
                        "variant": variant,
                        "baseline_ranked_ids": baseline["score"]["ranked_ids"],
                        "variant_ranked_ids": row["score"]["ranked_ids"],
                        "baseline_top_hit": baseline["score"]["top_hit"],
                        "variant_top_hit": row["score"]["top_hit"],
                    }
                )
    return pairs


def _normalize_candidate(raw: dict[str, Any], index: int) -> dict[str, Any]:
    payload_json = dict(raw.get("payload_json") or {})
    candidate = {
        "id": str(raw["id"]),
        "object_type": str(raw.get("object_type") or "evidence"),
        "scope": str(raw.get("scope") or "conversation"),
        "scope_canonical": str(raw.get("scope_canonical") or "chat"),
        "status": str(raw.get("status") or "active"),
        "privacy_level": int(raw.get("privacy_level", 0)),
        "sensitivity": str(raw.get("sensitivity") or "medium"),
        "intimacy_boundary": str(raw.get("intimacy_boundary") or "ordinary"),
        "platform_locked": int(raw.get("platform_locked", 0)),
        "platform_id": str(raw.get("platform_id") or "default"),
        "platform_id_lock": raw.get("platform_id_lock"),
        "temporal_type": str(raw.get("temporal_type") or "unknown"),
        "valid_from": str(raw.get("valid_from") or ""),
        "valid_to": str(raw.get("valid_to") or ""),
        "canonical_text": str(raw["canonical_text"]),
        "payload_json": payload_json,
        "vitality": float(raw.get("vitality", 0.25)),
        "maya_score": float(raw.get("maya_score", 0.0)),
        "rank": float(raw.get("rank", index + 1)),
        "rrf_score": float(raw.get("rrf_score", 1.0 / (index + 1))),
        "retrieval_sources": list(raw.get("retrieval_sources") or ["benchmark_fixture"]),
        "created_at": str(raw.get("created_at") or "2026-06-17T12:00:00+00:00"),
        "updated_at": str(raw.get("updated_at") or "2026-06-17T12:00:00+00:00"),
    }
    return candidate


def _context_for_case(case: BenchmarkCase) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="bench_user",
        conversation_id=f"cnv_{case.case_id}",
        source_message_id=f"msg_{case.case_id}",
        workspace_id=None,
        assistant_mode_id=case.mode,
        recent_messages=[
            ExtractionContextMessage(
                role=message["role"],
                content=message["content"],
            )
            for message in case.recent_context
        ],
        privacy_enforcement="off",
    )


def _resolved_policy(mode_id: str):
    loader = ManifestLoader(_MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _plan_for_case(case: BenchmarkCase, privacy_ceiling: int) -> RetrievalPlan:
    exact_facets: list[ExactFacet] = []
    for value in case.exact_facets:
        try:
            exact_facets.append(ExactFacet(value))
        except ValueError:
            continue
    return RetrievalPlan(
        original_query=case.query,
        assistant_mode_id=case.mode,
        workspace_id=None,
        conversation_id=f"cnv_{case.case_id}",
        fts_queries=[case.query],
        sub_query_plans=[
            {
                "text": case.query,
                "fts_queries": [case.query],
            }
        ],
        query_type=case.query_type,
        scope_filter=[],
        status_filter=[],
        vector_limit=0,
        max_candidates=max(10, len(case.candidates)),
        max_context_items=8,
        privacy_ceiling=privacy_ceiling,
        privacy_enforcement="off",
        retrieval_levels=[0],
        consequence_search_enabled=False,
        require_evidence_regrounding=False,
        skip_retrieval=False,
        exact_recall_mode=case.exact_recall_mode,
        exact_facets=exact_facets,
    )


def _detected_needs(values: tuple[str, ...]) -> list[DetectedNeed]:
    needs: list[DetectedNeed] = []
    for value in values:
        try:
            need_type = NeedTrigger(value)
        except ValueError:
            continue
        needs.append(
            DetectedNeed(
                need_type=need_type,
                confidence=0.8,
                reasoning="applicability card benchmark fixture",
            )
        )
    return needs


def _card_batch_size_for_variant(variant: VariantName) -> int:
    if variant == "cards_single":
        return 1
    if variant == "cards_batch_8":
        return 8
    return 4


def _parse_variants(raw: str) -> tuple[VariantName, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        return _DEFAULT_VARIANTS
    invalid = [value for value in values if value not in _ALLOWED_VARIANTS]
    if invalid:
        raise ValueError(f"Unknown applicability card benchmark variant(s): {invalid}")
    return values  # type: ignore[return-value]


def _parse_invalid_count(
    diagnostics: list[dict[str, Any]],
    *,
    error: dict[str, Any] | None,
) -> int:
    count = 1 if error else 0
    for diagnostic in diagnostics:
        event = str(diagnostic.get("event") or "")
        if event in {
            "invalid_structured_output",
            "malformed_domain_output",
            "missing_after_retry",
            "card_parse_invalid",
            "card_missing_after_retry",
        }:
            count += 1
    return count


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


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = assert_outside_repo(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temp_path.replace(destination)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return destination


__all__ = [
    "_DEFAULT_CASES_PATH",
    "_MINIMAX_M3_MODEL",
    "BenchmarkCase",
    "expand_cases",
    "load_cases",
    "normalize_scored",
    "score_output",
    "summarize_run",
]
