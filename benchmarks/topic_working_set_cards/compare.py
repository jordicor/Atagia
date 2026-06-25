"""Compare legacy JSON Topic Working Set planning with current cards.

This is a shadow benchmark: it does not change production ingestion. It runs
the previous single structured-JSON planner prompt against the current card
planner and writes comparable artifacts for small Gemini/OpenRouter runs.
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
from typing import Any, Literal, cast

from dotenv import load_dotenv

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.llm_output_limits import TOPIC_WORKING_SET_MAX_OUTPUT_TOKENS
from atagia.memory.topic_working_set import (
    _CONTENT_ACTIONS,
    _TopicBoundary,
    _TopicCardPlan,
    _TopicContent,
    _TopicRoute,
    _card_lines,
    _line_tokens,
    _message_ids_from_messages,
    _parse_boundary_card_output,
    _parse_content_card_output,
    _parse_route_card_output,
    _topic_card_plan_to_structured_plan,
    _topic_ids_from_snapshot,
    _topics_by_id_from_snapshot,
    _valid_message_ids_from_tokens,
    TopicUpdateActionType,
    TopicWorkingSetPlan,
    TopicWorkingSetUpdater,
)
from atagia.models.schemas_memory import IntimacyBoundary
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
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
    "json_legacy",
    "cards_current",
    "cards_split_route_v1",
    "cards_split_route_v2",
    "cards_split_route_v3",
    "cards_split_route_v4",
    "cards_split_route_v5",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES_PATH = (
    _PROJECT_ROOT / "benchmarks" / "topic_working_set_cards" / "cases.jsonl"
)
_DIRECT_GEMINI_FLASH_LITE_MODEL = "google/gemini-3.1-flash-lite"
_DIRECT_MINIMAX_M3_MODEL = "minimax/MiniMax-M3"
_ALLOWED_VARIANTS: tuple[VariantName, ...] = (
    "json_legacy",
    "cards_current",
    "cards_split_route_v1",
    "cards_split_route_v2",
    "cards_split_route_v3",
    "cards_split_route_v4",
    "cards_split_route_v5",
)
_DEFAULT_VARIANTS: tuple[VariantName, ...] = ("json_legacy", "cards_current")
_FIXED_CLOCK = datetime(2026, 6, 16, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    conversation_id: str
    snapshot: dict[str, Any]
    messages: list[dict[str, Any]]
    expected: dict[str, Any]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CardResult:
    card_name: str
    target_id: str | None
    raw_output: str | None
    parsed: Any
    parse_valid: bool
    error: str | None = None


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
            "Comma-separated variants: json_legacy,cards_current,"
            "cards_split_route_v1,cards_split_route_v2,cards_split_route_v3,"
            "cards_split_route_v4,cards_split_route_v5"
        ),
    )
    parser.add_argument("--json-model", default=_DIRECT_GEMINI_FLASH_LITE_MODEL)
    parser.add_argument("--card-model", default=_DIRECT_GEMINI_FLASH_LITE_MODEL)
    parser.add_argument(
        "--model",
        default=None,
        help="Convenience override that sets both --json-model and --card-model.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm-progress-every", type=int, default=0)
    parser.add_argument(
        "--llm-call-delay-ms",
        type=int,
        default=0,
        help=(
            "Serialize benchmark LLM calls and sleep this many milliseconds "
            "before each call. Useful for direct providers with tight rate limits."
        ),
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(Path(args.cases), limit=args.limit)
    variants = _parse_variants(args.variants)
    repetitions = max(1, int(args.repetitions))
    json_model = str(args.model or args.json_model)
    card_model = str(args.model or args.card_model)

    output_dir = (
        resolve_output_dir("topic_working_set_cards")
        if args.output_dir is None
        else assert_outside_repo(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = replace(
        Settings.from_env(),
        llm_forced_global_model=json_model,
    )
    client = build_llm_client(settings)
    recorder = LLMCallRecorder(progress_interval=args.llm_progress_every)
    install_llm_call_recorder(client, recorder)
    llm_call_delay_ms = max(0, int(args.llm_call_delay_ms))
    if llm_call_delay_ms:
        install_llm_call_delay(client, delay_seconds=llm_call_delay_ms / 1000.0)

    started_at = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for case in cases:
            for variant in variants:
                with recorder.context(
                    benchmark="topic_working_set_cards",
                    case_id=case.case_id,
                    variant=variant,
                    repetition=repetition + 1,
                ):
                    row = await run_one_variant(
                        client=client,
                        case=case,
                        variant=variant,
                        json_model=json_model,
                        card_model=card_model,
                        repetition=repetition + 1,
                    )
                rows.append(row)
                print(
                    f"{variant} {case.case_id} rep={repetition + 1} "
                    f"exact={row['score']['exact_match']} "
                    f"wall_ms={row['wall_time_ms']:.0f}"
                )

    finished_at = datetime.now(timezone.utc)
    summary = summarize_run(
        rows,
        recorder=recorder,
        variants=variants,
        started_at=started_at,
        finished_at=finished_at,
        json_model=json_model,
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
    json_model: str,
    card_model: str,
    repetition: int,
) -> dict[str, Any]:
    started = perf_counter()
    if variant == "json_legacy":
        result = await run_json_legacy(
            client=client,
            model=json_model,
            case=case,
        )
    elif variant == "cards_split_route_v1":
        result = await run_cards_split_route_v1(
            client=client,
            model=card_model,
            case=case,
        )
    elif variant == "cards_split_route_v2":
        result = await run_cards_split_route_v2(
            client=client,
            model=card_model,
            case=case,
        )
    elif variant == "cards_split_route_v3":
        result = await run_cards_split_route_v3(
            client=client,
            model=card_model,
            case=case,
        )
    elif variant == "cards_split_route_v4":
        result = await run_cards_split_route_v4(
            client=client,
            model=card_model,
            case=case,
        )
    elif variant == "cards_split_route_v5":
        result = await run_cards_split_route_v5(
            client=client,
            model=card_model,
            case=case,
        )
    else:
        result = await run_cards_current(
            client=client,
            model=card_model,
            case=case,
        )
    wall_time_ms = (perf_counter() - started) * 1000.0
    plan = cast(TopicWorkingSetPlan, result["plan"])
    normalized_plan = normalize_plan(plan)
    topic_projection = project_topic_state(case.snapshot, normalized_plan)
    score = score_plan(normalized_plan, case.expected)
    return {
        "case_id": case.case_id,
        "conversation_id": case.conversation_id,
        "notes": case.notes,
        "variant": variant,
        "repetition": repetition,
        "model": json_model if variant == "json_legacy" else card_model,
        "wall_time_ms": wall_time_ms,
        "expected": case.expected,
        "plan": normalized_plan,
        "route_signature": route_signature(normalized_plan),
        "topic_projection": topic_projection,
        "score": score,
        **{key: value for key, value in result.items() if key != "plan"},
    }


async def run_json_legacy(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    prompt = legacy_topic_working_set_prompt(
        conversation_id=case.conversation_id,
        snapshot=case.snapshot,
        messages=case.messages,
    )
    request = LLMCompletionRequest(
        model=model,
        messages=[
            LLMMessage(
                role="system",
                content="Maintain conversation topic working sets as structured JSON.",
            ),
            LLMMessage(role="user", content=prompt),
        ],
        max_output_tokens=TOPIC_WORKING_SET_MAX_OUTPUT_TOKENS,
        response_schema=TopicWorkingSetPlan.model_json_schema(),
        metadata={
            "user_id": "benchmark-user",
            "conversation_id": case.conversation_id,
            "purpose": "topic_working_set_update",
            "topic_working_set_shadow_variant": "json_legacy",
            **TopicWorkingSetUpdater._intimacy_metadata_from_snapshot(case.snapshot),
        },
    )
    try:
        structured = await client.complete_structured_with_response(
            request,
            TopicWorkingSetPlan,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": getattr(exc, "output_text", None),
            "error": _error_payload(exc),
            "parse_invalid_count": 1,
            "card_outputs": {},
        }
    return {
        "plan": structured.value,
        "raw_output": structured.response.output_text,
        "error": None,
        "parse_invalid_count": 0,
        "card_outputs": {},
    }


async def run_cards_current(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    """Run the production card flow: split route (existing + uncovered new-topic),
    per-target content and boundary cards, and an LLM artifact-link card."""
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card(
        updater=updater,
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    uncovered_messages = _messages_not_covered_by_routes(case.messages, existing_routes)
    if uncovered_messages:
        new_result = await _run_new_topic_track_card(
            updater=updater,
            client=client,
            model=model,
            case=case,
            uncovered_messages=uncovered_messages,
        )
    else:
        new_result = CardResult(
            "new_topic_track",
            None,
            "skip: all messages already routed",
            (),
            True,
        )
    card_outputs["new_topic_track"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_track", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "content", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries: dict[str, _TopicBoundary] = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "boundary", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifact_result = await _run_artifact_card(
        client=client,
        model=model,
        case=case,
        routes=routes,
    )
    if artifact_result is not None:
        card_outputs["artifact"] = _jsonable_card_result(artifact_result)
        if artifact_result.error:
            errors.append({"card": "artifact", "message": artifact_result.error})
        artifacts = cast(dict[str, tuple[str, ...]], artifact_result.parsed)
    else:
        artifacts = {}

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def run_cards_split_route_v1(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card_v1(
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route_v1"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route_v1", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    new_result = await _run_new_topic_route_card_v1(
        client=client,
        model=model,
        case=case,
        existing_routes=existing_routes,
    )
    card_outputs["new_topic_route_v1"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_route_v1", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "content", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "boundary", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifact_result = await _run_artifact_card(
        client=client,
        model=model,
        case=case,
        routes=routes,
    )
    if artifact_result is not None:
        card_outputs["artifact"] = _jsonable_card_result(artifact_result)
        if artifact_result.error:
            errors.append({"card": "artifact", "message": artifact_result.error})
        artifacts = cast(dict[str, tuple[str, ...]], artifact_result.parsed)
    else:
        artifacts = {}

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def run_cards_split_route_v2(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card_v1(
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route_v1"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route_v1", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    new_result = await _run_new_topic_needed_card_v2(
        client=client,
        model=model,
        case=case,
        existing_routes=existing_routes,
    )
    card_outputs["new_topic_needed_v2"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_needed_v2", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append(
                {
                    "card": "content",
                    "target_id": result.target_id,
                    "message": result.error,
                }
            )
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "boundary", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifact_result = await _run_artifact_card(
        client=client,
        model=model,
        case=case,
        routes=routes,
    )
    if artifact_result is not None:
        card_outputs["artifact"] = _jsonable_card_result(artifact_result)
        if artifact_result.error:
            errors.append({"card": "artifact", "message": artifact_result.error})
        artifacts = cast(dict[str, tuple[str, ...]], artifact_result.parsed)
    else:
        artifacts = {}

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def run_cards_split_route_v3(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card_v1(
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route_v1"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route_v1", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    new_result = await _run_new_topic_track_card_v3(
        client=client,
        model=model,
        case=case,
        existing_routes=existing_routes,
    )
    card_outputs["new_topic_track_v3"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_track_v3", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append(
                {
                    "card": "content",
                    "target_id": result.target_id,
                    "message": result.error,
                }
            )
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "boundary", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifact_result = await _run_artifact_card(
        client=client,
        model=model,
        case=case,
        routes=routes,
    )
    if artifact_result is not None:
        card_outputs["artifact"] = _jsonable_card_result(artifact_result)
        if artifact_result.error:
            errors.append({"card": "artifact", "message": artifact_result.error})
        artifacts = cast(dict[str, tuple[str, ...]], artifact_result.parsed)
    else:
        artifacts = {}

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def run_cards_split_route_v4(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card_v1(
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route_v1"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route_v1", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    uncovered_messages = _messages_not_covered_by_routes(
        case.messages,
        existing_routes,
    )
    if uncovered_messages:
        new_result = await _run_new_topic_track_card_v4(
            client=client,
            model=model,
            case=case,
            uncovered_messages=uncovered_messages,
        )
    else:
        new_result = CardResult(
            "new_topic_track_v4",
            None,
            "skip: all messages already routed",
            (),
            True,
        )
    card_outputs["new_topic_track_v4"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_track_v4", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append(
                {
                    "card": "content",
                    "target_id": result.target_id,
                    "message": result.error,
                }
            )
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append({"card": "boundary", "target_id": result.target_id, "message": result.error})
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifacts = _artifact_ids_from_route_messages(case=case, routes=routes)
    if artifacts:
        card_outputs["artifact_deterministic_v4"] = _jsonable_card_result(
            CardResult(
                "artifact_deterministic_v4",
                None,
                "derived from route source message artifacts",
                artifacts,
                True,
            )
        )

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def run_cards_split_route_v5(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(_FIXED_CLOCK),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model=model),
    )
    card_outputs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    existing_result = await _run_existing_route_card_v1(
        client=client,
        model=model,
        case=case,
    )
    card_outputs["existing_route_v1"] = _jsonable_card_result(existing_result)
    if existing_result.error:
        errors.append({"card": "existing_route_v1", "message": existing_result.error})
    existing_routes = cast(tuple[_TopicRoute, ...], existing_result.parsed)

    uncovered_messages = _messages_not_covered_by_routes(
        case.messages,
        existing_routes,
    )
    if uncovered_messages:
        new_result = await _run_new_topic_track_card_v4(
            client=client,
            model=model,
            case=case,
            uncovered_messages=uncovered_messages,
        )
    else:
        new_result = CardResult(
            "new_topic_track_v4",
            None,
            "skip: all messages already routed",
            (),
            True,
        )
    card_outputs["new_topic_track_v4"] = _jsonable_card_result(new_result)
    if new_result.error:
        errors.append({"card": "new_topic_track_v4", "message": new_result.error})
    new_routes = cast(tuple[_TopicRoute, ...], new_result.parsed)

    routes = _dedupe_routes([*existing_routes, *new_routes])
    if not routes:
        return {
            "plan": TopicWorkingSetPlan(actions=[], nothing_to_update=True),
            "raw_output": None,
            "error": errors or None,
            "parse_invalid_count": sum(
                1 for output in card_outputs.values() if not output["parse_valid"]
            ),
            "card_outputs": card_outputs,
        }

    content_routes = tuple(route for route in routes if route.action in _CONTENT_ACTIONS)
    content_results = await _run_content_cards(
        updater=updater,
        client=client,
        model=model,
        case=case,
        routes=content_routes,
    )
    contents: dict[str, _TopicContent] = {}
    for result in content_results:
        card_outputs[f"content:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append(
                {
                    "card": "content",
                    "target_id": result.target_id,
                    "message": result.error,
                }
            )
        if result.target_id is not None:
            contents[result.target_id] = cast(_TopicContent, result.parsed)

    boundary_results = await _run_boundary_cards_v5(
        client=client,
        model=model,
        case=case,
        routes=content_routes,
        contents=contents,
    )
    boundaries: dict[str, _TopicBoundary] = {}
    for result in boundary_results:
        card_outputs[f"boundary:{result.target_id}"] = _jsonable_card_result(result)
        if result.error:
            errors.append(
                {
                    "card": "boundary",
                    "target_id": result.target_id,
                    "message": result.error,
                }
            )
        if result.target_id is not None and isinstance(result.parsed, _TopicBoundary):
            boundaries[result.target_id] = result.parsed

    artifacts = _artifact_ids_from_route_messages(case=case, routes=routes)
    if artifacts:
        card_outputs["artifact_deterministic_v5"] = _jsonable_card_result(
            CardResult(
                "artifact_deterministic_v5",
                None,
                "derived from route source message artifacts",
                artifacts,
                True,
            )
        )

    plan = _topic_card_plan_to_structured_plan(
        _TopicCardPlan(
            routes=routes,
            contents=contents,
            boundaries=boundaries,
            artifacts=artifacts,
        )
    )
    parse_invalid_count = sum(
        1 for output in card_outputs.values() if not output["parse_valid"]
    )
    return {
        "plan": plan,
        "raw_output": None,
        "error": errors or None,
        "parse_invalid_count": parse_invalid_count,
        "card_outputs": card_outputs,
    }


async def _run_existing_route_card_v1(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> CardResult:
    prompt = _build_existing_route_prompt_v1(case)
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_route_card",
        conversation_id=case.conversation_id,
        prompt=prompt,
        shadow_card="existing_route_v1",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("existing_route_v1", None, None, (), False, _error_message(exc))
    parsed = tuple(
        route
        for route in _parse_route_card_output(
            response.output_text,
            valid_topic_ids=_topic_ids_from_snapshot(case.snapshot),
            valid_message_ids=_message_ids_from_messages(case.messages),
        )
        if route.action is not TopicUpdateActionType.CREATE
    )
    return CardResult(
        "existing_route_v1",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_new_topic_route_card_v1(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    existing_routes: tuple[_TopicRoute, ...],
) -> CardResult:
    prompt = _build_new_topic_route_prompt_v1(case, existing_routes=existing_routes)
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_route_card",
        conversation_id=case.conversation_id,
        prompt=prompt,
        shadow_card="new_topic_route_v1",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("new_topic_route_v1", None, None, (), False, _error_message(exc))
    parsed = tuple(
        route
        for route in _parse_route_card_output(
            response.output_text,
            valid_topic_ids=_topic_ids_from_snapshot(case.snapshot),
            valid_message_ids=_message_ids_from_messages(case.messages),
        )
        if route.action is TopicUpdateActionType.CREATE
    )
    return CardResult(
        "new_topic_route_v1",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_new_topic_needed_card_v2(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    existing_routes: tuple[_TopicRoute, ...],
) -> CardResult:
    prompt = _build_new_topic_needed_prompt_v2(case, existing_routes=existing_routes)
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_route_card",
        conversation_id=case.conversation_id,
        prompt=prompt,
        shadow_card="new_topic_needed_v2",
        shadow_variant="cards_split_route_v2",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("new_topic_needed_v2", None, None, (), False, _error_message(exc))
    parsed = _parse_new_topic_needed_output_v2(
        response.output_text,
        valid_message_ids=_message_ids_from_messages(case.messages),
    )
    return CardResult(
        "new_topic_needed_v2",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_new_topic_track_card_v3(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    existing_routes: tuple[_TopicRoute, ...],
) -> CardResult:
    prompt = _build_new_topic_track_prompt_v3(case, existing_routes=existing_routes)
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_route_card",
        conversation_id=case.conversation_id,
        prompt=prompt,
        shadow_card="new_topic_track_v3",
        shadow_variant="cards_split_route_v3",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("new_topic_track_v3", None, None, (), False, _error_message(exc))
    parsed = _parse_new_topic_track_output_v3(
        response.output_text,
        valid_message_ids=_message_ids_from_messages(case.messages),
    )
    return CardResult(
        "new_topic_track_v3",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_new_topic_track_card_v4(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    uncovered_messages: list[dict[str, Any]],
) -> CardResult:
    prompt = _build_new_topic_track_prompt_v4(
        case,
        uncovered_messages=uncovered_messages,
    )
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_route_card",
        conversation_id=case.conversation_id,
        prompt=prompt,
        shadow_card="new_topic_track_v4",
        shadow_variant="cards_split_route_v4",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("new_topic_track_v4", None, None, (), False, _error_message(exc))
    parsed = _parse_new_topic_track_output_v3(
        response.output_text,
        valid_message_ids=_message_ids_from_messages(uncovered_messages),
    )
    return CardResult(
        "new_topic_track_v4",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_existing_route_card(
    *,
    updater: TopicWorkingSetUpdater,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
) -> CardResult:
    request = _request_with_model(
        updater._card_request(
            card_name="route",
            user_id="benchmark-user",
            conversation_id=case.conversation_id,
            prompt=updater._build_existing_route_prompt(
                conversation_id=case.conversation_id,
                snapshot=case.snapshot,
                messages=case.messages,
            ),
            snapshot=case.snapshot,
        ),
        model=model,
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("existing_route", None, None, (), False, _error_message(exc))
    routes = tuple(
        route
        for route in _parse_route_card_output(
            response.output_text,
            valid_topic_ids=_topic_ids_from_snapshot(case.snapshot),
            valid_message_ids=_message_ids_from_messages(case.messages),
            conversation_id=case.conversation_id,
        )
        if route.action is not TopicUpdateActionType.CREATE
    )
    return CardResult(
        "existing_route",
        None,
        response.output_text,
        routes,
        _plain_card_output_valid(response.output_text, bool(routes)),
    )


async def _run_new_topic_track_card(
    *,
    updater: TopicWorkingSetUpdater,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    uncovered_messages: list[dict[str, Any]],
) -> CardResult:
    request = _request_with_model(
        updater._card_request(
            card_name="route",
            user_id="benchmark-user",
            conversation_id=case.conversation_id,
            prompt=updater._build_new_topic_track_prompt(
                conversation_id=case.conversation_id,
                snapshot=case.snapshot,
                messages=uncovered_messages,
            ),
            snapshot=case.snapshot,
        ),
        model=model,
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("new_topic_track", None, None, (), False, _error_message(exc))
    parsed = _parse_new_topic_track_output_v3(
        response.output_text,
        valid_message_ids=_message_ids_from_messages(uncovered_messages),
    )
    return CardResult(
        "new_topic_track",
        None,
        response.output_text,
        parsed,
        _plain_card_output_valid(response.output_text, bool(parsed)),
    )


async def _run_content_cards(
    *,
    updater: TopicWorkingSetUpdater,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    routes: tuple[_TopicRoute, ...],
) -> list[CardResult]:
    if not routes:
        return []
    topics_by_id = _topics_by_id_from_snapshot(case.snapshot)
    semaphore = asyncio.Semaphore(2)

    async def run_one(route: _TopicRoute) -> CardResult:
        async with semaphore:
            request = _request_with_model(
                updater._card_request(
                    card_name="content",
                    user_id="benchmark-user",
                    conversation_id=case.conversation_id,
                    prompt=updater._build_content_prompt(
                        conversation_id=case.conversation_id,
                        snapshot=case.snapshot,
                        messages=case.messages,
                        route=route,
                        existing_topic=topics_by_id.get(route.target_id),
                    ),
                    snapshot=case.snapshot,
                    target_id=route.target_id,
                ),
                model=model,
            )
            try:
                response = await client.complete(request)
            except Exception as exc:  # noqa: BLE001
                return CardResult(
                    "content",
                    route.target_id,
                    None,
                    _TopicContent(),
                    False,
                    _error_message(exc),
                )
            content = _parse_content_card_output(response.output_text)
            has_signal = bool(
                content.title
                or content.summary
                or content.active_goal
                or content.open_questions
                or content.decisions
            )
            return CardResult(
                "content",
                route.target_id,
                response.output_text,
                content,
                _plain_card_output_valid(response.output_text, has_signal),
            )

    return list(await asyncio.gather(*(run_one(route) for route in routes)))


async def _run_boundary_cards(
    *,
    updater: TopicWorkingSetUpdater,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    routes: tuple[_TopicRoute, ...],
    contents: dict[str, _TopicContent],
) -> list[CardResult]:
    if not routes:
        return []
    semaphore = asyncio.Semaphore(2)

    async def run_one(route: _TopicRoute) -> CardResult:
        async with semaphore:
            request = _request_with_model(
                updater._card_request(
                    card_name="boundary",
                    user_id="benchmark-user",
                    conversation_id=case.conversation_id,
                    prompt=updater._build_target_boundary_prompt(
                        conversation_id=case.conversation_id,
                        messages=case.messages,
                        route=route,
                        content=contents.get(route.target_id, _TopicContent()),
                    ),
                    snapshot=case.snapshot,
                    target_id=route.target_id,
                ),
                model=model,
            )
            try:
                response = await client.complete(request)
            except Exception as exc:  # noqa: BLE001
                return CardResult(
                    "boundary",
                    route.target_id,
                    None,
                    None,
                    False,
                    _error_message(exc),
                )
            boundaries = _parse_boundary_card_output(
                response.output_text,
                valid_target_ids={route.target_id},
            )
            boundary = boundaries.get(route.target_id)
            return CardResult(
                "boundary",
                route.target_id,
                response.output_text,
                boundary,
                _required_card_output_valid(response.output_text, boundary is not None),
            )

    return list(await asyncio.gather(*(run_one(route) for route in routes)))


async def _run_boundary_cards_v5(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    routes: tuple[_TopicRoute, ...],
    contents: dict[str, _TopicContent],
) -> list[CardResult]:
    if not routes:
        return []
    semaphore = asyncio.Semaphore(2)

    async def run_one(route: _TopicRoute) -> CardResult:
        async with semaphore:
            request = _plain_card_request(
                model=model,
                purpose="topic_working_set_boundary_card",
                conversation_id=case.conversation_id,
                prompt=_build_boundary_prompt_v5(
                    case,
                    route=route,
                    content=contents.get(route.target_id, _TopicContent()),
                ),
                shadow_card="boundary_v5",
                shadow_variant="cards_split_route_v5",
            )
            try:
                response = await client.complete(request)
            except Exception as exc:  # noqa: BLE001
                return CardResult(
                    "boundary_v5",
                    route.target_id,
                    None,
                    None,
                    False,
                    _error_message(exc),
                )
            boundaries = _parse_boundary_card_output(
                response.output_text,
                valid_target_ids={route.target_id},
            )
            boundary = boundaries.get(route.target_id)
            return CardResult(
                "boundary_v5",
                route.target_id,
                response.output_text,
                boundary,
                _required_card_output_valid(response.output_text, boundary is not None),
            )

    return list(await asyncio.gather(*(run_one(route) for route in routes)))


async def _run_artifact_card(
    *,
    client: LLMClient[Any],
    model: str,
    case: BenchmarkCase,
    routes: tuple[_TopicRoute, ...],
) -> CardResult | None:
    valid_artifact_ids = _provided_artifact_ids(case.messages)
    if not routes or not valid_artifact_ids:
        return None
    request = _plain_card_request(
        model=model,
        purpose="topic_working_set_boundary_card",
        conversation_id=case.conversation_id,
        prompt=_build_artifact_prompt(case, routes=routes),
        shadow_card="artifact",
        shadow_variant="cards_current",
    )
    try:
        response = await client.complete(request)
    except Exception as exc:  # noqa: BLE001
        return CardResult("artifact", None, None, {}, False, _error_message(exc))
    artifacts = _parse_artifact_card_output(
        response.output_text,
        valid_target_ids={route.target_id for route in routes},
        valid_artifact_ids=valid_artifact_ids,
    )
    return CardResult(
        "artifact",
        None,
        response.output_text,
        artifacts,
        _plain_card_output_valid(response.output_text, bool(artifacts)),
    )


def _build_artifact_prompt(
    case: BenchmarkCase,
    *,
    routes: tuple[_TopicRoute, ...],
) -> str:
    route_payload = [
        {
            "action": route.action.value,
            "target_id": route.target_id,
            "source_message_ids": list(route.source_message_ids),
        }
        for route in routes
    ]
    return "\n".join(
        [
            "Link provided artifacts to topics only when the topic depends on the attachment.",
            "Write one line per link, or exactly: none",
            "Do not write JSON.",
            "Use only artifact_ids from artifact_refs in the messages.",
            "Format: target_id artifact_id",
            "Example: tmp1 art_example",
            f"conversation_id={case.conversation_id}",
            "<routes>",
            json_utils.dumps(route_payload, indent=2, sort_keys=True),
            "</routes>",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(
                _message_payload_for_prompt(case.messages),
                indent=2,
                sort_keys=True,
            ),
            "</messages>",
        ]
    )


def _parse_artifact_card_output(
    text: str,
    *,
    valid_target_ids: set[str],
    valid_artifact_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    lines = _card_lines(text)
    if not lines or any(line.strip("`*_.,;[](){}\"'").casefold() == "none" for line in lines):
        return {}
    artifacts: dict[str, list[str]] = {}
    for line in lines:
        tokens = _line_tokens(line)
        if len(tokens) < 2:
            continue
        target_id = tokens[0]
        artifact_id = tokens[1]
        if target_id not in valid_target_ids or artifact_id not in valid_artifact_ids:
            continue
        rows = artifacts.setdefault(target_id, [])
        if artifact_id not in rows:
            rows.append(artifact_id)
    return {target_id: tuple(values) for target_id, values in artifacts.items()}


def _provided_artifact_ids(messages: list[dict[str, Any]]) -> set[str]:
    artifact_ids: set[str] = set()
    for message in messages:
        for artifact_ref in TopicWorkingSetUpdater._message_artifact_refs(message):
            artifact_id = artifact_ref.get("artifact_id")
            if artifact_id:
                artifact_ids.add(str(artifact_id))
    return artifact_ids


def legacy_topic_working_set_prompt(
    *,
    conversation_id: str,
    snapshot: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    boundary_values = ", ".join(boundary.value for boundary in IntimacyBoundary)
    message_payload = [
        {
            "id": str(message["id"]),
            "seq": message.get("seq"),
            "role": str(message["role"]),
            "text": TopicWorkingSetUpdater._message_text_for_topic_prompt(message),
            "raw_text_included": TopicWorkingSetUpdater._message_raw_text_allowed(message),
            "content_kind": message.get("content_kind") or "text",
            "policy_reason": message.get("policy_reason") or "normal",
            "created_at": message.get("created_at"),
            "artifact_refs": TopicWorkingSetUpdater._message_artifact_refs(message),
        }
        for message in messages
    ]
    return "\n".join(
        [
            "Update the read-only Topic Working Set for this conversation.",
            "Return only JSON matching the provided schema.",
            "Do not include markdown fences, preambles, tags, or explanations.",
            "Anything outside the first JSON object will be ignored.",
            "Do not create dataset-specific or benchmark-specific topics.",
            "Prefer updating existing topics when the new messages continue the same thread.",
            "Use source_message_ids only from the provided message IDs.",
            (
                "Use wire sentinels when a field is not applicable or should not "
                'be changed: omit any field other than action, or use "" for '
                "absent string fields; -1.0 for absent confidence fields; -1 "
                "for absent privacy_level."
            ),
            (
                'intimacy_boundary="" means not provided/no boundary change; '
                'intimacy_boundary="ordinary" means explicitly ordinary.'
            ),
            f"Valid intimacy_boundary values are exactly: {boundary_values}.",
            (
                "For each created or updated topic, set intimacy_boundary "
                "semantically. Use ordinary unless the topic itself is private "
                "romantic/intimate context, an intimacy boundary, or ambiguous "
                "intimate context."
            ),
            (
                "For non-ordinary intimacy_boundary values, set privacy_level "
                "to at least 2 and avoid exposing sensitive wording in topic "
                "titles where a neutral local label is enough."
            ),
            (
                "Use artifact_ids only from provided artifact_refs when a topic "
                "depends on an attachment."
            ),
            f"conversation_id={conversation_id}",
            "<existing_topic_snapshot>",
            json_utils.dumps(snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(message_payload, indent=2, sort_keys=True),
            "</messages>",
        ]
    )


def _plain_card_request(
    *,
    model: str,
    purpose: str,
    conversation_id: str,
    prompt: str,
    shadow_card: str,
    shadow_variant: str = "cards_split_route_v1",
) -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model=model,
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "Maintain a conversation Topic Working Set from data. "
                    "Write only the requested plain-text lines. No JSON. No explanation."
                ),
            ),
            LLMMessage(role="user", content=prompt),
        ],
        max_output_tokens=192,
        metadata={
            "user_id": "benchmark-user",
            "conversation_id": conversation_id,
            "purpose": purpose,
            "topic_working_set_card": shadow_card,
            "topic_working_set_shadow_variant": shadow_variant,
        },
    )


def _build_existing_route_prompt_v1(case: BenchmarkCase) -> str:
    return "\n".join(
        [
            "Decide whether this message batch changes an existing Topic Working Set topic.",
            "Only consider existing topics from the snapshot.",
            "Never create a new topic in this card.",
            "Write one line per touched existing topic, or exactly: none",
            "Allowed actions:",
            "update = the same topic continues and should remain active",
            "park = the user pauses, postpones, defers, or puts this topic aside",
            "reopen = the user resumes a parked topic",
            "close = the user says this topic is done, finished, resolved, or no longer active",
            "Format: action topic_id message_id [message_id ...]",
            "Use only topic ids from the snapshot.",
            "Use only message ids from the provided messages.",
            "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            f"conversation_id={case.conversation_id}",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(
                _message_payload_for_prompt(case.messages),
                indent=2,
                sort_keys=True,
            ),
            "</messages>",
        ]
    )


def _build_new_topic_route_prompt_v1(
    case: BenchmarkCase,
    *,
    existing_routes: tuple[_TopicRoute, ...],
) -> str:
    existing_route_payload = [
        {
            "action": route.action.value,
            "target_id": route.target_id,
            "source_message_ids": list(route.source_message_ids),
        }
        for route in existing_routes
    ]
    return "\n".join(
        [
            "Decide whether this message batch starts a new ongoing Topic Working Set topic.",
            "Only create topics that are not covered by an existing topic route.",
            "Never update, park, reopen, or close existing topics in this card.",
            "Write one create line per new topic, or exactly: none",
            "Create a topic when the message introduces work, a discussion, a decision thread, or attachment-driven context likely to matter later.",
            "Do not create a topic when the batch only continues, pauses, resumes, or closes an existing topic.",
            "For creates, invent a temporary target id like tmp1, tmp2.",
            "Format: create tmp_id message_id [message_id ...]",
            "Use only message ids from the provided messages.",
            "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            f"conversation_id={case.conversation_id}",
            "<existing_routes_already_selected>",
            json_utils.dumps(existing_route_payload, indent=2, sort_keys=True),
            "</existing_routes_already_selected>",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(
                _message_payload_for_prompt(case.messages),
                indent=2,
                sort_keys=True,
            ),
            "</messages>",
        ]
    )


def _build_new_topic_needed_prompt_v2(
    case: BenchmarkCase,
    *,
    existing_routes: tuple[_TopicRoute, ...],
) -> str:
    existing_route_payload = [
        {
            "action": route.action.value,
            "target_id": route.target_id,
            "source_message_ids": list(route.source_message_ids),
        }
        for route in existing_routes
    ]
    return "\n".join(
        [
            "Decide if this message batch needs one new local conversation topic.",
            "Only answer about a topic not already covered by the selected existing routes.",
            "Answer yes when the messages introduce a subject the assistant should keep as local context for later turns in this same conversation.",
            "Subjects include tasks, decisions, problems, plans, personal situations, attachments, or ongoing discussions.",
            "Privacy or sensitivity is not a reason to answer none; another card decides privacy.",
            "Answer none when the batch only continues, pauses, resumes, or closes an existing topic.",
            "Write exactly one line.",
            "Format when needed: yes message_id [message_id ...]",
            "Otherwise write exactly: none",
            "Use only message ids from the provided messages.",
            "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            f"conversation_id={case.conversation_id}",
            "<existing_routes_already_selected>",
            json_utils.dumps(existing_route_payload, indent=2, sort_keys=True),
            "</existing_routes_already_selected>",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(
                _message_payload_for_prompt(case.messages),
                indent=2,
                sort_keys=True,
            ),
            "</messages>",
        ]
    )


def _parse_new_topic_needed_output_v2(
    text: str,
    *,
    valid_message_ids: tuple[str, ...],
) -> tuple[_TopicRoute, ...]:
    lines = _card_lines(text)
    if not lines:
        return ()
    if all(line.strip("`*_.,;[](){}\"'").casefold() == "none" for line in lines):
        return ()
    valid_message_id_set = set(valid_message_ids)
    routes: list[_TopicRoute] = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].strip("`*_.,;[](){}\"'").casefold() != "yes":
            continue
        source_message_ids = _valid_message_ids_from_tokens(
            tokens[1:],
            valid_message_id_set=valid_message_id_set,
        )
        if not source_message_ids:
            continue
        routes.append(
            _TopicRoute(
                action=TopicUpdateActionType.CREATE,
                target_id=f"tmp{len(routes) + 1}",
                source_message_ids=tuple(source_message_ids),
            )
        )
        break
    return tuple(routes)


def _build_new_topic_track_prompt_v3(
    case: BenchmarkCase,
    *,
    existing_routes: tuple[_TopicRoute, ...],
) -> str:
    existing_route_payload = [
        {
            "action": route.action.value,
            "target_id": route.target_id,
            "source_message_ids": list(route.source_message_ids),
        }
        for route in existing_routes
    ]
    return "\n".join(
        [
            "This card only filters obvious no-topic batches.",
            "Default answer: track.",
            "Write ignore only when there is no local subject to carry forward.",
            "Ignore pure greetings, thanks, empty chatter, or messages already covered by selected existing routes.",
            "Track any new subject the assistant may need as local conversation context.",
            "Track tasks, decisions, problems, plans, personal situations, attachments, and ongoing discussions.",
            "Track private or sensitive subjects too; privacy is handled by a later card.",
            "Write exactly one line.",
            "Format when tracking: track message_id [message_id ...]",
            "Otherwise write exactly: ignore",
            "Use only message ids from the provided messages.",
            "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            f"conversation_id={case.conversation_id}",
            "<existing_routes_already_selected>",
            json_utils.dumps(existing_route_payload, indent=2, sort_keys=True),
            "</existing_routes_already_selected>",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<messages>",
            json_utils.dumps(
                _message_payload_for_prompt(case.messages),
                indent=2,
                sort_keys=True,
            ),
            "</messages>",
        ]
    )


def _parse_new_topic_track_output_v3(
    text: str,
    *,
    valid_message_ids: tuple[str, ...],
) -> tuple[_TopicRoute, ...]:
    lines = _card_lines(text)
    if not lines:
        return ()
    if all(line.strip("`*_.,;[](){}\"'").casefold() == "ignore" for line in lines):
        return ()
    valid_message_id_set = set(valid_message_ids)
    routes: list[_TopicRoute] = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].strip("`*_.,;[](){}\"'").casefold() != "track":
            continue
        source_message_ids = _valid_message_ids_from_tokens(
            tokens[1:],
            valid_message_id_set=valid_message_id_set,
        )
        if not source_message_ids:
            continue
        routes.append(
            _TopicRoute(
                action=TopicUpdateActionType.CREATE,
                target_id=f"tmp{len(routes) + 1}",
                source_message_ids=tuple(source_message_ids),
            )
        )
        break
    return tuple(routes)


def _build_new_topic_track_prompt_v4(
    case: BenchmarkCase,
    *,
    uncovered_messages: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "This card only sees messages not already assigned to an existing topic.",
            "Decide whether these remaining messages introduce one new local topic.",
            "Default answer: track.",
            "Write ignore only when there is no local subject to carry forward.",
            "Ignore pure greetings, thanks, empty chatter, or non-subject fragments.",
            "Track any subject the assistant may need as local conversation context.",
            "Track tasks, decisions, problems, plans, personal situations, attachments, and ongoing discussions.",
            "Track private or sensitive subjects too; privacy is handled by a later card.",
            "Write exactly one line.",
            "Format when tracking: track message_id [message_id ...]",
            "Otherwise write exactly: ignore",
            "Use only message ids from the provided messages.",
            "Do not write titles, summaries, goals, privacy, artifacts, or status fields.",
            f"conversation_id={case.conversation_id}",
            "<existing_topic_snapshot>",
            json_utils.dumps(case.snapshot, indent=2, sort_keys=True),
            "</existing_topic_snapshot>",
            "<uncovered_messages>",
            json_utils.dumps(
                _message_payload_for_prompt(uncovered_messages),
                indent=2,
                sort_keys=True,
            ),
            "</uncovered_messages>",
        ]
    )


def _build_boundary_prompt_v5(
    case: BenchmarkCase,
    *,
    route: _TopicRoute,
    content: _TopicContent,
) -> str:
    boundary_values = ", ".join(boundary.value for boundary in IntimacyBoundary)
    source_messages = [
        message
        for message in case.messages
        if str(message.get("id") or "") in set(route.source_message_ids)
    ]
    content_payload = {
        "title": content.title,
        "summary": content.summary,
        "active_goal": content.active_goal,
        "open_questions": list(content.open_questions),
        "decisions": list(content.decisions),
    }
    route_payload = {
        "action": route.action.value,
        "target_id": route.target_id,
        "source_message_ids": list(route.source_message_ids),
    }
    return "\n".join(
        [
            "Classify privacy and intimacy boundary for exactly one topic target.",
            "Always write one line. Never write none.",
            "Do not write JSON.",
            "Use ordinary unless the topic itself is private romantic/intimate context, a stated relationship boundary, or ambiguous intimate context.",
            "For non-ordinary boundaries, use privacy_level at least 2.",
            "If intimate context is present but the exact boundary is unclear, use ambiguous_intimate.",
            f"Valid intimacy_boundary values: {boundary_values}.",
            "Format: target_id intimacy_boundary privacy_level confidence",
            "Examples:",
            "tmp1 ordinary 0 0.7",
            "tpc_private ambiguous_intimate 2 0.8",
            f"conversation_id={case.conversation_id}",
            "<target_route>",
            json_utils.dumps(route_payload, indent=2, sort_keys=True),
            "</target_route>",
            "<target_content_draft>",
            json_utils.dumps(content_payload, indent=2, sort_keys=True),
            "</target_content_draft>",
            "<source_messages>",
            json_utils.dumps(
                _message_payload_for_prompt(source_messages),
                indent=2,
                sort_keys=True,
            ),
            "</source_messages>",
        ]
    )


def _message_payload_for_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(message["id"]),
            "seq": message.get("seq"),
            "role": str(message["role"]),
            "text": TopicWorkingSetUpdater._message_text_for_topic_prompt(message),
            "raw_text_included": TopicWorkingSetUpdater._message_raw_text_allowed(message),
            "content_kind": message.get("content_kind") or "text",
            "policy_reason": message.get("policy_reason") or "normal",
            "created_at": message.get("created_at"),
            "artifact_refs": TopicWorkingSetUpdater._message_artifact_refs(message),
        }
        for message in messages
    ]


def normalize_plan(plan: TopicWorkingSetPlan) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for action in plan.actions:
        if action.action is TopicUpdateActionType.NOOP:
            continue
        actions.append(
            {
                "action": action.action.value,
                "topic_id": _none_if_empty(action.topic_id),
                "parent_topic_id": _none_if_empty(action.parent_topic_id),
                "title": _none_if_empty(action.title),
                "summary": _none_if_empty(action.summary),
                "active_goal": _none_if_empty(action.active_goal),
                "open_questions": _dedupe_strings(action.open_questions),
                "decisions": _dedupe_strings(action.decisions),
                "artifact_ids": _dedupe_strings(action.artifact_ids),
                "source_message_ids": _dedupe_strings(action.source_message_ids),
                "confidence": _none_if_float_sentinel(action.confidence),
                "privacy_level": _none_if_int_sentinel(action.privacy_level),
                "intimacy_boundary": _none_if_empty(action.intimacy_boundary),
                "intimacy_boundary_confidence": _none_if_float_sentinel(
                    action.intimacy_boundary_confidence
                ),
            }
        )
    return {
        "nothing_to_update": bool(plan.nothing_to_update),
        "actions": actions,
    }


def route_signature(plan: dict[str, Any]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for index, action in enumerate(plan.get("actions") or [], start=1):
        action_name = str(action.get("action") or "")
        target = (
            f"create:{index}"
            if action_name == TopicUpdateActionType.CREATE.value
            else str(action.get("topic_id") or "")
        )
        signature.append(
            {
                "action": action_name,
                "target": target,
                "source_message_ids": sorted(action.get("source_message_ids") or []),
                "artifact_ids": sorted(action.get("artifact_ids") or []),
                "privacy_level": action.get("privacy_level"),
                "intimacy_boundary": action.get("intimacy_boundary"),
            }
        )
    return signature


def score_plan(plan: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    output_actions = list(plan.get("actions") or [])
    expected_actions = list(expected.get("actions") or [])
    used_output_indexes: set[int] = set()
    matches: list[dict[str, Any]] = []
    missing_actions: list[dict[str, Any]] = []
    requirement_failures: list[dict[str, Any]] = []

    for expected_action in expected_actions:
        match_index = _find_matching_action(
            output_actions,
            expected_action,
            used_output_indexes=used_output_indexes,
        )
        if match_index is None:
            missing_actions.append(expected_action)
            continue
        used_output_indexes.add(match_index)
        output_action = output_actions[match_index]
        failures = _requirement_failures(output_action, expected_action)
        requirement_failures.extend(failures)
        matches.append(
            {
                "expected": expected_action,
                "output": output_action,
                "requirement_failures": failures,
            }
        )

    unexpected_actions = [
        action
        for index, action in enumerate(output_actions)
        if index not in used_output_indexes
    ]
    nothing_expected = bool(expected.get("nothing_to_update", False))
    nothing_ok = (
        bool(plan.get("nothing_to_update"))
        if nothing_expected
        else not bool(plan.get("nothing_to_update")) or bool(expected_actions) is False
    )
    exact_match = (
        not missing_actions
        and not unexpected_actions
        and not requirement_failures
        and nothing_ok
        and len(output_actions) == len(expected_actions)
    )
    return {
        "exact_match": exact_match,
        "expected_action_count": len(expected_actions),
        "output_action_count": len(output_actions),
        "matched_action_count": len(matches),
        "missing_actions": missing_actions,
        "unexpected_actions": unexpected_actions,
        "requirement_failures": requirement_failures,
        "matches": matches,
        "nothing_to_update_ok": nothing_ok,
        "required_action_recall": _safe_div(len(matches), len(expected_actions)),
    }


def project_topic_state(snapshot: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    topics: dict[str, dict[str, Any]] = {}
    for status, key in (("active", "active_topics"), ("parked", "parked_topics")):
        for row in snapshot.get(key) or []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            topic_id = str(row["id"])
            topics[topic_id] = {
                "id": topic_id,
                "status": str(row.get("status") or status),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "active_goal": row.get("active_goal"),
                "open_questions": list(row.get("open_questions") or []),
                "decisions": list(row.get("decisions") or []),
                "artifact_ids": list(row.get("artifact_ids") or []),
                "privacy_level": row.get("privacy_level"),
                "intimacy_boundary": row.get("intimacy_boundary"),
            }
    created_count = 0
    for action in plan.get("actions") or []:
        action_name = str(action.get("action") or "")
        if action_name == TopicUpdateActionType.CREATE.value:
            created_count += 1
            topic_id = f"created_{created_count}"
            topics[topic_id] = {
                "id": topic_id,
                "status": "active",
                "title": action.get("title"),
                "summary": action.get("summary") or "",
                "active_goal": action.get("active_goal"),
                "open_questions": list(action.get("open_questions") or []),
                "decisions": list(action.get("decisions") or []),
                "artifact_ids": list(action.get("artifact_ids") or []),
                "privacy_level": action.get("privacy_level", 0),
                "intimacy_boundary": action.get("intimacy_boundary") or "ordinary",
            }
            continue
        topic_id = str(action.get("topic_id") or "")
        if not topic_id or topic_id not in topics:
            continue
        topic = topics[topic_id]
        if action_name == TopicUpdateActionType.PARK.value:
            topic["status"] = "parked"
        elif action_name == TopicUpdateActionType.REOPEN.value:
            topic["status"] = "active"
        elif action_name == TopicUpdateActionType.CLOSE.value:
            topic["status"] = "closed"
        for field_name in (
            "title",
            "summary",
            "active_goal",
            "privacy_level",
            "intimacy_boundary",
        ):
            if action.get(field_name) is not None:
                topic[field_name] = action[field_name]
        for list_field in ("open_questions", "decisions", "artifact_ids"):
            if action.get(list_field):
                topic[list_field] = list(action[list_field])
    rows = sorted(topics.values(), key=lambda row: str(row["id"]))
    return {
        "active_topics": [row for row in rows if row["status"] == "active"],
        "parked_topics": [row for row in rows if row["status"] == "parked"],
        "closed_topics": [row for row in rows if row["status"] == "closed"],
    }


def summarize_run(
    rows: list[dict[str, Any]],
    *,
    recorder: LLMCallRecorder,
    variants: tuple[VariantName, ...],
    started_at: datetime,
    finished_at: datetime,
    json_model: str,
    card_model: str,
    cases_path: Path,
) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        latencies = [float(row["wall_time_ms"]) for row in variant_rows]
        exact = sum(1 for row in variant_rows if row["score"]["exact_match"])
        failed = sum(1 for row in variant_rows if row.get("error"))
        parse_invalid = sum(int(row.get("parse_invalid_count") or 0) for row in variant_rows)
        recall_values = [
            float(row["score"].get("required_action_recall") or 0.0)
            for row in variant_rows
        ]
        llm_records = recorder.records_for_context(variant=variant)
        by_variant[variant] = {
            "cases": len(variant_rows),
            "exact_match_count": exact,
            "exact_match_rate": _safe_div(exact, len(variant_rows)),
            "mean_required_action_recall": (
                sum(recall_values) / len(recall_values) if recall_values else 1.0
            ),
            "failed_trials": failed,
            "parse_invalid_count": parse_invalid,
            "wall_time_ms": _latency_summary(latencies),
            "llm_call_summary": summarize_llm_calls(llm_records),
            "mismatch_case_ids": [
                row["case_id"]
                for row in variant_rows
                if not row["score"]["exact_match"]
            ],
        }
    return {
        "benchmark": "topic_working_set_cards",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "cases_path": str(cases_path),
        "json_model": json_model,
        "card_model": card_model,
        "variants": by_variant,
        "pairwise": pairwise_disagreements(rows),
        "llm_call_summary": recorder.summary(),
    }


def pairwise_disagreements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], int(row["repetition"])), {})[
            row["variant"]
        ] = row
    disagreements: list[dict[str, Any]] = []
    for (case_id, repetition), variants in sorted(grouped.items()):
        json_row = variants.get("json_legacy")
        card_row = variants.get("cards_current")
        if json_row is None or card_row is None:
            continue
        route_equal = json_row["route_signature"] == card_row["route_signature"]
        projection_equal = json_row["topic_projection"] == card_row["topic_projection"]
        if route_equal and projection_equal:
            continue
        disagreements.append(
            {
                "case_id": case_id,
                "repetition": repetition,
                "route_signature_equal": route_equal,
                "topic_projection_equal": projection_equal,
                "json_legacy_signature": json_row["route_signature"],
                "cards_current_signature": card_row["route_signature"],
            }
        )
    return {
        "comparable_pairs": sum(
            1
            for variants in grouped.values()
            if "json_legacy" in variants and "cards_current" in variants
        ),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def load_cases(path: Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            case = BenchmarkCase(
                case_id=str(payload.get("id") or f"case_{line_number}"),
                conversation_id=str(
                    payload.get("conversation_id") or "topic_working_set_cards"
                ),
                snapshot=dict(payload.get("snapshot") or {}),
                messages=list(payload.get("messages") or []),
                expected=dict(payload.get("expected") or {}),
                notes=str(payload.get("notes") or ""),
            )
            _validate_case(case, line_number=line_number)
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _parse_variants(raw: str) -> tuple[VariantName, ...]:
    variants: list[VariantName] = []
    allowed = set(_ALLOWED_VARIANTS)
    for piece in raw.split(","):
        value = piece.strip()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unknown variant {value!r}; allowed: {sorted(allowed)}")
        variants.append(cast(VariantName, value))
    if not variants:
        raise ValueError("At least one variant is required")
    return tuple(variants)


def _validate_case(case: BenchmarkCase, *, line_number: int) -> None:
    if not case.messages:
        raise ValueError(f"case line {line_number}: messages must not be empty")
    for message in case.messages:
        for key in ("id", "role", "text"):
            if key not in message:
                raise ValueError(f"case line {line_number}: message missing {key!r}")
    for expected_action in case.expected.get("actions") or []:
        action = expected_action.get("action")
        if action not in {action_type.value for action_type in TopicUpdateActionType}:
            raise ValueError(f"case line {line_number}: unknown expected action {action!r}")


def _find_matching_action(
    output_actions: list[dict[str, Any]],
    expected_action: dict[str, Any],
    *,
    used_output_indexes: set[int],
) -> int | None:
    expected_name = str(expected_action.get("action") or "")
    expected_topic_id = expected_action.get("topic_id")
    for index, output_action in enumerate(output_actions):
        if index in used_output_indexes:
            continue
        if output_action.get("action") != expected_name:
            continue
        if expected_topic_id and output_action.get("topic_id") != expected_topic_id:
            continue
        if expected_name != TopicUpdateActionType.CREATE.value and expected_topic_id is None:
            continue
        return index
    return None


def _requirement_failures(
    output_action: dict[str, Any],
    expected_action: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for field_name in ("source_message_ids", "artifact_ids"):
        required_values = set(expected_action.get(field_name) or [])
        if not required_values:
            continue
        output_values = set(output_action.get(field_name) or [])
        missing_values = sorted(required_values - output_values)
        if missing_values:
            failures.append({"field": field_name, "missing": missing_values})
    if expected_action.get("requires_title") and not output_action.get("title"):
        failures.append({"field": "title", "reason": "required"})
    min_privacy_level = expected_action.get("min_privacy_level")
    if min_privacy_level is not None:
        output_privacy = output_action.get("privacy_level")
        if output_privacy is None or int(output_privacy) < int(min_privacy_level):
            failures.append(
                {
                    "field": "privacy_level",
                    "minimum": int(min_privacy_level),
                    "actual": output_privacy,
                }
            )
    expected_boundary = expected_action.get("intimacy_boundary")
    if expected_boundary and output_action.get("intimacy_boundary") != expected_boundary:
        failures.append(
            {
                "field": "intimacy_boundary",
                "expected": expected_boundary,
                "actual": output_action.get("intimacy_boundary"),
            }
        )
    forbidden_boundary = expected_action.get("intimacy_boundary_not")
    if forbidden_boundary and output_action.get("intimacy_boundary") == forbidden_boundary:
        failures.append(
            {
                "field": "intimacy_boundary",
                "forbidden": forbidden_boundary,
                "actual": output_action.get("intimacy_boundary"),
            }
        )
    return failures


def _request_with_model(
    request: LLMCompletionRequest,
    *,
    model: str,
) -> LLMCompletionRequest:
    return request.model_copy(update={"model": model})


def _dedupe_routes(routes: list[_TopicRoute]) -> tuple[_TopicRoute, ...]:
    deduped: list[_TopicRoute] = []
    seen: set[tuple[str, str]] = set()
    for route in routes:
        key = (route.action.value, route.target_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return tuple(deduped)


def _messages_not_covered_by_routes(
    messages: list[dict[str, Any]],
    routes: tuple[_TopicRoute, ...],
) -> list[dict[str, Any]]:
    covered_message_ids = {
        message_id
        for route in routes
        for message_id in route.source_message_ids
    }
    return [
        message
        for message in messages
        if str(message.get("id") or "") not in covered_message_ids
    ]


def _artifact_ids_from_route_messages(
    *,
    case: BenchmarkCase,
    routes: tuple[_TopicRoute, ...],
) -> dict[str, tuple[str, ...]]:
    messages_by_id = {
        str(message["id"]): message
        for message in case.messages
        if message.get("id")
    }
    artifacts_by_target: dict[str, tuple[str, ...]] = {}
    for route in routes:
        artifact_ids: list[str] = []
        seen: set[str] = set()
        for message_id in route.source_message_ids:
            message = messages_by_id.get(message_id)
            if message is None:
                continue
            for artifact_ref in TopicWorkingSetUpdater._message_artifact_refs(message):
                artifact_id = str(artifact_ref.get("artifact_id") or "").strip()
                if not artifact_id or artifact_id in seen:
                    continue
                seen.add(artifact_id)
                artifact_ids.append(artifact_id)
        if artifact_ids:
            artifacts_by_target[route.target_id] = tuple(artifact_ids)
    return artifacts_by_target


def _jsonable_card_result(result: CardResult) -> dict[str, Any]:
    return {
        "card_name": result.card_name,
        "target_id": result.target_id,
        "raw_output": result.raw_output,
        "parsed": _json_safe(result.parsed),
        "parse_valid": result.parse_valid,
        "error": result.error,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, _TopicRoute):
        return {
            "action": value.action.value,
            "target_id": value.target_id,
            "source_message_ids": list(value.source_message_ids),
        }
    if isinstance(value, _TopicContent):
        return {
            "title": value.title,
            "summary": value.summary,
            "active_goal": value.active_goal,
            "open_questions": list(value.open_questions),
            "decisions": list(value.decisions),
        }
    if isinstance(value, _TopicBoundary):
        return {
            "boundary": value.boundary.value,
            "privacy_level": value.privacy_level,
            "confidence": value.confidence,
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, TopicUpdateActionType):
        return value.value
    if isinstance(value, IntimacyBoundary):
        return value.value
    return value


def _plain_card_output_valid(raw_output: str | None, has_signal: bool) -> bool:
    if raw_output is None:
        return False
    stripped = raw_output.strip()
    if not stripped:
        return False
    return has_signal or stripped.strip("`*_.,;[](){}\"'").casefold() == "none"


def _required_card_output_valid(raw_output: str | None, has_signal: bool) -> bool:
    return bool(raw_output is not None and raw_output.strip() and has_signal)


def _none_if_empty(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _none_if_float_sentinel(value: float) -> float | None:
    return None if value == -1.0 else float(value)


def _none_if_int_sentinel(value: int) -> int | None:
    return None if value == -1 else int(value)


def _dedupe_strings(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "mean": None, "max": None}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 0.50),
        "p95": _percentile(sorted_values, 0.95),
        "mean": sum(sorted_values) / len(sorted_values),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(
        len(sorted_values) - 1,
        max(0, round((len(sorted_values) - 1) * percentile)),
    )
    return sorted_values[index]


def _error_message(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "reason": getattr(exc, "reason", None),
        "details": list(getattr(exc, "details", ()) or ()),
    }


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
