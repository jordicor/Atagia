"""One frozen native-link challenger vs the imported production card.

Known regression fixtures are reported separately from an independently authored
generalization set. Repeats and option/ID perturbations are not new independent
cases. This isolates linking, not the complete consequence detector.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter
from typing import Any, Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.consequence_detector import ConsequenceDetector, _decode_native_choice
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import LLMCompletionRequest, RetryPolicy
from atagia.services.providers import build_llm_client
from atagia.services.prompt_authority import process_authority_context

from benchmarks.consequence_detection_cards.compare import _DEFAULT_CASES_PATH, load_cases
from benchmarks.consequence_detection_cards.link_challenger import build_challenger_request
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder
from benchmarks.output_root import assert_outside_repo

_NEW_CASES = Path(__file__).with_name("link_generalization_cases.jsonl")
_AUTHORITY = process_authority_context(
    privacy_enforcement="off", user_id="bench_user", purpose="consequence_detection"
)


class LinkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: str
    message: str
    recent_assistant_messages: list[dict[str, str]]
    expected_link_id: str | None
    notes: str = ""
    cohort: Literal["existing", "new"] = "new"

    @model_validator(mode="after")
    def validate_labels(self) -> LinkCase:
        ids = [message["id"] for message in self.recent_assistant_messages]
        if any(not item or item == "none" for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique, nonempty, and not 'none'")
        if self.expected_link_id is not None and self.expected_link_id not in ids:
            raise ValueError("Expected link must be an eligible candidate or null")
        if not self.case_id or not self.message:
            raise ValueError("Case ID and message must not be empty")
        return self


def load_link_cases(path: Path) -> list[LinkCase]:
    cases = [LinkCase.model_validate_json(line) for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines() if line.strip()]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Require nonempty cases with unique IDs")
    return cases


def known_cases() -> list[LinkCase]:
    return [LinkCase(
        case_id=case.case_id,
        category="historical",
        message=case.message,
        recent_assistant_messages=list(case.recent_assistant_messages),
        expected_link_id=case.expected_link_id,
        cohort="existing",
    ) for case in load_cases(_DEFAULT_CASES_PATH) if case.expected_is_consequence]


def relabel_case(case: LinkCase) -> LinkCase:
    """Change opaque IDs, NOT chronology or text; option order is also reversed."""
    # An ID literally quoted in the feedback is part of the input relationship.
    # Keep those IDs intact; reversing the option order remains a valid probe.
    if any(message["id"] in case.message for message in case.recent_assistant_messages):
        return case
    mapping = {
        message["id"]: f"candidate_{len(case.recent_assistant_messages) - index:03d}"
        for index, message in enumerate(case.recent_assistant_messages)
    }
    return case.model_copy(update={
        "recent_assistant_messages": [
            {**message, "id": mapping[message["id"]]}
            for message in case.recent_assistant_messages
        ],
        "expected_link_id": mapping.get(case.expected_link_id),
    })


def production_request(detector: ConsequenceDetector, case: LinkCase) -> LLMCompletionRequest:
    return detector._card_request(
        card_name="link", message_text=case.message, role="user",
        conversation_context=ExtractionConversationContext(
            user_id="bench_user", conversation_id="bench_link",
            source_message_id="bench_feedback", workspace_id="bench_workspace",
            assistant_mode_id="general_qa", recent_messages=[], privacy_enforcement="off",
        ),
        recent_assistant_messages=cast(list[dict[str, Any]], case.recent_assistant_messages),
        authority_context=_AUTHORITY,
    )


def build_requests(detector: ConsequenceDetector, case: LinkCase, *, reversed_options: bool):
    before = production_request(detector, case)
    after = build_challenger_request(
        before, message_text=case.message, role="user",
        recent_assistant_messages=cast(list[dict[str, Any]], case.recent_assistant_messages),
        authority_context=_AUTHORITY,
    )
    requests = {"before": before, "after": after}
    if reversed_options:
        for request in requests.values():
            assert request.choice_questions is not None
            question = request.choice_questions["likely_action_message_id"]
            question.criteria = dict(reversed(tuple(question.criteria.items())))
    return requests


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["error"] is None]
    return {
        "attempts": len(rows),
        "trials": len(successful),
        "matches": sum(row["correct"] for row in successful),
        "technical_failures": sum(row["error"] is not None for row in rows),
        "incorrect_positive_links": sum(
            row["error"] is None and row["selected"] is not None and not row["correct"]
            for row in rows
        ),
        "false_abstentions": sum(
            row["error"] is None and row["selected"] is None
            and row["expected"] is not None for row in rows
        ),
        "latency_p50_ms": median(row["elapsed_ms"] for row in successful) if successful else None,
        "input_tokens": sum(row["usage"].get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row["usage"].get("output_tokens", 0) for row in rows),
    }


def completed_keys(
    rows: list[dict[str, Any]], *, retry_transient: bool
) -> set[tuple[str, str, str]]:
    """Only authorized transient errors may precede a repeated trial slot."""
    completed: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["case_id"], row["variant"], row["arm"])
        if key in completed:
            raise ValueError("Duplicate response: semantic results must never be retried")
        if row["error"] is None:
            completed.add(key)
        elif not retry_transient or row["error"]["type"] != "TransientLLMError":
            raise ValueError("Continuation requires an authorized transient failure")
    return completed


async def run(args: argparse.Namespace) -> None:
    load_dotenv()
    new_cases = load_link_cases(Path(args.cases))
    cases = known_cases() + new_cases
    output_dir = assert_outside_repo(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Settings.from_env()
    settings = replace(
        base, llm_forced_global_model=None, llm_finite_decisions_enabled=True,
        llm_component_models={**base.llm_component_models, "consequence_link": args.model},
        llm_request_timeout_seconds=30.0, llm_output_limit_retry_attempts=0,
        llm_run_guard_enabled=True, llm_run_guard_mode="enforce",
        llm_run_guard_max_total_calls=200, llm_run_guard_max_total_tokens=250_000,
        llm_run_guard_max_reported_cost_usd=0.15, llm_run_guard_max_total_failed_calls=1,
    )
    if len(cases) > 40:
        raise ValueError("This bounded comparison accepts at most 40 independent cases")
    # Request construction is offline. All cases, prompts, and order are retained
    # before the first paid call; expected labels never enter model input.
    detector = ConsequenceDetector(
        llm_client=cast(Any, None),
        clock=FrozenClock(datetime(2026, 9, 17, tzinfo=timezone.utc)), settings=settings,
    )
    trials = []
    for repeat in range(2):
        for case in cases:
            trials.append((case, f"base_{repeat}", False))
    for case in new_cases:
        trials.append((relabel_case(case), "relabeled", True))
    prepared = []
    for index, (case, variant, reversed_options) in enumerate(trials):
        requests = build_requests(detector, case, reversed_options=reversed_options) if (
            case.recent_assistant_messages
        ) else {}
        prepared.append({
            "case": case.model_dump(), "variant": variant,
            "order": ["before", "after"] if index % 2 == 0 else ["after", "before"],
            "requests": {arm: request.model_dump(mode="json") for arm, request in requests.items()},
        })
    source_paths = [Path(__file__), Path(__file__).with_name("link_challenger.py"),
                    Path("src/atagia/memory/consequence_detector.py"),
                    Path(args.cases), _DEFAULT_CASES_PATH]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "code_head": subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "independent_cases": len(cases), "new_cases": len(new_cases),
        "trials_per_arm": len(trials), "repetitions": 2,
        "perturbation": "New cases also relabel IDs and reverse options once; chronology unchanged.",
        "scope": "Link card only; no free-text extraction or end-to-end quality claim.",
        "promotion_rule": "No historical regressions; new-case accuracy not lower; no increase in incorrect positive links on absent/ambiguous targets. Report all changes; one frozen challenger, no retuning.",
        "jev_input_price_estimate_usd_per_million": 0.042,
    }
    rows: list[dict[str, Any]] = []
    prior_calls: list[dict[str, Any]] = []
    if args.resume_from:
        prior_dir = Path(args.resume_from)
        prior_manifest = json.loads((prior_dir / "manifest.json").read_text(encoding="utf-8"))
        prior_summary = json.loads((prior_dir / "summary.json").read_text(encoding="utf-8"))
        retry_not_before = prior_summary.get("retry_not_before")
        if retry_not_before and datetime.now(timezone.utc) < datetime.fromisoformat(retry_not_before):
            raise ValueError(f"Provider cooldown has not elapsed: {retry_not_before}")
        prior_requests = json.loads((prior_dir / "requests.json").read_text(encoding="utf-8"))
        if prior_requests != prepared:
            raise ValueError("Refusing continuation after any input or request change")
        # The orchestration file changes to add continuation; the prompt,
        # production builder, and both fixture sources must remain identical.
        for path in source_paths[1:]:
            key = str(path)
            if manifest["source_sha256"][key] != prior_manifest["source_sha256"][key]:
                raise ValueError(f"Frozen source changed: {key}")
        rows = json.loads((prior_dir / "per_case.json").read_text(encoding="utf-8"))
        prior_calls = json.loads((prior_dir / "llm_calls.json").read_text(encoding="utf-8"))
        manifest["resumed_from"] = str(prior_dir.resolve())
        manifest["prior_attempts_retained"] = len(rows)
        manifest["continuation_policy"] = "Retain all attempts; skip every semantic result; retry only authorized transient failures, once per continuation. Stop on any new technical failure."
    already_completed = completed_keys(rows, retry_transient=args.retry_transient)
    attempt_counts = Counter((row["case_id"], row["variant"], row["arm"]) for row in rows)
    # Explicit arrays retain option order even though the atomic JSON writer
    # sorts object keys. Historical requests can also be reconstructed from
    # the frozen sources and the recorded variant; chronology never changes.
    write_json_atomic(output_dir / "option_order.json", [{
        "case_id": trial["case"]["case_id"], "variant": trial["variant"],
        "arms": {arm: list(request["choice_questions"]["likely_action_message_id"]["criteria"])
                 for arm, request in trial["requests"].items()},
    } for trial in prepared])
    write_json_atomic(output_dir / "requests.json", prepared)
    write_json_atomic(output_dir / "manifest.json", manifest)
    remaining_calls = 200 - len(prior_calls)
    remaining_tokens = 250_000 - sum(
        int(call.get("token_counts", {}).get("total_tokens", 0)) for call in prior_calls
    )
    remaining_cost = 0.15 - sum(
        float(call.get("cost_counts", {}).get("cost", 0)) for call in prior_calls
    )
    if min(remaining_calls, remaining_tokens, remaining_cost) <= 0:
        raise ValueError("The cumulative experiment budget is exhausted; no further calls allowed")
    settings = replace(
        settings, llm_run_guard_max_total_calls=remaining_calls,
        llm_run_guard_max_total_tokens=remaining_tokens,
        llm_run_guard_max_reported_cost_usd=remaining_cost,
    )
    client = build_llm_client(settings, retry_policy=RetryPolicy(attempts=1))
    client._interactive_retry_policy = RetryPolicy(attempts=1)
    client._extraction_retry_policy = RetryPolicy(attempts=1)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    complete = False
    try:
        for trial in prepared:
            case = trial["case"]
            for arm in trial["order"]:
                key = (case["case_id"], trial["variant"], arm)
                if key in already_completed:
                    continue
                started = perf_counter()
                selected = None
                error = None
                answers = {}
                usage = {}
                try:
                    if trial["requests"]:
                        request = LLMCompletionRequest.model_validate(trial["requests"][arm])
                        with recorder.context(arm=arm, case_id=case["case_id"], variant=trial["variant"]):
                            response = await client.complete(request)
                        choice = _decode_native_choice(request, response.choice_answers)
                        selected = None if choice == "none" else choice
                        answers = {key: value.model_dump() for key, value in response.choice_answers.items()}
                        usage = response.usage
                except Exception as exc:  # noqa: BLE001 - retain evidence, then stop
                    error = {"type": type(exc).__name__, "message": str(exc),
                             "retry_after_seconds": getattr(exc, "retry_after_seconds", None)}
                attempt_counts[key] += 1
                rows.append({
                    "case_id": case["case_id"], "cohort": case["cohort"],
                    "category": case["category"], "variant": trial["variant"], "arm": arm,
                    "expected": case["expected_link_id"], "selected": selected,
                    "correct": error is None and selected == case["expected_link_id"],
                    "error": error, "answers": answers, "usage": usage,
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "skipped_empty_pool": not bool(trial["requests"]),
                    "attempt": attempt_counts[key],
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
                write_json_atomic(output_dir / "per_case.json", rows)
                write_json_atomic(output_dir / "llm_calls.json", prior_calls + [
                    {**call, "sequence": len(prior_calls) + index + 1}
                    for index, call in enumerate(recorder.records())
                ])
                print(f"{arm} {case['case_id']} {trial['variant']} correct={rows[-1]['correct']} error={error is not None}", flush=True)
                if error is not None:
                    raise RuntimeError("Stopped on the first technical failure; artifacts retained")
        complete = True
    finally:
        write_json_atomic(output_dir / "run_guard.json", client.llm_run_guard_snapshot())
        await client.aclose()
        summary = {
            **manifest, "complete": complete, "finished_at": datetime.now(timezone.utc).isoformat(),
            "arms": {arm: {
                "all": summarize([row for row in rows if row["arm"] == arm]),
                **{cohort: summarize([row for row in rows if row["arm"] == arm and row["cohort"] == cohort]) for cohort in ("existing", "new")},
            } for arm in ("before", "after")},
        }
        if not complete and rows and rows[-1]["error"] is not None:
            last_error = rows[-1]["error"]
            summary["retryable"] = last_error["type"] == "TransientLLMError"
            if summary["retryable"]:
                summary["retry_not_before"] = (
                    datetime.now(timezone.utc)
                    + timedelta(seconds=max(600, last_error.get("retry_after_seconds") or 0))
                ).isoformat()
        write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "arms": summary["arms"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(_NEW_CASES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="typesafe/jev-latest")
    parser.add_argument("--resume-from", help="Continue unchanged requests, preserving all prior evidence")
    parser.add_argument("--retry-transient", action="store_true",
                        help="Authorized only: retry transient failures, never semantic results")
    args = parser.parse_args()
    if args.model != "typesafe/jev-latest":
        parser.error("This comparison is bounded to the native Jev backend")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
