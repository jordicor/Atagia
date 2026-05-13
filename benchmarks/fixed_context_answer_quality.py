"""Fixed-context answer quality ceiling runs for benchmark answers."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.broad_list_coverage import (
    AnswerRecord,
    CoverageSpec,
    StrictCoverageJudge,
    StrictCoverageVerdict,
    deterministic_coverage_verdict,
    load_locomo_records,
    load_model_sensitivity_records,
    load_specs,
)
from benchmarks.llm_metrics import summarize_llm_calls
from atagia.core.config import Settings
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.providers import build_llm_client


class AnswerProfile(BaseModel):
    """One answer model/budget profile for a fixed-context ceiling run."""

    model_config = ConfigDict(extra="forbid")

    label: str
    model: str
    max_output_tokens: int


class FixedContextAnswerResult(BaseModel):
    """One answer generated from an already-selected context."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question_id: str
    question_text: str
    ground_truth: str
    source_kind: str
    report_source: str
    source_answer_label: str | None = None
    answer_profile: str
    answer_model: str
    max_output_tokens: int
    context_chars: int
    answer: str = ""
    answer_error: str | None = None
    response_model: str | None = None
    elapsed_ms: float | None = None
    answer_usage: dict[str, Any] = Field(default_factory=dict)
    deterministic_verdict: StrictCoverageVerdict | None = None
    strict_verdict: StrictCoverageVerdict | None = None
    strict_judge_model: str | None = None


class FixedContextAnswerQualityReport(BaseModel):
    """Aggregate fixed-context answer quality report."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    spec_file: str
    input_reports: list[str]
    strict_judge_model: str | None
    answer_profiles: list[AnswerProfile]
    total_cases: int
    total_answer_generations: int
    summary_by_profile: dict[str, dict[str, Any]]
    llm_call_summary: dict[str, Any]
    results: list[FixedContextAnswerResult]


@dataclass(frozen=True)
class _RunCase:
    record: AnswerRecord
    case_id: str


def parse_answer_profile(raw: str) -> AnswerProfile:
    """Parse LABEL=MODEL[:MAX_OUTPUT_TOKENS]."""
    if "=" not in raw:
        raise ValueError("Answer profile must use LABEL=MODEL[:MAX_OUTPUT_TOKENS]")
    label, raw_model = raw.split("=", 1)
    label = label.strip()
    raw_model = raw_model.strip()
    if not label or not raw_model:
        raise ValueError("Answer profile label and model are required")
    model = raw_model
    max_output_tokens = 8192
    if ":" in raw_model:
        model, raw_tokens = raw_model.rsplit(":", 1)
        model = model.strip()
        try:
            max_output_tokens = int(raw_tokens)
        except ValueError as exc:
            raise ValueError("Answer profile max output tokens must be an integer") from exc
    if max_output_tokens < 1:
        raise ValueError("Answer profile max output tokens must be positive")
    return AnswerProfile(
        label=label,
        model=model,
        max_output_tokens=max_output_tokens,
    )


async def run_fixed_context_quality(
    *,
    records: list[AnswerRecord],
    specs: dict[str, CoverageSpec],
    profiles: list[AnswerProfile],
    strict_judge_model: str | None,
    spec_file: str,
    input_reports: list[str],
) -> FixedContextAnswerQualityReport:
    client = build_llm_client(Settings.from_env())
    judge = StrictCoverageJudge(client, strict_judge_model) if strict_judge_model else None
    cases = [
        _RunCase(record=record, case_id=_case_id(record))
        for record in records
        if record.question_id in specs and record.selected_context_text.strip()
    ]
    results: list[FixedContextAnswerResult] = []
    llm_records: list[dict[str, Any]] = []

    for case in cases:
        spec = specs[case.record.question_id]
        for profile in profiles:
            started_at = perf_counter()
            answer = ""
            answer_error = None
            response_model = None
            usage: dict[str, Any] = {}
            try:
                request = LLMCompletionRequest(
                    model=profile.model,
                    messages=_answer_messages(case.record),
                    temperature=0.0,
                    max_output_tokens=profile.max_output_tokens,
                    metadata={
                        "purpose": "benchmark_fixed_context_answer_generation",
                        "question": case.record.question_text,
                    },
                )
                complete = (
                    client.complete_streamed
                    if _uses_streaming_answer_generation(profile)
                    else client.complete
                )
                response = await complete(request)
                answer = response.output_text
                response_model = response.model
                usage = response.usage
            except Exception as exc:  # pragma: no cover - exercised by live runs.
                answer_error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            deterministic_verdict = None
            strict_verdict = None
            if answer:
                deterministic_verdict, _ = deterministic_coverage_verdict(answer, spec)
                if judge is not None:
                    strict_verdict = await judge.judge(
                        question=case.record.question_text,
                        prediction=answer,
                        ground_truth=case.record.ground_truth,
                        required_items=spec.required_items,
                    )
            result = FixedContextAnswerResult(
                case_id=case.case_id,
                question_id=case.record.question_id,
                question_text=case.record.question_text,
                ground_truth=case.record.ground_truth,
                source_kind=case.record.source_kind,
                report_source=case.record.report_source,
                source_answer_label=case.record.answer_label,
                answer_profile=profile.label,
                answer_model=profile.model,
                max_output_tokens=profile.max_output_tokens,
                context_chars=len(case.record.selected_context_text),
                answer=answer,
                answer_error=answer_error,
                response_model=response_model,
                elapsed_ms=elapsed_ms,
                answer_usage=usage,
                deterministic_verdict=deterministic_verdict,
                strict_verdict=strict_verdict,
                strict_judge_model=strict_judge_model if strict_verdict is not None else None,
            )
            results.append(result)
            llm_records.append(
                _llm_record(
                    purpose="benchmark_fixed_context_answer_generation",
                    model=profile.model,
                    response_model=response_model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    error=answer_error,
                )
            )

    return FixedContextAnswerQualityReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        spec_file=spec_file,
        input_reports=input_reports,
        strict_judge_model=strict_judge_model,
        answer_profiles=profiles,
        total_cases=len(cases),
        total_answer_generations=len(results),
        summary_by_profile=_summary_by_profile(results),
        llm_call_summary=summarize_llm_calls(llm_records),
        results=results,
    )


def _answer_messages(record: AnswerRecord) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "You are answering a fixed-context memory benchmark question. "
                "Use only the selected evidence provided by the benchmark. "
                "Include all directly supported items that answer the question, "
                "and do not add claims that are not supported by the evidence."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Selected evidence:\n{record.selected_context_text}\n\n"
                f"Question: {record.question_text}\n"
                "Answer:"
            ),
        ),
    ]


def _uses_streaming_answer_generation(profile: AnswerProfile) -> bool:
    return profile.model.lower().startswith("anthropic/") and profile.max_output_tokens > 8192


def _case_id(record: AnswerRecord) -> str:
    source = Path(record.report_source).stem
    label = record.answer_label or record.source_kind
    return f"{record.question_id}|{label}|{source}"


def _summary_by_profile(results: list[FixedContextAnswerResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[FixedContextAnswerResult]] = {}
    for result in results:
        grouped.setdefault(result.answer_profile, []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for profile, items in sorted(grouped.items()):
        strict_scores = [
            item.strict_verdict.binary_score
            for item in items
            if item.strict_verdict is not None
        ]
        unsupported_claims = sum(
            len(item.strict_verdict.unsupported_claims)
            for item in items
            if item.strict_verdict is not None
        )
        usage_records = [
            _llm_record(
                purpose="benchmark_fixed_context_answer_generation",
                model=item.answer_model,
                response_model=item.response_model,
                usage=item.answer_usage,
                latency_ms=item.elapsed_ms or 0.0,
                error=item.answer_error,
            )
            for item in items
        ]
        summary[profile] = {
            "answers": len(items),
            "errors": sum(1 for item in items if item.answer_error is not None),
            "strict_correct": sum(strict_scores),
            "strict_evaluated": len(strict_scores),
            "unsupported_claim_count": unsupported_claims,
            "llm_call_summary": summarize_llm_calls(usage_records),
        }
    return summary


def _llm_record(
    *,
    purpose: str,
    model: str,
    response_model: str | None,
    usage: dict[str, Any],
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "request_model": model,
        "response_model": response_model,
        "provider": None,
        "latency_ms": latency_ms,
        "token_counts": _token_counts(usage),
        "cost_counts": _cost_counts(usage),
        "error": {"message": error} if error is not None else None,
    }


def _token_counts(usage: dict[str, Any]) -> dict[str, float]:
    return {
        key: value
        for key, value in {
            "input_tokens": _first_number(usage, ("input_tokens",), ("prompt_tokens",)),
            "output_tokens": _first_number(
                usage,
                ("output_tokens",),
                ("completion_tokens",),
            ),
            "thinking_tokens": _first_number(
                usage,
                ("thinking_tokens",),
                ("reasoning_tokens",),
                ("completion_tokens_details", "reasoning_tokens"),
            ),
            "total_tokens": _first_number(usage, ("total_tokens",)),
            "cached_input_tokens": _first_number(
                usage,
                ("cache_read_input_tokens",),
                ("prompt_tokens_details", "cached_tokens"),
            ),
        }.items()
        if isinstance(value, int | float)
    }


def _cost_counts(usage: dict[str, Any]) -> dict[str, float]:
    return {
        key: value
        for key, value in {
            "cost": _first_number(usage, ("cost",)),
            "upstream_inference_cost": _first_number(
                usage,
                ("cost_details", "upstream_inference_cost"),
            ),
        }.items()
        if isinstance(value, int | float)
    }


def _first_number(usage: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = usage
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-context answers across model/budget profiles."
    )
    parser.add_argument("--spec-file", required=True, help="Coverage spec JSON.")
    parser.add_argument("--locomo-report", action="append", default=[])
    parser.add_argument("--model-sensitivity-report", action="append", default=[])
    parser.add_argument(
        "--questions",
        default=None,
        help="Optional comma-separated question ids to include.",
    )
    parser.add_argument(
        "--answer-profile",
        action="append",
        default=[],
        help="Repeatable LABEL=MODEL[:MAX_OUTPUT_TOKENS].",
    )
    parser.add_argument(
        "--strict-judge-model",
        default=None,
        help="Optional strict coverage judge model.",
    )
    parser.add_argument("--output", required=True, help="Output report JSON path.")
    return parser


async def _run_async(args: argparse.Namespace) -> FixedContextAnswerQualityReport:
    spec_path = Path(args.spec_file).expanduser()
    specs = load_specs(spec_path)
    allowed_questions = (
        {item.strip() for item in args.questions.split(",") if item.strip()}
        if args.questions
        else None
    )
    records: list[AnswerRecord] = []
    for raw_path in args.locomo_report:
        records.extend(load_locomo_records(Path(raw_path).expanduser()))
    for raw_path in args.model_sensitivity_report:
        records.extend(load_model_sensitivity_records(Path(raw_path).expanduser()))
    if allowed_questions is not None:
        records = [record for record in records if record.question_id in allowed_questions]
    records = _dedupe_fixed_context_records(records)
    profiles = [parse_answer_profile(item) for item in args.answer_profile]
    if not profiles:
        raise ValueError("At least one --answer-profile is required")
    return await run_fixed_context_quality(
        records=records,
        specs=specs,
        profiles=profiles,
        strict_judge_model=args.strict_judge_model,
        spec_file=str(spec_path),
        input_reports=[*args.locomo_report, *args.model_sensitivity_report],
    )


def _dedupe_fixed_context_records(records: list[AnswerRecord]) -> list[AnswerRecord]:
    deduped: list[AnswerRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            record.question_id,
            record.report_source,
            record.selected_context_text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def main() -> None:
    args = _build_parser().parse_args()
    report = asyncio.run(_run_async(args))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
    )
    print(
        f"Saved {report.total_answer_generations} fixed-context answers "
        f"for {report.total_cases} cases to {output_path}"
    )


if __name__ == "__main__":
    main()
