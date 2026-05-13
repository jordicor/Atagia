"""Compare exact/reconstructed LoCoMo product prompts with fixed evidence prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.broad_list_coverage import (
    CoverageSpec,
    StrictCoverageJudge,
    StrictCoverageVerdict,
    load_specs,
)
from benchmarks.fixed_context_answer_quality import AnswerProfile, parse_answer_profile
from benchmarks.llm_metrics import summarize_llm_calls
from benchmarks.trusted_eval import TRUSTED_EVALUATION_PROMPT_NOTE
from atagia.core.config import Settings
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.services.chat_support import build_system_prompt
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.providers import build_llm_client


_ASSISTANT_MODE_ID = "general_qa"
_DEFAULT_CONTRACT_BLOCK = """[Interaction Contract]
- directness: default
- brevity: default
- clarification_tolerance: default
- pace: default"""


class PromptReplayCase(BaseModel):
    """One frozen-context prompt replay case."""

    model_config = ConfigDict(extra="forbid")

    label: str
    question_id: str
    question_text: str
    ground_truth: str
    memory_block: str
    product_system_prompt: str
    product_prompt_source: str
    product_prompt_exact: bool
    source_artifact: str


class PromptReplayResult(BaseModel):
    """One answer for one case/prompt/model profile."""

    model_config = ConfigDict(extra="forbid")

    case_label: str
    question_id: str
    question_text: str
    ground_truth: str
    prompt_style: str
    product_prompt_source: str
    product_prompt_exact: bool
    answer_profile: str
    answer_model: str
    max_output_tokens: int
    response_model: str | None = None
    answer: str = ""
    answer_error: str | None = None
    elapsed_ms: float | None = None
    answer_usage: dict[str, Any] = Field(default_factory=dict)
    strict_verdict: StrictCoverageVerdict | None = None
    strict_judge_model: str | None = None


class ExactProductPromptReplayReport(BaseModel):
    """Aggregate exact/reconstructed product-prompt replay report."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    spec_file: str
    frozen_contexts_file: str | None
    locomo_reports: list[str]
    strict_judge_model: str | None
    answer_profiles: list[AnswerProfile]
    cases: list[PromptReplayCase]
    summary: dict[str, dict[str, Any]]
    llm_call_summary: dict[str, Any]
    results: list[PromptReplayResult]


async def run_prompt_replay(
    *,
    cases: list[PromptReplayCase],
    specs: dict[str, CoverageSpec],
    profiles: list[AnswerProfile],
    strict_judge_model: str | None,
    spec_file: str,
    frozen_contexts_file: str | None,
    locomo_reports: list[str],
) -> ExactProductPromptReplayReport:
    client = build_llm_client(Settings.from_env())
    judge = StrictCoverageJudge(client, strict_judge_model) if strict_judge_model else None
    results: list[PromptReplayResult] = []
    llm_records: list[dict[str, Any]] = []

    for case in cases:
        if case.question_id not in specs:
            continue
        for prompt_style in ("product", "evidence"):
            messages = _messages_for_prompt_style(case, prompt_style)
            for profile in profiles:
                started_at = perf_counter()
                response_model = None
                answer = ""
                answer_error = None
                usage: dict[str, Any] = {}
                try:
                    request = LLMCompletionRequest(
                        model=profile.model,
                        messages=messages,
                        temperature=0.0,
                        max_output_tokens=profile.max_output_tokens,
                        metadata={
                            "purpose": "benchmark_exact_product_prompt_replay",
                            "question": case.question_text,
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
                except Exception as exc:  # pragma: no cover - live provider path.
                    answer_error = f"{type(exc).__name__}: {exc}"
                elapsed_ms = (perf_counter() - started_at) * 1000.0
                verdict = None
                if judge is not None and answer:
                    spec = specs[case.question_id]
                    verdict = await judge.judge(
                        question=case.question_text,
                        prediction=answer,
                        ground_truth=case.ground_truth,
                        required_items=spec.required_items,
                    )
                result = PromptReplayResult(
                    case_label=case.label,
                    question_id=case.question_id,
                    question_text=case.question_text,
                    ground_truth=case.ground_truth,
                    prompt_style=prompt_style,
                    product_prompt_source=case.product_prompt_source,
                    product_prompt_exact=case.product_prompt_exact,
                    answer_profile=profile.label,
                    answer_model=profile.model,
                    max_output_tokens=profile.max_output_tokens,
                    response_model=response_model,
                    answer=answer,
                    answer_error=answer_error,
                    elapsed_ms=elapsed_ms,
                    answer_usage=usage,
                    strict_verdict=verdict,
                    strict_judge_model=strict_judge_model if verdict is not None else None,
                )
                results.append(result)
                llm_records.append(
                    _llm_record(
                        purpose="benchmark_exact_product_prompt_replay",
                        model=profile.model,
                        response_model=response_model,
                        usage=usage,
                        latency_ms=elapsed_ms,
                        error=answer_error,
                    )
                )

    return ExactProductPromptReplayReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        spec_file=spec_file,
        frozen_contexts_file=frozen_contexts_file,
        locomo_reports=locomo_reports,
        strict_judge_model=strict_judge_model,
        answer_profiles=profiles,
        cases=cases,
        summary=_summary(results),
        llm_call_summary=summarize_llm_calls(llm_records),
        results=results,
    )


def load_frozen_cases(path: Path, labels_by_qid: dict[str, str]) -> list[PromptReplayCase]:
    raw = json.loads(path.read_text())
    cases: list[PromptReplayCase] = []
    for qid, label in labels_by_qid.items():
        item = raw.get(qid)
        if not isinstance(item, dict):
            continue
        system_prompt = str(item.get("system_prompt") or "")
        memory_block = str(item.get("memory_block") or "")
        cases.append(
            PromptReplayCase(
                label=label,
                question_id=qid,
                question_text=str(item.get("question_text") or ""),
                ground_truth=str(item.get("ground_truth") or ""),
                memory_block=memory_block,
                product_system_prompt=system_prompt or _reconstruct_product_prompt(
                    memory_block,
                    trusted_evaluation=True,
                ),
                product_prompt_source=(
                    "stored_exact_system_prompt"
                    if system_prompt
                    else "reconstructed_from_memory_block"
                ),
                product_prompt_exact=bool(system_prompt),
                source_artifact=str(path),
            )
        )
    return cases


def load_locomo_summary_cases(
    paths: list[Path],
    labels_by_qid: dict[str, str],
) -> list[PromptReplayCase]:
    cases: list[PromptReplayCase] = []
    remaining = dict(labels_by_qid)
    for path in paths:
        data = json.loads(path.read_text())
        for conversation in data.get("conversations", []):
            for result in conversation.get("results", []):
                question = result.get("question") or {}
                qid = str(question.get("question_id") or "")
                label = remaining.get(qid)
                if label is None:
                    continue
                trace = result.get("trace") or {}
                memory_block = _context_text(trace.get("selected_memory_summaries"))
                trusted = bool(trace.get("trusted_evaluation"))
                cases.append(
                    PromptReplayCase(
                        label=label,
                        question_id=qid,
                        question_text=str(question.get("question_text") or ""),
                        ground_truth=str(question.get("ground_truth") or ""),
                        memory_block=memory_block,
                        product_system_prompt=_reconstruct_product_prompt(
                            memory_block,
                            trusted_evaluation=trusted,
                        ),
                        product_prompt_source="reconstructed_from_selected_summaries",
                        product_prompt_exact=False,
                        source_artifact=str(path),
                    )
                )
                remaining.pop(qid, None)
    return cases


def _reconstruct_product_prompt(memory_block: str, *, trusted_evaluation: bool) -> str:
    manifest = ManifestLoader(Path("manifests")).load_all()[_ASSISTANT_MODE_ID]
    policy = PolicyResolver().resolve(manifest, None, None)
    prompt = build_system_prompt(
        _ASSISTANT_MODE_ID,
        policy,
        _DEFAULT_CONTRACT_BLOCK,
        "",
        memory_block,
        "",
    )
    if trusted_evaluation:
        prompt = f"{prompt}\n\n{TRUSTED_EVALUATION_PROMPT_NOTE}"
    return prompt


def _messages_for_prompt_style(case: PromptReplayCase, prompt_style: str) -> list[LLMMessage]:
    if prompt_style == "product":
        return [
            LLMMessage(role="system", content=case.product_system_prompt),
            LLMMessage(role="user", content=case.question_text),
        ]
    if prompt_style == "evidence":
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
                    f"Selected evidence:\n{case.memory_block}\n\n"
                    f"Question: {case.question_text}\n"
                    "Answer:"
                ),
            ),
        ]
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def _context_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        for key in ("canonical_preview", "text", "summary", "source_quote"):
            raw = item.get(key)
            if isinstance(raw, str):
                parts.append(raw)
    return "\n".join(parts)


def _summary(results: list[PromptReplayResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[PromptReplayResult]] = {}
    for result in results:
        grouped.setdefault(f"{result.answer_profile}:{result.prompt_style}", []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for key, items in sorted(grouped.items()):
        summary[key] = {
            "answers": len(items),
            "errors": sum(1 for item in items if item.answer_error is not None),
            "strict_correct": sum(
                item.strict_verdict.binary_score
                for item in items
                if item.strict_verdict is not None
            ),
            "strict_evaluated": sum(1 for item in items if item.strict_verdict is not None),
            "unsupported_claim_count": sum(
                len(item.strict_verdict.unsupported_claims)
                for item in items
                if item.strict_verdict is not None
            ),
        }
    return summary


def _uses_streaming_answer_generation(profile: AnswerProfile) -> bool:
    return profile.model.lower().startswith("anthropic/") and profile.max_output_tokens > 8192


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
            "output_tokens": _first_number(usage, ("output_tokens",), ("completion_tokens",)),
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


def _parse_mapping(raw_values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError("Case mapping must use LABEL=QUESTION_ID")
        label, qid = raw.split("=", 1)
        label = label.strip()
        qid = qid.strip()
        if not label or not qid:
            raise ValueError("Case mapping label and question id are required")
        result[qid] = label
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay exact/reconstructed product prompts against fixed evidence prompts."
    )
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--frozen-contexts", default=None)
    parser.add_argument("--frozen-case", action="append", default=[])
    parser.add_argument("--locomo-report", action="append", default=[])
    parser.add_argument("--locomo-summary-case", action="append", default=[])
    parser.add_argument("--answer-profile", action="append", default=[])
    parser.add_argument("--strict-judge-model", default=None)
    parser.add_argument("--output", required=True)
    return parser


async def _run_async(args: argparse.Namespace) -> ExactProductPromptReplayReport:
    spec_path = Path(args.spec_file).expanduser()
    specs = load_specs(spec_path)
    cases: list[PromptReplayCase] = []
    frozen_contexts_file = None
    if args.frozen_contexts is not None:
        frozen_path = Path(args.frozen_contexts).expanduser()
        frozen_contexts_file = str(frozen_path)
        cases.extend(load_frozen_cases(frozen_path, _parse_mapping(args.frozen_case)))
    locomo_paths = [Path(path).expanduser() for path in args.locomo_report]
    cases.extend(load_locomo_summary_cases(locomo_paths, _parse_mapping(args.locomo_summary_case)))
    profiles = [parse_answer_profile(item) for item in args.answer_profile]
    if not cases:
        raise ValueError("At least one replay case is required")
    if not profiles:
        raise ValueError("At least one --answer-profile is required")
    return await run_prompt_replay(
        cases=cases,
        specs=specs,
        profiles=profiles,
        strict_judge_model=args.strict_judge_model,
        spec_file=str(spec_path),
        frozen_contexts_file=frozen_contexts_file,
        locomo_reports=[str(path) for path in locomo_paths],
    )


def main() -> None:
    args = _build_parser().parse_args()
    report = asyncio.run(_run_async(args))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    print(
        f"Saved {len(report.results)} exact-product prompt replay results "
        f"for {len(report.cases)} cases to {output_path}"
    )


if __name__ == "__main__":
    main()
