"""Strict broad-list coverage evaluation for benchmark answers."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atagia.core.config import Settings
from atagia.core.llm_output_limits import GENERIC_JUDGE_MAX_OUTPUT_TOKENS
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
from atagia.services.providers import build_llm_client


class CoverageItem(BaseModel):
    """One required item in an enumerable benchmark answer."""

    model_config = ConfigDict(extra="forbid")

    label: str
    aliases: list[str] = Field(default_factory=list)

    @property
    def match_terms(self) -> list[str]:
        return [self.label, *self.aliases]


class CoverageSpec(BaseModel):
    """Strict item checklist for one benchmark question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    required_items: list[CoverageItem]
    notes: str | None = None
    ground_truth_ambiguous: bool = False


class StrictCoverageVerdict(BaseModel):
    """Structured strict coverage judge result."""

    model_config = ConfigDict(extra="forbid")

    required_items_present: list[str] = Field(default_factory=list)
    required_items_missing: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    binary_score: int = Field(ge=0, le=1)
    rationale: str = ""


class AnswerRecord(BaseModel):
    """One already-generated benchmark answer to evaluate."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    ground_truth: str
    prediction: str
    report_source: str
    source_kind: str
    answer_label: str | None = None
    answer_model: str | None = None
    legacy_judge_score: int | None = None
    legacy_judge_reasoning: str | None = None
    selected_context_text: str = ""
    evidence_context_text: str = ""


class CoverageEvaluation(BaseModel):
    """Coverage evaluation for one answer."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    ground_truth: str
    prediction: str
    report_source: str
    source_kind: str
    answer_label: str | None = None
    answer_model: str | None = None
    legacy_judge_score: int | None = None
    legacy_judge_reasoning: str | None = None
    strict_coverage_score: int
    deterministic_item_checks: dict[str, bool]
    deterministic_verdict: StrictCoverageVerdict
    llm_judge_verdict: StrictCoverageVerdict | None = None
    llm_judge_model: str | None = None
    failure_categories: list[str] = Field(default_factory=list)
    missing_items_selected_context: list[str] = Field(default_factory=list)
    missing_items_evidence_context: list[str] = Field(default_factory=list)
    ground_truth_ambiguous: bool = False


class BroadListCoverageReport(BaseModel):
    """Report separating legacy judge scores from strict item coverage."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    spec_file: str
    input_reports: list[str]
    llm_judge_model: str | None = None
    llm_judge_run: bool = False
    total_answers: int
    legacy_total_correct: int
    strict_total_correct: int
    legacy_pass_strict_fail: int
    strict_pass_legacy_fail: int
    results: list[CoverageEvaluation]


class StrictCoverageJudge:
    """LLM judge that evaluates exact item coverage instead of loose equivalence."""

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    async def judge(
        self,
        *,
        question: str,
        prediction: str,
        ground_truth: str,
        required_items: list[CoverageItem],
    ) -> StrictCoverageVerdict:
        item_labels = [item.label for item in required_items]
        response = await self._llm_client.complete(
            LLMCompletionRequest(
                model=self._judge_model,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a strict coverage judge for broad-list "
                            "memory benchmark answers. A prediction passes only "
                            "when every required item is explicitly present and "
                            "there are no material unsupported claims. Return "
                            "strict JSON only with keys required_items_present, "
                            "required_items_missing, unsupported_claims, "
                            "binary_score, and rationale."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"Question: {question}\n"
                            f"Ground truth: {ground_truth}\n"
                            f"Required items: {json.dumps(item_labels, ensure_ascii=False)}\n"
                            f"Prediction: {prediction}"
                        ),
                    ),
                ],
                max_output_tokens=GENERIC_JUDGE_MAX_OUTPUT_TOKENS,
                response_schema=StrictCoverageVerdict.model_json_schema(),
                metadata={
                    "purpose": "benchmark_broad_list_coverage_judge",
                    "question": question,
                },
            )
        )
        return parse_strict_coverage_verdict(response.output_text)


def parse_strict_coverage_verdict(raw_output: str) -> StrictCoverageVerdict:
    """Parse a strict coverage judge JSON payload."""
    payload = _extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        raise ValueError("Strict coverage judge did not return a JSON object")
    return StrictCoverageVerdict.model_validate(payload)


def deterministic_coverage_verdict(
    prediction: str,
    spec: CoverageSpec,
) -> tuple[StrictCoverageVerdict, dict[str, bool]]:
    """Evaluate required items by exact configured labels/aliases."""
    normalized_prediction = _normalize_for_item_match(prediction)
    checks: dict[str, bool] = {}
    present: list[str] = []
    missing: list[str] = []
    for item in spec.required_items:
        matched = any(
            _normalize_for_item_match(term) in normalized_prediction
            for term in item.match_terms
        )
        checks[item.label] = matched
        if matched:
            present.append(item.label)
        else:
            missing.append(item.label)
    return (
        StrictCoverageVerdict(
            required_items_present=present,
            required_items_missing=missing,
            unsupported_claims=[],
            binary_score=1 if not missing else 0,
            rationale=(
                "All required items matched the deterministic checklist."
                if not missing
                else "Missing required items under deterministic checklist."
            ),
        ),
        checks,
    )


def build_coverage_report(
    *,
    records: list[AnswerRecord],
    specs: dict[str, CoverageSpec],
    spec_file: str,
    input_reports: list[str],
    llm_judge_results: dict[tuple[str, str | None, str], StrictCoverageVerdict] | None = None,
    llm_judge_model: str | None = None,
) -> BroadListCoverageReport:
    evaluations: list[CoverageEvaluation] = []
    for record in records:
        spec = specs.get(record.question_id)
        if spec is None:
            continue
        deterministic_verdict, checks = deterministic_coverage_verdict(
            record.prediction,
            spec,
        )
        llm_key = (record.question_id, record.answer_label, record.report_source)
        llm_verdict = (llm_judge_results or {}).get(llm_key)
        strict_verdict = llm_verdict or deterministic_verdict
        missing_in_selected = _items_present_in_context(
            deterministic_verdict.required_items_missing,
            spec,
            record.selected_context_text,
        )
        missing_in_evidence = _items_present_in_context(
            [
                item
                for item in deterministic_verdict.required_items_missing
                if item not in missing_in_selected
            ],
            spec,
            record.evidence_context_text,
        )
        evaluations.append(
            CoverageEvaluation(
                question_id=record.question_id,
                question_text=record.question_text,
                ground_truth=record.ground_truth,
                prediction=record.prediction,
                report_source=record.report_source,
                source_kind=record.source_kind,
                answer_label=record.answer_label,
                answer_model=record.answer_model,
                legacy_judge_score=record.legacy_judge_score,
                legacy_judge_reasoning=record.legacy_judge_reasoning,
                strict_coverage_score=strict_verdict.binary_score,
                deterministic_item_checks=checks,
                deterministic_verdict=deterministic_verdict,
                llm_judge_verdict=llm_verdict,
                llm_judge_model=llm_judge_model if llm_verdict is not None else None,
                failure_categories=_failure_categories(
                    record=record,
                    verdict=strict_verdict,
                    missing_in_selected=missing_in_selected,
                    missing_in_evidence=missing_in_evidence,
                    spec=spec,
                ),
                missing_items_selected_context=missing_in_selected,
                missing_items_evidence_context=missing_in_evidence,
                ground_truth_ambiguous=spec.ground_truth_ambiguous,
            )
        )

    legacy_correct = sum(1 for item in evaluations if item.legacy_judge_score == 1)
    strict_correct = sum(item.strict_coverage_score for item in evaluations)
    legacy_pass_strict_fail = sum(
        1
        for item in evaluations
        if item.legacy_judge_score == 1 and item.strict_coverage_score == 0
    )
    strict_pass_legacy_fail = sum(
        1
        for item in evaluations
        if item.legacy_judge_score == 0 and item.strict_coverage_score == 1
    )
    return BroadListCoverageReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        spec_file=spec_file,
        input_reports=input_reports,
        llm_judge_model=llm_judge_model,
        llm_judge_run=bool(llm_judge_results),
        total_answers=len(evaluations),
        legacy_total_correct=legacy_correct,
        strict_total_correct=strict_correct,
        legacy_pass_strict_fail=legacy_pass_strict_fail,
        strict_pass_legacy_fail=strict_pass_legacy_fail,
        results=evaluations,
    )


def load_specs(path: Path) -> dict[str, CoverageSpec]:
    raw = json.loads(path.read_text())
    specs = raw.get("questions", raw)
    return {
        spec.question_id: spec
        for spec in (CoverageSpec.model_validate(item) for item in specs)
    }


def load_locomo_records(path: Path) -> list[AnswerRecord]:
    data = json.loads(path.read_text())
    records: list[AnswerRecord] = []
    for conversation in data.get("conversations", []):
        for result in conversation.get("results", []):
            question = result.get("question") or {}
            trace = result.get("trace") or {}
            score_result = result.get("score_result") or {}
            records.append(
                AnswerRecord(
                    question_id=str(question.get("question_id") or ""),
                    question_text=str(question.get("question_text") or ""),
                    ground_truth=str(question.get("ground_truth") or ""),
                    prediction=str(result.get("prediction") or ""),
                    report_source=str(path),
                    source_kind="locomo_report",
                    answer_label=_locomo_answer_label(data),
                    answer_model=_model_info_value(data, "answer_model"),
                    legacy_judge_score=score_result.get("score"),
                    legacy_judge_reasoning=score_result.get("reasoning"),
                    selected_context_text=_context_text(
                        trace.get("selected_memory_summaries")
                    ),
                    evidence_context_text=_context_text(
                        trace.get("evidence_memory_summaries")
                    ),
                )
            )
    return records


def load_model_sensitivity_records(path: Path) -> list[AnswerRecord]:
    data = json.loads(path.read_text())
    contexts = data.get("contexts") or {}
    records: list[AnswerRecord] = []
    for result in data.get("results", []):
        question_id = str(result.get("question_id") or "")
        context = contexts.get(question_id) or {}
        judge = result.get("judge") or {}
        records.append(
            AnswerRecord(
                question_id=question_id,
                question_text=str(result.get("question_text") or context.get("question_text") or ""),
                ground_truth=str(result.get("ground_truth") or context.get("ground_truth") or ""),
                prediction=str(result.get("answer") or ""),
                report_source=str(path),
                source_kind="model_sensitivity_report",
                answer_label=str(result.get("tier") or ""),
                answer_model=result.get("answer_model"),
                legacy_judge_score=judge.get("score"),
                legacy_judge_reasoning=judge.get("reasoning"),
                selected_context_text=str(context.get("memory_block") or ""),
                evidence_context_text="",
            )
        )
    return records


async def run_llm_judge(
    *,
    records: list[AnswerRecord],
    specs: dict[str, CoverageSpec],
    judge_model: str,
) -> dict[tuple[str, str | None, str], StrictCoverageVerdict]:
    settings = Settings.from_env()
    client = build_llm_client(settings)
    judge = StrictCoverageJudge(client, judge_model)
    verdicts: dict[tuple[str, str | None, str], StrictCoverageVerdict] = {}
    for record in records:
        spec = specs.get(record.question_id)
        if spec is None:
            continue
        verdicts[(record.question_id, record.answer_label, record.report_source)] = (
            await judge.judge(
                question=record.question_text,
                prediction=record.prediction,
                ground_truth=record.ground_truth,
                required_items=spec.required_items,
            )
        )
    return verdicts


def _extract_json_payload(raw_output: str) -> Any | None:
    stripped = raw_output.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_for_item_match(value: str) -> str:
    return " ".join(
        value.casefold()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("-", " ")
        .split()
    )


def _items_present_in_context(
    item_labels: list[str],
    spec: CoverageSpec,
    context_text: str,
) -> list[str]:
    normalized_context = _normalize_for_item_match(context_text)
    by_label = {item.label: item for item in spec.required_items}
    present: list[str] = []
    for label in item_labels:
        item = by_label[label]
        if any(
            _normalize_for_item_match(term) in normalized_context
            for term in item.match_terms
        ):
            present.append(label)
    return present


def _failure_categories(
    *,
    record: AnswerRecord,
    verdict: StrictCoverageVerdict,
    missing_in_selected: list[str],
    missing_in_evidence: list[str],
    spec: CoverageSpec,
) -> list[str]:
    if verdict.binary_score == 1:
        return ["passed"]
    categories: list[str] = []
    if record.legacy_judge_score == 1:
        categories.append("legacy_judge_permissive")
    if missing_in_selected:
        categories.append("evidence_selected_but_omitted")
    if missing_in_evidence:
        categories.append("evidence_not_selected_or_not_retrieved")
    if verdict.unsupported_claims:
        categories.append("unsupported_claims")
    if spec.ground_truth_ambiguous:
        categories.append("ground_truth_ambiguity")
    missing_accounted_for = set(missing_in_selected) | set(missing_in_evidence)
    if any(item not in missing_accounted_for for item in verdict.required_items_missing):
        categories.append("evidence_not_retrieved_or_not_visible_in_context")
    if not categories:
        categories.append("unknown_without_context_artifact")
    return categories


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


def _model_info_value(data: dict[str, Any], key: str) -> str | None:
    model_info = data.get("model_info") or {}
    value = model_info.get(key)
    return str(value) if value is not None else None


def _locomo_answer_label(data: dict[str, Any]) -> str:
    answer_model = _model_info_value(data, "answer_model")
    return answer_model or "locomo_report"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate broad-list benchmark answers with strict item coverage."
    )
    parser.add_argument("--spec-file", required=True, help="Coverage spec JSON.")
    parser.add_argument(
        "--locomo-report",
        action="append",
        default=[],
        help="LoCoMo report JSON. Repeatable.",
    )
    parser.add_argument(
        "--model-sensitivity-report",
        action="append",
        default=[],
        help="Model sensitivity report JSON. Repeatable.",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Optional comma-separated question ids to include.",
    )
    parser.add_argument("--output", required=True, help="Output report JSON path.")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional strict LLM coverage judge model. If omitted, deterministic coverage only.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> BroadListCoverageReport:
    spec_path = Path(args.spec_file).expanduser()
    specs = load_specs(spec_path)
    allowed_questions = (
        {item.strip() for item in args.questions.split(",") if item.strip()}
        if args.questions
        else None
    )
    input_reports = [*args.locomo_report, *args.model_sensitivity_report]
    records: list[AnswerRecord] = []
    for raw_path in args.locomo_report:
        records.extend(load_locomo_records(Path(raw_path).expanduser()))
    for raw_path in args.model_sensitivity_report:
        records.extend(load_model_sensitivity_records(Path(raw_path).expanduser()))
    if allowed_questions is not None:
        records = [record for record in records if record.question_id in allowed_questions]
    llm_results = None
    if args.judge_model is not None:
        llm_results = await run_llm_judge(
            records=records,
            specs=specs,
            judge_model=args.judge_model,
        )
    return build_coverage_report(
        records=records,
        specs=specs,
        spec_file=str(spec_path),
        input_reports=input_reports,
        llm_judge_results=llm_results,
        llm_judge_model=args.judge_model,
    )


def main() -> None:
    args = _build_parser().parse_args()
    try:
        report = asyncio.run(_run_async(args))
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
    )
    print(
        f"Saved {report.total_answers} coverage evaluations to {output_path} "
        f"(legacy={report.legacy_total_correct}, strict={report.strict_total_correct})"
    )


if __name__ == "__main__":
    main()
