"""Tests for strict broad-list coverage evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.broad_list_coverage import (
    AnswerRecord,
    CoverageItem,
    CoverageSpec,
    StrictCoverageJudge,
    StrictCoverageVerdict,
    build_coverage_report,
    deterministic_coverage_verdict,
    load_locomo_records,
    parse_strict_coverage_verdict,
)
from atagia.services.llm_client import LLMCompletionResponse


def test_deterministic_coverage_requires_every_required_item() -> None:
    spec = CoverageSpec(
        question_id="synthetic-coverage:q1",
        required_items=[
            CoverageItem(label="amber sundial"),
            CoverageItem(label="folding beacon", aliases=["collapsible beacon"]),
            CoverageItem(label="ceramic weather sensor"),
        ],
    )

    verdict, checks = deterministic_coverage_verdict(
        "Mira completed the amber sundial and a folding beacon.",
        spec,
    )

    assert checks == {
        "amber sundial": True,
        "folding beacon": True,
        "ceramic weather sensor": False,
    }
    assert verdict.binary_score == 0
    assert verdict.required_items_missing == ["ceramic weather sensor"]


def test_report_separates_legacy_judge_from_strict_coverage() -> None:
    spec = CoverageSpec(
        question_id="fixture-broad-list",
        required_items=[
            CoverageItem(label="copper kite", aliases=["kite prototype"]),
            CoverageItem(label="maple lantern"),
        ],
    )
    record = AnswerRecord(
        question_id="fixture-broad-list",
        question_text="Which prototype exhibits did I finish this month?",
        ground_truth="I finished the copper kite and the maple lantern.",
        prediction="I finished the copper kite.",
        report_source="report.json",
        source_kind="unit",
        legacy_judge_score=1,
        legacy_judge_reasoning="Semantically close.",
        selected_context_text="The notes also mention the maple lantern.",
    )

    report = build_coverage_report(
        records=[record],
        specs={spec.question_id: spec},
        spec_file="spec.json",
        input_reports=["report.json"],
    )

    result = report.results[0]
    assert result.legacy_judge_score == 1
    assert result.strict_coverage_score == 0
    assert report.legacy_pass_strict_fail == 1
    assert result.missing_items_selected_context == ["maple lantern"]
    assert result.failure_categories == [
        "legacy_judge_permissive",
        "evidence_selected_but_omitted",
    ]


def test_parse_strict_coverage_verdict_accepts_embedded_json() -> None:
    verdict = parse_strict_coverage_verdict(
        'Result: {"required_items_present":["fair"],'
        '"required_items_missing":["dance competition"],'
        '"unsupported_claims":[],"binary_score":0,"rationale":"missing one"}'
    )

    assert verdict == StrictCoverageVerdict(
        required_items_present=["fair"],
        required_items_missing=["dance competition"],
        unsupported_claims=[],
        binary_score=0,
        rationale="missing one",
    )


def test_load_locomo_records_collects_selected_and_evidence_context(tmp_path: Path) -> None:
    report_path = tmp_path / "locomo-report.json"
    report_path.write_text(
        json.dumps(
            {
                "model_info": {"answer_model": "test-answer"},
                "conversations": [
                    {
                        "results": [
                            {
                                "question": {
                                    "question_id": "conv-1:q1",
                                    "question_text": "What?",
                                    "ground_truth": "A, B",
                                },
                                "prediction": "A",
                                "score_result": {
                                    "score": 1,
                                    "reasoning": "legacy pass",
                                },
                                "trace": {
                                    "selected_memory_summaries": [
                                        {"canonical_preview": "Selected fact B."}
                                    ],
                                    "evidence_memory_summaries": [
                                        {"canonical_preview": "Evidence fact C."}
                                    ],
                                },
                            }
                        ]
                    }
                ],
            }
        )
    )

    records = load_locomo_records(report_path)

    assert len(records) == 1
    assert records[0].answer_label == "test-answer"
    assert records[0].legacy_judge_score == 1
    assert records[0].selected_context_text == "Selected fact B."
    assert records[0].evidence_context_text == "Evidence fact C."


@pytest.mark.asyncio
async def test_strict_coverage_judge_uses_structured_json_contract() -> None:
    class FakeClient:
        async def complete(self, request):
            assert request.response_schema is not None
            assert request.metadata["purpose"] == "benchmark_broad_list_coverage_judge"
            return LLMCompletionResponse(
                provider="fake",
                model=request.model,
                output_text=json.dumps(
                    {
                        "required_items_present": ["fair"],
                        "required_items_missing": ["dance competition"],
                        "unsupported_claims": [],
                        "binary_score": 0,
                        "rationale": "missing one item",
                    }
                ),
            )

    judge = StrictCoverageJudge(FakeClient(), "anthropic/claude-opus-4-7")

    verdict = await judge.judge(
        question="Which events?",
        prediction="fair",
        ground_truth="fair, dance competition",
        required_items=[
            CoverageItem(label="fair"),
            CoverageItem(label="dance competition"),
        ],
    )

    assert verdict.binary_score == 0
    assert verdict.required_items_missing == ["dance competition"]
