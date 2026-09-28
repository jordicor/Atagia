"""Tests for the LoCoMo rejudge tool (CS-0.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.base import BenchmarkConversation, BenchmarkTurn
from benchmarks.locomo import rejudge
from benchmarks.locomo.night_run_artifacts import QuestionRecord
from benchmarks.locomo.rejudge import (
    RejudgeConfig,
    _Verdict,
    _actual_cost,
    _aggregate,
    _e1b_decomposition,
    _expected_ordering,
    _memory_error_counters,
    _needs_llm,
    _stored_verdict,
    _verdict_flips,
    run_rejudge,
)
from benchmarks.scorer import JudgeProtocol


def _rec(qid: str, category: int, strict: int) -> QuestionRecord:
    return QuestionRecord(
        conversation_id="conv-x",
        question_id=qid,
        category=category,
        question_text="q",
        ground_truth="g",
        evidence_turn_ids=[],
        prediction="p",
        strict_score=strict,
        strict_reasoning="stored",
        strict_judge_model="kimi",
    )


def test_needs_llm_per_protocol() -> None:
    passed = _rec("a", 1, 1)
    failed = _rec("b", 1, 0)
    assert _needs_llm(passed, JudgeProtocol.SOURCE_AWARE_STRICT) is False
    assert _needs_llm(failed, JudgeProtocol.SOURCE_AWARE_STRICT) is False
    assert _needs_llm(passed, JudgeProtocol.GOLD_ONLY_LENIENT) is True
    assert _needs_llm(failed, JudgeProtocol.GOLD_ONLY_LENIENT) is True
    # Two-stage: only strict failures are re-judged for memory_quality.
    assert _needs_llm(passed, JudgeProtocol.MEMORY_QUALITY) is False
    assert _needs_llm(failed, JudgeProtocol.MEMORY_QUALITY) is True


def test_stored_verdict_strict_pass_is_memory_quality_pass() -> None:
    verdict = _stored_verdict(_rec("a", 1, 1), JudgeProtocol.MEMORY_QUALITY)
    assert verdict.score == 1
    assert verdict.rejudged is False
    assert verdict.missing_info is False


def test_aggregate_and_ordering() -> None:
    verdicts = [
        _Verdict("a", "c", 1, 0, "memory_quality", 1, "r"),
        _Verdict("b", "c", 1, 1, "memory_quality", 1, "r"),
        _Verdict("c", "c", 2, 0, "memory_quality", 0, "r"),
    ]
    strict = _aggregate(
        [
            _Verdict("a", "c", 1, 0, "source_aware_strict", 0, "r"),
            _Verdict("b", "c", 1, 1, "source_aware_strict", 1, "r"),
            _Verdict("c", "c", 2, 0, "source_aware_strict", 0, "r"),
        ]
    )
    proto = _aggregate(verdicts)
    assert strict["passed"] == 1
    assert proto["passed"] == 2
    ordering = _expected_ordering(strict, proto, JudgeProtocol.MEMORY_QUALITY)
    assert ordering["satisfied"] is True


def test_verdict_flips() -> None:
    verdicts = [
        _Verdict("a", "c", 1, 0, "memory_quality", 1, "now pass"),
        _Verdict("b", "c", 1, 1, "memory_quality", 0, "now fail"),
    ]
    flips = _verdict_flips(verdicts)
    assert flips["strict_fail_to_protocol_pass"]["count"] == 1
    assert flips["strict_pass_to_protocol_fail"]["count"] == 1


def test_e1b_decomposition() -> None:
    verdicts = [
        # true extra (memory_quality pass, no missing info, true_addition_only)
        _Verdict(
            "a", "c", 1, 0, "memory_quality", 1, "true extra",
            missing_info=False, true_addition_only=True, rejudged=True,
        ),
        # false addition (hallucination)
        _Verdict(
            "b", "c", 1, 0, "memory_quality", 0, "false extra",
            missing_info=False, false_addition=True, rejudged=True,
        ),
        # misattribution
        _Verdict(
            "c", "c", 2, 0, "memory_quality", 0, "wrong person",
            missing_info=False, misattribution=True, rejudged=True,
        ),
        # omission -> NOT in the addition-flagged pool
        _Verdict(
            "d", "c", 1, 0, "memory_quality", 0, "missing",
            missing_info=True, rejudged=True,
        ),
    ]
    audit = _e1b_decomposition(verdicts)
    assert audit["addition_flagged_pool"] == 3
    assert audit["overall"]["true_addition_only"] == 1
    assert audit["overall"]["false_addition"] == 1
    assert audit["overall"]["misattribution"] == 1
    assert audit["overall"]["none_flagged"] == 0
    assert audit["overall"]["multi_flagged"] == 0


def test_e1b_pool_arithmetic_transparency() -> None:
    """none_flagged + multi_flagged make the pool arithmetic auditable."""
    verdicts = [
        # Double-flagged: counted in both false_addition and misattribution.
        _Verdict(
            "a", "c", 1, 0, "memory_quality", 0, "false and crossed",
            missing_info=False, false_addition=True, misattribution=True,
            rejudged=True,
        ),
        # None-flagged: memory_quality pass with no extras at all (plain
        # strict-judge disagreement).
        _Verdict(
            "b", "c", 1, 0, "memory_quality", 1, "strict was wrong",
            missing_info=False, false_addition=False, misattribution=False,
            true_addition_only=False, rejudged=True,
        ),
    ]
    audit = _e1b_decomposition(verdicts)
    overall = audit["overall"]
    assert overall["pool"] == 2
    assert overall["false_addition"] == 1
    assert overall["misattribution"] == 1
    assert overall["multi_flagged"] == 1
    assert overall["none_flagged"] == 1
    # pool == distinct flagged (2 bucket counts - 1 overlap) + none_flagged.
    distinct_flagged = (
        overall["true_addition_only"]
        + overall["false_addition"]
        + overall["misattribution"]
        - overall["multi_flagged"]
    )
    assert distinct_flagged + overall["none_flagged"] == overall["pool"]


def test_memory_error_counters() -> None:
    verdicts = [
        _Verdict("a", "c", 1, 0, "memory_quality", 0, "r", false_addition=True),
        _Verdict("b", "c", 1, 0, "memory_quality", 0, "r", misattribution=True),
        _Verdict("c", "c", 2, 1, "memory_quality", 1, "r"),
    ]
    counters = _memory_error_counters(verdicts, total_questions=3)
    assert counters["false_addition_count"] == 1
    assert counters["misattribution_count"] == 1
    assert counters["false_addition_rate"] == round(1 / 3, 4)


def test_actual_cost_uses_cached_price() -> None:
    records = [
        {"token_counts": {"input_tokens": 1_000_000, "cached_input_tokens": 900_000, "output_tokens": 100_000}}
    ]
    config = RejudgeConfig(report_specs=[], protocol=JudgeProtocol.MEMORY_QUALITY)
    cost = _actual_cost(records, config)
    # 100k non-cached at the input rate + 900k at the cached rate + 100k output.
    expected = (
        0.1 * config.input_price
        + 0.9 * config.cached_input_price
        + 0.1 * config.output_price
    )
    assert cost["cost_usd"] == round(expected, 4)
    assert cost["cached_input_tokens"] == 900_000
    # The cached rate must actually be cheaper for the split to matter.
    assert config.cached_input_price < config.input_price


def test_resolve_report_paths_prefers_recovery(tmp_path: Path) -> None:
    base = tmp_path / "locomo_conv-990001"
    recovery = tmp_path / "locomo_conv-990001_recovery_eval"
    base.mkdir()
    recovery.mkdir()
    (base / "locomo-report-20260101T000000Z.json").write_text("{}", encoding="utf-8")
    (recovery / "locomo-report-20260102T000000Z.json").write_text("{}", encoding="utf-8")
    resolved = rejudge.resolve_report_paths([tmp_path])
    assert len(resolved) == 1
    assert resolved[0].parent.name == "locomo_conv-990001_recovery_eval"


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        payload = {
            "verdict": 1,
            "missing_info": False,
            "false_addition": False,
            "misattribution": False,
            "true_addition_only": True,
            "reasoning": "true extra",
        }
        return LLMCompletionResponse(
            provider="kimi",
            model="kimi-k2.7-code-highspeed",
            output_text=json.dumps(payload),
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


@pytest.mark.asyncio
async def test_run_rejudge_memory_quality_two_stage(tmp_path: Path, monkeypatch) -> None:
    report = {
        "conversations": [
            {
                "conversation_id": "conv-x",
                "results": [
                    {
                        "question": {
                            "question_id": "conv-x:q1",
                            "category": 1,
                            "question_text": "q1",
                            "ground_truth": "g1",
                            "evidence_turn_ids": [],
                        },
                        "prediction": "p1",
                        "score_result": {"score": 1, "reasoning": "pass", "judge_model": "kimi"},
                        "trace": {},
                    },
                    {
                        "question": {
                            "question_id": "conv-x:q2",
                            "category": 1,
                            "question_text": "q2",
                            "ground_truth": "g2",
                            "evidence_turn_ids": [],
                        },
                        "prediction": "p2 plus extra",
                        "score_result": {"score": 0, "reasoning": "fail", "judge_model": "kimi"},
                        "trace": {},
                    },
                ],
            }
        ]
    }
    report_path = tmp_path / "locomo_conv-x" / "locomo-report-20260101T000000Z.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    fake_client = _FakeClient()
    conversation = BenchmarkConversation(
        conversation_id="conv-x",
        turns=[
            BenchmarkTurn(
                role="user",
                text="hello",
                speaker="Alice",
                session_id="session_1",
                timestamp="2023-01-01T00:00:00",
                turn_id="D97:9701",
            )
        ],
        questions=[],
    )
    monkeypatch.setattr(rejudge, "build_llm_client", lambda settings: fake_client)
    monkeypatch.setattr(rejudge, "install_llm_call_recorder", lambda client, recorder: None)
    monkeypatch.setattr(
        rejudge, "load_conversations", lambda data_path: {"conv-x": conversation}
    )

    config = RejudgeConfig(
        report_specs=[str(report_path)],
        protocol=JudgeProtocol.MEMORY_QUALITY,
    )
    result = await run_rejudge(config)

    # Only the strict FAILURE was re-judged (two-stage).
    assert fake_client.calls == 1
    assert result["strict_baseline"]["passed"] == 1
    assert result["protocol_result"]["passed"] == 2
    assert result["coverage"]["verdicts_produced"] == 2
    # The strict-pass verdict was reused without an LLM call.
    reused = [v for v in result["verdicts"] if v["question_id"] == "conv-x:q1"][0]
    assert reused["rejudged"] is False
