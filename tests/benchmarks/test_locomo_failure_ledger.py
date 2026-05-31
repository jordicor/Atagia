"""Tests for offline LoCoMo failure custody ledgers."""

from __future__ import annotations

import json
from pathlib import Path

from atagia.services.answer_postcondition import DEFAULT_ABSTENTION_TEXT
from benchmarks.locomo.failure_ledger import (
    _first_loss_is_partial,
    _first_loss_stage,
    build_failure_ledger,
    format_failure_ledger_summary,
    parse_report_spec,
)


def test_first_loss_stage_uses_mechanical_critical_evidence_counts() -> None:
    assert _first_loss_stage({"critical_evidence_count": 0}) == (
        "critical_custody_unavailable"
    )
    assert _first_loss_stage(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 0,
            "scored_count": 0,
            "selected_count": 0,
        }
    ) == "raw_absent"
    assert _first_loss_is_partial(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 1,
            "scored_count": 0,
            "selected_count": 0,
        }
    )
    assert _first_loss_stage(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 3,
            "scored_count": 1,
            "selected_count": 0,
        }
    ) == "raw_unscored"
    assert _first_loss_stage(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 3,
            "scored_count": 3,
            "selected_count": 0,
        }
    ) == "scored_unselected"
    assert _first_loss_is_partial(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 3,
            "scored_count": 3,
            "selected_count": 2,
        }
    )
    assert _first_loss_stage(
        {
            "critical_evidence_count": 3,
            "raw_candidate_count": 3,
            "scored_count": 3,
            "selected_count": 3,
        }
    ) == "critical_selected_all"


def test_failure_ledger_filters_report_conversations_and_excludes_text(tmp_path: Path) -> None:
    report_path = tmp_path / "locomo-report.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark_name": "LoCoMo",
                "timestamp": "2026-05-23T00:00:00+00:00",
                "total_questions": 3,
                "total_correct": 1,
                "overall_accuracy": 1 / 3,
                "model_info": {
                    "provider": "openrouter",
                    "answer_model": "openrouter/openai/gpt-chat-latest",
                    "judge_model": "openrouter/anthropic/claude-opus-4.7",
                    "activation_flags": {
                        "answer_postcondition_guard_enabled": True,
                    },
                    "internal_debug_blob": "drop me",
                },
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "results": [
                            _result(
                                "conv-a:q1",
                                score=0,
                                prediction=DEFAULT_ABSTENTION_TEXT,
                                trace=_trace(
                                    critical_counts={
                                        "critical_evidence_count": 2,
                                        "raw_candidate_count": 0,
                                        "scored_count": 0,
                                        "selected_count": 0,
                                        "absent_count": 2,
                                    },
                                    guard_status="abstained",
                                    shadow_state="retrieval_sufficient",
                                ),
                            ),
                            _result("conv-a:q2", score=1),
                        ],
                    },
                    {
                        "conversation_id": "conv-b",
                        "results": [_result("conv-b:q1", score=0)],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    ledger = build_failure_ledger([parse_report_spec(f"{report_path}::conv-a")])

    compacted_model_info = ledger["source_reports"][0]["model_info"]
    assert compacted_model_info["activation_flags"] == {
        "answer_postcondition_guard_enabled": True,
    }
    assert "internal_debug_blob" not in compacted_model_info
    assert ledger["source_result_summary"]["total_questions"] == 2
    assert ledger["source_result_summary"]["total_correct"] == 1
    assert ledger["ledger_summary"]["item_count"] == 1
    assert ledger["ledger_summary"]["failed_by_category_name"] == {"single-hop": 1}
    item = ledger["items"][0]
    assert item["conversation_id"] == "conv-a"
    assert item["question_id"] == "conv-a:q1"
    assert item["critical_evidence"]["first_loss_stage"] == "raw_absent"
    assert item["shadow_sufficiency"]["false_positive_zero_raw_critical"] is True
    assert item["shadow_sufficiency"]["false_positive_zero_selected_critical"] is True
    assert "known_hard_abstention_text" in item["diagnostic_labels"]
    assert "RAW QUESTION TEXT" not in json.dumps(ledger)
    assert "RAW GROUND TRUTH" not in json.dumps(ledger)
    assert "RAW PREDICTION" not in json.dumps(ledger)
    assert format_failure_ledger_summary(ledger).startswith("LoCoMo failure ledger:")


def test_failure_ledger_tracks_summary_only_selected_support(tmp_path: Path) -> None:
    report_path = tmp_path / "locomo-report.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark_name": "LoCoMo",
                "timestamp": "2026-05-23T00:00:00+00:00",
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "results": [
                            _result(
                                "conv-a:q1",
                                score=0,
                                trace=_trace(
                                    critical_counts={
                                        "critical_evidence_count": 1,
                                        "raw_candidate_count": 1,
                                        "scored_count": 1,
                                        "selected_count": 1,
                                        "absent_count": 0,
                                    },
                                    critical_items=[
                                        {
                                            "memory_id": "sum_1",
                                            "object_type": "summary_view",
                                            "candidate_kind": "summary_view",
                                            "source_kind": "summarized",
                                            "selected": True,
                                        }
                                    ],
                                    retrieval_custody=[
                                        {
                                            "candidate_id": "sum_1",
                                            "candidate_kind": "summary_view",
                                            "source_kind": "summarized",
                                            "channels": ["fts"],
                                            "selected": True,
                                            "composer_decision": "selected",
                                        }
                                    ],
                                ),
                            )
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ledger = build_failure_ledger([report_path])

    item = ledger["items"][0]
    assert item["critical_evidence"]["first_loss_stage"] == "critical_selected_all"
    assert item["critical_evidence"]["selected_summary_only"] is True
    assert item["retrieval"]["selected_summary_only"] is True
    assert item["retrieval"]["selected_candidate_kind_counts"] == {"summary_view": 1}
    assert "critical_selected_summary_only" in item["diagnostic_labels"]
    assert "retrieval_selected_summary_only" in item["diagnostic_labels"]


def _result(
    question_id: str,
    *,
    score: int,
    prediction: str = "RAW PREDICTION",
    trace: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "question": {
            "question_id": question_id,
            "question_text": "RAW QUESTION TEXT",
            "ground_truth": "RAW GROUND TRUTH",
            "category": 1,
            "evidence_turn_ids": ["D1:1"],
        },
        "prediction": prediction,
        "score_result": {
            "score": score,
            "reasoning": "RAW JUDGE REASON",
            "judge_model": "judge",
        },
        "memories_used": 1,
        "retrieval_time_ms": 12.0,
        "trace": trace or _trace(),
    }


def _trace(
    *,
    critical_counts: dict[str, int] | None = None,
    critical_items: list[dict[str, object]] | None = None,
    retrieval_custody: list[dict[str, object]] | None = None,
    guard_status: str = "passed",
    shadow_state: str = "retrieval_insufficient",
) -> dict[str, object]:
    return {
        "diagnosis_bucket": "answer_policy_or_grading",
        "sufficiency_diagnostic": "answer_or_judge_issue",
        "benchmark_privacy_enforcement": "off",
        "benchmark_answer_privacy_override": True,
        "benchmark_high_risk_secret_redaction_disabled": True,
        "evidence_turn_ids": ["D1:1"],
        "missing_evidence_turn_ids": [],
        "evidence_memory_count": 1,
        "active_evidence_count": 1,
        "selected_memory_ids": ["mem_1"],
        "selected_evidence_memory_ids": ["mem_1"],
        "critical_evidence_custody": {
            "counts": critical_counts
            or {
                "critical_evidence_count": 1,
                "raw_candidate_count": 1,
                "scored_count": 1,
                "selected_count": 1,
                "absent_count": 0,
            },
            "survival_stage_counts": {"selected": 1},
            "items": critical_items
            or [
                {
                    "memory_id": "mem_1",
                    "object_type": "evidence",
                    "candidate_kind": "evidence",
                    "source_kind": "extracted",
                    "selected": True,
                }
            ],
        },
        "retrieval_custody": retrieval_custody
        or [
            {
                "candidate_id": "mem_1",
                "candidate_kind": "evidence",
                "source_kind": "extracted",
                "channels": ["fts"],
                "selected": True,
                "composer_decision": "selected",
            }
        ],
        "retrieval_trace": {
            "custody": {
                "candidate_count_by_channel": {"fts": 1},
                "post_user_id_candidate_count": 1,
                "post_scope_coordinate_lifecycle_candidate_count": 1,
                "scored_candidate_count": 1,
                "selected_candidate_count": 1,
                "drop_counts_by_stage": {},
                "drop_counts_by_reason": {},
            },
            "runtime_diagnostics": {
                "db_query_count": 2,
                "db_query_count_by_operation": {"SELECT": 2},
                "hydration_timings_ms": {"context_composition": 3.5},
                "stage_timings_ms": {"candidate_search": 4.5},
                "lock_wait_count": 0,
                "sqlite_busy_count": 0,
            },
        },
        "answer_postcondition_guard": {
            "enabled": True,
            "status": guard_status,
            "retry_count": 0,
            "evidence_use_repair_count": 0,
            "evidence_use_repair_success_count": 0,
            "evidence_use_repair_failure_count": 0,
            "verdict": {
                "is_abstention": guard_status == "abstained",
                "pass_postcondition": guard_status == "passed",
            },
        },
        "shadow_sufficiency_diagnostics": {
            "state": shadow_state,
            "confidence": 0.8,
            "candidate_count": 1,
        },
        "llm_call_summary": {
            "total_calls": 1,
            "failed_calls": 0,
            "token_totals": {"total_tokens": 10},
            "model_call_counts": {"model": 1},
        },
    }
