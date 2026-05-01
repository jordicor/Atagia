"""Tests for shared benchmark retrieval custody records."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.custody_summary import (
    format_retrieval_custody_summary,
    summarize_retrieval_custody,
)
from benchmarks.retrieval_custody import build_retrieval_custody


def test_build_retrieval_custody_records_candidate_route_and_decisions() -> None:
    pipeline_result = SimpleNamespace(
        raw_candidates=[
            {
                "id": "mem_selected",
                "object_type": "evidence",
                "channel_ranks": {"fts": 1, "embedding": 3},
                "matched_sub_queries": ["project budget"],
                "rrf_score": 0.75,
            },
            {
                "id": "raw_1",
                "is_verbatim_evidence_window": True,
                "channel_ranks": {},
                "rrf_score": 0.1,
            },
        ],
        scored_candidates=[
            SimpleNamespace(
                memory_id="mem_selected",
                llm_applicability=0.9,
                retrieval_score=0.7,
                vitality_boost=0.0,
                confirmation_boost=0.0,
                need_boost=0.1,
                penalty=0.0,
                final_score=0.73,
            )
        ],
    )

    custody = build_retrieval_custody(
        pipeline_result=pipeline_result,
        selected_memory_ids=["mem_selected"],
        user_id="usr_benchmark",
    )

    assert custody[0]["candidate_id"] == "mem_selected"
    assert custody[0]["candidate_kind"] == "evidence"
    assert custody[0]["channels"] == ["embedding", "fts"]
    assert custody[0]["channel_ranks"] == {"fts": 1, "embedding": 3}
    assert custody[0]["matched_sub_queries"] == ["project budget"]
    assert custody[0]["scored"] is True
    assert custody[0]["scorer"]["llm_applicability"] == 0.9
    assert custody[0]["composer_decision"] == "selected"
    assert custody[0]["user_id"] == "usr_benchmark"
    assert custody[1]["candidate_kind"] == "verbatim_evidence_search_window"
    assert custody[1]["channels"] == ["verbatim_evidence_search"]
    assert custody[1]["filter_reason"] == "not_scored_or_filtered"
    assert custody[1]["composer_decision"] == "not_selected"


def test_build_retrieval_custody_prefers_core_candidate_custody() -> None:
    pipeline_result = SimpleNamespace(
        raw_candidates=[],
        scored_candidates=[],
        candidate_custody=[
            {
                "schema_version": 2,
                "candidate_id": "mem_1",
                "candidate_kind": "evidence",
                "selected": True,
            }
        ],
    )

    custody = build_retrieval_custody(
        pipeline_result=pipeline_result,
        selected_memory_ids=[],
        user_id="usr_benchmark",
        retrieval_event_id="ret_1",
    )

    assert custody == [
        {
            "schema_version": 2,
            "candidate_id": "mem_1",
            "candidate_kind": "evidence",
            "selected": True,
            "user_id": "usr_benchmark",
            "retrieval_event_id": "ret_1",
        }
    ]


def test_summarize_retrieval_custody_counts_channels_and_selected() -> None:
    summary = summarize_retrieval_custody(
        [
            [
                {
                    "candidate_kind": "evidence",
                    "channels": ["embedding", "fts"],
                    "composer_decision": "selected",
                    "selected": True,
                },
                {
                    "candidate_kind": "verbatim_evidence_search_window",
                    "channels": ["verbatim_evidence_search"],
                    "composer_decision": "not_selected",
                    "filter_reason": "not_scored_or_filtered",
                    "selected": False,
                },
            ]
        ]
    )

    assert summary == {
        "candidate_count": 2,
        "selected_count": 1,
        "channel_counts": {"embedding": 1, "fts": 1, "verbatim_evidence_search": 1},
        "selected_channel_counts": {"embedding": 1, "fts": 1},
        "candidate_kind_counts": {"evidence": 1, "verbatim_evidence_search_window": 1},
        "composer_decision_counts": {"not_selected": 1, "selected": 1},
        "filter_reason_counts": {"not_scored_or_filtered": 1},
    }


def test_format_retrieval_custody_summary_is_terminal_friendly() -> None:
    summary = {
        "candidate_count": 2,
        "selected_count": 1,
        "channel_counts": {"fts": 2},
        "selected_channel_counts": {"fts": 1},
        "candidate_kind_counts": {"memory": 2},
        "composer_decision_counts": {"selected": 1, "not_selected": 1},
        "filter_reason_counts": {"not_scored_or_filtered": 1},
    }

    assert format_retrieval_custody_summary(summary) == (
        "Retrieval custody: candidates=2 selected=1 "
        "channels=fts=2 selected_channels=fts=1 "
        "kinds=memory=2 "
        "decisions=not_selected=1 selected=1 "
        "filters=not_scored_or_filtered=1"
    )
