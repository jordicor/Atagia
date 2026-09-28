"""Tests for score-ordered lane fusion in candidate merging."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.candidate_diversity import early_diversity_select
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import RetrievalPlan
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.retrieval_pipeline import RetrievalPipeline

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"


class _StubProvider(LLMProvider):
    name = "stub-never-called"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError("Pool ordering must not issue LLM calls")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Pool ordering must not issue LLM calls")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


def _scorer() -> ApplicabilityScorer:
    return ApplicabilityScorer(
        llm_client=LLMClient(provider_name=_StubProvider.name, providers=[_StubProvider()]),
        clock=FrozenClock(datetime(2026, 3, 30, 21, 0, tzinfo=timezone.utc)),
        settings=_settings(),
    )


def _resolved_policy(mode_id: str = "personal_assistant"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _plan() -> RetrievalPlan:
    fts_queries = ["safe outage fix"]
    return RetrievalPlan(
        assistant_mode_id="personal_assistant",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=fts_queries,
        sub_query_plans=[
            {
                "text": fts_queries[0],
                "fts_queries": fts_queries,
            }
        ],
        query_type="default",
        scope_filter=[],
        status_filter=[],
        vector_limit=0,
        max_candidates=30,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
    )


def _candidate(
    memory_id: str,
    *,
    rrf_score: float,
    object_type: str = "evidence",
) -> dict[str, object]:
    return {
        "id": memory_id,
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "assistant_mode_id": "personal_assistant",
        "platform_id": "default",
        "platform_locked": 0,
        "platform_id_lock": None,
        "object_type": object_type,
        "scope": "conversation",
        "scope_canonical": "chat",
        "canonical_text": f"Candidate text for {memory_id}.",
        "payload_json": {},
        "privacy_level": 0,
        "sensitivity": "public",
        "temporal_type": "unknown",
        "valid_from": None,
        "valid_to": None,
        "status": "active",
        "updated_at": "2026-03-30T21:00:00+00:00",
        "rrf_score": rrf_score,
        "retrieval_sources": ["fts"],
    }


def test_fused_merge_admits_enriched_only_top_evidence_into_shortlist() -> None:
    # Regression shape from the retrieval audit: a candidate found only by the
    # enriched lane used to sit behind the whole base lane (25 candidates) and
    # never reached the rerank_top_k=15 shortlist.
    base = [
        _candidate(f"mem_base_{index}", rrf_score=0.50 - (index * 0.01))
        for index in range(25)
    ]
    enriched = [_candidate("mem_enriched_only_best", rrf_score=0.90)]

    merged = RetrievalPipeline._merge_candidates(base, enriched)

    assert merged[0]["id"] == "mem_enriched_only_best"
    shortlist = early_diversity_select(
        merged,
        query_type="default",
        shortlist_k=15,
    )
    assert "mem_enriched_only_best" in {candidate["id"] for candidate in shortlist}


def test_fused_merge_orders_union_by_retrieval_score() -> None:
    base = [
        _candidate("mem_base_low", rrf_score=0.20),
        _candidate("mem_base_high", rrf_score=0.80),
    ]
    enriched = [
        _candidate("mem_enriched_mid", rrf_score=0.50),
        _candidate("mem_enriched_best", rrf_score=0.90),
    ]

    merged = RetrievalPipeline._merge_candidates(base, enriched)

    assert [candidate["id"] for candidate in merged] == [
        "mem_enriched_best",
        "mem_base_high",
        "mem_enriched_mid",
        "mem_base_low",
    ]


def test_fused_merge_keeps_base_first_order_on_equal_scores() -> None:
    base = [
        _candidate("mem_base_1", rrf_score=0.40),
        _candidate("mem_base_2", rrf_score=0.40),
    ]
    enriched = [
        _candidate("mem_enriched_1", rrf_score=0.40),
        _candidate("mem_base_1", rrf_score=0.40, object_type="belief"),
    ]

    merged = RetrievalPipeline._merge_candidates(base, enriched)

    assert [candidate["id"] for candidate in merged] == [
        "mem_base_1",
        "mem_base_2",
        "mem_enriched_1",
    ]


def test_fused_merge_uses_explicit_subquery_coverage_to_break_score_ties() -> None:
    base = [_candidate("mem_base", rrf_score=0.60)]
    enriched = [_candidate("mem_enriched", rrf_score=0.60)]
    base[0]["rrf_subquery_coverage"] = 0.2
    enriched[0]["rrf_subquery_coverage"] = 0.8

    merged = RetrievalPipeline._merge_candidates(base, enriched)

    assert [candidate["id"] for candidate in merged] == [
        "mem_enriched",
        "mem_base",
    ]


def test_fused_merge_records_lane_provenance() -> None:
    base = [_candidate("mem_shared", rrf_score=0.30)]
    enriched = [
        _candidate("mem_shared", rrf_score=0.70),
        _candidate("mem_enriched_only", rrf_score=0.60),
    ]

    merged = RetrievalPipeline._merge_candidates(base, enriched)

    by_id = {candidate["id"]: candidate for candidate in merged}
    assert by_id["mem_shared"]["retrieval_lanes"] == ["base", "enriched"]
    assert by_id["mem_enriched_only"]["retrieval_lanes"] == ["enriched"]
    # The higher-scoring enriched view still wins the dedupe.
    assert by_id["mem_shared"]["rrf_score"] == 0.7


def test_merge_single_lane_early_returns_tag_provenance() -> None:
    base = [_candidate("mem_base", rrf_score=0.30)]

    merged_base = RetrievalPipeline._merge_candidates(base, [])
    merged_enriched = RetrievalPipeline._merge_candidates(
        [], [_candidate("mem_enriched", rrf_score=0.30)]
    )

    assert merged_base[0]["retrieval_lanes"] == ["base"]
    assert merged_enriched[0]["retrieval_lanes"] == ["enriched"]


def test_pool_order_is_identical_across_privacy_modes() -> None:
    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline._scorer = _scorer()
    resolved_policy = _resolved_policy()
    retrieval_plan = _plan()
    # More candidates than the preservation quota, distinct scores, and a
    # single object type so preferred-type ordering cannot mask differences.
    candidates = [
        _candidate(f"mem_{index:02d}", rrf_score=1.0 - (index * 0.05))
        for index in range(14)
    ]

    off_order = pipeline._filter_candidates_for_policy_mode(
        list(candidates),
        resolved_policy,
        [],
        retrieval_plan=retrieval_plan,
        privacy_enforcement="off",
    )
    enforce_order = pipeline._filter_candidates_for_policy_mode(
        list(candidates),
        resolved_policy,
        [],
        retrieval_plan=retrieval_plan,
        privacy_enforcement="enforce",
    )

    off_ids = [str(candidate["id"]) for candidate in off_order]
    enforce_ids = [str(candidate["id"]) for candidate in enforce_order]
    assert off_ids == enforce_ids
    # The top-retrieval-score preservation quota leads the pool in both modes.
    assert off_ids[:10] == [f"mem_{index:02d}" for index in range(10)]
    # Pure reorder: no candidate is dropped in privacy-off mode.
    assert sorted(off_ids) == sorted(str(candidate["id"]) for candidate in candidates)


@pytest.mark.asyncio
async def test_shared_shortlist_helper_applies_optional_fused_guard_ordering() -> None:
    class GuardOrderingStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        async def order_candidates_with_guards(
            self,
            candidates: list[dict[str, object]],
            _plan: RetrievalPlan,
            *,
            user_id: str,
        ) -> list[dict[str, object]]:
            self.calls.append(
                (user_id, [str(candidate["id"]) for candidate in candidates])
            )
            return list(reversed(candidates))

        @staticmethod
        def candidate_guard_priority(
            _candidate: dict[str, object],
            _plan: RetrievalPlan,
        ) -> tuple[int, int]:
            return (0, 0)

    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline._settings = replace(
        _settings(),
        fused_candidate_guard_ordering_enabled=True,
    )
    guard_ordering = GuardOrderingStub()
    pipeline._candidate_search = guard_ordering
    candidates = [
        _candidate("mem_first", rrf_score=0.5),
        _candidate("mem_guarded", rrf_score=0.5),
    ]

    shortlist = await pipeline._select_scoring_shortlist(
        candidates,
        retrieval_plan=_plan(),
        shortlist_k=1,
        user_id="usr_1",
    )

    assert guard_ordering.calls == [
        ("usr_1", ["mem_first", "mem_guarded"]),
    ]
    assert [candidate["id"] for candidate in shortlist] == ["mem_guarded"]


@pytest.mark.asyncio
async def test_broad_list_diversity_cannot_cross_fused_guard_buckets() -> None:
    class GuardOrderingStub:
        async def order_candidates_with_guards(
            self,
            candidates: list[dict[str, object]],
            _plan: RetrievalPlan,
            *,
            user_id: str,
        ) -> list[dict[str, object]]:
            assert user_id == "usr_1"
            return candidates

        @staticmethod
        def candidate_guard_priority(
            candidate: dict[str, object],
            _plan: RetrievalPlan,
        ) -> tuple[int, int]:
            return (1, 0) if candidate["id"] == "mem_stale" else (0, 0)

    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline._settings = replace(
        _settings(),
        fused_candidate_guard_ordering_enabled=True,
    )
    pipeline._candidate_search = GuardOrderingStub()
    candidates = [
        _candidate("mem_fresh_1", rrf_score=0.9),
        _candidate("mem_fresh_2", rrf_score=0.8),
        _candidate("mem_stale", rrf_score=0.7),
    ]
    candidates[0]["canonical_text"] = "same fresh callback topic alpha"
    candidates[1]["canonical_text"] = "same fresh callback topic beta"
    candidates[2]["canonical_text"] = "unrelated stale cluster omega"
    broad_plan = _plan().model_copy(update={"query_type": "broad_list"})

    shortlist = await pipeline._select_scoring_shortlist(
        candidates,
        retrieval_plan=broad_plan,
        shortlist_k=2,
        user_id="usr_1",
    )

    assert [candidate["id"] for candidate in shortlist] == [
        "mem_fresh_1",
        "mem_fresh_2",
    ]


def test_fused_merge_demotes_coverage_siblings_below_retrieved_evidence() -> None:
    # Review finding: coverage siblings used to inherit neighbor scores up to
    # 0.70 as their own rrf_score and flood the score-ordered shortlist,
    # displacing genuinely retrieved evidence.
    base = [
        _candidate(f"mem_real_{index}", rrf_score=0.50 - (index * 0.01))
        for index in range(15)
    ]
    coverage = [
        RetrievalPipeline._annotate_source_message_coverage_candidate(
            _candidate(f"mem_coverage_{index}", rrf_score=0.0),
            source_message_id="msg_rank1",
            source_metadata={
                "rrf_score": 1.0 - (index * 0.05),
                "matched_sub_queries": [],
            },
        )
        for index in range(12)
    ]

    merged = RetrievalPipeline._merge_candidates(
        base,
        coverage,
        lane_labels=("base", "coverage"),
    )

    merged_ids = [str(candidate["id"]) for candidate in merged]
    assert merged_ids[:15] == [f"mem_real_{index}" for index in range(15)]
    assert set(merged_ids[15:]) == {f"mem_coverage_{index}" for index in range(12)}
    shortlist = early_diversity_select(
        merged,
        query_type="default",
        shortlist_k=15,
    )
    assert {str(candidate["id"]) for candidate in shortlist} == {
        f"mem_real_{index}" for index in range(15)
    }
    # The inherited neighbor score survives as provenance metadata, never as
    # the candidate's own retrieval score.
    coverage_by_id = {str(candidate["id"]): candidate for candidate in merged[15:]}
    assert coverage_by_id["mem_coverage_0"]["coverage_inherited_score"] == 0.7
    assert coverage_by_id["mem_coverage_0"]["rrf_score"] == 0.0
